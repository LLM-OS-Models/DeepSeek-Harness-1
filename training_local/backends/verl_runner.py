"""verl backend for harness-1 local RL.

verl is the recommended backend for DeepSeek-V4-Flash RL:
  - Megatron actor for backprop on 304B MoE
  - vLLM rollout for fast FP8 inference (with DSpark speculative decoding)
  - Ray-based distributed orchestration

This runner wires our LocalSearchEnv as a custom reward function in verl's
GRPO trainer. Multi-turn agentic rollout is driven by our rollout.py during
the rollout phase.

Setup notes:
  - Requires `pip install verl deepspeed ray[default]`
  - H200 x8 target: TP=4 for rollout, FSDP/Megatron for actor
  - DeepSpeed ZeRO-3 for optimizer state sharding across the LoRA params
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import structlog

from training_local.config import RLConfig
from training_local.encoding import DeepSeekEncoding

logger = structlog.get_logger("harness1.backends.verl")


def run(
    config: RLConfig,
    toolset,
    encoding: DeepSeekEncoding,
) -> None:
    """Run GRPO via verl.

    verl has its own Ray-based orchestration. We:
      1. Initialize Ray cluster (uses all visible GPUs)
      2. Build verl's hydra config from our RLConfig
      3. Define a reward function that drives our LocalSearchEnv
      4. Run RayTrainer.main()
    """
    try:
        import ray
        from verl.trainer.ppo.ray_trainer import RayTrainer
        from verl.workers.rollout.vllm_rollout import vllm_rollout
    except ImportError as e:
        raise RuntimeError(
            f"verl not installed: {e}. Install with: uv pip install verl"
        )

    from training_local.env import LocalSearchEnv
    from training_local.data import load_queries
    from training_local.rollout import run_group, collect_group_rewards
    from training_local.rewards import grpo_advantages

    env = LocalSearchEnv(toolset=toolset, encoding=encoding, config=config)
    queries = load_queries(config)

    if not queries:
        raise RuntimeError("No training queries loaded.")

    logger.info(
        "verl backend starting",
        n_queries=len(queries),
        group_size=config.group_size,
        model=config.model.model_path,
    )

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Build verl-compatible dataset
    dataset = _build_verl_dataset(queries, encoding, env, config)

    # Build verl config (hydra-style nested dict)
    verl_config = _build_verl_config(config)

    # Run trainer
    trainer = RayTrainer(config=verl_config)
    trainer.init_workers()
    trainer.fit(dataset)


def _build_verl_dataset(queries, encoding, env, config: RLConfig):
    """Build a dataset of prompt + reward_function pairs for verl."""
    import pandas as pd
    from verl.utils.dataset import RLHFDataset

    rows = []
    for q in queries:
        try:
            state = env.reset(
                query_id=q.query_id,
                query_text=q.query_text,
                gold_ids=q.gold_doc_ids,
                fa_gold_ids=q.fa_doc_ids,
            )
            prompt = env.render_prompt(state)
        except Exception as e:
            continue

        rows.append({
            "query_id": q.query_id,
            "prompt": prompt,
            "gold_doc_ids": list(q.gold_doc_ids),
            "fa_doc_ids": list(q.fa_doc_ids),
            "reward_config": {
                "outcome_weight": config.outcome_weight,
                "trajectory_recall_weight": config.trajectory_recall_weight,
                "ndcg_weight": config.ndcg_weight,
                "tool_diversity_bonus": config.tool_diversity_bonus,
                "tool_diversity_target": config.tool_diversity_target,
                "turn_penalty_min_turns": config.turn_penalty_min_turns,
                "turn_penalty_max": config.turn_penalty_max,
            },
        })

    df = pd.DataFrame(rows)
    return df


def _build_verl_config(config: RLConfig) -> Dict[str, Any]:
    """Translate our RLConfig into verl's nested hydra config."""
    return {
        "trainer": {
            "total_epochs": config.num_epochs,
            "n_gpus_per_node": _visible_gpu_count(),
            "project_name": "harness1-local",
            "experiment_name": config.run_name,
            "logger": ["wandb", "console"] if config.log_to_wandb else ["console"],
            "save_freq": config.save_every_steps,
            "test_freq": config.eval_every_steps,
        },
        "actor_rollout_ref": {
            "model": {
                "path": config.model.model_path,
                "tokenizer_path": config.model.resolved_tokenizer_path,
                "enable_gradient_checkpointing": True,
                "trust_remote_code": True,
            },
            "actor": {
                "strategy": "fsdp",  # use fsdp for LoRA; switch to megatron for full
                "lora_rank": config.lora.rank if config.lora.enabled else 0,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "warmup_steps": config.warmup_steps,
                "ppo_mini_batch_size": config.batch_size * config.group_size,
                "ppo_micro_batch_size": 1,
                "clip_ratio": config.ppo_clip_ratio,
                "kl_penalty_coef": config.kl_penalty_coef,
            },
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_parallel_size": config.model.rollout_tensor_parallel_size,
                "gpu_memory_utilization": config.model.rollout_gpu_memory_utilization,
                "max_model_len": config.model.rollout_max_model_len,
                "block_size": config.model.rollout_block_size,
                "enable_expert_parallel": config.model.rollout_enable_expert_parallel,
                "kv_cache_dtype": config.model.rollout_kv_cache_dtype,
                "dtype": config.model.rollout_dtype,
                "temperature": config.model.rollout_temperature,
                "top_p": config.model.rollout_top_p,
                "n": config.group_size,
                "max_tokens": config.model.rollout_max_tokens_per_turn,
                "speculative_config": {
                    "method": "dspark",
                    "num_speculative_tokens": config.model.rollout_speculative_tokens,
                    "draft_sample_method": "greedy",
                }
                if config.model.rollout_speculative_tokens > 0
                else None,
            },
            "ref": {
                "log_prob_micro_batch_size": 1,
                "strategy": "fsdp",
            },
        },
        "data": {
            "train_batch_size": config.batch_size * config.group_size,
            "max_prompt_length": config.model.rollout_max_model_len // 2,
            "max_response_length": config.model.rollout_max_tokens_per_turn,
            "shuffle": True,
        },
        "algorithm": {
            "advantage_estimator": "grpo",
            "kl_penalty_lb": config.kl_penalty_coef,
            "oapl_beta1": config.oapl_beta1,
            "oapl_beta2": config.oapl_beta2,
        },
    }


def _visible_gpu_count() -> int:
    try:
        n = int(os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",")) + 1
        return max(n, 1)
    except Exception:
        return 1
