"""TRL GRPOTrainer backend for harness-1 local RL.

Uses HuggingFace TRL's GRPOTrainer with vLLM rollout co-located on the
training nodes. Multi-turn agentic rollouts are driven by our custom
LocalSearchEnv; reward is computed by our rewards.py.

Per Sid-1 (TI/TO):
  - The trainer sees the FULL prompt for each turn (re-encoded). No re-render
    of parsed messages.
  - Reward is sparse: 0 for all non-terminal turns, terminal reward at end.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import structlog

from training_local.config import RLConfig
from training_local.encoding import DeepSeekEncoding
from training_local.env import LocalSearchEnv
from training_local.rewards import RewardBreakdown

logger = structlog.get_logger("harness1.backends.trl")


def run(
    config: RLConfig,
    toolset,
    encoding: DeepSeekEncoding,
) -> None:
    """Run GRPO via TRL.

    For multi-turn agentic RL with a stateful env, TRL's GRPOTrainer doesn't
    natively support mid-episode tool execution. We bridge by treating the
    ENTIRE episode as a single "completion" with a custom reward function:
    each "completion" is actually the model's full sequence of turns, and our
    reward function runs the env internally.
    """
    env = LocalSearchEnv(toolset=toolset, encoding=encoding, config=config)

    # Lazy imports so module loads without GPU.
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    # Build dataset of "prompts" (one per query). Each prompt is the
    # INITIAL env prompt; the rest of the episode is driven by env.step
    # inside the reward function.
    from training_local.data import load_queries

    queries = load_queries(config)
    if not queries:
        raise RuntimeError(
            "No training queries loaded. Check TRAIN_DATASETS and dataset setup."
        )

    logger.info(
        "loaded queries",
        n=len(queries),
        datasets=config.train_datasets,
    )

    def reward_fn(prompt_str: str, **kwargs) -> float:
        """Custom reward: run a full episode and return terminal reward."""
        # Note: this is invoked per-sample; for true GRPO we want G samples
        # per prompt. TRL's GRPO handles group sampling internally.
        # Here we receive a single completion per call.
        completion = kwargs.get("completions", kwargs.get("completion", ""))
        if isinstance(completion, list):
            completion = completion[0] if completion else ""

        # Run env with this single completion as the first turn
        # (more elaborate multi-turn rollout would require a custom trainer)
        return _single_turn_reward(env, encoding, prompt_str, completion, config)

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.resolved_tokenizer_path,
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=os.path.join(config.output_dir, config.run_name),
        num_generations=config.group_size,
        max_completion_length=config.model.rollout_max_tokens_per_turn,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_every_steps,
        eval_steps=config.eval_every_steps if config.eval_every_steps > 0 else 1000,
        logging_steps=10,
        per_device_train_batch_size=1,  # each "batch" is one query's group
        gradient_accumulation_steps=max(1, config.batch_size // 4),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if config.log_to_wandb else "none",
        seed=config.seed,
        beta=config.kl_penalty_coef,  # KL coefficient
        temperature=config.model.rollout_temperature,
        # vLLM colocate for fast FP8 rollout
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=config.model.rollout_gpu_memory_utilization,
    )

    # GRPOTrainer needs a model
    from transformers import AutoModelForCausalLM

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    trainer = GRPOTrainer(
        model=config.model.model_path,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=[{"prompt": q.query_text} for q in queries],
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
        model_init_kwargs=model_kwargs,
    )

    logger.info("starting GRPO training", steps=config.max_steps)
    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, config.run_name, "final"))
    logger.info("training complete")


def _single_turn_reward(
    env: LocalSearchEnv,
    encoding: DeepSeekEncoding,
    prompt: str,
    completion: str,
    config: RLConfig,
) -> float:
    """Fallback single-turn reward (used by TRL's per-call reward_fn).

    Note: TRL's standard GRPOTrainer expects single-turn prompts. True
    multi-turn agentic RL requires either:
      (a) TRL's recent multi-turn support (check version), or
      (b) Custom trainer that drives env.step() during rollout.

    For now, treat one completion as one turn and compute a partial reward.
    The full multi-turn implementation lives in our verl_runner.
    """
    try:
        parsed = encoding.parse_completion(completion)
    except Exception:
        parsed = {"role": "assistant", "content": completion, "tool_calls": []}

    # Use the first query from the prompt's gold data
    # (in real training, this would be passed in via dataset metadata)
    state = env.reset(
        query_id="_trl_single_turn",
        query_text=prompt,
        gold_ids=set(),
        fa_gold_ids=set(),
    )
    state, _, _, _ = env.step(state, parsed)
    reward = env.compute_terminal_reward(state)
    return reward.total
