"""Hyperparameters and runtime configuration for local RL/SFT training.

All values reflect best-practice defaults synthesized from:
  - Original harness-1 launch_rl.sh (4-component recall reward, GROUP_SIZE=8)
  - Sid-1 technical report (NDCG reward, TI/TO protocol, length scheduling)
  - KARL (arXiv:2603.05218) (OAPL KL params, compression, nugget eval)
  - DeepSeek-V4-Flash model card (FP8 KV cache, expert parallel)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name)
    if not v:
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class ModelConfig:
    """Base model + tokenizer configuration."""

    model_path: str = _env_str(
        "HARNESS1_MODEL_PATH",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    tokenizer_path: Optional[str] = None

    # FP8 quantization for the rollout (vLLM) side. Training side uses BF16
    # for stable backprop; FP8 weights are loaded into vLLM for fast rollout.
    rollout_dtype: str = "fp8"
    train_dtype: str = "bfloat16"

    # vLLM rollout engine settings
    rollout_tensor_parallel_size: int = _env_int("ROLLOUT_TP_SIZE", 4)
    rollout_gpu_memory_utilization: float = _env_float("ROLLOUT_GPU_MEM_UTIL", 0.45)
    rollout_max_model_len: int = _env_int("ROLLOUT_MAX_MODEL_LEN", 131072)
    rollout_block_size: int = 256
    rollout_enable_expert_parallel: bool = True
    rollout_kv_cache_dtype: str = "fp8"
    rollout_speculative_tokens: int = 7

    # Thinking mode for DeepSeek encoding
    thinking_mode: str = _env_str("THINKING_MODE", "thinking")
    reasoning_effort: str = _env_str("REASONING_EFFORT", "high")

    # Sampling
    rollout_temperature: float = _env_float("ROLLOUT_TEMPERATURE", 1.0)
    rollout_top_p: float = _env_float("ROLLOUT_TOP_P", 0.95)
    rollout_max_tokens_per_turn: int = _env_int("ROLLOUT_MAX_TOKENS_PER_TURN", 4096)

    @property
    def resolved_tokenizer_path(self) -> str:
        return self.tokenizer_path or self.model_path


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    enabled: bool = True
    rank: int = _env_int("LORA_RANK", 32)
    alpha: int = _env_int("LORA_ALPHA", 64)
    dropout: float = _env_float("LORA_DROPOUT", 0.05)
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # MoE: also target expert linear layers (per Sid-1, target all linears)
    target_expert_modules: bool = True


@dataclass
class RLConfig:
    """RL training hyperparameters (GRPO + Sid-1/KARL techniques)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # ===== Data =====
    train_datasets: List[str] = field(
        default_factory=lambda: _env_list("TRAIN_DATASETS", ["sec"])
    )
    rl_query_split: str = _env_str("RL_QUERY_SPLIT", "train")
    rl_collection_split: str = _env_str("RL_COLLECTION_SPLIT", "train")
    max_train_queries: Optional[int] = _env_int("MAX_TRAIN_QUERIES", 0) or None
    eval_datasets: List[str] = field(
        default_factory=lambda: _env_list("EVAL_DATASETS", ["browsecompplus"])
    )

    # ===== GRPO =====
    group_size: int = _env_int("GROUP_SIZE", 8)  # Sid-1: large enough for >0.95 format pass
    batch_size: int = _env_int("BATCH_SIZE", 32)  # queries per step (after * group_size)
    num_epochs: int = _env_int("NUM_EPOCHS", 3)
    learning_rate: float = _env_float("LEARNING_RATE", 1e-5)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.0)
    warmup_steps: int = _env_int("WARMUP_STEPS", 50)
    save_every_steps: int = _env_int("SAVE_EVERY_STEPS", 50)
    eval_every_steps: int = _env_int("EVAL_EVERY_STEPS", 100)
    max_steps: Optional[int] = _env_int("MAX_STEPS", 0) or None

    # ===== Rollout length scheduling (Sid-1) =====
    max_turns_start: int = _env_int("MAX_TURNS_START", 32)
    max_turns_end: int = _env_int("MAX_TURNS_END", 128)
    max_turns_schedule_steps: int = _env_int("MAX_TURNS_SCHEDULE_STEPS", 500)

    # ===== Reward weights =====
    # Original 4-component recall reward
    recall_beta: float = _env_float("RECALL_BETA", 2.0)  # F_beta favors recall 2x over precision
    outcome_weight: float = _env_float("OUTCOME_WEIGHT", 0.7)
    trajectory_recall_weight: float = _env_float("TRAJECTORY_RECALL_WEIGHT", 0.3)
    final_answer_bonus: float = _env_float("FINAL_ANSWER_BONUS", 1.0)
    final_answer_recall_weight: float = _env_float("FINAL_ANSWER_RECALL_WEIGHT", 0.8)
    trajectory_fa_recall_weight: float = _env_float("TRAJECTORY_FA_RECALL_WEIGHT", 0.4)
    fa_miss_penalty_weight: float = _env_float("FA_MISS_PENALTY_WEIGHT", 0.35)

    # Sid-1 NDCG-style ranking reward (in addition to recall)
    ndcg_weight: float = _env_float("NDCG_WEIGHT", 0.2)

    # Tool diversity bonus (anti-reward-hacking, original harness-1)
    tool_diversity_bonus: float = _env_float("TOOL_DIVERSITY_BONUS", 0.25)
    tool_diversity_target: int = _env_int("TOOL_DIVERSITY_TARGET", 6)

    # Turn penalty (encourage efficiency)
    turn_penalty_min_turns: int = _env_int("TURN_PENALTY_MIN_TURNS", 20)
    turn_penalty_max: float = _env_float("TURN_PENALTY_MAX", 0.02)

    # ===== KL penalty / OAPL (KARL) =====
    # KARL OAPL: two KL coefficients
    kl_penalty_coef: float = _env_float("KL_PENALTY_COEF", 0.005)
    oapl_beta1: float = _env_float("OAPL_BETA1", 1.0)  # value smoothing
    oapl_beta2: float = _env_float("OAPL_BETA2", 1.0)  # regularization strength
    oapl_iterations: int = _env_int("OAPL_ITERATIONS", 2)

    # PPO clipping
    ppo_clip_ratio: float = _env_float("PPO_CLIP_RATIO", 0.2)
    ppo_clip_value: float = _env_float("PPO_CLIP_VALUE", 0.2)

    # ===== Context compression (KARL) =====
    enable_compression: bool = True
    compression_char_threshold: int = _env_int("COMPRESSION_CHAR_THRESHOLD", 150_000)

    # ===== v8d features (original harness-1) =====
    v8d_subtractive_curation: bool = True
    v8d_importance_tagging: bool = True
    v8d_auto_populate_first_search: bool = True
    v8d_verify_tool: bool = True
    v8d_token_budget_marker: bool = True

    # ===== Checkpointing =====
    output_dir: str = _env_str("OUTPUT_DIR", "outputs/rl_runs")
    run_name: str = _env_str("RUN_NAME", "rl_local_default")
    init_from_checkpoint: Optional[str] = _env_str("INIT_FROM_CHECKPOINT", "") or None
    reference_model_path: Optional[str] = None  # defaults to base model

    # ===== Infrastructure =====
    num_train_workers: int = _env_int("NUM_TRAIN_WORKERS", 4)  # FSDP/Megatron workers
    num_rollout_workers: int = _env_int("NUM_ROLLOUT_WORKERS", 4)  # vLLM TP
    seed: int = _env_int("SEED", 42)
    smoke_test: bool = bool(_env_int("SMOKE_TEST", 0))
    log_to_wandb: bool = bool(_env_int("LOG_WANDB", 1))


@dataclass
class SFTConfig:
    """SFT warm-start training hyperparameters."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Data
    train_datasets: List[str] = field(
        default_factory=lambda: _env_list("SFT_TRAIN_DATASETS", ["browsecompplus", "sec"])
    )
    data_dir: str = _env_str("SFT_DATA_DIR", "tmp/sft_data")
    max_train_trajectories: Optional[int] = None

    # Training
    num_epochs: int = _env_int("SFT_EPOCHS", 3)
    batch_size: int = _env_int("SFT_BATCH_SIZE", 8)
    gradient_accumulation_steps: int = _env_int("SFT_GRAD_ACCUM", 4)
    learning_rate: float = _env_float("SFT_LEARNING_RATE", 5e-5)
    warmup_ratio: float = _env_float("SFT_WARMUP_RATIO", 0.03)
    weight_decay: float = _env_float("SFT_WEIGHT_DECAY", 0.0)
    max_grad_norm: float = _env_float("SFT_MAX_GRAD_NORM", 1.0)

    # Pass-rate filtering for synthetic data (KARL)
    min_pass_rate: float = _env_float("SFT_MIN_PASS_RATE", 0.1)
    max_pass_rate: float = _env_float("SFT_MAX_PASS_RATE", 0.9)

    # Checkpointing
    output_dir: str = _env_str("SFT_OUTPUT_DIR", "outputs/sft_runs")
    run_name: str = _env_str("SFT_RUN_NAME", "sft_warmup")
    save_every_steps: int = _env_int("SFT_SAVE_EVERY_STEPS", 100)

    # Infrastructure
    num_train_workers: int = _env_int("SFT_NUM_WORKERS", 8)
    seed: int = _env_int("SEED", 42)
    log_to_wandb: bool = bool(_env_int("LOG_WANDB", 1))
    smoke_test: bool = bool(_env_int("SMOKE_TEST", 0))
