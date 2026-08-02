"""Harness-1 local training package.

Replaces the original Tinker-based training pipeline with a local verl/TRL-based
pipeline targeting DeepSeek-V4-Flash-0731 (FP8) + LoRA on multi-GPU (H200 x8).

Modules:
    encoding:        Wrapper around DeepSeek-V4 custom encoding (encoding_dsv4).
    config:          All hyperparameters as dataclasses.
    rewards:         4-component recall reward + NDCG (Sid-1 inspired).
    tools_adapter:   Bridges harness ToolSet to verl tool manager interface.
    env:             LocalSearchEnv — multi-turn agentic search environment.
    agent:           DeepSeekPolicyInferenceModel — policy interface for verl.
    data:            SearchDataset adapter for verl prompt format.
    checkpoint:      HF + LoRA adapter save/load.
    rollout:         Multi-turn rollout driver orchestrating env + policy.
    train_rl:        Main GRPO RL training entrypoint.
    train_sft:       SFT warm-start training entrypoint.
    smoke_test:      Quick wiring check (no GPU needed).
"""

from training_local.config import RLConfig, SFTConfig, ModelConfig

__all__ = ["RLConfig", "SFTConfig", "ModelConfig"]
__version__ = "0.2.0"
