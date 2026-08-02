"""Checkpoint save/load utilities for LoRA + HF model.

Handles:
  - LoRA adapter save/load (PEFT format)
  - Merged full-model save (for vLLM serving after training)
  - Optimizer state save (for resumption)
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from training_local.config import LoRAConfig, ModelConfig


def save_lora_adapter(model, output_dir: str, step: Optional[int] = None) -> str:
    """Save the LoRA adapter only (small footprint, ~MB)."""
    out = Path(output_dir)
    if step is not None:
        out = out / f"step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    return str(out)


def save_merged_model(model, tokenizer, output_dir: str) -> str:
    """Merge LoRA into base and save full model (for vLLM serving).

    Per Swift docs (DeepSeek-V4 FP8+LoRA): set `--merge_lora false` to keep
    LoRA-only checkpoints for training; merge only when deploying.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged.save_pretrained(str(out), safe_serialization=True)
            tokenizer.save_pretrained(str(out))
            return str(out)
    except Exception as e:
        raise RuntimeError(f"Failed to merge LoRA: {e}")

    raise RuntimeError("Model is not a LoRA-wrapped model; nothing to merge.")


def load_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    lora_config: LoRAConfig,
    dtype: str = "bfloat16",
):
    """Load a HF base model with a LoRA adapter applied."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    return model


def save_training_state(
    output_dir: str,
    step: int,
    optimizer,
    scheduler,
    rng_state,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save full training state for resumption."""
    import torch

    out = Path(output_dir) / f"checkpoint-{step}"
    out.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_state": rng_state,
        "extra": extra or {},
    }
    torch.save(state, out / "training_state.pt")
    return str(out)


def load_training_state(checkpoint_dir: str, optimizer=None, scheduler=None):
    """Load training state (does not restore model — use load_lora_adapter)."""
    import torch

    path = Path(checkpoint_dir) / "training_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No training state at {path}")

    state = torch.load(path, map_location="cpu", weights_only=False)
    if optimizer and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state
