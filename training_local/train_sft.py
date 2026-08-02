"""SFT warm-start training for harness-1 local.

Trains a LoRA adapter on pre-generated search trajectories using DeepSeek-V4-Flash
encoding. Output is a LoRA adapter compatible with train_rl.py.

Usage:
    python -m training_local.train_sft --data-dir tmp/sft_data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from training_local.config import SFTConfig
from training_local.encoding import DeepSeekEncoding

logger = structlog.get_logger("harness1.train_sft")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="Override SFT_DATA_DIR")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def trajectory_to_messages(traj: Dict[str, Any], query_text: str) -> List[Dict[str, Any]]:
    """Convert a stored SFT trajectory into DeepSeek message format.

    The original generate_sft_data.py emits:
        {
            "trajectory": {"actions_and_observations": [...]},
            "system_prompt": str,
            "query_text": str,
        }
    """
    system_prompt = traj.get("system_prompt", "")
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    inner = traj.get("trajectory", traj)
    ano = inner.get("actions_and_observations", [])
    for entry in ano:
        if entry.get("type") == "action":
            tools = entry.get("tools", [])
            params = entry.get("params", [])
            sources = entry.get("sources", [])
            reasoning = entry.get("reasoning", "")

            content = entry.get("text", "")
            tool_calls = []
            for i, tool_name in enumerate(tools):
                args = params[i] if i < len(params) else {}
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                })

            messages.append({
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
                "tool_calls": tool_calls,
            })
        elif entry.get("type") == "observation":
            observations = entry.get("observations", [])
            obs_text = "\n\n".join(str(o) for o in observations)
            messages.append({
                "role": "tool",
                "name": entry.get("tool_name", "search_corpus"),
                "content": obs_text,
                "tool_call_id": f"call_{len(messages) // 2}",
            })

    return messages


def build_sft_examples(
    trajectories: List[Dict[str, Any]],
    encoding: DeepSeekEncoding,
) -> List[Dict[str, Any]]:
    """Convert all trajectories into DeepSeek-encoded prompt + completion pairs."""
    examples = []
    for traj in trajectories:
        query_text = traj.get("query_text", "")
        messages = trajectory_to_messages(traj, query_text)
        if len(messages) < 2:
            continue
        try:
            full_text = encoding.encode_messages(messages)
            examples.append({
                "text": full_text,
                "messages": messages,
            })
        except Exception as e:
            logger.warning("Failed to encode trajectory", error=str(e))
            continue
    return examples


def run():
    args = parse_args()
    config = SFTConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.smoke_test:
        config.smoke_test = True

    encoding = DeepSeekEncoding(
        model_path=config.model.model_path,
        thinking_mode=config.model.thinking_mode,
        reasoning_effort=config.model.reasoning_effort,
    )

    if config.smoke_test:
        logger.info("smoke test mode; skipping data load + training")
        return

    from training_local.data import load_sft_trajectories
    trajectories = load_sft_trajectories(config)
    logger.info("loaded trajectories", n=len(trajectories))

    examples = build_sft_examples(trajectories, encoding)
    logger.info("encoded examples", n=len(examples))

    if not examples:
        raise RuntimeError("No SFT examples produced. Check data format.")

    # Save as JSON for inspection
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_examples.json", "w") as f:
        json.dump(examples[:5], f, indent=2, default=str)
    logger.info("wrote sample examples", path=str(output_dir / "train_examples.json"))

    # Train via TRL SFTTrainer
    _train_with_trl(examples, config, output_dir)


def _train_with_trl(examples, config: SFTConfig, output_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.resolved_tokenizer_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        save_steps=config.save_every_steps,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if config.log_to_wandb else "none",
        seed=config.seed,
        max_length=config.model.rollout_max_model_len,
    )

    ds = Dataset.from_list([{"text": ex["text"]} for ex in examples])
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    logger.info("SFT complete", adapter_path=str(output_dir / "final"))


if __name__ == "__main__":
    run()
