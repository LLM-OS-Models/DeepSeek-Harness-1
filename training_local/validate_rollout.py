"""Quick end-to-end rollout validation on real DeepSeek-V4-Flash.

Verifies the minimum viable loop without Chroma / reward backend:
  1. encoding_dsv4 loads from snapshot
  2. vLLM TP=2 starts with FP8 + DSpark
  3. encode_messages → sample → parse_completion runs cleanly
  4. parse_completion returns well-formed DSML tool calls

Usage (2 GPUs):
  CUDA_VISIBLE_DEVICES=6,7 uv run python -m training_local.validate_rollout

Designed to fail loudly on the first concrete issue so we can fix-and-push.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

# vLLM's import chain touches CUDA before forking TP workers, which fails with
# "Cannot re-initialize CUDA in forked subprocess" under the default fork
# start method. Force spawn globally before any vLLM import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# vLLM 0.25 defaults VLLM_USE_FLASHINFER_SAMPLER=True, which forces a flashinfer
# import. flashinfer.ai doesn't ship torch 2.11 wheels yet, so disable it and
# fall back to the native PyTorch top-k/top-p sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from training_local.config import ModelConfig
from training_local.encoding import DeepSeekEncoding
from training_local._vllm_runtime_patches import apply_all as _apply_vllm_patches

# MHC forward_cuda fallback for Python 3.12 + tilelang incompat.
_apply_vllm_patches()


def _banner(s: str) -> None:
    print(f"\n{'=' * 60}\n{s}\n{'=' * 60}", flush=True)


def _ok(s: str) -> None:
    print(f"  ✓ {s}", flush=True)


def _fail(s: str) -> None:
    print(f"  ✗ {s}", flush=True)


def main() -> int:
    _banner("harness-1 local — rollout validation (real model)")

    # ---------- 1. encoding ----------
    _banner("[1/4] encoding_dsv4 module load + render")
    cfg = ModelConfig()
    enc = DeepSeekEncoding(
        model_path=cfg.model_path,
        thinking_mode=cfg.thinking_mode,
        reasoning_effort=cfg.reasoning_effort,
    )
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a search agent. Use the available tools to find "
                "evidence. Stop when you have enough."
            ),
        },
        {
            "role": "user",
            "content": "What is the revenue of ACME Corp in fiscal year 2024?",
        },
    ]
    try:
        prompt_text = enc.encode_messages(messages)
    except Exception as e:
        _fail(f"encode_messages raised: {e!r}")
        return 1
    if not prompt_text or len(prompt_text) < 50:
        _fail(f"prompt suspiciously short: len={len(prompt_text)}")
        return 1
    _ok(f"encoded prompt ({len(prompt_text)} chars)")
    print(f"\n----- PROMPT HEAD (first 400 chars) -----\n{prompt_text[:400]}\n----- END -----\n", flush=True)

    # ---------- 2. vLLM start ----------
    _banner("[2/4] vLLM start (TP=2, FP8, DSpark 7 tokens)")
    from vllm import LLM, SamplingParams  # noqa: F401

    t0 = time.time()
    llm = LLM(
        model=cfg.model_path,
        dtype=cfg.rollout_dtype,
        tensor_parallel_size=2,
        gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEM_UTIL", "0.90")),
        max_model_len=int(os.environ.get("ROLLOUT_MAX_MODEL_LEN", "8192")),
        block_size=256,
        trust_remote_code=True,
        enable_expert_parallel=True,
        kv_cache_dtype="fp8",
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 7,
            "draft_sample_method": "greedy",
        },
    )
    _ok(f"vLLM ready in {time.time() - t0:.1f}s")

    # ---------- 3. sample ----------
    _banner("[3/4] sample one completion")
    sp = SamplingParams(
        n=1, temperature=0.7, top_p=0.95, max_tokens=512,
    )
    t0 = time.time()
    outputs = llm.generate([prompt_text], sp)
    elapsed = time.time() - t0
    completion_text = outputs[0].outputs[0].text
    finish = outputs[0].outputs[0].finish_reason
    n_tokens = len(outputs[0].outputs[0].token_ids)
    if not completion_text:
        _fail(f"empty completion, finish={finish}")
        return 1
    _ok(f"got {n_tokens} tokens in {elapsed:.2f}s (finish={finish})")
    print(f"\n----- COMPLETION (first 800 chars) -----\n{completion_text[:800]}\n----- END -----\n", flush=True)

    # ---------- 4. parse ----------
    _banner("[4/4] parse_completion")
    try:
        parsed = enc.parse_completion(completion_text)
    except Exception as e:
        _fail(f"parse_completion raised: {e!r}")
        return 1
    role = parsed.get("role")
    content = parsed.get("content")
    reasoning = parsed.get("reasoning_content")
    tool_calls = parsed.get("tool_calls", [])
    _ok(f"role={role!r}, content_len={len(content) if content else 0}, "
        f"reasoning_len={len(reasoning) if reasoning else 0}, "
        f"tool_calls={len(tool_calls)}")
    if tool_calls:
        tc0 = tool_calls[0]
        name = tc0.get("function", {}).get("name") if "function" in tc0 else tc0.get("name")
        _ok(f"first tool call name: {name!r}")
    print(f"\n----- PARSED (truncated) -----\n{str(parsed)[:600]}\n----- END -----\n", flush=True)

    _banner("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
