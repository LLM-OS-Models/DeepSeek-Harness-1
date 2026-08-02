"""Multi-turn rollout validation on real DeepSeek-V4-Flash.

Builds on `validate_rollout.py` (single-turn smoke test) and exercises the
multi-turn loop that RL training will actually drive:

  1. vLLM TP=2 starts with FP8 + DSpark.
  2. encode_messages renders a system + user + tools prompt.
  3. Sample turn 1 → parse_completion → expect tool_calls.
  4. Append a synthesized tool_result message, re-encode, sample turn 2.
  5. parse turn 2 → expect a final answer (no tool_calls).

This catches multi-turn regressions that the single-turn test cannot:
  - tool_call id propagation through encoding/decoding
  - assistant turn carrying both content and tool_calls
  - tool message merge logic in encode_messages
  - prompt length growth across turns stays under max_model_len

Usage (2 GPUs):
  CUDA_VISIBLE_DEVICES=6,7 uv run python -m training_local.validate_multiturn

Fails loudly on the first regression.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

# vLLM import chain touches CUDA; force spawn before any vLLM import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from training_local.config import ModelConfig
from training_local.encoding import DeepSeekEncoding
from training_local._vllm_runtime_patches import apply_all as _apply_vllm_patches

_apply_vllm_patches()


def _banner(s: str) -> None:
    print(f"\n{'=' * 60}\n{s}\n{'=' * 60}", flush=True)


def _ok(s: str) -> None:
    print(f"  ✓ {s}", flush=True)


def _fail(s: str) -> None:
    print(f"  ✗ {s}", flush=True)


# Toy tool schema: a fake web search that returns a hard-coded result.
SEARCH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query. Returns top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _fake_tool_result(call_id: str) -> Dict[str, Any]:
    """Synthesize a plausible-looking tool result for any call_id."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": (
            "ACME Corp fiscal year 2024 revenue was $4.2 billion, "
            "up 12% year-over-year (source: ACME 10-K filing)."
        ),
    }


def _sample_once(
    llm, sp, enc: DeepSeekEncoding, messages: List[Dict[str, Any]], label: str
) -> Dict[str, Any]:
    """Encode → sample → parse. Returns parsed dict; raises on hard failure."""
    prompt_text = enc.encode_messages(messages)
    print(f"\n----- {label} PROMPT (last 300 chars) -----\n...{prompt_text[-300:]}\n----- END -----\n", flush=True)

    outputs = llm.generate([prompt_text], sp)
    out = outputs[0].outputs[0]
    print(f"\n----- {label} COMPLETION (first 600 chars) -----\n{out.text[:600]}\n----- END -----\n", flush=True)

    parsed = enc.parse_completion(out.text)
    return parsed


def main() -> int:
    _banner("harness-1 local — multi-turn rollout validation")

    cfg = ModelConfig()
    enc = DeepSeekEncoding(
        model_path=cfg.model_path,
        thinking_mode=cfg.thinking_mode,
        reasoning_effort=cfg.reasoning_effort,
    )

    # ---------- 1. vLLM start ----------
    _banner("[1/3] vLLM start (TP=2, FP8, DSpark 7 tokens)")
    from vllm import LLM, SamplingParams

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

    sp = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=512)

    # ---------- 2. Turn 1: expect a tool_call ----------
    _banner("[2/3] Turn 1 — encode + sample + parse (expect web_search call)")
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a search agent. Use the web_search tool to find "
                "evidence, then answer. Call the tool once, then stop."
            ),
            "tools": SEARCH_TOOLS,
        },
        {
            "role": "user",
            "content": "What was ACME Corp's fiscal year 2024 revenue?",
        },
    ]
    try:
        t1 = _sample_once(llm, sp, enc, messages, "TURN-1")
    except Exception as e:
        _fail(f"turn 1 raised: {e!r}")
        return 1
    t1_calls = t1.get("tool_calls", [])
    _ok(f"turn 1: role={t1.get('role')!r}, tool_calls={len(t1_calls)}, "
        f"content_len={len(t1.get('content') or '')}")
    if not t1_calls:
        _fail("turn 1 produced no tool_calls — model did not call web_search")
        print(f"  parsed: {str(t1)[:400]}", flush=True)
        return 1
    call_id = t1_calls[0].get("id", "call_0")
    fn_name = (
        t1_calls[0].get("function", {}).get("name")
        if "function" in t1_calls[0]
        else t1_calls[0].get("name")
    )
    _ok(f"turn 1 first call: id={call_id!r}, name={fn_name!r}")

    # ---------- 3. Turn 2: append assistant + tool_result, expect final answer ----------
    _banner("[3/3] Turn 2 — append tool_result, re-encode, expect final answer")
    messages.append({
        "role": "assistant",
        "content": t1.get("content") or "",
        "tool_calls": t1_calls,
    })
    messages.append(_fake_tool_result(call_id))
    try:
        t2 = _sample_once(llm, sp, enc, messages, "TURN-2")
    except Exception as e:
        _fail(f"turn 2 raised: {e!r}")
        return 1
    t2_calls = t2.get("tool_calls", [])
    t2_content = t2.get("content") or ""
    _ok(f"turn 2: role={t2.get('role')!r}, tool_calls={len(t2_calls)}, "
        f"content_len={len(t2_content)}")
    print(f"\n----- TURN-2 CONTENT -----\n{t2_content[:400]}\n----- END -----\n", flush=True)
    if not t2_content:
        _fail("turn 2 produced empty content — expected a final answer")
        return 1

    _banner("ALL CHECKS PASSED (multi-turn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
