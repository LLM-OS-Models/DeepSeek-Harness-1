"""DeepSeek policy interface for vLLM-backed rollout.

Used by:
  - training_local.rollout: standalone rollout driver (no verl)
  - training_local.train_sft: SFT inference for verification / data prep

For RL training via verl, the verl library manages its own vLLM rollout internally;
this class is used for SFT and for verl-bypass paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from training_local.config import ModelConfig
from training_local.encoding import DeepSeekEncoding


@dataclass
class Completion:
    """A single completion from the policy."""

    text: str
    token_ids: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"
    prompt_token_ids: Optional[List[int]] = None


class DeepSeekPolicyInferenceModel:
    """vLLM-backed DeepSeek-V4-Flash policy.

    Lazy-loads vLLM on first use so that the rest of the code can run on
    machines without GPUs (e.g., for config validation, smoke tests).
    """

    def __init__(
        self,
        config: ModelConfig,
        encoding: Optional[DeepSeekEncoding] = None,
    ):
        self.config = config
        self.encoding = encoding or DeepSeekEncoding(
            model_path=config.model_path,
            thinking_mode=config.thinking_mode,
            reasoning_effort=config.reasoning_effort,
        )
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        # Apply runtime patches (MHC tilelang fallback for Py3.12).
        from training_local._vllm_runtime_patches import apply_all
        apply_all()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.resolved_tokenizer_path,
            trust_remote_code=True,
        )

        kwargs = dict(
            model=self.config.model_path,
            dtype=self.config.rollout_dtype,
            tensor_parallel_size=self.config.rollout_tensor_parallel_size,
            gpu_memory_utilization=self.config.rollout_gpu_memory_utilization,
            max_model_len=self.config.rollout_max_model_len,
            block_size=self.config.rollout_block_size,
            trust_remote_code=True,
            enable_expert_parallel=self.config.rollout_enable_expert_parallel,
            kv_cache_dtype=self.config.rollout_kv_cache_dtype,
        )
        # DSpark speculative decoding (DeepSeek-V4-Flash native)
        if self.config.rollout_speculative_tokens > 0:
            kwargs["speculative_config"] = {
                "method": "dspark",
                "num_speculative_tokens": self.config.rollout_speculative_tokens,
                "draft_sample_method": "greedy",
            }

        self._llm = LLM(**kwargs)
        self._SamplingParams = SamplingParams

    def sample(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> List[Completion]:
        """Sample n completions from the policy."""
        self._ensure_loaded()
        params = self._SamplingParams(
            n=n,
            temperature=temperature if temperature is not None else self.config.rollout_temperature,
            top_p=self.config.rollout_top_p,
            max_tokens=max_tokens or self.config.rollout_max_tokens_per_turn,
            stop=stop,
        )
        outputs = self._llm.generate([prompt], params)
        results: List[Completion] = []
        for out in outputs[0].outputs:
            results.append(
                Completion(
                    text=out.text,
                    token_ids=list(out.token_ids),
                    logprobs=_extract_logprobs(out),
                    finish_reason=out.finish_reason,
                    prompt_token_ids=list(outputs[0].prompt_token_ids),
                )
            )
        return results

    def encode_prompt(self, messages: List[dict]) -> str:
        """Convenience: encode messages via DeepSeek encoding."""
        return self.encoding.encode_messages(messages)


def _extract_logprobs(output) -> Optional[List[float]]:
    """Extract logprob values from vLLM RequestOutput."""
    if not hasattr(output, "logprobs") or output.logprobs is None:
        return None
    result = []
    for step_lp in output.logprobs:
        if step_lp:
            top = next(iter(step_lp.values()))
            result.append(top.logprob if hasattr(top, "logprob") else top.get("logprob", 0.0))
    return result
