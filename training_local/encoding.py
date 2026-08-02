"""DeepSeek-V4-Flash encoding wrapper.

DeepSeek-V4-Flash uses a CUSTOM encoding (not Jinja chat template) shipped in
the model repo under `encoding/encoding_dsv4.py`. This module loads that file
dynamically and exposes a stable Python interface.

Per Sid-1 (TI/TO principle): never reconstruct messages from parsed dicts.
Always preserve exact token sequences across tool-call boundaries to avoid
log-probability collapse at boundary tokens.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

ThinkingMode = Literal["thinking", "chat"]
ReasoningEffort = Literal["low", "high", "max"]


def _resolve_snapshot_path(model_path: str) -> str:
    """Resolve a HF model id or local path to the snapshot directory.

    Handles three cases:
      1. Direct path to a snapshot dir (contains `encoding/`).
      2. HF cache `models--org--name/snapshots/<rev>/`.
      3. HF model id `org/name` -> resolved via local cache.
    """
    if os.path.isdir(model_path) and os.path.isdir(os.path.join(model_path, "encoding")):
        return model_path

    if "models--" in model_path:
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            revs = sorted(os.listdir(snapshots_dir))
            if revs:
                return os.path.join(snapshots_dir, revs[0])

    if "/" in model_path and not os.path.exists(model_path):
        org, name = model_path.split("/", 1)
        cache_root = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        cache_root = os.path.expanduser(cache_root)
        hub_dir = os.path.join(cache_root, "hub", f"models--{org}--{name}")
        if os.path.isdir(hub_dir):
            snapshots_dir = os.path.join(hub_dir, "snapshots")
            revs = sorted(os.listdir(snapshots_dir))
            if revs:
                return os.path.join(snapshots_dir, revs[0])

    raise FileNotFoundError(
        f"Could not resolve DeepSeek-V4-Flash encoding for: {model_path}"
    )


_ENCODING_MODULE_CACHE: Dict[str, Any] = {}


def load_encoding_module(model_path: str):
    """Load `encoding_dsv4.py` from the model snapshot.

    The file is a self-contained Python module (no external deps beyond stdlib).
    Returns the loaded module.
    """
    if model_path in _ENCODING_MODULE_CACHE:
        return _ENCODING_MODULE_CACHE[model_path]

    snapshot = _resolve_snapshot_path(model_path)
    encoding_file = Path(snapshot) / "encoding" / "encoding_dsv4.py"
    if not encoding_file.is_file():
        raise FileNotFoundError(
            f"encoding_dsv4.py not found at: {encoding_file}\n"
            f"Snapshot contents: {os.listdir(snapshot)}"
        )

    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoding_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ENCODING_MODULE_CACHE[model_path] = module
    return module


@dataclass
class DeepSeekEncoding:
    """Stateful encoder/decoder wrapper around DeepSeek-V4-Flash encoding."""

    model_path: str
    thinking_mode: ThinkingMode = "thinking"
    reasoning_effort: ReasoningEffort = "high"

    def __post_init__(self):
        self._mod = load_encoding_module(self.model_path)

    def encode_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Render an OpenAI-compatible message list into a prompt string."""
        return self._mod.encode_messages(
            messages,
            thinking_mode=self.thinking_mode,
            reasoning_effort=self.reasoning_effort,
        )

    def parse_completion(self, text: str) -> Dict[str, Any]:
        """Parse a model completion text into structured content.

        Returns dict with keys: role, content, reasoning_content, tool_calls.
        tool_calls are in OpenAI format: {id, type: "function", function: {name, arguments}}.
        """
        # vLLM strips the EOS token from `RequestOutput.text` even when
        # finish_reason='stop' (EOS was generated). The DSv4 parser strictly
        # requires the EOS sentinel at the end of the text. Re-attach it when
        # missing so callers can pass the stripped text directly.
        eos = "<｜end▁of▁sentence｜>"
        if not text.endswith(eos):
            text = text + eos

        # The DSv4 parser treats the FIRST </think> as the end of the thinking
        # block and rejects any subsequent </think> as an unexpected special
        # token. With high/max reasoning_effort the model sometimes emits two
        # or three </think> tokens (e.g. when it briefly answers, then reopens
        # thinking). Collapse extra </think> occurrences in the post-thinking
        # region so we can recover the structured content instead of failing.
        if self.thinking_mode == "thinking":
            think_end = "</think>"
            first = text.find(think_end)
            if first != -1:
                split = first + len(think_end)
                head = text[:split]
                tail = text[split:].replace(think_end, "")
                text = head + tail

        return self._mod.parse_message_from_completion_text(
            text, thinking_mode=self.thinking_mode
        )

    def render_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Render tool definitions into the prompt-injectable tools section."""
        return self._mod.render_tools(tools)


def default_stop_tokens(tokenizer) -> List[int]:
    """Return end-of-turn stop token IDs for DeepSeek-V4.

    DeepSeek-V4 uses a single end-of-turn token; the encoding module emits
    the proper separator. We use the tokenizer's eos token id and the
    end-of-turn marker if available.
    """
    stop = []
    for token_str in ("<|end▁of▁sentence|>", "<|eot_id>", "<|im_end|>", "<|/think|>", "<|/answer|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid >= 0 and tid not in stop:
                stop.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop:
        stop.append(tokenizer.eos_token_id)
    return stop
