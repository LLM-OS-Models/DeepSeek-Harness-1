#!/usr/bin/env bash
# Quick end-to-end rollout validation on real DeepSeek-V4-Flash.
#
# Verifies the minimum viable loop without Chroma / reward backend:
#   1. encoding_dsv4 loads from snapshot
#   2. vLLM TP=2 starts with FP8 + DSpark
#   3. encode_messages → sample → parse_completion runs cleanly
#   4. parse_completion returns well-formed DSML tool calls
#
# Usage:
#   bash training_local/validate_rollout.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash training_local/validate_rollout.sh
#   ROLLOUT_MAX_MODEL_LEN=4096 bash training_local/validate_rollout.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Clear PYTHONPATH so the uv venv is the only Python source. The host may
# have an unrelated user-site (e.g. ~/.local/lib/python3.12/site-packages) on
# PYTHONPATH that ships a different torch ABI and shadows the venv.
unset PYTHONPATH

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.90}"
# Smoke test uses low reasoning effort by default. The model snapshot's
# encoding maps "high" → "Reasoning Effort: Absolute maximum..." which pushes
# the model to over-think and occasionally emit multiple </think> tokens.
# "low" adds no prefix and gives cleaner structured output for validation.
export REASONING_EFFORT="${REASONING_EFFORT:-low}"

# Re-apply vLLM patches in case venv was reinstalled.
uv run --no-sync python -m training_local.patch_vllm || true

exec uv run python -m training_local.validate_rollout "$@"
