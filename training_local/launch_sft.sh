#!/usr/bin/env bash
# Launch harness-1 SFT warm-start training.
#
# Usage:
#   bash training_local/launch_sft.sh
#   SFT_DATA_DIR=/path/to/data bash training_local/launch_sft.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Clear PYTHONPATH to avoid host system site-packages contaminating the venv.
unset PYTHONPATH

# Same vLLM env hardening as launch_rl.sh — see that file for rationale.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

export HARNESS1_MODEL_PATH="${HARNESS1_MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export SFT_DATA_DIR="${SFT_DATA_DIR:-tmp/sft_data}"
export SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft_runs}"
export SFT_RUN_NAME="${SFT_RUN_NAME:-sft_warmup_$(date +%Y%m%d_%H%M%S)}"
export SFT_EPOCHS="${SFT_EPOCHS:-3}"
export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-5e-5}"
export LORA_RANK="${LORA_RANK:-32}"

echo "============================================================"
echo "Harness-1 SFT warm-start"
echo "  model       : ${HARNESS1_MODEL_PATH}"
echo "  data dir    : ${SFT_DATA_DIR}"
echo "  output      : ${SFT_OUTPUT_DIR}/${SFT_RUN_NAME}"
echo "============================================================"

# Re-apply vLLM patches in case venv was reinstalled.
uv run --no-sync python -m training_local.patch_vllm || true

uv run python -m training_local.train_sft "$@"
