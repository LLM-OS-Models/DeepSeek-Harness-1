#!/usr/bin/env bash
# Launch harness-1 local RL training with sensible defaults.
#
# Usage:
#   bash training_local/launch_rl.sh                  # default run
#   SMOKE_TEST=1 bash training_local/launch_rl.sh     # smoke test (no GPU)
#   BACKEND=verl bash training_local/launch_rl.sh     # force verl backend
#
# Required env vars:
#   OPENAI_API_KEY     — used by retrieval/verification tools
#   CHROMA_API_KEY     — Chroma vector store
#   CHROMA_DATABASE    — Chroma database name
#
# Optional env vars (see training_local/config.py for full list):
#   HARNESS1_MODEL_PATH        — model id or local path
#   TRAIN_DATASETS             — comma-separated dataset names
#   GROUP_SIZE, BATCH_SIZE     — GRPO config
#   ROLLOUT_TP_SIZE            — vLLM tensor parallel size
#   OUTPUT_DIR, RUN_NAME       — checkpoint location

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Clear PYTHONPATH so the uv venv is the only Python source. The host may
# have an unrelated system-Python site-packages on PYTHONPATH that breaks
# ABI for C-extension packages (numpy, torch).
unset PYTHONPATH

# Load .env.local if present
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

# ===== Defaults =====
export HARNESS1_MODEL_PATH="${HARNESS1_MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export TRAIN_DATASETS="${TRAIN_DATASETS:-sec}"
export RL_QUERY_SPLIT="${RL_QUERY_SPLIT:-train}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export LORA_RANK="${LORA_RANK:-32}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-131072}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl_runs}"
export RUN_NAME="${RUN_NAME:-rl_dsv4_flash_$(date +%Y%m%d_%H%M%S)}"
export BACKEND="${BACKEND:-auto}"

# ===== KL / OAPL =====
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.005}"
export OAPL_BETA1="${OAPL_BETA1:-1.0}"
export OAPL_BETA2="${OAPL_BETA2:-1.0}"

# ===== Reward weights =====
export OUTCOME_WEIGHT="${OUTCOME_WEIGHT:-0.7}"
export TRAJECTORY_RECALL_WEIGHT="${TRAJECTORY_RECALL_WEIGHT:-0.3}"
export NDCG_WEIGHT="${NDCG_WEIGHT:-0.2}"
export TOOL_DIVERSITY_BONUS="${TOOL_DIVERSITY_BONUS:-0.25}"
export TOOL_DIVERSITY_TARGET="${TOOL_DIVERSITY_TARGET:-6}"
export TURN_PENALTY_MIN_TURNS="${TURN_PENALTY_MIN_TURNS:-20}"
export TURN_PENALTY_MAX="${TURN_PENALTY_MAX:-0.02}"

# ===== Safety checks =====
if [[ "${SMOKE_TEST:-0}" != "1" ]]; then
  for required in OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE; do
    if [[ -z "${!required:-}" ]]; then
      echo "ERROR: $required is not set. Add it to .env.local" >&2
      exit 1
    fi
  done
fi

# ===== Launch =====
echo "============================================================"
echo "Harness-1 local RL"
echo "  backend     : ${BACKEND}"
echo "  model       : ${HARNESS1_MODEL_PATH}"
echo "  datasets    : ${TRAIN_DATASETS}"
echo "  group/batch : ${GROUP_SIZE}/${BATCH_SIZE}"
echo "  LoRA rank   : ${LORA_RANK}"
echo "  rollout TP  : ${ROLLOUT_TP_SIZE}"
echo "  output      : ${OUTPUT_DIR}/${RUN_NAME}"
echo "  smoke       : ${SMOKE_TEST:-0}"
echo "============================================================"

uv run python -m training_local.train_rl --backend "${BACKEND}" "$@"
