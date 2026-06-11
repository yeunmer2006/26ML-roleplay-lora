#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ml_roleplay

cleanup() {
  unset JUDGE_API_KEY JUDGE_BASE_URL JUDGE_MODEL
}
trap cleanup EXIT

read -rsp "Judge API Key: " JUDGE_API_KEY
printf '\n'
export JUDGE_API_KEY
export JUDGE_BASE_URL="https://api.minimaxi.com/v1"
export JUDGE_MODEL="MiniMax-M3"

BASE_MODEL="$(python scripts/resource_manager.py resolve-model \
  --project-root "${SCRIPT_DIR}")"

python scripts/eval.py compare \
  --base_model "${BASE_MODEL}" \
  --adapter output/experiments/smoke_20260610_232946/final_model \
  --dataset processed \
  --output_dir output/evaluations/smoke_001
