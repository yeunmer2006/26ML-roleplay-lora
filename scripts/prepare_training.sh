#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-ml_roleplay}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_DIR="${MODEL_DIR:-}"
DATASET_ID="${DATASET_ID:-KaraKaraWitch/PIPPA-ShareGPT-formatted}"
DATA_DIR="${DATA_DIR:-processed}"

log() {
  printf '[prepare] %s\n' "$*"
}

if ! command -v conda >/dev/null 2>&1; then
  for candidate in "${HOME}/miniconda3/etc/profile.d/conda.sh" \
                   "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 Miniconda/Anaconda。" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "无法激活 Conda 环境: ${ENV_NAME}" >&2
  exit 1
fi

log "环境: ${CONDA_DEFAULT_ENV} ($(python --version 2>&1))"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"需要 Python 3.10，当前为 {sys.version.split()[0]}")
PY

log "安装/校准训练依赖（不会重装 PyTorch）"
python -m pip install -r "${PROJECT_ROOT}/requirements-training.txt"

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}, CUDA build: {torch.version.cuda}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {props.total_memory / 2**30:.1f} GiB")
else:
    print("警告: 当前进程看不到 CUDA；可完成下载准备，但不能训练。")
PY

RESOURCE_ARGS=(
  prepare
  --project-root "${PROJECT_ROOT}"
  --model-id "${MODEL_ID}"
  --dataset-id "${DATASET_ID}"
  --data-dir "${DATA_DIR}"
)
if [[ -n "${MODEL_DIR}" ]]; then
  RESOURCE_ARGS+=(--model-dir "${MODEL_DIR}")
fi

log "按本地优先顺序检查模型和数据"
RESOURCE_JSON="$(python "${PROJECT_ROOT}/scripts/resource_manager.py" "${RESOURCE_ARGS[@]}")"

log "准备完成"
RESOURCE_JSON="${RESOURCE_JSON}" python - <<'PY'
import json
import os

resources = json.loads(os.environ["RESOURCE_JSON"])
print(f"[prepare] 模型: {resources['model_path']}")
print(f"[prepare] 数据: {resources['data_path']}")
PY
log "冒烟测试: bash scripts/run_training.sh smoke"
log "正式训练: bash scripts/run_training.sh train"
