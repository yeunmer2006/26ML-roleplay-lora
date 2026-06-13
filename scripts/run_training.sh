#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-ml_roleplay}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_DIR="${MODEL_DIR:-}"
DATA_DIR="${DATA_DIR:-processed}"
MODE="${1:-}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/run_training.sh smoke
  bash scripts/run_training.sh train
  bash scripts/run_training.sh train --resume output/experiments/run_001/checkpoint-250

可选参数:
  --resume PATH       从 checkpoint 继续训练
  --output-dir PATH   指定实验输出目录
  --skip-benchmark    跳过正式训练前的 50-step 时间预检
EOF
}

if [[ "${MODE}" != "smoke" && "${MODE}" != "train" ]]; then
  usage >&2
  exit 1
fi
shift

RESUME=""
OUTPUT_DIR=""
SKIP_BENCHMARK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME="${2:?--resume 需要路径}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir 需要路径}"
      shift 2
      ;;
    --skip-benchmark)
      SKIP_BENCHMARK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

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
  echo "未找到 conda。" >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，无法启动 QLoRA 训练")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
PY

MODEL_ARGS=(
  resolve-model
  --project-root "${PROJECT_ROOT}"
  --model-id "${MODEL_ID}"
)
if [[ -n "${MODEL_DIR}" ]]; then
  MODEL_ARGS+=(--model-dir "${MODEL_DIR}")
fi
MODEL_DIR="$(python "${PROJECT_ROOT}/scripts/resource_manager.py" "${MODEL_ARGS[@]}")"
DATA_DIR="$(python "${PROJECT_ROOT}/scripts/resource_manager.py" prepare-data \
  --project-root "${PROJECT_ROOT}" \
  --data-dir "${DATA_DIR}")"
printf '[train] 模型: %s\n' "${MODEL_DIR}"
printf '[train] 数据: %s\n' "${DATA_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/output/experiments/${MODE}_${timestamp}"
elif [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

if [[ -n "${RESUME}" && "${RESUME}" != /* ]]; then
  RESUME="${PROJECT_ROOT}/${RESUME}"
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/console.log"

run_logged() {
  printf '[train] command:' | tee -a "${LOG_FILE}"
  printf ' %q' "$@" | tee -a "${LOG_FILE}"
  printf '\n' | tee -a "${LOG_FILE}"
  "$@" 2>&1 | tee -a "${LOG_FILE}"
}

COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --model_path "${MODEL_DIR}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${MODE}" == "smoke" ]]; then
  run_logged python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/train_smoke.yaml" \
    --max_train_samples 100 \
    --max_eval_samples 20 \
    --max_steps 10 \
    "${COMMON_ARGS[@]}"
  exit 0
fi

if [[ -z "${RESUME}" && "${SKIP_BENCHMARK}" -eq 0 ]]; then
  BENCHMARK_DIR="${OUTPUT_DIR}/benchmark"
  run_logged python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/train_4060.yaml" \
    --max_train_samples 4000 \
    --max_eval_samples 200 \
    --benchmark_steps 50 \
    --max_runtime_minutes 110 \
    --output_dir "${BENCHMARK_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_path "${MODEL_DIR}"
fi

TRAIN_ARGS=(
  python "${PROJECT_ROOT}/scripts/train.py"
  --config "${PROJECT_ROOT}/configs/train_4060.yaml"
  --max_train_samples 4000
  --max_eval_samples 200
  "${COMMON_ARGS[@]}"
)
if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME}")
fi
run_logged "${TRAIN_ARGS[@]}"
