#!/bin/bash
# LSDBench
# Usage: bash run_lsdbench.sh <model_path> [gpu_id] [fps] [max_frames]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH=${1:?"Usage: bash run_lsdbench.sh <model_path> [gpu_id] [fps] [max_frames]"}
GPU_ID=${2:-0}
FPS=${3:-2.0}
MAX_FRAMES=${4:-32}

EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_lsdbench.py}"

eval "$(inference_paths lsdbench "$MODEL_PATH")"
OUTPUT_FILE="${OUT_DIR}/lsdbench_fps${FPS}_max${MAX_FRAMES}.json"
LOG_FILE="${LOG_DIR}/lsdbench_${MODEL_NAME}_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export FORCE_QWENVL_VIDEO_READER=decord

BATCH_SIZE=${BATCH_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}

if ! python -c "import decord" 2>/dev/null; then
    echo "[FATAL] decord not installed. Run: pip install decord"
    exit 1
fi

echo "============================================"
echo "LSDBench eval (vLLM, decord)"
echo "  Model:  ${MODEL_PATH}"
echo "  GPU:    ${GPU_ID}    fps=${FPS}, max_frames=${MAX_FRAMES}"
echo "  Data:   ${DATA_LSDBENCH_JSON}"
echo "  Video:  ${DATA_LSDBENCH_VIDEO_DIR}"
echo "  Output: ${OUTPUT_FILE}"
echo "============================================"

python -u "${EVAL_PY}" \
    --ckpt_path "${MODEL_PATH}" \
    --data_path "${DATA_LSDBENCH_JSON}" \
    --video_dir "${DATA_LSDBENCH_VIDEO_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --fps ${FPS} \
    --max_frames ${MAX_FRAMES} \
    --batch_size ${BATCH_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature 0.0 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "LSDBench done — results: ${OUTPUT_FILE}"
