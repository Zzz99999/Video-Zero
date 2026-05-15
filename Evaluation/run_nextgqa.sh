#!/bin/bash
# NExT-GQA: VideoQA + (optional) temporal grounding.
# Usage: bash run_nextgqa.sh <model_path> [gpu_id] [fps] [max_frames] [mode] [split]
#   mode  : qa | grounding | both   (default qa)
#   split : test | val              (default test)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH=${1:?"Usage: bash run_nextgqa.sh <model_path> [gpu_id] [fps] [max_frames] [mode] [split]"}
GPU_ID=${2:-0}
FPS=${3:-2.0}
MAX_FRAMES=${4:-32}
MODE=${5:-qa}
SPLIT=${6:-test}
NFRAMES=${NFRAMES:-0}

EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_nextgqa.py}"

eval "$(inference_paths nextgqa "$MODEL_PATH")"
if [ "$NFRAMES" -gt 0 ]; then
    OUTPUT_FILE="${OUT_DIR}/nextgqa_${SPLIT}_${MODE}_${NFRAMES}f.json"
else
    OUTPUT_FILE="${OUT_DIR}/nextgqa_${SPLIT}_${MODE}_fps${FPS}_max${MAX_FRAMES}.json"
fi
LOG_FILE="${LOG_DIR}/nextgqa_${MODEL_NAME}_${SPLIT}_${MODE}_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

BATCH_SIZE=${BATCH_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}

if [ ! -d "${DATA_NEXTGQA_VIDEO_DIR}" ]; then
    echo "[ERROR] video dir not found: ${DATA_NEXTGQA_VIDEO_DIR}"
    exit 1
fi

echo "============================================"
echo "NExT-GQA eval"
echo "  Model:  ${MODEL_PATH}"
echo "  GPU:    ${GPU_ID}    Mode: ${MODE}    Split: ${SPLIT}"
echo "  Sampling: fps=${FPS}, max_frames=${MAX_FRAMES}$([ "$NFRAMES" -gt 0 ] && echo " (override nframes=${NFRAMES})")"
echo "  Data:   ${DATA_NEXTGQA_DIR}"
echo "  Video:  ${DATA_NEXTGQA_VIDEO_DIR}"
echo "  Output: ${OUTPUT_FILE}"
echo "============================================"

if [ "$NFRAMES" -gt 0 ]; then
    NFRAMES_ARGS="--nframes ${NFRAMES}"
else
    NFRAMES_ARGS="--fps ${FPS} --max_frames ${MAX_FRAMES}"
fi

python -u "${EVAL_PY}" \
    --ckpt_path "${MODEL_PATH}" \
    --data_dir "${DATA_NEXTGQA_DIR}" \
    --video_dir "${DATA_NEXTGQA_VIDEO_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --split "${SPLIT}" \
    --mode "${MODE}" \
    ${NFRAMES_ARGS} \
    --batch_size ${BATCH_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature 0.0 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "NExT-GQA done — results: ${OUTPUT_FILE}"
