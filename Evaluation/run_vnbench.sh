#!/bin/bash
# VNBench (VideoNIAH) — 4-try strict eval.
# Usage: bash run_vnbench.sh <model_path> [gpu_id] [fps] [max_frames]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH=${1:?"Usage: bash run_vnbench.sh <model_path> [gpu_id] [fps] [max_frames]"}
GPU_ID=${2:-0}
FPS=${3:-2.0}
MAX_FRAMES=${4:-32}
NFRAMES=${NFRAMES:-0}

EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_vnbench.py}"

eval "$(inference_paths vnbench "$MODEL_PATH")"
if [ "$NFRAMES" -gt 0 ]; then
    OUTPUT_FILE="${OUT_DIR}/vnbench_${NFRAMES}f.json"
else
    OUTPUT_FILE="${OUT_DIR}/vnbench_fps${FPS}_max${MAX_FRAMES}.json"
fi
LOG_FILE="${LOG_DIR}/vnbench_${MODEL_NAME}_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

BATCH_SIZE=${BATCH_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}

echo "============================================"
echo "VNBench eval (vLLM)"
echo "  Model:      ${MODEL_PATH}"
echo "  GPU:        ${GPU_ID}    fps=${FPS}, max_frames=${MAX_FRAMES}$([ "$NFRAMES" -gt 0 ] && echo " (override nframes=${NFRAMES})")"
echo "  Annotation: ${DATA_VNBENCH_ANNO}"
echo "  Video dir:  ${DATA_VNBENCH_VIDEO_DIR}"
echo "  Output:     ${OUTPUT_FILE}"
echo "============================================"

if [ "$NFRAMES" -gt 0 ]; then
    NFRAMES_ARGS="--nframes ${NFRAMES}"
else
    NFRAMES_ARGS="--fps ${FPS} --max_frames ${MAX_FRAMES}"
fi

python -u "${EVAL_PY}" \
    --ckpt_path "${MODEL_PATH}" \
    --annotation_path "${DATA_VNBENCH_ANNO}" \
    --video_dir "${DATA_VNBENCH_VIDEO_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    ${NFRAMES_ARGS} \
    --batch_size ${BATCH_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature 0.0 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "VNBench done — results: ${OUTPUT_FILE}"
