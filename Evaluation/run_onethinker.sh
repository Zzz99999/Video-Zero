#!/bin/bash
# OneThinker 7-task suite (one ckpt, all tasks in benchmarks.txt).
# Requires bash >= 4 (for `mapfile`).
# Usage: bash run_onethinker.sh <model_path> [gpu_id] [bench_filter]
#   bench_filter: "all" (default) | "mmvu,charades" | "eval_mmvu.json"
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH=${1:?"Usage: bash run_onethinker.sh <model_path> [gpu_id] [bench_filter]"}
GPU_ID=${2:-0}
BENCH_FILTER=${3:-all}

EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_bench.py}"
BENCHMARKS_FILE="${SCRIPT_DIR}/benchmarks.txt"

eval "$(inference_paths onethinker "$MODEL_PATH")"

NFRAMES=${NFRAMES:-0}
FPS=${FPS:-2}
MAX_FRAMES=${MAX_FRAMES:-32}
MAX_PIXELS_VIDEO=${MAX_PIXELS_VIDEO:-$((256*32*32))}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
SUFFIX="${SUFFIX:-_cot_greedy_fps2}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Read benchmarks.txt and apply filter
mapfile -t ALL_BENCHES < <(grep -vE '^\s*(#|$)' "$BENCHMARKS_FILE")
SELECTED=()
if [ "$BENCH_FILTER" = "all" ] || [ -z "$BENCH_FILTER" ]; then
    SELECTED=("${ALL_BENCHES[@]}")
else
    IFS=',' read -ra FILTERS <<< "$BENCH_FILTER"
    for b in "${ALL_BENCHES[@]}"; do
        for f in "${FILTERS[@]}"; do
            f="${f// /}"
            if [ "$b" = "$f" ] || [ "${b}" = "eval_${f}.json" ] || [ "${b%.json}" = "eval_${f}" ]; then
                SELECTED+=("$b")
                break
            fi
        done
    done
fi

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "[ERROR] filter '${BENCH_FILTER}' matched nothing"
    exit 1
fi

echo "============================================"
echo "OneThinker eval"
echo "  Model:    ${MODEL_PATH}"
echo "  GPU:      ${GPU_ID}    fps=${FPS} max_frames=${MAX_FRAMES}"
echo "  Dataset:  ${DATA_ONETHINKER_ROOT}"
echo "  Out dir:  ${OUT_DIR}"
echo "  Benches (${#SELECTED[@]}):"
for b in "${SELECTED[@]}"; do echo "    - $b"; done
echo "============================================"

if [ ! -f "$EVAL_PY" ]; then
    echo "[ERROR] eval py not found: $EVAL_PY"
    exit 1
fi
for b in "${SELECTED[@]}"; do
    if [ ! -f "${DATA_ONETHINKER_ROOT}/${b}" ]; then
        echo "[ERROR] dataset not found: ${DATA_ONETHINKER_ROOT}/${b}"
        exit 1
    fi
done

TOTAL=${#SELECTED[@]}
CURRENT=0
FAILED=0

for bench in "${SELECTED[@]}"; do
    CURRENT=$((CURRENT + 1))
    JSON_PATH="${DATA_ONETHINKER_ROOT}/${bench}"
    BENCH_NAME="$(basename "${bench%.json}")"
    BENCH_LOG="${LOG_DIR}/onethinker_${MODEL_NAME}_${BENCH_NAME}_${TIMESTAMP}.log"

    echo ""
    echo "----- [${CURRENT}/${TOTAL}] ${bench} ($(date)) -----"

    if python -u "${EVAL_PY}" \
        --model_path "${MODEL_PATH}" \
        --input_json "${JSON_PATH}" \
        --out_dir "${OUT_DIR}" \
        --suffix "${SUFFIX}" \
        --base_prefix "${DATA_ONETHINKER_ROOT}" \
        --batch_size ${BATCH_SIZE} \
        --max_tokens ${MAX_TOKENS} \
        --temperature ${TEMPERATURE} \
        --max_pixels_video ${MAX_PIXELS_VIDEO} \
        --nframes ${NFRAMES} \
        --fps ${FPS} \
        --max_frames ${MAX_FRAMES} \
        --max_model_len ${MAX_MODEL_LEN} \
        --gpu_memory_utilization ${GPU_MEM_UTIL} \
        2>&1 | tee -a "$BENCH_LOG"; then
        echo "[OK] ${bench}"
    else
        echo "[FAIL] ${bench}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "OneThinker done: $((TOTAL - FAILED))/${TOTAL} succeeded — results in ${OUT_DIR}"
exit $FAILED
