#!/bin/bash
# Shared runtime env vars. All run_*.sh source paths.sh first, then this.

export DECORD_EOF_RETRY_MAX=2048001
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# HF cache: default to OFFLINE so eval runs never silently re-download mid-experiment.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# export FORCE_QWENVL_VIDEO_READER=decord

# Helper: derive output / log dirs.
# Usage:  eval "$(inference_paths <bench_name> <MODEL_PATH>)"
# Sets:   $OUT_DIR / $LOG_DIR / $TIMESTAMP / $MODEL_NAME
inference_paths() {
    local bench="$1"
    local model_path="$2"
    local model_name
    model_name="$(basename "$model_path")"
    local ts
    ts="$(date +"%Y%m%d_%H%M%S")"
    local out_dir="${INFERENCE_RESULTS_ROOT}/${bench}/${model_name}"
    local log_dir="${INFERENCE_LOGS_ROOT}/${bench}"
    mkdir -p "$out_dir" "$log_dir"
    cat <<EOF
MODEL_NAME='${model_name}'
TIMESTAMP='${ts}'
OUT_DIR='${out_dir}'
LOG_DIR='${log_dir}'
EOF
}
export -f inference_paths
