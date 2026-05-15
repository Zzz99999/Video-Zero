#!/bin/bash
# Centralized dataset / output paths shared by every run_*.sh.
# Override any variable by `export`-ing it before launching, e.g.:
#   export INFERENCE_DATA_ROOT=/mnt/datasets

export INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INFERENCE_DATA_ROOT="${INFERENCE_DATA_ROOT:-${INFERENCE_ROOT}/datasets}"
export INFERENCE_RESULTS_ROOT="${INFERENCE_RESULTS_ROOT:-${INFERENCE_ROOT}/_results}"
export INFERENCE_LOGS_ROOT="${INFERENCE_LOGS_ROOT:-${INFERENCE_ROOT}/_logs}"

# OneThinker 7-task suite
export DATA_ONETHINKER_ROOT="${DATA_ONETHINKER_ROOT:-${INFERENCE_DATA_ROOT}/OneThinker-eval}"

# NExT-GQA / LSDBench / VideoNIAH
export DATA_NEXTGQA_DIR="${DATA_NEXTGQA_DIR:-${INFERENCE_DATA_ROOT}/NExT-GQA/datasets/nextgqa}"
export DATA_NEXTGQA_VIDEO_DIR="${DATA_NEXTGQA_VIDEO_DIR:-${INFERENCE_DATA_ROOT}/NExT-GQA/NExTVideo}"
export DATA_LSDBENCH_JSON="${DATA_LSDBENCH_JSON:-${INFERENCE_DATA_ROOT}/LSDBench/test.json}"
export DATA_LSDBENCH_VIDEO_DIR="${DATA_LSDBENCH_VIDEO_DIR:-${INFERENCE_DATA_ROOT}/LSDBench}"
export DATA_VNBENCH_ANNO="${DATA_VNBENCH_ANNO:-${INFERENCE_DATA_ROOT}/VNBench/VNBench-main-4try.json}"
export DATA_VNBENCH_VIDEO_DIR="${DATA_VNBENCH_VIDEO_DIR:-${INFERENCE_DATA_ROOT}/VNBench/videos}"
