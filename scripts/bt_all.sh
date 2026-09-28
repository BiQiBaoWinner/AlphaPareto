#!/usr/bin/env bash
# Centralized Backtest Script for Alpha Projects
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/out/logs/bt"
mkdir -p "${LOG_DIR}"

run_backtest() {
    local PROJECT_DIR_NAME="$1" # e.g., AlphaPareto, Alpha158
    local INSTRUMENTS="$2"      # csi300, csi800, all
    local PROJECT_ALIAS="$3"    # Name used inside backtest.py for reporting
    local BENCHMARK="$4"        # SH000300, SH000906, SH000001
    local CUDA_ID="${5:-0}"

    local WORK_DIR="${BASE_DIR}/${PROJECT_DIR_NAME}"
    cd "${WORK_DIR}"

    echo "[INFO] Starting Backtest: Project=${PROJECT_ALIAS}, Pool=${INSTRUMENTS}, Benchmark=${BENCHMARK}"

    CUDA_VISIBLE_DEVICES="${CUDA_ID}" python backtest.py \
        --instruments "${INSTRUMENTS}" \
        --which-project "${PROJECT_ALIAS}" \
        --benchmark "${BENCHMARK}" \
        --topk 50 \
        --dropN 5 \
        2>&1 | tee "${LOG_DIR}/${PROJECT_ALIAS}_${INSTRUMENTS}_backtest.log" &
}

# Example Backtests (Uncomment to use)
# AlphaPareto
# run_backtest "AlphaPareto" "csi300" "AlphaPareto" "SH000300" 0
# run_backtest "AlphaPareto" "csi800" "AlphaPareto" "SH000906" 0

# AlphaForge
# run_backtest "AlphaForge" "csi300" "AlphaForge" "SH000300" 0

wait
echo "[INFO] All backtests completed."
