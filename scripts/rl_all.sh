#!/usr/bin/env bash
# Centralized Reinforcement Learning (RL) Launch Script for Alpha Projects
set -euo pipefail

# Project configurations
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/out/logs"
mkdir -p "${LOG_DIR}"

# Function to run RL for a specific project
run_rl() {
    local PROJECT_NAME="$1" # e.g., AlphaPareto, AlphaPareto_wo_llm
    local INSTRUMENTS="$2"  # csi300, csi800, all
    local CAPACITY="$3"     # 10, 20
    local LLM_MODEL="${4:-NoneLLM}"
    local C0="${5:-0}"
    local CUDA_ID="${6:-0}"

    local WORK_DIR="${BASE_DIR}/${PROJECT_NAME}/scripts"
    
    if [[ ! -d "${WORK_DIR}" ]]; then
        echo "[ERROR] Project directory ${WORK_DIR} not found."
        return 1
    fi

    echo "[INFO] Starting RL: Project=${PROJECT_NAME}, Pool=${INSTRUMENTS}, Capacity=${CAPACITY}, LLM=${LLM_MODEL}"
    
    # Identify the correct script (some projects might use different sub-paths, adjust as needed)
    local SCRIPT_PATH="${WORK_DIR}/rl.py"
    
    CUDA_VISIBLE_DEVICES="${CUDA_ID}" python "${SCRIPT_PATH}" \
        --pool-capacity "${CAPACITY}" \
        --instruments "${INSTRUMENTS}" \
        --leading-days 20 \
        --llm "${LLM_MODEL}" \
        --c0 "${C0}" \
        2>&1 | tee "${LOG_DIR}/${PROJECT_NAME}_${INSTRUMENTS}_${CAPACITY}_${LLM_MODEL}.log" &
}

# Example usage (commented out by default)
# AlphaPareto csi300
# run_rl "AlphaPareto" "csi300" 10 "deepseek-r1:1.5b" 0.09 0
# run_rl "AlphaPareto" "csi300" 20 "deepseek-r1:1.5b" 0.09 0

# AlphaPareto wo LLM
# run_rl "AlphaPareto_wo_llm" "csi300" 10 "NoneLLM" 0.09 1

wait
echo "[INFO] All RL processes finished."
