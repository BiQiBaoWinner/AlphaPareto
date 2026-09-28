#!/bin/bash
set -euo pipefail

ALPHAPARETO_WO_LLM_ROOT="${ALPHAPARETO_WO_LLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
OUT_DIR="${ALPHAPARETO_WO_LLM_OUT_DIR:-$ALPHAPARETO_WO_LLM_ROOT/out}"
LOG_DIR="${ALPHAPARETO_WO_LLM_LOG_DIR:-$OUT_DIR/logs}"

# Optional: source "${CONDA_BASE}/bin/activate" && conda activate your-env
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 python "${ALPHAPARETO_WO_LLM_ROOT}/scripts/rl.py" --pool-capacity 10 --instruments 'csi300' --leading-days 20 --llm 'deepseek-llm-7b-chat' 2>&1 | tee "${LOG_DIR}/llm_ds7b_csi300_10_20_output_newrange2.log" &
CUDA_VISIBLE_DEVICES=0 python "${ALPHAPARETO_WO_LLM_ROOT}/scripts/rl.py" --pool-capacity 20 --instruments 'csi300' --leading-days 20 --llm 'deepseek-llm-7b-chat' 2>&1 | tee "${LOG_DIR}/llm_ds7b_csi300_20_20_output_newrange2.log" &
# CUDA_VISIBLE_DEVICES=0 python "${ALPHAPARETO_WO_LLM_ROOT}/scripts/rl.py" --pool-capacity 50 --instruments 'csi300' --leading-days 20 2>&1 | tee "${LOG_DIR}/llm_csi300_50_20_output.log" &
# CUDA_VISIBLE_DEVICES=0 python "${ALPHAPARETO_WO_LLM_ROOT}/scripts/rl.py" --pool-capacity 100 --instruments 'csi300' --leading-days 20 2>&1 | tee "${LOG_DIR}/llm_csi300_100_20_output.log" &
