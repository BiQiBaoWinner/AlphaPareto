#!/bin/bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
OUT_DIR="${ALPHAGEN_OUT_DIR:-$REPO_ROOT/out}"
LOG_DIR="${ALPHAGEN_LOG_DIR:-$OUT_DIR/logs}"

# Optional: source "${CONDA_BASE}/bin/activate" && conda activate your-env
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 10 --instruments 'csi800' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi800_10_20_output_newrange2.log" &
CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 20 --instruments 'csi800' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi800_20_20_output_newrange2.log" &
# CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 50 --instruments 'csi300' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi300_50_20_output.log" &
# CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 100 --instruments 'csi300' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi300_100_20_output.log" &
#
# CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 10 --instruments 'csi1000' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi1000_10_20_output.log" &
# CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 20 --instruments 'csi1000' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi1000_20_20_output.log" &
# CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 50 --instruments 'csi500' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi500_50_20_output.log" &
# CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/scripts/rl.py" --pool-capacity 100 --instruments 'csi500' --leading-days 20 2>&1 | tee "${LOG_DIR}/csi500_100_20_output.log" &
