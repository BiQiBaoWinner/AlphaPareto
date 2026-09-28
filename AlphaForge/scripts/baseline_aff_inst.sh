#!/usr/bin/env bash
set -euo pipefail

# Source the common flow functions from the relative path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/baseline_aff_common_flow.sh"

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${PROJECT_DIR}/out/AlphaForge"
cd "${PROJECT_DIR}"

# Handle symlink for 'out' directory if necessary
# Note: In a submission version, users might prefer direct directories,
# but we keep the logic clean.
if [[ ! -e out ]]; then
  echo "[INFO] Creating out directory"
  mkdir -p out
elif [[ -L out ]]; then
  echo "[INFO] 'out' is already a symlink: $(readlink out)"
else
  echo "[INFO] 'out' directory exists."
fi

# Run for specific pools
# run_aff_for_pool "csi300"
run_aff_for_pool "csi800"
# run_aff_for_pool "all"
