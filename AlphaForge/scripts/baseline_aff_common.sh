#!/usr/bin/env bash
set -euo pipefail

run_aff_for_pool() {
  local INSTRUMENTS="$1"

  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  FREQ="day"
  CUDA_ID=0
  SAVE_NAME="aff_baseline"

  # stage1 setup (ZOO_SIZE=40 as per user script)
  ZOO_SIZE=40
  CORR_THRESH=0.7
  IC_THRESH=0.03
  ICIR_THRESH=0.1

  # stage2 setup
  N_FACTORS_LIST=(10 20)
  WINDOW="inf"

  CLEAN_PKL=0
  RUN_STAGE1=1
  SEEDS_LIST=(0 42 128 256 512)
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  cd "${PROJECT_DIR}"

  for SEED in "${SEEDS_LIST[@]}"; do
    SEEDS="[$SEED]"

    if [[ "${RUN_STAGE1}" == "1" ]]; then
      if [[ "${CLEAN_PKL}" == "1" ]]; then
        rm -rf pkl
      fi

      echo "[INFO] Stage1 pool=${INSTRUMENTS} seed=${SEED}"
      python train_AFF.py \
        --instruments="${INSTRUMENTS}" \
        --freq="${FREQ}" \
        --seeds="${SEEDS}" \
        --cuda="${CUDA_ID}" \
        --save_name="${SAVE_NAME}" \
        --zoo_size="${ZOO_SIZE}" \
        --corr_thresh="${CORR_THRESH}" \
        --ic_thresh="${IC_THRESH}" \
        --icir_thresh="${ICIR_THRESH}"
    fi

    for N_FACTORS in "${N_FACTORS_LIST[@]}"; do
      if (( N_FACTORS > ZOO_SIZE )); then
        echo "[WARN] skip N_FACTORS=${N_FACTORS} > ZOO_SIZE=${ZOO_SIZE}"
        continue
      fi

      echo "[INFO] Stage2 pool=${INSTRUMENTS} n_factors=${N_FACTORS} seed=${SEED}"
      python combine_AFF.py \
        --instruments="${INSTRUMENTS}" \
        --freq="${FREQ}" \
        --seeds="${SEEDS}" \
        --cuda="${CUDA_ID}" \
        --save_name="${SAVE_NAME}" \
        --n_factors="${N_FACTORS}" \
        --window="${WINDOW}"
    done
  done
}
