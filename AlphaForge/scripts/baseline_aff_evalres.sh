#!/usr/bin/env bash
# Result Evaluation Script

# Base directory for the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Evaluate results
# ROOT_PATH is typically where the stage2 results are saved
ROOT_PATH="./out/results/AlphaForge/"

CUDA_VISIBLE_DEVICES=0 python Eval_Res.py --instruments all --root_path "${ROOT_PATH}"
