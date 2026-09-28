#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/out/logs}"
mkdir -p "$LOG_DIR"

# Run baseline models: GP and DSO
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/gp.py" 2>&1 | tee "$LOG_DIR/baseline_gp_output_newrange2.log" &
# Optional: use a dedicated env (e.g. tf1) for dso
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/dso.py" 2>&1 | tee "$LOG_DIR/baseline_dso_output.log" &

# Run baseline models: MLP and XGB
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/mlp_qlib.py" --instrument "csi300" \
                                                                                            --train-start "2013-07-01" \
                                                                                            --train-end "2023-06-30" \
                                                                                            --valid-start "2023-07-01" \
                                                                                            --valid-end "2024-06-30" \
                                                                                            --test-start "2024-07-01" \
                                                                                            --test-end "2025-06-30" \
                                                                                            --hidden-sizes 1024 \
                                                                                            --lr 1e-5 \
                                                                                            --batch-size 10240 \
                                                                                            --epochs 1000 \
                                                                                            --patience 15 2>&1 | tee "$LOG_DIR/mlp_qlib_csi300_output.log" &
# pid_300=$!

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/xgb_qlib.py" --instrument "csi300" \
                                                                                            --train-start "2013-07-01" \
                                                                                            --train-end "2023-06-30" \
                                                                                            --valid-start "2023-07-01" \
                                                                                            --valid-end "2024-06-30" \
                                                                                            --test-start "2024-07-01" \
                                                                                            --test-end "2025-06-30" 2>&1 | tee "$LOG_DIR/xgb_qlib_csi300_output.log" &

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/mlp_qlib.py" --instrument "csi800" \
                                                                                            --train-start "2013-07-01" \
                                                                                            --train-end "2023-06-30" \
                                                                                            --valid-start "2023-07-01" \
                                                                                            --valid-end "2024-06-30" \
                                                                                            --test-start "2024-07-01" \
                                                                                            --test-end "2025-06-30" \
                                                                                            --hidden-sizes 1024 \
                                                                                            --lr 1e-5 \
                                                                                            --batch-size 10240 \
                                                                                            --epochs 1000 \
                                                                                            --patience 15 2>&1 | tee "$LOG_DIR/mlp_qlib_csi800_output.log" &
# pid_800=$!

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/xgb_qlib.py" --instrument "csi800" \
                                                                                            --train-start "2013-07-01" \
                                                                                            --train-end "2023-06-30" \
                                                                                            --valid-start "2023-07-01" \
                                                                                            --valid-end "2024-06-30" \
                                                                                            --test-start "2024-07-01" \
                                                                                            --test-end "2025-06-30" 2>&1 | tee "$LOG_DIR/xgb_qlib_csi800_output.log" &


# wait "$pid_300" "$pid_800"
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/mlp_qlib.py" --instrument "all" \
                                                                                            --train-start "2013-07-01" \
                                                                                            --train-end "2023-06-30" \
                                                                                            --valid-start "2023-07-01" \
                                                                                            --valid-end "2024-06-30" \
                                                                                            --test-start "2024-07-01" \
                                                                                            --test-end "2025-06-30" \
                                                                                            --hidden-sizes 1024 \
                                                                                            --lr 1e-5 \
                                                                                            --batch-size 10240 \
                                                                                            --epochs 1000 \
                                                                                            --patience 15 2>&1 | tee "$LOG_DIR/mlp_qlib_mkt_output.log"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/xgb_qlib.py" --instrument "all" \
                                                                                            --train-start "2013-07-01" \
                                                                                            --train-end "2023-06-30" \
                                                                                            --valid-start "2023-07-01" \
                                                                                            --valid-end "2024-06-30" \
                                                                                            --test-start "2024-07-01" \
                                                                                            --test-end "2025-06-30" 2>&1 | tee "$LOG_DIR/xgb_qlib_mkt_output.log" &
