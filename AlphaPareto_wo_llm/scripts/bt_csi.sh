#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
python "$REPO_ROOT/backtest.py" --pool-capacity 10 --instruments 'csi300' &
python "$REPO_ROOT/backtest.py" --pool-capacity 20 --instruments 'csi300' &
python "$REPO_ROOT/backtest.py" --pool-capacity 50 --instruments 'csi300' &
python "$REPO_ROOT/backtest.py" --pool-capacity 100 --instruments 'csi300' &
python "$REPO_ROOT/backtest.py" --pool-capacity 10 --instruments 'csi500' &
python "$REPO_ROOT/backtest.py" --pool-capacity 20 --instruments 'csi500' &
# python "$REPO_ROOT/backtest.py" --pool-capacity 50 --instruments 'csi500' &
# python "$REPO_ROOT/backtest.py" --pool-capacity 100 --instruments 'csi500' &
