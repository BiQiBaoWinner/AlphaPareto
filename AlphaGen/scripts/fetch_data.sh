#!/bin/bash
# Optional: wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
python "$REPO_ROOT/data_collection/fetch_baostock_data.py"
