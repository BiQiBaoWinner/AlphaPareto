#!/bin/bash
# Optional: download a qlib binary tarball and extract under ~/.qlib/qlib_data/cn_data (see Qlib docs).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
python "$REPO_ROOT/data_collection/fetch_baostock_data.py"
