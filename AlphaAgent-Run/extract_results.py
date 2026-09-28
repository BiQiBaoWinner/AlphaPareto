#!/usr/bin/env python3
"""
Extract factor expressions from AlphaAgent run logs, evaluate IC, and produce
stepover_test_results.json.

This script:
1. Walks the AlphaAgent log directory to find FactorTask objects in pkl files
2. Groups factors by iteration round (using __session__ directory timestamps)
3. Evaluates each round's factors independently (IC-weighted combination per round)
4. Selects the best round by combined valid IC
5. Outputs the best round's results as stepover_test_results.json

Usage:
    python extract_results.py --log_dir ./AlphaAgent/log_csi300_seed0 --output ./results/csi300/seed_0/stepover_test_results.json
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ===================== Time Configuration =====================
TRAIN_START = "2013-07-01"
TRAIN_END   = "2023-06-30"
VALID_START = "2023-07-01"
VALID_END   = "2024-06-30"
TEST_START  = "2024-07-01"
TEST_END    = "2025-06-30"
DATA_START  = "2012-07-01"
DATA_END    = "2025-06-30"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    SCRIPT_DIR / "AlphaAgent" / "alphaagent" / "scenarios" / "qlib"
    / "experiment" / "factor_data_template" / "daily_pv.h5"
)


# ============================================================
# Step 1: Extract factor expressions from pkl log files
# ============================================================

def normalize_expr(expr):
    """Normalize a factor expression string."""
    if expr is None:
        return None
    expr = str(expr).strip()
    if not expr:
        return None
    return " ".join(expr.split())


def _register_factor(expr, name, factors, seen, source):
    """Register a factor expression if it's new."""
    expr = normalize_expr(expr)
    if expr is None or expr in seen:
        return
    seen.add(expr)
    factors.append({
        "expression": expr,
        "name": name if name else f"f{len(factors)}",
        "source": source,
    })


def _dig(obj, factors, seen, depth, source):
    """Recursively extract factor_expression from objects, skipping workspace data."""
    if depth > 15 or obj is None:
        return

    # Direct hit: FactorTask object with factor_expression
    if hasattr(obj, "factor_expression"):
        try:
            expr = getattr(obj, "factor_expression")
            name = getattr(obj, "factor_name", f"f{len(factors)}")
            _register_factor(expr, name, factors, seen, source)
        except Exception:
            pass
        return  # Don't recurse into FactorTask's sub_workspace_list

    # Skip large workspace objects
    cls_name = type(obj).__name__
    if "Workspace" in cls_name or "FBWorkspace" in cls_name:
        return

    # Recurse into sub_tasks (contains FactorTask objects)
    if hasattr(obj, "sub_tasks"):
        try:
            items = getattr(obj, "sub_tasks")
            if items is not None:
                for item in items:
                    _dig(item, factors, seen, depth + 1, source)
        except Exception:
            pass

    # Recurse into based_experiments
    if hasattr(obj, "based_experiments"):
        try:
            items = getattr(obj, "based_experiments")
            if items is not None:
                for item in items:
                    _dig(item, factors, seen, depth + 1, source)
        except Exception:
            pass

    # Recurse into lists and tuples
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _dig(item, factors, seen, depth + 1, source)

    # Recurse into dicts
    if isinstance(obj, dict):
        for v in obj.values():
            _dig(v, factors, seen, depth + 1, source)


def _get_outer_round_boundaries(log_root):
    """
    Detect OUTER hypothesis round boundaries from __session__ directory.

    AlphaAgent has two nested loops:
      - Outer: evolving_n iterations (propose → construct → calculate → backtest → feedback)
      - Inner: CoSTEER code refinement iterations within factor_calculate

    The __session__/ directory numbers are a FLAT global counter across both loops.
    We detect outer rounds by finding directories containing "factor_propose" step files,
    since factor_propose only runs once per outer iteration.

    Returns list of (outer_round_idx, start_mtime) sorted by time.
    """
    session_dirs = list(log_root.rglob("__session__"))
    if not session_dirs:
        return None

    session_dir = session_dirs[0]
    boundaries = []
    outer_idx = 0

    # Sort by numeric directory name
    iter_dirs = sorted(
        [d for d in session_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    for d in iter_dirs:
        # Check if this directory contains a factor_propose step
        has_propose = any(
            "factor_propose" in f.name
            for f in d.iterdir()
            if not f.is_dir()
        )
        if has_propose:
            # Get the earliest file time as the start of this outer round
            mtimes = []
            for f in d.iterdir():
                if not f.is_dir():
                    try:
                        mtimes.append(f.stat().st_mtime)
                    except OSError:
                        pass
            if mtimes:
                boundaries.append((outer_idx, min(mtimes)))
                outer_idx += 1

    return boundaries if boundaries else None


def _assign_round(pkl_mtime, boundaries):
    """Assign a pkl file to an outer round based on its modification time."""
    if boundaries is None:
        return 0

    assigned = 0
    for round_idx, start_time in boundaries:
        if pkl_mtime >= start_time - 120:  # 2 min tolerance
            assigned = round_idx
    return assigned


def extract_factors_by_round(log_dir):
    """
    Extract factor expressions grouped by iteration round.

    Uses __session__ directory timestamps to determine which round each
    experiment generation pkl belongs to.

    Returns:
        dict: {round_num: [factor_dicts]} or None if extraction fails
    """
    log_root = Path(log_dir)

    if not log_root.exists():
        print(f"[ERROR] Log directory not found: {log_dir}", file=sys.stderr)
        return None

    # Get outer round boundaries from __session__ (using factor_propose as delimiter)
    boundaries = _get_outer_round_boundaries(log_root)
    if boundaries:
        print(f"  Found {len(boundaries)} outer hypothesis rounds in __session__")
    else:
        print(f"  [WARN] No __session__ found, all factors assigned to round 0")

    # Collect all experiment generation pkl files
    target_pkls = sorted(log_root.rglob("experiment generation/**/*.pkl"))
    target_pkls += sorted(log_root.rglob("hypothesis gen*/**/*.pkl"))
    target_pkls = list(dict.fromkeys(target_pkls))

    if not target_pkls:
        print(f"[WARN] No 'experiment generation' pkl found, fallback to full scan",
              file=sys.stderr)
        all_pkls = [
            p for p in sorted(log_root.rglob("*.pkl"))
            if not any(skip in str(p) for skip in [
                "evolving code", "evolving feedback", "/coder/", "/runner/",
                "RD-Agent_workspace", "__session__"
            ])
        ]
        target_pkls = all_pkls

    if not target_pkls:
        print(f"[ERROR] No pkl files found in: {log_dir}", file=sys.stderr)
        return None

    print(f"  Scanning {len(target_pkls)} pkl files ...")

    # Extract factors and assign to rounds
    all_seen = set()
    rounds = {}

    for pkl_file in sorted(target_pkls, key=lambda p: p.stat().st_mtime):
        pkl_mtime = pkl_file.stat().st_mtime
        round_num = _assign_round(pkl_mtime, boundaries)

        factors_this_file = []
        seen_this_file = set()
        try:
            with open(pkl_file, "rb") as f:
                obj = pickle.load(f)
            _dig(obj, factors_this_file, seen_this_file, 0, str(pkl_file))
        except Exception:
            pass

        for fac in factors_this_file:
            expr = fac["expression"]
            if expr not in all_seen:
                all_seen.add(expr)
                rounds.setdefault(round_num, []).append(fac)

    return rounds if rounds else None


# ============================================================
# Step 2: IC Computation Utilities
# ============================================================

def calc_cross_sectional_ic(df, factor_col, label_col):
    """
    Compute mean cross-sectional IC (Pearson correlation per day, then average).
    This is the standard IC metric used in quant factor evaluation.
    """
    if df.empty:
        return 0.0
    ic_values = []
    for _, group in df.groupby(level="datetime", sort=True):
        if group[factor_col].notna().sum() < 2 or group[label_col].notna().sum() < 2:
            continue
        ic = group[factor_col].corr(group[label_col])
        if pd.notna(ic):
            ic_values.append(float(ic))
    return float(np.mean(ic_values)) if ic_values else 0.0


# ============================================================
# Step 3: Factor Evaluation Engine (uses AlphaAgent's expr_parser)
# ============================================================

_modules_loaded = False
_parse_expression = None
_parse_symbol = None
_function_lib_ns = None
_bridge_translate = None


def _ensure_modules():
    """Lazy-load AlphaAgent's expr_parser and function_lib modules."""
    global _modules_loaded, _parse_expression, _parse_symbol, _function_lib_ns, _bridge_translate
    if _modules_loaded:
        return

    # AlphaAgent should be importable if pip install -e was done
    from alphaagent.components.coder.factor_coder.expr_parser import (
        parse_expression,
        parse_symbol,
    )
    import alphaagent.components.coder.factor_coder.function_lib as flib
    from alphaagent.components.coder.factor_coder.alphapareto_bridge import translate

    _parse_expression = parse_expression
    _parse_symbol = parse_symbol
    _bridge_translate = translate
    _function_lib_ns = {k: v for k, v in vars(flib).items() if not k.startswith("__")}
    _function_lib_ns["pd"] = pd
    _function_lib_ns["np"] = np
    _modules_loaded = True


def _fast_ts_corr(df1, df2, p=5):
    """
    Vectorized rolling correlation (replaces the slow groupby.apply version
    in function_lib for TS_CORR).
    """
    if not (isinstance(df1, pd.Series) and isinstance(df2, pd.Series)):
        return _function_lib_ns["TS_CORR"](df1, df2, p)
    try:
        w1 = df1.unstack(level="instrument")
        w2 = df2.unstack(level="instrument")
        result = w1.rolling(p, min_periods=2).corr(w2)
        out = result.stack()
        out.index.names = ["datetime", "instrument"]
        return out
    except Exception:
        return _function_lib_ns["TS_CORR"](df1, df2, p)


def eval_factor_on_h5(expr_str, df):
    """
    Evaluate a single factor expression on a price-volume DataFrame using
    AlphaAgent's expr_parser + function_lib.

    Accepts both alphapareto format (e.g. Div($close,10)) and function_lib
    format (e.g. TS_STD($close, 10)). Alphapareto expressions are translated
    to function_lib format first via the bridge.

    Returns: pd.Series with MultiIndex(datetime, instrument)
    """
    _ensure_modules()

    # Translate alphapareto format to function_lib format
    try:
        funclib_expr = _bridge_translate(expr_str)
    except Exception as e:
        # Check if it looks like alphapareto format (starts with an operator name)
        _ap_ops = {"Abs", "Log", "Add", "Sub", "Mul", "Div", "Greater", "Less",
                   "Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad",
                   "Delta", "WMA", "EMA", "Cov", "Corr"}
        first_token = expr_str.strip().split("(")[0].strip()
        if first_token in _ap_ops:
            raise ValueError(f"Malformed alphapareto expression: {e}")
        # Might be function_lib format already — pass through
        funclib_expr = expr_str

    parsed = _parse_symbol(funclib_expr, df.columns)
    parsed = _parse_expression(parsed)

    # Replace column names: $close -> df['$close']
    for col in df.columns:
        parsed = parsed.replace(col[1:], f"df['{col}']")

    ns = dict(_function_lib_ns)
    ns["df"] = df
    ns["TS_CORR"] = _fast_ts_corr  # Use vectorized version

    result = eval(parsed, ns)
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    return result.astype(float)


def compute_label(df):
    """
    Compute forward return label: close[t+2]/close[t+1] - 1
    (matches the Ref($close,-2)/Ref($close,-1)-1 in qlib configs)
    """
    close = df["$close"]
    label = close.groupby(level="instrument").transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    label.name = "label"
    return label


# ============================================================
# Step 3b: Alphapareto Format Conversion
# ============================================================

_bridge_loaded = False
_bridge_normalize = None

# Mapping from function_lib names to alphapareto operator names
_FUNCLIB_TO_AP = {
    "ABS": "Abs", "LOG": "Log",
    "TS_MEAN": "Mean", "TS_SUM": "Sum", "TS_STD": "Std", "TS_VAR": "Var",
    "TS_MAX": "Max", "TS_MIN": "Min", "TS_MEDIAN": "Med", "TS_MAD": "Mad",
    "DELAY": "Ref", "DELTA": "Delta", "WMA": "WMA", "EMA": "EMA",
    "TS_CORR": "Corr", "TS_COVARIANCE": "Cov",
    "ADD": "Add", "SUBTRACT": "Sub", "MULTIPLY": "Mul", "DIVIDE": "Div",
}

import re as _re


def _ensure_bridge():
    global _bridge_loaded, _bridge_normalize
    if _bridge_loaded:
        return
    try:
        from alphaagent.components.coder.factor_coder.alphapareto_bridge import normalize
        _bridge_normalize = normalize
    except ImportError:
        pass
    _bridge_loaded = True


def _funclib_to_alphapareto(expr_str):
    """Best-effort rename of function_lib operators to alphapareto names."""
    result = expr_str
    for old, new in sorted(_FUNCLIB_TO_AP.items(), key=lambda x: -len(x[0])):
        result = _re.sub(r'\b' + old + r'\b', new, result)
    return result


def _to_alphapareto_format(expr_str):
    """Convert an expression to canonical alphapareto format if possible."""
    _ensure_bridge()
    if _bridge_normalize is None:
        # Fallback: best-effort rename only
        return _funclib_to_alphapareto(expr_str)

    # Try direct parse (already alphapareto format)
    try:
        return _bridge_normalize(expr_str)
    except Exception:
        pass

    # Try after renaming function_lib → alphapareto
    try:
        return _bridge_normalize(_funclib_to_alphapareto(expr_str))
    except Exception:
        pass

    # Fallback
    return expr_str


# ============================================================
# Step 4: Per-Round Evaluation Pipeline
# ============================================================

def _eval_single_factor(expr, idx, df, label):
    """Evaluate one factor expression and return its IC record.

    Never returns None — failed factors get IC=0 to preserve factor count.
    """
    try:
        fv = eval_factor_on_h5(expr, df)
        if fv is None or (hasattr(fv, 'empty') and fv.empty):
            print(f"      Factor {idx}: FAILED - empty result (IC set to 0)")
            return {
                "idx": idx, "expr": expr,
                "valid_ic": 0.0, "test_ic": 0.0, "sign": 1.0, "failed": True,
            }

        fv.name = f"f{idx}"
        joined = pd.concat([fv, label], axis=1).dropna()
        if joined.empty:
            print(f"      Factor {idx}: FAILED - no valid data after join (IC set to 0)")
            return {
                "idx": idx, "expr": expr,
                "valid_ic": 0.0, "test_ic": 0.0, "sign": 1.0, "failed": True,
            }

        dt = joined.index.get_level_values("datetime")
        valid_data = joined[(dt >= VALID_START) & (dt <= VALID_END)]
        test_data  = joined[(dt >= TEST_START)  & (dt <= TEST_END)]

        valid_ic = calc_cross_sectional_ic(valid_data, f"f{idx}", "label")
        test_ic  = calc_cross_sectional_ic(test_data,  f"f{idx}", "label")

        return {
            "idx": idx,
            "expr": expr,
            "valid_ic": float(valid_ic),
            "test_ic": float(test_ic),
            "sign": -1.0 if valid_ic < 0 else 1.0,
            "factor_values": joined[f"f{idx}"],
        }
    except Exception as e:
        print(f"      Factor {idx}: FAILED - {type(e).__name__}: {e} (IC set to 0)")
        return {
            "idx": idx, "expr": expr,
            "valid_ic": 0.0, "test_ic": 0.0, "sign": 1.0, "failed": True,
        }


def _compute_combo_ic(records, df, label, period_start, period_end):
    """Backward-compat shim: OLS combo IC, fit on TRAIN, eval on [period_start, period_end]."""
    return _compute_combo_ic_ols(records, label, TRAIN_START, TRAIN_END, period_start, period_end)


def _compute_combo_ic_ols(records, label, fit_start, fit_end, eval_start, eval_end):
    """OLS regression of factors on label on [fit_start, fit_end],
    then evaluate cross-sectional IC of predictions on [eval_start, eval_end].

    No regularization. Uses cached factor_values from _eval_single_factor.

    Robustness:
      - replace +/-inf with NaN, then dropna (inf can slip past a plain dropna
        and makes LAPACK's gelsd return LinAlgError, which numpy wraps as a
        confusing SystemError "returned a result with an exception set").
      - drop constant columns (zero variance → rank-deficient → unstable coef).
      - fall back to pseudo-inverse via scipy/numpy SVD if lstsq still fails.
    """
    valid_records = [
        r for r in records
        if not r.get("failed") and r.get("factor_values") is not None
    ]
    if not valid_records:
        return 0.0

    factor_df = pd.concat(
        [r["factor_values"].rename(f"f{r['idx']}") for r in valid_records],
        axis=1,
    )
    data = pd.concat([factor_df, label.rename("label")], axis=1)
    # kill inf before dropna — this is the common cause of LAPACK errors.
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return 0.0

    dt = data.index.get_level_values("datetime")
    fit_df = data[(dt >= fit_start) & (dt <= fit_end)]
    eval_df = data[(dt >= eval_start) & (dt <= eval_end)]
    if fit_df.empty or eval_df.empty:
        return 0.0

    factor_cols = [c for c in data.columns if c != "label"]
    # drop columns that are constant (or near-constant) on the fit window —
    # they contribute no information and break OLS rank.
    fit_std = fit_df[factor_cols].std()
    keep_cols = [c for c in factor_cols if fit_std.get(c, 0) > 1e-12]
    if not keep_cols:
        print(f"      [WARN] OLS: all factor columns constant on fit window")
        return 0.0

    X_fit = np.ascontiguousarray(fit_df[keep_cols].values, dtype=np.float64)
    y_fit = np.ascontiguousarray(fit_df["label"].values,  dtype=np.float64)
    X_fit_aug = np.column_stack([X_fit, np.ones(len(X_fit))])

    try:
        coef, *_ = np.linalg.lstsq(X_fit_aug, y_fit, rcond=None)
    except Exception as e:
        # LAPACK sometimes fails even on clean inputs due to internal SVD
        # convergence issues. Fall back to pinv (always succeeds on finite data).
        print(f"      [WARN] OLS lstsq failed ({type(e).__name__}); falling back to pinv")
        try:
            coef = np.linalg.pinv(X_fit_aug) @ y_fit
        except Exception as e2:
            print(f"      [WARN] OLS pinv also failed: {type(e2).__name__}: {e2}")
            return 0.0

    X_eval = np.ascontiguousarray(eval_df[keep_cols].values, dtype=np.float64)
    y_pred = X_eval @ coef[:-1] + coef[-1]
    eval_out = eval_df[["label"]].copy()
    eval_out["combo"] = y_pred
    return calc_cross_sectional_ic(eval_out, "combo", "label")


def evaluate_factors_by_round(rounds, data_path):
    """
    Evaluate factors grouped by round.
    For each round, compute IC-weighted combination valid_ic and test_ic.
    Select the best round by combined valid_ic.

    Returns:
        dict with per-round results and best-round selection
    """
    print(f"  Loading data: {data_path}")
    df = pd.read_hdf(data_path, key="data")
    if df.index.names == ["instrument", "datetime"]:
        df = df.swaplevel().sort_index()

    label = compute_label(df)

    round_results = {}
    global_idx = 0

    for round_num in sorted(rounds.keys()):
        factors = rounds[round_num]
        exprs = [f["expression"] for f in factors]
        print(f"\n  --- Round {round_num} ({len(exprs)} factors) ---")

        # Evaluate each factor in this round (never drops — failed factors get IC=0)
        records = []
        for expr in exprs:
            print(f"    factor_expression: {expr}")
            rec = _eval_single_factor(expr, global_idx, df, label)
            records.append(rec)
            failed_tag = " [FAILED]" if rec.get("failed") else ""
            print(f"      Factor {global_idx}: valid_ic={rec['valid_ic']:.6f}, "
                  f"test_ic={rec['test_ic']:.6f}{failed_tag}")
            global_idx += 1

        assert len(records) == len(exprs), \
            f"Round {round_num}: expected {len(exprs)} factors, got {len(records)}"

        # Compute IC-weighted combination for this round
        combo_valid_ic = _compute_combo_ic(records, df, label, VALID_START, VALID_END)
        combo_test_ic = _compute_combo_ic(records, df, label, TEST_START, TEST_END)

        # Compute weights for output
        abs_sum = sum(abs(r["valid_ic"]) for r in records)
        if abs_sum > 0:
            for r in records:
                r["weight"] = abs(r["valid_ic"]) / abs_sum
        else:
            eq = 1.0 / len(records)
            for r in records:
                r["weight"] = eq

        round_results[round_num] = {
            "records": records,
            "combo_valid_ic": float(combo_valid_ic),
            "combo_test_ic": float(combo_test_ic),
            "n_factors": len(records),
            "best_single_valid_ic": float(max(r["valid_ic"] for r in records)),
        }

        print(f"    Round {round_num} summary: "
              f"combo_valid_ic={combo_valid_ic:.6f}, "
              f"combo_test_ic={combo_test_ic:.6f}, "
              f"n_factors={len(records)}")

    if not round_results:
        return None

    # Select best round by combo_valid_ic
    best_round = max(round_results, key=lambda r: round_results[r]["combo_valid_ic"])
    best = round_results[best_round]

    print(f"\n  >>> Best round: {best_round} "
          f"(combo_valid_ic={best['combo_valid_ic']:.6f}, "
          f"combo_test_ic={best['combo_test_ic']:.6f})")

    # Build output
    records = best["records"]

    # Normalize expressions to alphapareto format
    normalized_exprs = []
    for r in records:
        normalized_exprs.append(_to_alphapareto_format(r["expr"]))

    return {
        # Best round results (primary output)
        "best_round":    best_round,
        "exprs":         normalized_exprs,
        "ics_ret":       [float(r["test_ic"]) for r in records],
        "weights":       [float(r["weight"]) for r in records],
        "best_valid_ic": float(best["best_single_valid_ic"]),
        "combo_valid_ic": float(best["combo_valid_ic"]),
        "test_ic_ret":   float(best["combo_test_ic"]),
        # Per-round summary for reference
        "all_rounds": {
            str(rn): {
                "combo_valid_ic": rv["combo_valid_ic"],
                "combo_test_ic": rv["combo_test_ic"],
                "n_factors": rv["n_factors"],
                "best_single_valid_ic": rv["best_single_valid_ic"],
            }
            for rn, rv in round_results.items()
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract factors from AlphaAgent logs and evaluate IC"
    )
    parser.add_argument(
        "--log_dir", default="./AlphaAgent/log",
        help="Path to AlphaAgent log directory",
    )
    parser.add_argument(
        "--output", default="./stepover_test_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--data_path", default=str(DEFAULT_DATA_PATH),
        help="Path to daily_pv.h5 (generated by AlphaAgent on first run)",
    )
    args = parser.parse_args()

    # Step 1: Extract factor expressions grouped by round
    print("=" * 60)
    print("Step 1: Extracting factor expressions by round ...")
    print("=" * 60)
    rounds = extract_factors_by_round(args.log_dir)
    if not rounds:
        print("[ERROR] No factors found! Check log directory.", file=sys.stderr)
        sys.exit(1)

    total_factors = sum(len(v) for v in rounds.values())
    print(f"\nFound {total_factors} unique factor expressions in {len(rounds)} rounds:")
    for rn in sorted(rounds.keys()):
        factors = rounds[rn]
        print(f"\n  Round {rn} ({len(factors)} factors):")
        for i, f in enumerate(factors):
            print(f"    [{i}] {f['name']}: {f['expression'][:75]}...")

    # Step 2: Evaluate IC per round and select best
    print("\n" + "=" * 60)
    print("Step 2: Evaluating factor IC per round ...")
    print("=" * 60)
    data_path = Path(args.data_path)
    if not data_path.exists():
        # Try alternative location
        alt_path = (
            SCRIPT_DIR / "AlphaAgent" / "git_ignore_folder"
            / "factor_implementation_source_data" / "daily_pv.h5"
        )
        if alt_path.exists():
            data_path = alt_path
            print(f"  Using alternative data path: {data_path}")
        else:
            print(f"[ERROR] Data file not found: {args.data_path}", file=sys.stderr)
            print(f"  Also checked: {alt_path}", file=sys.stderr)
            print(f"  Please ensure AlphaAgent has been run at least once to generate data.",
                  file=sys.stderr)
            sys.exit(1)

    result = evaluate_factors_by_round(rounds, data_path)
    if result is None:
        print("[ERROR] No valid factors after evaluation!", file=sys.stderr)
        sys.exit(2)

    # Step 3: Save output
    print("\n" + "=" * 60)
    print("Step 3: Saving results ...")
    print("=" * 60)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n  Output: {args.output}")
    print(f"  Best round:      {result['best_round']}")
    print(f"  Factors:         {len(result['exprs'])}")
    print(f"  combo_valid_ic:  {result['combo_valid_ic']:.6f}")
    print(f"  test_ic_ret:     {result['test_ic_ret']:.6f}")
    print(f"\n  All rounds:")
    for rn, rv in sorted(result["all_rounds"].items(), key=lambda x: int(x[0])):
        marker = " <<<" if int(rn) == result["best_round"] else ""
        print(f"    Round {rn}: combo_valid={rv['combo_valid_ic']:.6f}, "
              f"combo_test={rv['combo_test_ic']:.6f}, "
              f"n_factors={rv['n_factors']}{marker}")
    print("=" * 60)


if __name__ == "__main__":
    main()