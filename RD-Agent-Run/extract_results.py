#!/usr/bin/env python3
"""
Extract factor expressions from RD-Agent run logs, evaluate IC, and produce
stepover_test_results.json (same format as AlphaAgent).

RD-Agent generates Python code (factor.py) for each factor. This script:
1. Walks the RD-Agent log directory to find FactorTask objects in pkl files
2. Groups factors by iteration round (using __session__ directory structure)
3. Executes each factor's code to get factor values, then computes IC
4. Selects the best round by combined valid IC
5. Outputs the best round's results as stepover_test_results.json

Usage:
    python extract_results.py --log_dir ./RD-Agent/log_all_seed0 --output ./results/all/seed_0/stepover_test_results.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ===================== Alphapareto Translation =====================
# Translate RD-Agent Python factor code to alphapareto expression format
# using LLM (DeepSeek) + alphapareto_bridge validation.

# Available alphapareto operators (for LLM prompt)
_AP_OPERATORS = """
Unary:   Abs(x), Log(x)
Binary:  Add(x,y), Sub(x,y), Mul(x,y), Div(x,y), Greater(x,y), Less(x,y)
Rolling: Ref(x,d), Mean(x,d), Sum(x,d), Std(x,d), Var(x,d),
         Max(x,d), Min(x,d), Med(x,d), Mad(x,d), Delta(x,d), WMA(x,d), EMA(x,d)
Pair:    Cov(x,y,d), Corr(x,y,d)
Features: $open, $close, $high, $low, $volume
"""

_bridge_normalize = None
_bridge_loaded = False


_bridge_module = None  # full module reference for AST types


def _ensure_bridge():
    """Try to import alphapareto_bridge.normalize for validation."""
    global _bridge_normalize, _bridge_loaded, _bridge_module
    if _bridge_loaded:
        return
    _bridge_loaded = True
    # Try to find alphapareto_bridge from AlphaAgent
    try:
        from alphaagent.components.coder.factor_coder import alphapareto_bridge as mod
        _bridge_normalize = mod.normalize
        _bridge_module = mod
        return
    except ImportError:
        pass
    # Try local copy
    bridge_path = SCRIPT_DIR / "alphapareto_bridge.py"
    if bridge_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("alphapareto_bridge", bridge_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _bridge_normalize = mod.normalize
        _bridge_module = mod


def _call_llm(prompt, api_key=None):
    """Call DeepSeek API to translate factor code to alphapareto expression."""
    try:
        import openai
    except ImportError:
        return None

    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"      [WARN] LLM call failed: {e}")
        return None


def translate_to_alphapareto(factor_name, formulation, code, api_key=None):
    """
    Get alphapareto expression for a factor.

    Priority strategy:
    1. If formulation is already a valid alphapareto expression, normalize and return it.
    2. Fallback: use LLM to translate Python code to alphapareto.
    """
    _ensure_bridge()

    # Strategy 1: formulation is already a valid alphapareto expression
    if formulation and _bridge_normalize:
        try:
            canonical = _bridge_normalize(formulation)
            return canonical
        except Exception:
            pass  # formulation is not valid alphapareto, continue to LLM

    # Strategy 2: LLM translation (fallback)
    prompt = f"""Convert the following quantitative factor into a single alphapareto expression.

Available operators:
{_AP_OPERATORS}

Rules:
- Output ONLY one line: the alphapareto expression. No explanation.
- Use $close, $open, $high, $low, $volume as features.
- Rolling window d must be one of: 1, 5, 10, 20, 40
- Nest operators, e.g. Div(Sub($close,Ref($close,20)),Ref($close,20))
- If the factor is too complex to express exactly, approximate with the closest expression.

Factor name: {factor_name}
Formulation: {formulation}

Python code:
```python
{code[:1500]}
```

Alphapareto expression:"""

    raw = _call_llm(prompt, api_key=api_key)
    if not raw:
        return formulation  # final fallback

    # Clean up LLM output: take first non-empty line, strip markdown
    for line in raw.split("\n"):
        line = line.strip().strip("`").strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            expr = line
            break
    else:
        return formulation

    # Validate with bridge if available
    if _bridge_normalize:
        try:
            return _bridge_normalize(expr)
        except Exception:
            # LLM output didn't parse — return raw expression
            return expr

    return expr


def batch_translate_to_alphapareto(records, api_key=None):
    """Translate a list of factor records to alphapareto format.
    Tries bridge.normalize(formulation) first; falls back to LLM translation."""
    _ensure_bridge()
    exprs = []
    for i, r in enumerate(records):
        name = r.get("factor_name", r.get("expr", f"factor_{i}"))
        formulation = r.get("factor_formulation", r.get("expr", ""))
        code = r.get("code", "")

        # Fast path: formulation is already valid alphapareto
        if formulation and _bridge_normalize:
            try:
                canonical = _bridge_normalize(formulation)
                print(f"    [{i}] {name} -> {canonical[:60]} (from formulation)")
                exprs.append(canonical)
                continue
            except Exception:
                pass

        # Slow path: LLM translation
        print(f"    [{i}] Translating {name} via LLM ...", end=" ")
        ap_expr = translate_to_alphapareto(name, formulation, code, api_key=api_key)
        print(f"-> {ap_expr[:60]}")
        exprs.append(ap_expr)
    return exprs


# ===================== Independent Alphapareto Expression Evaluator =====================
# Pure pandas implementation — NO dependency on AlphaAgent modules.
# Evaluates alphapareto expressions (e.g. "Std($close,10)") directly on price-volume DataFrames.


def _grouped_rolling(series, window, func, min_periods=1):
    """Apply rolling function grouped by instrument."""
    return series.groupby(level="instrument").transform(
        lambda x: func(x.rolling(window, min_periods=min_periods))
    )


def _eval_ast(node, df):
    """Recursively evaluate a parsed alphapareto AST node on a DataFrame.
    df has MultiIndex(datetime, instrument) and columns like $open, $close, etc.
    Returns pd.Series with the same MultiIndex."""

    # AST types from bridge module (loaded via _ensure_bridge)
    FeatureNode = _bridge_module.FeatureNode
    ConstantNode = _bridge_module.ConstantNode
    OpNode = _bridge_module.OpNode

    if isinstance(node, FeatureNode):
        col = f"${node.name}"
        if col not in df.columns:
            raise ValueError(f"Feature {col} not found in data columns: {list(df.columns)}")
        return df[col].copy()

    if isinstance(node, ConstantNode):
        return pd.Series(node.value, index=df.index, dtype=float)

    if isinstance(node, OpNode):
        op = node.op_name
        args = node.args

        # --- Unary operators ---
        if op == "Abs":
            return _eval_ast(args[0], df).abs()
        elif op == "Log":
            v = _eval_ast(args[0], df)
            return np.log(v.replace(0, np.nan))

        # --- Binary operators ---
        elif op == "Add":
            return _eval_ast(args[0], df) + _eval_ast(args[1], df)
        elif op == "Sub":
            return _eval_ast(args[0], df) - _eval_ast(args[1], df)
        elif op == "Mul":
            return _eval_ast(args[0], df) * _eval_ast(args[1], df)
        elif op == "Div":
            a = _eval_ast(args[0], df)
            b = _eval_ast(args[1], df)
            result = a / b.replace(0, np.nan)
            return result.replace([np.inf, -np.inf], np.nan)
        elif op == "Greater":
            a = _eval_ast(args[0], df)
            b = _eval_ast(args[1], df)
            return pd.concat([a, b], axis=1).max(axis=1)
        elif op == "Less":
            a = _eval_ast(args[0], df)
            b = _eval_ast(args[1], df)
            return pd.concat([a, b], axis=1).min(axis=1)

        # --- Rolling operators (operand, window) ---
        elif op == "Ref":
            operand = _eval_ast(args[0], df)
            d = int(ConstantNode(args[1].value).value) if isinstance(args[1], ConstantNode) else int(args[1].value)
            return operand.groupby(level="instrument").shift(d)
        elif op == "Mean":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.mean())
        elif op == "Sum":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.sum())
        elif op == "Std":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.std())
        elif op == "Var":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.var())
        elif op == "Max":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.max())
        elif op == "Min":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.min())
        elif op == "Med":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return _grouped_rolling(operand, d, lambda r: r.median())
        elif op == "Mad":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return operand.groupby(level="instrument").transform(
                lambda x: x.rolling(d, min_periods=1).apply(
                    lambda w: np.abs(w - w.mean()).mean(), raw=True
                )
            )
        elif op == "Delta":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return operand - operand.groupby(level="instrument").shift(d)
        elif op == "WMA":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            weights = np.arange(1, d + 1, dtype=float)
            weights /= weights.sum()
            return operand.groupby(level="instrument").transform(
                lambda x: x.rolling(d, min_periods=1).apply(
                    lambda w: np.dot(w[-len(weights):], weights[-len(w):]) if len(w) > 0 else np.nan,
                    raw=True,
                )
            )
        elif op == "EMA":
            operand = _eval_ast(args[0], df)
            d = int(args[1].value)
            return operand.groupby(level="instrument").transform(
                lambda x: x.ewm(span=d, adjust=False).mean()
            )

        # --- Pair rolling operators (operand1, operand2, window) ---
        elif op == "Cov":
            a = _eval_ast(args[0], df)
            b = _eval_ast(args[1], df)
            d = int(args[2].value)
            # Unstack, rolling cov, restack
            try:
                wa = a.unstack(level="instrument")
                wb = b.unstack(level="instrument")
                result = wa.rolling(d, min_periods=2).cov(wb)
                return result.stack().reorder_levels(["datetime", "instrument"]).sort_index()
            except Exception:
                # Fallback: per-group
                combined = pd.DataFrame({"a": a, "b": b})
                return combined.groupby(level="instrument").apply(
                    lambda g: g["a"].rolling(d, min_periods=2).cov(g["b"])
                ).droplevel(0).sort_index()
        elif op == "Corr":
            a = _eval_ast(args[0], df)
            b = _eval_ast(args[1], df)
            d = int(args[2].value)
            try:
                wa = a.unstack(level="instrument")
                wb = b.unstack(level="instrument")
                result = wa.rolling(d, min_periods=2).corr(wb)
                return result.stack().reorder_levels(["datetime", "instrument"]).sort_index()
            except Exception:
                combined = pd.DataFrame({"a": a, "b": b})
                return combined.groupby(level="instrument").apply(
                    lambda g: g["a"].rolling(d, min_periods=2).corr(g["b"])
                ).droplevel(0).sort_index()

        else:
            raise ValueError(f"Unknown operator: {op}")

    raise ValueError(f"Unknown AST node type: {type(node)}")


def eval_alphapareto_expr(expr_str, df):
    """
    Evaluate an alphapareto expression on a price-volume DataFrame.
    Returns pd.Series with MultiIndex(datetime, instrument), or None on failure.
    Pure pandas implementation — no AlphaAgent dependency.
    """
    _ensure_bridge()
    if _bridge_module is None:
        print(f"      [eval_alphapareto] bridge module not loaded")
        return None
    try:
        ast = _bridge_module._parse(expr_str)
        result = _eval_ast(ast, df)
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        return result.astype(float)
    except Exception as e:
        print(f"      [eval_alphapareto] {expr_str[:60]}... -> {type(e).__name__}: {e}")
        return None


# ===================== Time Configuration =====================
# Must match the FactorBasePropSetting in conf.py
TRAIN_START = "2013-07-01"
TRAIN_END   = "2023-06-30"
VALID_START = "2023-07-01"
VALID_END   = "2024-06-30"
TEST_START  = "2024-07-01"
TEST_END    = "2025-06-30"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    SCRIPT_DIR / "RD-Agent" / "rdagent" / "scenarios" / "qlib"
    / "experiment" / "factor_data_template" / "daily_pv_all.h5"
)
# Fallback: factor implementation source data
ALT_DATA_PATH = (
    SCRIPT_DIR / "RD-Agent" / "git_ignore_folder"
    / "factor_implementation_source_data" / "daily_pv_all.h5"
)


# ============================================================
# Step 1: Extract factor info and code from pkl log files
# ============================================================

def _extract_factor_info(obj):
    """Extract factor_name, factor_formulation, factor_description, and code from a FactorTask-like object."""
    info = {}
    info["factor_name"] = getattr(obj, "factor_name", getattr(obj, "name", "unknown"))
    info["factor_formulation"] = getattr(obj, "factor_formulation", "")
    info["factor_description"] = getattr(obj, "factor_description", getattr(obj, "description", ""))
    return info


def _extract_code_from_workspace(ws):
    """Get factor.py code from a workspace object."""
    if ws is None:
        return None
    file_dict = getattr(ws, "file_dict", None)
    if file_dict and "factor.py" in file_dict:
        return file_dict["factor.py"]
    # Try workspace_path
    ws_path = getattr(ws, "workspace_path", None)
    if ws_path:
        factor_py = Path(ws_path) / "factor.py"
        if factor_py.exists():
            return factor_py.read_text(encoding="utf-8")
    return None


def _get_based_factor_names(obj):
    """Collect factor names from all based_experiments to exclude accumulated factors."""
    names = set()
    based = getattr(obj, "based_experiments", None) or []
    for exp in based:
        for t in (getattr(exp, "sub_tasks", None) or []):
            n = getattr(t, "factor_name", None) or getattr(t, "name", None)
            if n:
                names.add(n)
    return names


def _dig_factors(obj, factors, seen, depth, source):
    """Recursively dig into objects to find FactorTask + workspace pairs.
    Only extracts NEW factors for this round (excludes based_experiments accumulation)."""
    if depth > 15 or obj is None:
        return

    # Check for Experiment objects with sub_tasks + sub_workspace_list
    if hasattr(obj, "sub_tasks") and hasattr(obj, "sub_workspace_list"):
        sub_tasks = getattr(obj, "sub_tasks", None) or []
        sub_ws = getattr(obj, "sub_workspace_list", None) or []

        # Identify factor names from based_experiments to exclude them
        based_names = _get_based_factor_names(obj)

        for i, task in enumerate(sub_tasks):
            if not hasattr(task, "factor_name"):
                continue
            info = _extract_factor_info(task)
            key = info["factor_name"]

            # Skip factors inherited from based_experiments
            if key in based_names:
                continue

            code = _extract_code_from_workspace(sub_ws[i]) if i < len(sub_ws) else None
            # Allow factors without `code` as long as we have either a
            # formulation (preferred — alphapareto bridge can normalize it)
            # or at least a name. This recovers Loop_0 / partial-failure rounds
            # where coding stage produced no factor.py file.
            if key not in seen and (code or info.get("factor_formulation")):
                seen.add(key)
                factors.append({
                    "factor_name": info["factor_name"],
                    "factor_formulation": info["factor_formulation"],
                    "factor_description": info["factor_description"],
                    "code": code,
                    "source": source,
                })
        # Do NOT recurse into based_experiments
        return

    # Bare FactorTask object (no Experiment wrapper) — happens in
    # direct_exp_gen/experiment generation/*.pkl, which is a list[FactorTask].
    # Loop_0 sometimes only has this (no Experiment ever materialized) so
    # without this branch we'd silently lose the round.
    if hasattr(obj, "factor_name") and not hasattr(obj, "sub_tasks"):
        info = _extract_factor_info(obj)
        key = info["factor_name"]
        if key not in seen and info.get("factor_formulation"):
            seen.add(key)
            factors.append({
                "factor_name": info["factor_name"],
                "factor_formulation": info["factor_formulation"],
                "factor_description": info["factor_description"],
                "code": None,
                "source": source,
            })
        return

    # Recurse into lists/tuples/dicts (but not arbitrary objects that might be experiments)
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _dig_factors(item, factors, seen, depth + 1, source)
    elif isinstance(obj, dict):
        for v in obj.values():
            _dig_factors(v, factors, seen, depth + 1, source)


def _get_session_round_map(log_root):
    """
    Map __session__ subdirectories to outer round numbers.
    RD-Agent's __session__ has numbered dirs; each corresponds to one RDLoop iteration.
    """
    session_dirs = list(log_root.rglob("__session__"))
    if not session_dirs:
        return None

    session_dir = session_dirs[0]
    iter_dirs = sorted(
        [d for d in session_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    # Each numbered dir in __session__ is one outer round
    round_map = {}  # dir_name -> round_idx
    for idx, d in enumerate(iter_dirs):
        round_map[d.name] = idx
        # Also store mtime for pkl assignment
        round_map[f"_mtime_{d.name}"] = d.stat().st_mtime
    round_map["_total"] = len(iter_dirs)
    return round_map


def extract_factors_by_round(log_dir):
    """
    Extract factor code grouped by iteration round.

    Returns:
        dict: {round_num: [factor_dicts]}
    """
    log_root = Path(log_dir)
    if not log_root.exists():
        print(f"[ERROR] Log directory not found: {log_dir}", file=sys.stderr)
        return None

    # Find all pkl files (skip workspace/cache dirs)
    all_pkls = sorted(
        p for p in log_root.rglob("*.pkl")
        if not any(skip in str(p) for skip in [
            "pickle_cache", "RD-Agent_workspace", "__pycache__",
        ])
    )

    if not all_pkls:
        print(f"[ERROR] No pkl files found in: {log_dir}", file=sys.stderr)
        return None

    # Count Loop_<N> directories to show how many rounds exist
    loop_dirs = sorted(
        [d for d in log_root.iterdir() if d.is_dir() and d.name.startswith("Loop_")],
        key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else 0,
    )
    if loop_dirs:
        print(f"  Found {len(loop_dirs)} Loop directories: {[d.name for d in loop_dirs]}")
    else:
        print(f"  [WARN] No Loop_<N> directories found")

    print(f"  Scanning {len(all_pkls)} pkl files ...")

    # Per-round: collect factors from each pkl, keyed by (round, pkl_path)
    # so we can pick the best pkl per round (most factors, latest file)
    round_pkl_factors = {}  # {round_num: [(pkl_path, {name: factor_dict})]}

    for pkl_file in all_pkls:
        # Determine round from path: look for Loop_<N> in the path
        round_num = 0
        parts = pkl_file.parts
        for part in parts:
            if part.startswith("Loop_"):
                try:
                    round_num = int(part.split("_", 1)[1])
                except (ValueError, IndexError):
                    pass
                break

        factors_this_file = []
        seen_this_file = set()
        try:
            with open(pkl_file, "rb") as f:
                obj = pickle.load(f)
            _dig_factors(obj, factors_this_file, seen_this_file, 0, str(pkl_file))
        except Exception:
            pass

        if factors_this_file:
            if round_num not in round_pkl_factors:
                round_pkl_factors[round_num] = []
            fmap = {}
            for fac in factors_this_file:
                fmap[fac["factor_name"]] = fac
            round_pkl_factors[round_num].append((str(pkl_file), fmap))

    # For each round, pick the pkl whose factor count is closest to 10
    # (prefer exactly 10 > less than 10 > more than 10).
    # If tied, pick the latest pkl file (last in sorted order).
    rounds = {}
    for rn in sorted(round_pkl_factors.keys()):
        pkl_entries = round_pkl_factors[rn]

        # Score each pkl: 0 = exactly 10, 1 = less than 10, 2 = more than 10
        def _score(entry):
            n = len(entry[1])
            if n == 10:
                return (0, -n)
            elif n < 10:
                return (1, -n)  # prefer more factors
            else:
                return (2, n)   # prefer fewer factors

        # Sort by score, then by pkl path (later files last = preferred)
        best_pkl_path, best_fmap = min(pkl_entries, key=lambda e: (_score(e), e[0]))

        factors_list = list(best_fmap.values())

        # Safety cap: if still > 10, take only the first 10
        if len(factors_list) > 10:
            print(f"  [CAP] Round {rn}: {len(factors_list)} factors found, capping to 10")
            factors_list = factors_list[:10]

        print(f"  Round {rn}: {len(factors_list)} factors (from {Path(best_pkl_path).name})")
        rounds[rn] = factors_list

    return rounds if rounds else None


# ============================================================
# Step 2: IC Computation
# ============================================================

def calc_cross_sectional_ic(df, factor_col, label_col):
    """Compute mean cross-sectional IC (Pearson correlation per day, then average)."""
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
# Step 3: Factor Execution via code
# ============================================================

def execute_factor_code(code, data_path):
    """
    Execute a factor.py code string and return the result DataFrame.

    The code is expected to read source data, compute a factor,
    and write result.h5.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Write factor code
        (tmpdir / "factor.py").write_text(code, encoding="utf-8")

        # Symlink source data files
        data_src = data_path.parent
        for f in data_src.iterdir():
            if f.is_file() and f.suffix in (".h5", ".csv", ".pkl", ".parquet"):
                target = tmpdir / f.name
                if not target.exists():
                    try:
                        os.symlink(f, target)
                    except OSError:
                        import shutil
                        shutil.copy2(f, target)

        # RD-Agent generated code reads "daily_pv.h5", but the actual file
        # may be named "daily_pv_all.h5". Create an alias if needed.
        daily_pv_link = tmpdir / "daily_pv.h5"
        if not daily_pv_link.exists():
            daily_pv_all = tmpdir / "daily_pv_all.h5"
            if daily_pv_all.exists():
                try:
                    os.symlink(daily_pv_all, daily_pv_link)
                except OSError:
                    import shutil
                    shutil.copy2(daily_pv_all, daily_pv_link)

        # Execute
        try:
            subprocess.check_output(
                [sys.executable, "factor.py"],
                cwd=str(tmpdir),
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            msg = e.output.decode()[:200] if hasattr(e, "output") and e.output else str(e)
            raise RuntimeError(f"Execution failed: {msg}")

        # Read result
        result_path = tmpdir / "result.h5"
        if not result_path.exists():
            raise RuntimeError("No result.h5 produced")

        df = pd.read_hdf(result_path)
        return df


def compute_label(df):
    """Compute forward return label: close[t+2]/close[t+1] - 1."""
    close = df["$close"]
    label = close.groupby(level="instrument").transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    label.name = "label"
    return label


# ============================================================
# Step 4: Per-Round Evaluation
# ============================================================

def _compute_ic_from_series(fv, idx, label):
    """Compute valid/test IC from a factor value Series. Returns (valid_ic, test_ic) or None."""
    if fv is None:
        return None

    fv.name = f"f{idx}"

    # Ensure MultiIndex alignment
    if fv.index.names != ["datetime", "instrument"]:
        if "instrument" in fv.index.names and "datetime" in fv.index.names:
            pass
        else:
            return None

    joined = pd.concat([fv, label], axis=1).dropna()
    if joined.empty:
        return None

    dt = joined.index.get_level_values("datetime")
    valid_data = joined[(dt >= VALID_START) & (dt <= VALID_END)]
    test_data  = joined[(dt >= TEST_START)  & (dt <= TEST_END)]

    valid_ic = calc_cross_sectional_ic(valid_data, f"f{idx}", "label")
    test_ic  = calc_cross_sectional_ic(test_data,  f"f{idx}", "label")
    return float(valid_ic), float(test_ic)


def _eval_single_factor(factor_dict, idx, data_path, pv_df, label):
    """
    Evaluate one factor. Strategy:
    1. Translate Python code → alphapareto expression (via LLM)
    2. Try evaluating the alphapareto expression via function_lib
    3. Fallback: execute original Python code
    Never returns None — failed factors get IC=0.
    """
    code = factor_dict["code"]
    factor_name = factor_dict.get("factor_name", f"factor_{idx}")
    factor_formulation = factor_dict.get("factor_formulation", "")
    expr_display = factor_formulation or factor_name

    def _make_result(valid_ic, test_ic, ap_expr, failed=False, factor_values=None):
        return {
            "idx": idx,
            "expr": ap_expr or expr_display,
            "code": code,
            "factor_name": factor_name,
            "factor_formulation": factor_formulation,
            "alphapareto_expr": ap_expr,
            "valid_ic": float(valid_ic),
            "test_ic": float(test_ic),
            "sign": -1.0 if valid_ic < 0 else 1.0,
            "failed": failed,
            "factor_values": factor_values,  # cached Series for combo IC
        }

    # Step 1: Translate to alphapareto
    ap_expr = translate_to_alphapareto(factor_name, factor_formulation, code)
    print(f"    factor_expression: {ap_expr}")

    # Step 2: Try evaluating alphapareto expression
    fv = eval_alphapareto_expr(ap_expr, pv_df)
    if fv is not None:
        ic_result = _compute_ic_from_series(fv, idx, label)
        if ic_result is not None:
            valid_ic, test_ic = ic_result
            return _make_result(valid_ic, test_ic, ap_expr, factor_values=fv)

    # Step 3: Fallback — execute original Python code
    print(f"      [alphapareto eval failed, trying Python code ...]")
    try:
        result_df = execute_factor_code(code, data_path)
        if result_df is not None and not result_df.empty:
            if isinstance(result_df, pd.DataFrame):
                fv = result_df.iloc[:, 0]
            else:
                fv = result_df
            ic_result = _compute_ic_from_series(fv, idx, label)
            if ic_result is not None:
                valid_ic, test_ic = ic_result
                return _make_result(valid_ic, test_ic, ap_expr, factor_values=fv)
    except Exception as e:
        print(f"      Factor {idx}: Python code also FAILED - {type(e).__name__}: {e}")

    # Both methods failed
    print(f"      Factor {idx}: FAILED (IC set to 0)")
    return _make_result(0.0, 0.0, ap_expr, failed=True)


def _compute_combo_ic(records, label, period_start, period_end):
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
      - fall back to pseudo-inverse via SVD if lstsq still fails.
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
    # kill inf before dropna — common cause of LAPACK errors.
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return 0.0

    dt = data.index.get_level_values("datetime")
    fit_df = data[(dt >= fit_start) & (dt <= fit_end)]
    eval_df = data[(dt >= eval_start) & (dt <= eval_end)]
    if fit_df.empty or eval_df.empty:
        return 0.0

    factor_cols = [c for c in data.columns if c != "label"]
    # drop columns that are constant (or near-constant) on the fit window.
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
    """Evaluate factors grouped by round. Same logic as AlphaAgent's extract_results."""
    print(f"  Loading data: {data_path}")
    pv_df = pd.read_hdf(data_path, key="data")
    if pv_df.index.names == ["instrument", "datetime"]:
        pv_df = pv_df.swaplevel().sort_index()

    label = compute_label(pv_df)

    round_results = {}
    global_idx = 0

    for round_num in sorted(rounds.keys()):
        factors = rounds[round_num]
        print(f"\n  --- Round {round_num} ({len(factors)} factors) ---")

        if not factors:
            print(f"    [SKIP] Round {round_num} has 0 factors, skipping.")
            round_results[round_num] = {
                "records": [],
                "combo_valid_ic": 0.0,
                "combo_test_ic": 0.0,
                "n_factors": 0,
                "best_single_valid_ic": 0.0,
            }
            continue

        records = []
        for fac in factors:
            print(f"    [{global_idx}] {fac['factor_name']}")
            rec = _eval_single_factor(fac, global_idx, data_path, pv_df, label)
            records.append(rec)
            failed_tag = " [FAILED]" if rec.get("failed") else ""
            print(f"      valid_ic={rec['valid_ic']:.6f}, test_ic={rec['test_ic']:.6f}{failed_tag}")
            global_idx += 1

        # Compute combo IC
        combo_valid_ic = _compute_combo_ic(records, label, VALID_START, VALID_END)
        combo_test_ic = _compute_combo_ic(records, label, TEST_START, TEST_END)

        # Compute weights
        abs_sum = sum(abs(r["valid_ic"]) for r in records)
        if abs_sum > 0:
            for r in records:
                r["weight"] = abs(r["valid_ic"]) / abs_sum
        else:
            eq = 1.0 / len(records) if records else 0.0
            for r in records:
                r["weight"] = eq

        non_failed = [r for r in records if not r.get("failed")]
        best_single = float(max((r["valid_ic"] for r in records), default=0.0))

        round_results[round_num] = {
            "records": records,
            "combo_valid_ic": float(combo_valid_ic),
            "combo_test_ic": float(combo_test_ic),
            "n_factors": len(records),
            "best_single_valid_ic": best_single,
        }

        print(f"    Round {round_num} summary: "
              f"combo_valid_ic={combo_valid_ic:.6f}, "
              f"combo_test_ic={combo_test_ic:.6f}, "
              f"n_factors={len(records)}")

    if not round_results:
        return None

    # Only consider rounds that have factors when picking the best round
    non_empty_rounds = [r for r in round_results if round_results[r]["n_factors"] > 0]
    if not non_empty_rounds:
        print("[ERROR] All rounds have 0 factors!")
        return None
    best_round = max(non_empty_rounds, key=lambda r: round_results[r]["combo_valid_ic"])
    best = round_results[best_round]

    print(f"\n  >>> Best round: {best_round} "
          f"(combo_valid_ic={best['combo_valid_ic']:.6f}, "
          f"combo_test_ic={best['combo_test_ic']:.6f})")

    records = best["records"]

    # Translate factor expressions to alphapareto format
    print("\n  Translating factors to alphapareto format ...")
    alphapareto_exprs = batch_translate_to_alphapareto(records)

    return {
        "best_round":    best_round,
        "exprs":         alphapareto_exprs,
        "ics_ret":       [float(r["test_ic"]) for r in records],
        "weights":       [float(r["weight"]) for r in records],
        "best_valid_ic": float(best["best_single_valid_ic"]),
        "combo_valid_ic": float(best["combo_valid_ic"]),
        "test_ic_ret":   float(best["combo_test_ic"]),
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
        description="Extract factors from RD-Agent logs and evaluate IC"
    )
    parser.add_argument(
        "--log_dir", default="./RD-Agent/log_RDAgent",
        help="Path to RD-Agent log directory",
    )
    parser.add_argument(
        "--output", default="./stepover_test_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--data_path", default=None,
        help="Path to daily_pv_all.h5",
    )
    args = parser.parse_args()

    # Resolve data path
    if args.data_path:
        data_path = Path(args.data_path)
    elif DEFAULT_DATA_PATH.exists():
        data_path = DEFAULT_DATA_PATH
    elif ALT_DATA_PATH.exists():
        data_path = ALT_DATA_PATH
    else:
        # Try finding any daily_pv*.h5 under RD-Agent
        candidates = list((SCRIPT_DIR / "RD-Agent").rglob("daily_pv*.h5"))
        if candidates:
            data_path = candidates[0]
        else:
            print("[ERROR] No data file found. Run generate.py first.", file=sys.stderr)
            sys.exit(1)

    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Extract
    print("=" * 60)
    print("Step 1: Extracting factor code by round ...")
    print("=" * 60)
    rounds = extract_factors_by_round(args.log_dir)
    if not rounds:
        print("[ERROR] No factors found! Check log directory.", file=sys.stderr)
        sys.exit(1)

    total_factors = sum(len(v) for v in rounds.values())
    print(f"\nFound {total_factors} unique factors in {len(rounds)} rounds:")
    for rn in sorted(rounds.keys()):
        factors = rounds[rn]
        print(f"\n  Round {rn} ({len(factors)} factors):")
        for i, f in enumerate(factors):
            print(f"    [{i}] {f['factor_name']}: {(f.get('factor_formulation') or '')[:60]}")

    # Step 2: Evaluate
    print("\n" + "=" * 60)
    print("Step 2: Evaluating factor IC per round ...")
    print("=" * 60)

    result = evaluate_factors_by_round(rounds, data_path)
    if result is None:
        print("[ERROR] No valid factors after evaluation!", file=sys.stderr)
        sys.exit(2)

    # Step 3: Save
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