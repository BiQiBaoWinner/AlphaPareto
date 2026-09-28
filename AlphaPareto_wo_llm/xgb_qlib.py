import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

import gymnasium as gym
import sys
sys.modules['gym'] = gym

from qlib.contrib.model.xgboost import XGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

from alphagen.utils.random import reseed_everything
from alphagen_qlib.stock_data import initialize_qlib

DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
]

DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]


@dataclass
class TrainConfig:
    seed: int = 2
    experiment_name: str = "xgb_qlib_baseline"
    instrument: str = "csi300"
    train_start: str = "2012-01-01"
    train_end: str = "2021-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-06-30"
    qlib_data: str = "~/.qlib/qlib_data/cn_data_rolling"
    data_device: str = "cpu"
    handler_class: str = "Alpha158"
    handler_module: str = "qlib.contrib.data.handler"
    handler_kwargs: Dict[str, Union[str, float, int]] = field(default_factory=dict)
    n_estimators: int = 2000
    learning_rate: float = 0.001
    max_depth: int = 8
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    tree_method: str = "hist"
    n_jobs: int = max(1, os.cpu_count() or 1)
    eval_metric: str = "rmse"
    early_stopping_rounds: int = 30
    xgb_verbose: int = 50
    # Modified default output dir
    output_dir: Union[str, Path] = "out/results/baselines/xgb_qlib"
    grid_params: Dict[str, Tuple[Union[int, float], ...]] = field(default_factory=dict)


def _build_param_grid(grid: Dict[str, Tuple[Union[int, float], ...]]) -> List[Dict[str, Union[int, float]]]:
    if not grid:
        return [{}]
    keys = [key for key, values in grid.items() if values]
    if not keys:
        return [{}]
    combos: List[Dict[str, Union[int, float]]] = []
    for values in product(*(grid[key] for key in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def _build_dataset_config(config: TrainConfig) -> Dict[str, object]:
    handler_kwargs = {
        "start_time": config.train_start,
        "end_time": config.test_end,
        "fit_start_time": config.train_start,
        "fit_end_time": config.train_end,
        "instruments": config.instrument,
    }
    handler_kwargs.update(config.handler_kwargs)
    handler_kwargs.setdefault("infer_processors", DEFAULT_INFER_PROCESSORS)
    handler_kwargs.setdefault("learn_processors", DEFAULT_LEARN_PROCESSORS)

    raw_segments = {
        "train": (config.train_start, config.train_end),
        "valid": (config.valid_start, config.valid_end),
        "test": (config.test_start, config.test_end),
    }
    segments = {name: span for name, span in raw_segments.items() if all(span)}
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": config.handler_class,
                "module_path": config.handler_module,
                "kwargs": handler_kwargs,
            },
            "segments": segments,
        },
    }


def _prepare_model_and_fit_kwargs(
    config: TrainConfig,
    params_override: Dict[str, Union[int, float]],
) -> Tuple[Dict[str, Union[int, float, str]], Dict[str, Union[int, float]]]:
    model_kwargs: Dict[str, Union[int, float, str]] = {
        "objective": "reg:squarederror",
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "reg_alpha": config.reg_alpha,
        "gamma": config.gamma,
        "tree_method": config.tree_method,
        "nthread": config.n_jobs,
        "seed": config.seed,
        "eval_metric": config.eval_metric,
        "verbosity": 0,
    }
    override = dict(params_override)
    fit_kwargs: Dict[str, Union[int, float]] = {}

    for key in ("num_boost_round", "n_estimators"):
        if key in override:
            fit_kwargs["num_boost_round"] = int(override.pop(key))
            break
    if "num_boost_round" not in fit_kwargs:
        fit_kwargs["num_boost_round"] = int(config.n_estimators)

    if "early_stopping_rounds" in override:
        fit_kwargs["early_stopping_rounds"] = int(override.pop("early_stopping_rounds"))
    else:
        fit_kwargs["early_stopping_rounds"] = int(config.early_stopping_rounds)

    if "verbose_eval" in override:
        fit_kwargs["verbose_eval"] = int(override.pop("verbose_eval"))
    else:
        fit_kwargs["verbose_eval"] = int(config.xgb_verbose)

    if "n_jobs" in override:
        override["nthread"] = int(override.pop("n_jobs"))
    if "nthread" in override:
        override["nthread"] = int(override["nthread"])

    if "max_depth" in override:
        override["max_depth"] = int(override["max_depth"])
    if "min_child_weight" in override:
        override["min_child_weight"] = float(override["min_child_weight"])

    model_kwargs.update(override)
    return model_kwargs, fit_kwargs


def _compute_metrics(pred: pd.Series, label: pd.Series) -> Dict[str, float]:
    joined = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    if joined.empty:
        return {"ic": float("nan"), "ic_ir": float("nan")}

    ic_daily = joined.groupby(level=0).apply(lambda df: df["pred"].corr(df["label"]))
    ic_daily = ic_daily.dropna()
    if ic_daily.empty:
        return {"ic": float("nan"), "ic_ir": float("nan")}

    ic = ic_daily.mean()
    ic_std = ic_daily.std()
    ic_ir = ic / ic_std if ic_std and ic_std > 1e-6 else float("nan")
    return {"ic": float(ic), "ic_ir": float(ic_ir)}


def _evaluate_segments(model: XGBModel, dataset: DatasetH) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for segment in ("train", "valid", "test"):
        try:
            label_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        except Exception:
            metrics[segment] = {"ic": float("nan"), "ic_ir": float("nan")}
            continue
        if label_df is None or label_df.empty:
            metrics[segment] = {"ic": float("nan"), "ic_ir": float("nan")}
            continue
        label_series = label_df.iloc[:, 0].astype("float32")
        pred = model.predict(dataset, segment)
        metrics[segment] = _compute_metrics(pred, label_series)
    return metrics


def _run_single(
    config: TrainConfig,
    dataset: DatasetH,
    dataset_config: Dict[str, object],
    params_override: Dict[str, Union[int, float]],
    grid_index: int,
    grid_total: int,
) -> Dict[str, object]:
    params_for_log = dict(params_override)
    model_kwargs, fit_kwargs = _prepare_model_and_fit_kwargs(config, params_for_log)
    model_config = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": model_kwargs,
    }

    run_name = f"seed_{config.seed}_grid_{grid_index:03d}"
    with R.start(experiment_name=config.experiment_name, recorder_name=run_name):
        model: XGBModel = init_instance_by_config(model_config)
        log_payload = {
            "model": model_kwargs,
            "fit": fit_kwargs,
            "grid": params_for_log,
            "dataset": {
                "handler": dataset_config["kwargs"]["handler"]["kwargs"],
                "segments": dataset_config["kwargs"].get("segments", {}),
            },
        }
        R.log_params(**flatten_dict(log_payload))
        model.fit(dataset, **fit_kwargs)
        recorder = R.get_recorder()
        sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
        sr.generate()
        try:
            SigAnaRecord(recorder=recorder).generate()
        except Exception:
            pass
        metrics = _evaluate_segments(model, dataset)
        recorder.log_metrics(**flatten_dict(metrics))

    params_used = dict(model_kwargs)
    params_used.update(fit_kwargs)
    params_used["recorder_name"] = run_name
    params_used["grid_index"] = grid_index
    params_used["grid_total"] = grid_total
    return {"metrics": metrics, "params": params_used, "recorder_name": run_name}


def _save_results(config: TrainConfig, metrics: Dict[str, Dict[str, float]], best_params: Dict[str, Union[int, float]]) -> None:
    base_dir = Path(config.output_dir)
    output_dir = base_dir / config.instrument / f"seed_{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
    payload = {
        "config": config_dict,
        "metrics": metrics,
        "best_params": best_params
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args(seed) -> TrainConfig:
    parser = argparse.ArgumentParser(description="XGBoost Qlib baseline for AlphaGen task")
    parser.add_argument("--experiment-name", dest="experiment_name", type=str, default=TrainConfig.experiment_name)
    parser.add_argument("--instrument", type=str, default=TrainConfig.instrument)
    parser.add_argument("--train-start", dest="train_start", type=str, default=TrainConfig.train_start)
    parser.add_argument("--train-end", dest="train_end", type=str, default=TrainConfig.train_end)
    parser.add_argument("--valid-start", dest="valid_start", type=str, default=TrainConfig.valid_start)
    parser.add_argument("--valid-end", dest="valid_end", type=str, default=TrainConfig.valid_end)
    parser.add_argument("--test-start", dest="test_start", type=str, default=TrainConfig.test_start)
    parser.add_argument("--test-end", dest="test_end", type=str, default=TrainConfig.test_end)
    parser.add_argument("--qlib-data", dest="qlib_data", type=str, default=TrainConfig.qlib_data)
    parser.add_argument("--data-device", dest="data_device", type=str, default=TrainConfig.data_device)
    parser.add_argument("--handler-class", dest="handler_class", type=str, default=TrainConfig.handler_class)
    parser.add_argument("--handler-module", dest="handler_module", type=str, default=TrainConfig.handler_module)
    parser.add_argument("--handler-json", dest="handler_json", type=str, default="")
    parser.add_argument("--n-estimators", dest="n_estimators", type=int, default=TrainConfig.n_estimators)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--max-depth", dest="max_depth", type=int, default=TrainConfig.max_depth)
    parser.add_argument("--min-child-weight", dest="min_child_weight", type=float, default=TrainConfig.min_child_weight)
    parser.add_argument("--subsample", type=float, default=TrainConfig.subsample)
    parser.add_argument("--colsample-bytree", dest="colsample_bytree", type=float, default=TrainConfig.colsample_bytree)
    parser.add_argument("--reg-lambda", dest="reg_lambda", type=float, default=TrainConfig.reg_lambda)
    parser.add_argument("--reg-alpha", dest="reg_alpha", type=float, default=TrainConfig.reg_alpha)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--tree-method", dest="tree_method", type=str, default=TrainConfig.tree_method)
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=TrainConfig.n_jobs)
    parser.add_argument("--eval-metric", dest="eval_metric", type=str, default=TrainConfig.eval_metric)
    parser.add_argument("--early-stopping", dest="early_stopping_rounds", type=int, default=TrainConfig.early_stopping_rounds)
    parser.add_argument("--xgb-verbose", dest="xgb_verbose", type=int, default=TrainConfig.xgb_verbose)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--grid-json", dest="grid_json", type=str, default="")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.endswith("_qlib"):
        output_dir = output_dir.rstrip("/") + "_qlib"

    handler_kwargs: Dict[str, Union[str, float, int]] = {
        "start_time": args.train_start,
        "end_time": args.test_end,
        "fit_start_time": args.train_start,
        "fit_end_time": args.train_end,
        "instruments": args.instrument,
    }
    if args.handler_json:
        try:
            handler_loaded = json.loads(args.handler_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse handler_json: {exc}") from exc
        if not isinstance(handler_loaded, dict):
            raise ValueError("handler_json must be a JSON object mapping handler kwargs to values.")
        handler_kwargs.update(handler_loaded)

    config = TrainConfig(
        seed=seed,
        experiment_name=args.experiment_name,
        instrument=args.instrument,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        qlib_data=args.qlib_data,
        data_device=args.data_device,
        handler_class=args.handler_class,
        handler_module=args.handler_module,
        handler_kwargs=handler_kwargs,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        eval_metric=args.eval_metric,
        early_stopping_rounds=args.early_stopping_rounds,
        xgb_verbose=args.xgb_verbose,
        output_dir=output_dir,
    )

    grid_params: Dict[str, Tuple[Union[int, float], ...]] = {}
    if args.grid_json:
        try:
            loaded = json.loads(args.grid_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse grid_json: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("grid_json must be a JSON object mapping parameter names to value lists.")
        for key, values in loaded.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"Grid values for {key} must be a list or tuple.")
            processed = tuple(values)
            if not processed:
                continue
            grid_params[key] = processed
    config.grid_params = grid_params
    return config


def main(seed) -> None:
    config = parse_args(seed=seed)
    reseed_everything(config.seed)
    initialize_qlib(os.path.expanduser(config.qlib_data))
    dataset_config = _build_dataset_config(config)
    dataset: DatasetH = init_instance_by_config(dataset_config)

    param_grid = _build_param_grid(config.grid_params)
    best_info: Union[Dict[str, object], None] = None
    best_valid_ic = float("-inf")

    for idx, params in enumerate(param_grid, 1):
        print(f"[Grid] ({idx}/{len(param_grid)}) testing params: {params}")
        reseed_everything(config.seed)
        run_info = _run_single(config, dataset, dataset_config, params, idx, len(param_grid))
        valid_ic = run_info["metrics"].get("valid", {}).get("ic", float("nan"))
        print(f"[Grid] params {params} => valid IC {valid_ic:.6f}")
        if math.isnan(valid_ic):
            if best_info is None:
                best_info = run_info
            continue
        if valid_ic > best_valid_ic:
            best_valid_ic = valid_ic
            best_info = run_info

    if best_info is None:
        raise RuntimeError("Grid search failed to produce a trained model.")

    best_valid_ic_value = best_info["metrics"].get("valid", {}).get("ic", float("nan"))
    print(f"[Grid] Best params: {best_info['params']} with valid IC {best_valid_ic_value:.6f}")

    _save_results(config, best_info["metrics"], best_info["params"])
    print(json.dumps(best_info["metrics"], indent=2))


if __name__ == "__main__":
    # seeds = (0, 42, 128, 256, 512, 1024, 2023, 2024, 2025, 2048)
    seeds = (0, 42, 128, 256, 512)
    for seed in seeds:
        main(seed)