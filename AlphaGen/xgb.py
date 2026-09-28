import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

import xgboost as xgb

from alphagen.utils.random import reseed_everything
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.correlation import batch_pearsonr
from alphagen_generic.features import open_, close, high, low, volume, vwap, target
from alphagen_qlib.stock_data import StockData, initialize_qlib


@dataclass
class TrainConfig:
    seed: int = 2
    instrument: str = "csi300"
    train_start: str = "2012-01-01"
    train_end: str = "2021-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-06-30"
    # qlib_data: str = "~/.qlib/qlib_data/cn_data_2024h1"
    qlib_data: str = "~/.qlib/qlib_data/cn_data_rolling"
    data_device: str = "cpu"
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
    output_dir: Union[str, Path] = "out/results/baselines/xgb"
    grid_params: Dict[str, Tuple[Union[int, float], ...]] = field(default_factory=dict)


def _resolve_device(name: str) -> torch.device:
    return torch.device(name)

def _mask_to_numpy(features: Tensor, target: Tensor, mask: Tensor, split: str) -> Tuple[np.ndarray, np.ndarray]:
    sample_count = int(mask.sum().item())
    if sample_count == 0:
        raise RuntimeError(f"No valid samples available for {split}.")
    X = features[mask].numpy()
    y = target[mask].numpy()
    return X, y


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _default_grid_from_config(config: TrainConfig) -> Dict[str, Tuple[Union[int, float], ...]]:
    n_estimators = tuple(sorted({
        config.n_estimators,
        min(4000, max(200, int(config.n_estimators * 1.5)))
    }))
    learning_rates = tuple(sorted({
        _clamp(config.learning_rate * 0.5, 1e-4, 0.2),
        _clamp(config.learning_rate * 2.0, 1e-4, 0.2)
    }))
    max_depth = tuple(sorted({
        max(3, config.max_depth - 2),
        min(12, config.max_depth + 2)
    }))
    subsample = tuple(sorted({
        _clamp(config.subsample, 0.5, 0.95),
        _clamp(config.subsample + 0.1, 0.5, 0.95)
    }))
    colsample = tuple(sorted({
        _clamp(config.colsample_bytree, 0.5, 0.95),
        _clamp(config.colsample_bytree + 0.1, 0.5, 0.95)
    }))
    min_child_weight = (max(0.1, config.min_child_weight),)
    reg_lambda = (max(1e-4, config.reg_lambda),)
    reg_alpha = (max(0.0, config.reg_alpha),)
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rates,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "min_child_weight": min_child_weight,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha
    }


def _build_param_grid(grid: Dict[str, Tuple[Union[int, float], ...]]) -> List[Dict[str, Union[int, float]]]:
    keys = [key for key, values in grid.items() if values]
    if not keys:
        raise ValueError("Parameter grid is empty; provide at least one hyperparameter candidate.")
    combos: List[Dict[str, Union[int, float]]] = []
    for values in product(*(grid[key] for key in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def _build_feature_matrix(stock_data: StockData) -> Tuple[Tensor, Tensor, Tensor, int, int]:
    exprs = [open_, close, high, low, volume, vwap]
    tensors = []
    for expr in exprs:
        val = normalize_by_day(expr.evaluate(stock_data))
        tensors.append(torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
    features = torch.stack(tensors, dim=-1)
    target_raw = target.evaluate(stock_data)
    valid_mask = ~torch.isnan(target_raw)
    target_tensor = normalize_by_day(target_raw)
    target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    flat_features = features.reshape(-1, features.shape[-1]).contiguous()
    flat_target = target_tensor.reshape(-1).contiguous()
    flat_mask = valid_mask.reshape(-1)
    return flat_features.cpu(), flat_target.cpu(), flat_mask.cpu(), stock_data.n_days, stock_data.n_stocks


def _train_xgb(
    config: TrainConfig,
    train_features: Tensor,
    train_target: Tensor,
    train_mask: Tensor,
    valid_features: Tensor,
    valid_target: Tensor,
    valid_mask: Tensor,
    params_override: Dict[str, Union[int, float]]
) -> Tuple[xgb.XGBRegressor, Dict[str, Union[int, float]]]:
    X_train, y_train = _mask_to_numpy(train_features, train_target, train_mask, "training")
    X_valid, y_valid = _mask_to_numpy(valid_features, valid_target, valid_mask, "validation")
    indices = np.arange(X_train.shape[0])
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    model_kwargs: Dict[str, Union[int, float, str]] = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "reg_alpha": config.reg_alpha,
        "gamma": config.gamma,
        "tree_method": config.tree_method,
        "n_jobs": config.n_jobs,
        "random_state": config.seed,
        "eval_metric": config.eval_metric
    }
    model_kwargs.update(params_override)
    model = xgb.XGBRegressor(**model_kwargs)

    if valid_mask.sum().item() > 0:
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  verbose=config.xgb_verbose)
    else:
        model.fit(X_train, y_train, verbose=config.xgb_verbose)

    return model, model_kwargs


def _predict_full(model: xgb.XGBRegressor, features: Tensor, mask: Tensor) -> Tensor:
    preds = np.zeros(features.shape[0], dtype=np.float32)
    mask_np = mask.numpy()
    if mask_np.sum() == 0:
        return torch.from_numpy(preds)
    inputs = features[mask].numpy()
    preds_mask = model.predict(inputs)
    preds[mask_np] = preds_mask
    return torch.from_numpy(preds)


def _metrics(pred: Tensor, target: Tensor, mask: Tensor, n_days: int, n_stocks: int) -> Dict[str, float]:
    pred_matrix = pred.reshape(n_days, n_stocks)
    target_matrix = target.reshape(n_days, n_stocks)
    mask_matrix = mask.reshape(n_days, n_stocks)
    pred_matrix = pred_matrix.clone()
    target_matrix = target_matrix.clone()
    pred_matrix[~mask_matrix] = torch.nan
    target_matrix[~mask_matrix] = torch.nan
    ic_daily = batch_pearsonr(pred_matrix, target_matrix)
    ic = ic_daily.mean().item()
    ic_std = ic_daily.std().item()
    ic_ir = ic / ic_std if ic_std > 1e-6 else float("nan")
    return {
        "ic": ic,
        "ic_ir": ic_ir
    }


def _save_results(config: TrainConfig, metrics: Dict[str, Dict[str, float]], best_params: Dict[str, Union[int, float]]) -> None:
    # output_dir = _prepare_output_dir(config.output_dir, config.seed)
    # output_path = output_dir / "results.json"
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
    parser = argparse.ArgumentParser(description="XGBoost baseline for AlphaGen task")
    # parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--instrument", type=str, default=TrainConfig.instrument)
    parser.add_argument("--train-start", dest="train_start", type=str, default=TrainConfig.train_start)
    parser.add_argument("--train-end", dest="train_end", type=str, default=TrainConfig.train_end)
    parser.add_argument("--valid-start", dest="valid_start", type=str, default=TrainConfig.valid_start)
    parser.add_argument("--valid-end", dest="valid_end", type=str, default=TrainConfig.valid_end)
    parser.add_argument("--test-start", dest="test_start", type=str, default=TrainConfig.test_start)
    parser.add_argument("--test-end", dest="test_end", type=str, default=TrainConfig.test_end)
    parser.add_argument("--qlib-data", dest="qlib_data", type=str, default=TrainConfig.qlib_data)
    parser.add_argument("--data-device", dest="data_device", type=str, default=TrainConfig.data_device)
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
    
    config = TrainConfig(
        seed=seed,
        instrument=args.instrument,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        qlib_data=args.qlib_data,
        data_device=args.data_device,
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
        output_dir=args.output_dir
    )

    grid_params: Dict[str, Tuple[Union[int, float], ...]] = {}
    if args.grid_json:
        try:
            loaded = json.loads(args.grid_json)
            if not isinstance(loaded, dict):
                raise ValueError("grid_json must be a JSON object mapping parameter names to value lists.")
            for key, values in loaded.items():
                if not isinstance(values, (list, tuple)):
                    raise ValueError(f"Grid values for {key} must be a list or tuple.")
                processed = tuple(values)
                if not processed:
                    continue
                grid_params[key] = processed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse grid_json: {exc}") from exc
    if not grid_params:
        grid_params = _default_grid_from_config(config)
    else:
        # ensure all entries are tuples
        grid_params = {k: tuple(v) for k, v in grid_params.items() if v}
    config.grid_params = grid_params
    return config


def main(seed) -> None:
    
    config = parse_args(seed=seed)
    reseed_everything(config.seed)
    initialize_qlib(os.path.expanduser(config.qlib_data))
    data_device = _resolve_device(config.data_device)
    
    data_train = StockData(config.instrument, config.train_start, config.train_end, device=data_device)
    data_valid = StockData(config.instrument, config.valid_start, config.valid_end, device=data_device)
    data_test = StockData(config.instrument, config.test_start, config.test_end, device=data_device)

    train_feat, train_tgt, train_mask, n_days_train, n_stocks_train = _build_feature_matrix(data_train)
    valid_feat, valid_tgt, valid_mask, n_days_valid, n_stocks_valid = _build_feature_matrix(data_valid)
    test_feat, test_tgt, test_mask, n_days_test, n_stocks_test = _build_feature_matrix(data_test)

    param_grid = _build_param_grid(config.grid_params)
    best_model: Union[xgb.XGBRegressor, None] = None
    best_params: Dict[str, Union[int, float]] = {}
    best_valid_ic = float("-inf")

    for idx, params in enumerate(param_grid, 1):
        print(f"[Grid] ({idx}/{len(param_grid)}) testing params: {params}")
        model, params_used = _train_xgb(
            config,
            train_feat,
            train_tgt,
            train_mask,
            valid_feat,
            valid_tgt,
            valid_mask,
            params_override=params
        )
        pred_valid_tmp = _predict_full(model, valid_feat, valid_mask)
        valid_metrics_tmp = _metrics(pred_valid_tmp, valid_tgt, valid_mask, n_days_valid, n_stocks_valid)
        valid_ic = valid_metrics_tmp["ic"]
        print(f"[Grid] params {params} => valid IC {valid_ic:.6f}")
        if valid_ic > best_valid_ic:
            best_valid_ic = valid_ic
            best_model = model
            best_params = params_used

    if best_model is None:
        raise RuntimeError("Grid search failed to produce a trained model.")

    print(f"[Grid] Best params: {best_params} with valid IC {best_valid_ic:.6f}")

    pred_train = _predict_full(best_model, train_feat, train_mask)
    pred_valid = _predict_full(best_model, valid_feat, valid_mask)
    pred_test = _predict_full(best_model, test_feat, test_mask)

    metrics = {
        "train": _metrics(pred_train, train_tgt, train_mask, n_days_train, n_stocks_train),
        "valid": _metrics(pred_valid, valid_tgt, valid_mask, n_days_valid, n_stocks_valid),
        "test": _metrics(pred_test, test_tgt, test_mask, n_days_test, n_stocks_test)
    }

    _save_results(config, metrics, best_params)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    seeds = (0, 42, 128, 256, 512, 1024, 2023, 2024, 2025, 2048)
    for seed in seeds:
        main(seed)
