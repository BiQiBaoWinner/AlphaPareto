import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch import nn

import gymnasium as gym
import sys
sys.modules['gym'] = gym

from qlib.contrib.model.pytorch_nn import DNNModelPytorch
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
    experiment_name: str = "mlp_qlib_baseline"
    instrument: str = "csi300"
    train_start: str = "2012-01-01" 
    train_end: str = "2021-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-06-30"
    qlib_data: str = "~/.qlib/qlib_data/cn_data_rolling"
    device: str = "auto"
    handler_class: str = "Alpha158"
    handler_module: str = "qlib.contrib.data.handler"
    handler_kwargs: Dict[str, Union[str, float, int]] = field(default_factory=dict)
    hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8192
    epochs: int = 50
    patience: int = 5
    # Modified default output dir
    output_dir: Union[str, Path] = "out/results/baselines/mlp_qlib"


def _resolve_device(device_flag: str) -> str:
    if device_flag.lower() == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_flag


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


def _compute_metrics(pred: Union[pd.Series, pd.DataFrame], label: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0] if not pred.empty else pd.Series(dtype="float32")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0] if not label.empty else pd.Series(dtype="float32")

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


class AlphaMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp_qlib(config: TrainConfig, dataset: DatasetH) -> Tuple[DNNModelPytorch, Dict[str, Union[int, float, str]]]:
    train_features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
    feature_dim = train_features.shape[1]
    hidden_sizes = config.hidden_sizes if config.hidden_sizes else tuple()
    model_backbone = AlphaMLP(feature_dim, hidden_sizes, config.dropout)

    train_size = len(train_features)
    steps_per_epoch = max(1, train_size // max(1, config.batch_size))
    max_steps = max(config.epochs * steps_per_epoch, config.epochs)
    eval_steps = max(1, steps_per_epoch)
    device_flag = _resolve_device(config.device)

    model = DNNModelPytorch(
        lr=config.lr,
        max_steps=max_steps,
        batch_size=config.batch_size,
        early_stop_rounds=config.patience,
        eval_steps=eval_steps,
        optimizer="adam",
        loss="mse",
        GPU=device_flag,
        seed=config.seed,
        weight_decay=config.weight_decay,
        init_model=model_backbone,
    )
    model.fit(dataset, verbose=True)

    train_summary: Dict[str, Union[int, float, str]] = {
        "train_size": int(train_size),
        "feature_dim": int(feature_dim),
        "steps_per_epoch": int(steps_per_epoch),
        "max_steps": int(max_steps),
        "eval_steps": int(eval_steps),
        "device": device_flag,
    }
    return model, train_summary


def _evaluate_segments(model: DNNModelPytorch, dataset: DatasetH) -> Dict[str, Dict[str, float]]:
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
        pred = model.predict(dataset, segment)
        metrics[segment] = _compute_metrics(pred, label_df)
    return metrics


def _save_results(
    config: TrainConfig,
    metrics: Dict[str, Dict[str, float]],
    training_info: Dict[str, Union[int, float, str]],
) -> None:
    base_dir = Path(config.output_dir)
    output_dir = base_dir / config.instrument / f"seed_{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
    config_dict["hidden_sizes"] = list(config.hidden_sizes)

    payload = {
        "config": config_dict,
        "metrics": metrics,
        "training_info": training_info,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args(seed) -> TrainConfig:
    parser = argparse.ArgumentParser(description="MLP Qlib baseline for AlphaGen task")
    parser.add_argument("--experiment-name", dest="experiment_name", type=str, default=TrainConfig.experiment_name)
    parser.add_argument("--instrument", type=str, default=TrainConfig.instrument)
    parser.add_argument("--train-start", dest="train_start", type=str, default=TrainConfig.train_start)
    parser.add_argument("--train-end", dest="train_end", type=str, default=TrainConfig.train_end)
    parser.add_argument("--valid-start", dest="valid_start", type=str, default=TrainConfig.valid_start)
    parser.add_argument("--valid-end", dest="valid_end", type=str, default=TrainConfig.valid_end)
    parser.add_argument("--test-start", dest="test_start", type=str, default=TrainConfig.test_start)
    parser.add_argument("--test-end", dest="test_end", type=str, default=TrainConfig.test_end)
    parser.add_argument("--qlib-data", dest="qlib_data", type=str, default=TrainConfig.qlib_data)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--handler-class", dest="handler_class", type=str, default=TrainConfig.handler_class)
    parser.add_argument("--handler-module", dest="handler_module", type=str, default=TrainConfig.handler_module)
    parser.add_argument("--handler-json", dest="handler_json", type=str, default="")
    parser.add_argument("--hidden-sizes", dest="hidden_sizes", type=int, nargs="*", default=list(TrainConfig.hidden_sizes))
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=TrainConfig.output_dir)
    args = parser.parse_args()
    hidden = tuple(args.hidden_sizes) if args.hidden_sizes else tuple()
    handler_kwargs: Dict[str, Union[str, float, int]] = {}
    if args.handler_json:
        try:
            loaded = json.loads(args.handler_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse handler_json: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("handler_json must be a JSON object mapping handler kwargs to values.")
        handler_kwargs.update(loaded)
    
    # Ensure output_dir ends with _qlib
    output_dir = args.output_dir
    if not output_dir.endswith("_qlib"):
        output_dir = output_dir.rstrip("/") + "_qlib"

    return TrainConfig(
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
        device=args.device,
        handler_class=args.handler_class,
        handler_module=args.handler_module,
        handler_kwargs=handler_kwargs,
        hidden_sizes=hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=output_dir
    )


def main(seed) -> None:
    config = parse_args(seed=seed)
    reseed_everything(config.seed)
    initialize_qlib(os.path.expanduser(config.qlib_data))
    dataset_config = _build_dataset_config(config)
    dataset: DatasetH = init_instance_by_config(dataset_config)

    recorder_name = f"seed_{config.seed}"
    metrics: Dict[str, Dict[str, float]] = {}
    training_summary: Dict[str, Union[int, float, str]]

    with R.start(experiment_name=config.experiment_name, recorder_name=recorder_name):
        model, training_summary = _train_mlp_qlib(config, dataset)
        recorder = R.get_recorder()

        log_payload = {
            "model": {
                "hidden_sizes": list(config.hidden_sizes),
                "dropout": config.dropout,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "patience": config.patience,
                "device": training_summary.get("device"),
            },
            "training": training_summary,
            "dataset": {
                "handler": dataset_config["kwargs"]["handler"]["kwargs"],
                "segments": dataset_config["kwargs"].get("segments", {}),
            },
        }
        R.log_params(**flatten_dict(log_payload))

        sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
        sr.generate()
        try:
            SigAnaRecord(recorder=recorder).generate()
        except Exception:
            pass

        metrics = _evaluate_segments(model, dataset)
        recorder.log_metrics(**flatten_dict(metrics))

    _save_results(config, metrics, training_summary)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    # seeds = (0, 42, 128, 256, 512, 1024, 2023, 2024, 2025, 2048)
    seeds = (0, 42, 128, 256, 512)
    for seed in seeds:
        main(seed)