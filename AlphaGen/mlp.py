import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    device: str = "cuda:0"
    # device: str = "auto"
    hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8192
    eval_batch_size: int = 65536
    epochs: int = 50
    patience: int = 5
    output_dir: Union[str, Path] = "out/results/baselines/mlp"


def _auto_device(device_flag: str) -> torch.device:
    if device_flag != "auto":
        return torch.device(device_flag)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_dataset(features: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, split: str) -> TensorDataset:
    sample_count = int(mask.sum().item())
    if sample_count == 0:
        raise RuntimeError(f"No valid samples available for {split}.")
    return TensorDataset(features[mask], target[mask])


def _build_feature_matrix(stock_data: StockData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
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


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_mlp(
    config: TrainConfig,
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    train_mask: torch.Tensor,
    valid_features: torch.Tensor,
    valid_target: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device
) -> FeedForwardNet:
    train_ds = _make_dataset(train_features, train_target, train_mask, "training")
    valid_ds = _make_dataset(valid_features, valid_target, valid_mask, "validation")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)

    model = FeedForwardNet(train_features.shape[1], config.hidden_sizes, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    epochs_without_improve = 0

    for epoch in range(config.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item() * len(batch_x)
        val_loss /= len(valid_ds)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_full(model: FeedForwardNet, features: torch.Tensor, mask: torch.Tensor,
                  batch_size: int, device: torch.device) -> torch.Tensor:
    preds = torch.zeros(features.shape[0], dtype=torch.float32)
    if mask.sum() == 0:
        return preds
    dataset = TensorDataset(features[mask])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    gathered = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            out = model(batch_x.to(device)).cpu()
            gathered.append(out)
    preds_idx = torch.cat(gathered)
    preds[idx] = preds_idx
    return preds


def _metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             n_days: int, n_stocks: int) -> Dict[str, float]:
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


def _save_results(config: TrainConfig, metrics: Dict[str, Dict[str, float]]) -> None:
    # output_dir = _prepare_output_dir(config.output_dir, config.seed)
    # output_path = output_dir / "results.json"
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    output_path = os.path.join(config.output_dir, f"{config.instrument}_{config.seed}_results.json")
    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
    payload = {
        "config": config_dict,
        "metrics": metrics
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args(seed) -> TrainConfig:
    parser = argparse.ArgumentParser(description="MLP baseline for AlphaGen task")
    # parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--instrument", type=str, default=TrainConfig.instrument)
    parser.add_argument("--train-start", dest="train_start", type=str, default=TrainConfig.train_start)
    parser.add_argument("--train-end", dest="train_end", type=str, default=TrainConfig.train_end)
    parser.add_argument("--valid-start", dest="valid_start", type=str, default=TrainConfig.valid_start)
    parser.add_argument("--valid-end", dest="valid_end", type=str, default=TrainConfig.valid_end)
    parser.add_argument("--test-start", dest="test_start", type=str, default=TrainConfig.test_start)
    parser.add_argument("--test-end", dest="test_end", type=str, default=TrainConfig.test_end)
    parser.add_argument("--qlib-data", dest="qlib_data", type=str, default=TrainConfig.qlib_data)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--hidden-sizes", dest="hidden_sizes", type=int, nargs="*", default=list(TrainConfig.hidden_sizes))
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=TrainConfig.output_dir)
    args = parser.parse_args()
    hidden = tuple(args.hidden_sizes) if args.hidden_sizes else TrainConfig.hidden_sizes
    
    return TrainConfig(
        seed=seed,
        instrument=args.instrument,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        qlib_data=args.qlib_data,
        device=args.device,
        hidden_sizes=hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )


def main(seed) -> None:
    
    config = parse_args(seed=seed)
    reseed_everything(config.seed)
    initialize_qlib(os.path.expanduser(config.qlib_data))
    device = _auto_device(config.device)

    stock_device = device
    data_train = StockData(config.instrument, config.train_start, config.train_end, device=stock_device)
    data_valid = StockData(config.instrument, config.valid_start, config.valid_end, device=stock_device)
    data_test = StockData(config.instrument, config.test_start, config.test_end, device=stock_device)

    train_feat, train_tgt, train_mask, n_days_train, n_stocks_train = _build_feature_matrix(data_train)
    valid_feat, valid_tgt, valid_mask, n_days_valid, n_stocks_valid = _build_feature_matrix(data_valid)
    test_feat, test_tgt, test_mask, n_days_test, n_stocks_test = _build_feature_matrix(data_test)

    model = _train_mlp(
        config,
        train_feat,
        train_tgt,
        train_mask,
        valid_feat,
        valid_tgt,
        valid_mask,
        device
    )

    pred_train = _predict_full(model, train_feat, train_mask, config.eval_batch_size, device)
    pred_valid = _predict_full(model, valid_feat, valid_mask, config.eval_batch_size, device)
    pred_test = _predict_full(model, test_feat, test_mask, config.eval_batch_size, device)

    metrics = {
        "train": _metrics(pred_train, train_tgt, train_mask, n_days_train, n_stocks_train),
        "valid": _metrics(pred_valid, valid_tgt, valid_mask, n_days_valid, n_stocks_valid),
        "test": _metrics(pred_test, test_tgt, test_mask, n_days_test, n_stocks_test)
    }

    _save_results(config, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    seeds = (0, 42, 128, 256, 512, 1024, 2023, 2024, 2025, 2048)
    for seed in seeds:
        main(seed)
