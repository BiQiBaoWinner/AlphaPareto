import os
import json
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from gan.utils import load_pickle, get_blds_list_df
from gan.utils.data import get_data_by_year
from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret
from alphagen.utils.pytorch_utils import normalize_by_day


def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw["exprs"]
    exprs = [eval(expr_raw.replace("open", "open_").replace("$", "")) for expr_raw in exprs_raw]
    weights = raw["weights"]
    return exprs, weights


def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
        return load_alpha_pool(raw)


def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list = []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i : i + chunk_size], y[i : i + chunk_size]))
    return torch.cat(spearmanr_list, dim=0)


def get_tensor_metrics_raw(x, y):
    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0)
    ric_s = torch.nan_to_num(ric_s, nan=0)
    ret_s = torch.nan_to_num(ret_s, nan=0)

    return ic_s, ric_s, ret_s


def get_tensor_metrics_raw_chunked(x, y, device, chunk_days=200):
    n_days = x.shape[0]
    ic_chunks = []
    ric_chunks = []
    ret_chunks = []
    for start in range(0, n_days, chunk_days):
        end = min(start + chunk_days, n_days)
        x_chunk = x[start:end]
        y_chunk = y[start:end]
        if device.type == "cuda":
            x_chunk = x_chunk.to(device)
            y_chunk = y_chunk.to(device)
        with torch.no_grad():
            ic_s, ric_s, ret_s = get_tensor_metrics_raw(x_chunk, y_chunk)
        ic_chunks.append(ic_s.cpu())
        ric_chunks.append(ric_s.cpu())
        ret_chunks.append(ret_s.cpu())
        if device.type == "cuda":
            del x_chunk, y_chunk, ic_s, ric_s, ret_s
            torch.cuda.empty_cache()
    ic_s = torch.cat(ic_chunks, dim=0)
    ric_s = torch.cat(ric_chunks, dim=0)
    ret_s = torch.cat(ret_chunks, dim=0)
    return ic_s, ric_s, ret_s


def _move_stockdata_to_cpu(data_obj):
    data_obj.data = data_obj.data.cpu()
    data_obj.device = torch.device("cpu")
    return data_obj


def main(
    instruments: str = "csi500",
    freq: str = "day",
    seeds: str = "[0]",
    cuda: int = 0,
    save_name: str = "test",
    n_factors: int = 10,
    max_candidates: int = 20,
    window: Union[int, str] = "inf",
    metrics_chunk_days: int = 50,
    metrics_device: str = "auto",
):
    if isinstance(seeds, str):
        seeds = eval(seeds)
    assert isinstance(seeds, list)

    if isinstance(window, str):
        assert window == "inf"
        window = float("inf")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    if metrics_device == "cpu":
        device = torch.device("cpu")
    elif metrics_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if cuda >= 0 and torch.cuda.is_available() else "cpu")

    train_end = "2023-06-30"

    returned = get_data_by_year(
        train_start="2013-07-01",
        train_end="2023-06-30",
        valid_start="2023-07-01",
        valid_end="2024-06-30",
        test_start="2024-07-01",
        test_end="2025-06-30",
        instruments=instruments,
        target=target,
        freq=freq,
    )
    data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = returned

    # Keep data on CPU to avoid GPU OOM.
    data_all = _move_stockdata_to_cpu(data_all)
    data = _move_stockdata_to_cpu(data)
    data_valid = _move_stockdata_to_cpu(data_valid)
    data_valid_withhead = _move_stockdata_to_cpu(data_valid_withhead)
    data_test = _move_stockdata_to_cpu(data_test)
    data_test_withhead = _move_stockdata_to_cpu(data_test_withhead)

    for seed in seeds:
        path = f"out/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final.pkl"
        tensor_save_path = f"out/{save_name}_{instruments}_{train_end}_{seed}/"
        name = f"{train_end}_{n_factors}_{window}_{seed}"
        zoo = load_pickle(path)

        df = get_blds_list_df([zoo]).sort_values("score", ascending=False, key=lambda x: abs(x))
        df = df.head(max_candidates)

        cache_dir = os.path.join(tensor_save_path, f"flow_cache_{max_candidates}")
        os.makedirs(cache_dir, exist_ok=True)

        # Precompute target tensor once on CPU.
        tgt_tensor = target.evaluate(data_all)
        if tgt_tensor.dim() == 2:
            tgt_tensor_2d = tgt_tensor
            tgt_tensor_3d = tgt_tensor.unsqueeze(-1)
        else:
            tgt_tensor_2d = tgt_tensor[..., 0]
            tgt_tensor_3d = tgt_tensor
        # MOD: build date index aligned with target tensor days.
        if data_all.max_future_days == 0:
            date_index = data_all._dates[data_all.max_backtrack_days:]
        else:
            date_index = data_all._dates[data_all.max_backtrack_days : -data_all.max_future_days]
        date_index = date_index.astype(str)

        # Stream each factor to disk and write ic/ric/ret to disk-backed memmaps.
        n_days = tgt_tensor_2d.shape[0]
        n_exprs = len(df)
        ic_mm_path = os.path.join(cache_dir, "ic_s.mm")
        ric_mm_path = os.path.join(cache_dir, "ric_s.mm")
        ret_mm_path = os.path.join(cache_dir, "ret_s.mm")
        ic_mm = np.memmap(ic_mm_path, dtype="float32", mode="w+", shape=(n_days, n_exprs))
        ric_mm = np.memmap(ric_mm_path, dtype="float32", mode="w+", shape=(n_days, n_exprs))
        ret_mm = np.memmap(ret_mm_path, dtype="float32", mode="w+", shape=(n_days, n_exprs))

        for idx, expr in enumerate(df["exprs"].to_list()):
            fct_tensor = expr.evaluate(data_all)
            fct_tensor = normalize_by_day(fct_tensor)

            ic_s, ric_s, ret_s = get_tensor_metrics_raw_chunked(
                fct_tensor,
                tgt_tensor_2d,
                device,
                chunk_days=metrics_chunk_days,
            )
            ic_mm[:, idx] = ic_s.numpy()
            ric_mm[:, idx] = ric_s.numpy()
            ret_mm[:, idx] = ret_s.numpy()

            torch.save(fct_tensor.cpu(), os.path.join(cache_dir, f"fct_{idx}.pt"))

            del fct_tensor

        ic_mm.flush()
        ric_mm.flush()
        ret_mm.flush()

        # shift = 21
        shift = 20

        pred_list = []
        ics_list = []
        rics_list = []
        good_idx_list = []
        weights_list = []
        # MOD: keep per-date indices/weights for saving.
        # MOD: record test-set exprs/weights for json output.
        test_start_idx = len(tgt_tensor_3d) - data_test.n_days
        test_records = []

        pbar = range(len(tgt_tensor_3d) - data_test.n_days - data_valid.n_days, len(tgt_tensor_3d))
        for cur in pbar:
            if np.isfinite(window):
                begin = cur - window - shift
            else:
                begin = 0

            cur_ic = torch.from_numpy(ic_mm[begin : cur - shift])
            cur_ric = torch.from_numpy(ric_mm[begin : cur - shift])
            cur_ret = torch.from_numpy(ret_mm[begin : cur - shift])

            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)
            ret_mean = cur_ret.mean(dim=0)
            ret_std = cur_ret.std(dim=0)

            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std
            retir = ret_mean / ret_std

            metrics = dict(
                ic=ic_mean.detach().cpu().numpy(),
                ic_std=ic_std.detach().cpu().numpy(),
                icir=icir.detach().cpu().numpy(),
                ric=ric_mean.detach().cpu().numpy(),
                ric_std=ric_std.detach().cpu().numpy(),
                ricir=ricir.detach().cpu().numpy(),
                ret=ret_mean.detach().cpu().numpy(),
                ret_std=ret_std.detach().cpu().numpy(),
                retir=retir.detach().cpu().numpy(),
            )
            tmp = pd.DataFrame(metrics).sort_values("ricir", ascending=False, key=lambda x: abs(x))

            aaaa = tmp[(tmp["ric"] > 0.02) & (tmp["ricir"] > 0.2)]
            if len(aaaa) < 1:
                aaaa = tmp.iloc[:1]

            good_idx = aaaa.iloc[:n_factors].index.to_list()
            good_idx_list.append(good_idx)
            # MOD: store per-date factor indices.

            # Load only required factors from disk for the current window.
            factors = []
            for idx in good_idx:
                fct = torch.load(os.path.join(cache_dir, f"fct_{idx}.pt"), map_location="cpu")
                factors.append(fct)
            fct_stack = torch.stack(factors, dim=-1)

            x = fct_stack[begin : cur - shift]
            y = tgt_tensor_3d[begin : cur - shift]
            to_pred = fct_stack[cur]
            y_true = tgt_tensor_3d[cur]

            y = y.reshape(-1, y.shape[-1])
            x = x.reshape(-1, x.shape[-1])

            to_select = torch.isfinite(y)[:, 0]
            y = y[to_select]
            x = x[to_select]

            to_pred = torch.nan_to_num(to_pred, nan=0)

            ones = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones], dim=-1)
            ones = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones], dim=-1)

            coef = torch.linalg.lstsq(x, y).solution

            pred = to_pred @ coef
            weights_list.append(coef.detach().cpu().numpy())
            # MOD: store per-date weights.

            if cur >= test_start_idx:
                exprs_for_day = [str(df["exprs"].iloc[i]) for i in good_idx]
                coef_np = coef.detach().cpu().numpy()
                test_records.append(
                    {
                        "date": str(date_index[cur]),
                        "exprs": exprs_for_day,
                        "weights": coef_np[:-1].tolist(),
                        # MOD: ensure bias is a scalar to avoid numpy deprecation warning.
                        "bias": float(np.asarray(coef_np[-1]).item()),
                    }
                )
                # MOD: save per-day exprs/weights for test set.

            cur_ic = batch_pearsonr(pred.T, y_true.T)[0]
            cur_ric = batch_spearmanr(pred.T, y_true.T)[0]
            ics_list.append(cur_ic.detach().cpu().numpy())
            rics_list.append(cur_ric.detach().cpu().numpy())

            pred_list.append(pred[:, 0])

        # MOD: persist weights and selected indices as pt files.
        max_len = max(len(w) for w in weights_list)
        weights_padded = np.zeros((len(weights_list), max_len), dtype=np.float32)
        good_idx_padded = np.full((len(good_idx_list), max_len - 1), -1, dtype=np.int64)
        for i, w in enumerate(weights_list):
            # MOD: flatten weights to handle (k, 1) or (k,) shapes.
            w_arr = np.asarray(w, dtype=np.float32).reshape(-1)
            weights_padded[i, : w_arr.shape[0]] = w_arr
        for i, idxs in enumerate(good_idx_list):
            idx_arr = np.asarray(idxs, dtype=np.int64)
            good_idx_padded[i, : idx_arr.shape[0]] = idx_arr
        weights_tensor = torch.from_numpy(weights_padded)
        good_idx_tensor = torch.from_numpy(good_idx_padded)
        torch.save(weights_tensor, f"{tensor_save_path}/weights_{name}.pt")
        torch.save(good_idx_tensor, f"{tensor_save_path}/good_idx_{name}.pt")

        # MOD: write test-set exprs/weights json.
        json_path = os.path.join(tensor_save_path, "exprs_weights.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(test_records, f, ensure_ascii=True, indent=2)

        num_1 = data_valid.n_days
        num_2 = data_test.n_days
        all_pred = torch.stack(pred_list, dim=0)
        all_pred = all_pred[-num_2 - num_1 : -num_1]
        torch.save(all_pred.detach().cpu(), f"{tensor_save_path}/pred_valid_{name}.pt")

        num_ = data_test.n_days
        all_pred = torch.stack(pred_list, dim=0)
        all_pred = all_pred[-num_:]
        torch.save(all_pred.detach().cpu(), f"{tensor_save_path}/pred_{name}.pt")
        # MOD: align test stock ids to data_all columns before building DataFrame.
        all_ids = pd.Index(data_all._stock_ids)
        test_ids = pd.Index(data_test._stock_ids)
        col_idx = all_ids.get_indexer(test_ids)
        if (col_idx < 0).any():
            missing = test_ids[col_idx < 0]
            raise ValueError(f"test stock ids missing in data_all: {list(missing[:10])}")
        pred_test = all_pred[:, col_idx]
        # MOD: persist pred as a DataFrame with date/stock index.
        pred_df = data_test.make_dataframe(pred_test.detach().cpu())
        pred_df_path = os.path.join(tensor_save_path, f"pred_{name}_df.pkl")
        pred_df.to_pickle(pred_df_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
