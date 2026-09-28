import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


SEEDS = (0, 42, 128, 256, 512)

ALL_INSTRUMENTS = ['csi300', 'csi800', 'all']

TRAIN_START = "2013-07-01"
TRAIN_END = "2023-06-30"
VALID_START = "2023-07-01"
VALID_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-06-30"

DATA_START = "2012-07-01"
DATA_END = "2025-06-30"


PARAM_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "fit_intercept": [True, False],
}



def init_qlib(data_path):
    import qlib
    from qlib.config import REG_CN
    data_path = os.path.expanduser(data_path)
    qlib.init(provider_uri=data_path, region=REG_CN)
    print(f"[ok] Qlib init done, data_path={data_path}")



def build_dataset(instruments):
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    segments = {
        "train": (TRAIN_START, TRAIN_END),
        "valid": (VALID_START, VALID_END),
        "test":  (TEST_START, TEST_END),
    }

    handler = Alpha158(
        instruments=instruments,
        start_time=DATA_START,
        end_time=DATA_END,
        infer_processors=[
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
        ],
        label=["Ref($close, -20) / $close - 1"],
    )

    dataset = DatasetH(handler=handler, segments=segments)

    for seg_name in ["train", "valid", "test"]:
        seg_data = dataset.prepare(seg_name, col_set="label", data_key="infer")
        if isinstance(seg_data, pd.DataFrame):
            seg_data = seg_data.iloc[:, 0]
        n_samples = len(seg_data)
        n_days = len(seg_data.index.get_level_values("datetime").unique())
        n_stocks = n_samples // max(n_days, 1)
        print(f"  {seg_name}: {n_samples} samples, {n_days} days, ~{n_stocks} stocks/day")

    feat_sample = dataset.prepare("train", col_set="feature", data_key="infer")
    n_features = feat_sample.shape[1]
    print(f"  features: {n_features}")
    print(f"  instruments: {instruments}")
    return dataset


# Qlib LinearModel

def get_linear_model(params):

    from qlib.contrib.model.linear import LinearModel
    return LinearModel(
        estimator="ridge",
        alpha=params["alpha"],
        fit_intercept=params["fit_intercept"],
    )



def calc_metrics(pred, label):
    df = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if df.empty:
        return {"IC": np.nan, "RankIC": np.nan, "ICIR": np.nan, "RankICIR": np.nan}

    daily_ic, daily_ric = [], []
    for _, g in df.groupby(level="datetime"):
        if len(g) < 5:
            continue
        p, l = g["pred"].values, g["label"].values
        ic, _ = stats.pearsonr(p, l)
        ric, _ = stats.spearmanr(p, l)
        daily_ic.append(ic)
        daily_ric.append(ric)

    daily_ic = np.array(daily_ic)
    daily_ric = np.array(daily_ric)

    return {
        "IC": float(np.mean(daily_ic)),
        "RankIC": float(np.mean(daily_ric)),
        "ICIR": float(np.mean(daily_ic) / (np.std(daily_ic) + 1e-8)),
        "RankICIR": float(np.mean(daily_ric) / (np.std(daily_ric) + 1e-8)),
    }



def search_hyperparams(dataset):
    print("\n" + "=" * 60)
    print("Alpha158 + Ridge Hyperparameter Search via sklearn ParameterGrid")
    print(f"  scoring=neg_MSE")
    print(f"  validation: {VALID_START} ~ {VALID_END}")
    print("=" * 60)

    for seg_name in ["train", "valid"]:
        seg_data = dataset.prepare(seg_name, col_set="label", data_key="infer")
        if isinstance(seg_data, pd.DataFrame):
            seg_data = seg_data.iloc[:, 0]
        n_samples = len(seg_data)
        n_days = len(seg_data.index.get_level_values("datetime").unique())
        seg_label = seg_name.capitalize()
        print(f"  {seg_label}: {n_samples} samples, {n_days} days")

    from sklearn.model_selection import ParameterGrid
    param_combinations = list(ParameterGrid(PARAM_GRID))
    n_total = len(param_combinations)
    print(f"  Total combinations: {n_total}")

    search_log = []
    best_score = -np.inf
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\n  [{i+1}/{n_total}] alpha={params['alpha']}, "
              f"fit_intercept={params['fit_intercept']}", end="  ")

        try:
            model = get_linear_model(params)
            model.fit(dataset)

            seg_label = dataset.prepare("valid", col_set="label", data_key="infer")
            if isinstance(seg_label, pd.DataFrame):
                seg_label = seg_label.iloc[:, 0]
            pred = model.predict(dataset, segment="valid")
            pred = pred.reindex(seg_label.index)

            common = pred.dropna().index.intersection(seg_label.dropna().index)
            pred_clean = pred.loc[common].values
            label_clean = seg_label.loc[common].values
            mse_val = float(np.mean((pred_clean - label_clean) ** 2))
            valid_score = -mse_val

            print(f"Valid MSE={mse_val:.6f}  (neg_MSE={valid_score:.6f})")

        except Exception as e:
            print(f"FAILED - {e}")
            valid_score = -np.inf

        search_log.append({
            "params": {k: float(v) if isinstance(v, float) else v
                       for k, v in params.items()},
            "mean_score": float(valid_score),
            "std_score": 0.0,
            "rank": -1,
        })

        if valid_score > best_score:
            best_score = valid_score
            best_params = params.copy()

    sorted_indices = sorted(range(len(search_log)),
                            key=lambda x: -search_log[x]["mean_score"])
    for rank, idx in enumerate(sorted_indices, 1):
        search_log[idx]["rank"] = rank

    print(f"\n  Best Valid neg_MSE: {best_score:.6f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")

    return best_params, search_log



def run_single(seed, dataset, params):

    print(f"  seed={seed}: training...", end=" ", flush=True)

    model = get_linear_model(params)
    model.fit(dataset)

    results = {}
    test_pred = None
    for seg_name in ["train", "valid", "test"]:
        seg_label = dataset.prepare(seg_name, col_set="label", data_key="infer")
        if isinstance(seg_label, pd.DataFrame):
            seg_label = seg_label.iloc[:, 0]
        pred_seg = model.predict(dataset, segment=seg_name)
        pred_seg = pred_seg.reindex(seg_label.index)
        metrics = calc_metrics(pred_seg, seg_label)
        results[seg_name] = metrics
        if seg_name == "test":
            test_pred = pred_seg

    print(f"train IC={results['train']['IC']:.4f} | "
          f"valid IC={results['valid']['IC']:.4f} | "
          f"test IC={results['test']['IC']:.4f}  "
          f"RankIC={results['test']['RankIC']:.4f}")

    return results, test_pred



def run_experiment(instruments, output_dir):
    print("\n" + "=" * 60)
    print(f"Alpha158 + Ridge Linear: instruments={instruments}")
    print(f"  train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  valid: {VALID_START} ~ {VALID_END}")
    print(f"  test:  {TEST_START} ~ {TEST_END}")
    print("=" * 60)

    dataset = build_dataset(instruments)
    os.makedirs(output_dir, exist_ok=True)

    best_params, search_log = search_hyperparams(dataset)

    search_path = os.path.join(output_dir, f"alpha158_{instruments}_search.json")
    sorted_log = sorted(search_log, key=lambda x: -x["mean_score"])

    best_params_save = {k: float(v) if isinstance(v, float) else v
                        for k, v in best_params.items()}
    search_save = {
        "best_params": best_params_save,
        "best_cv_score": float(sorted_log[0]["mean_score"]),
        "n_combinations": len(search_log),
        "cv_method": "sklearn ParameterGrid + Qlib LinearModel (train on train, eval on valid)",
        "scoring": "neg_MSE",
        "all_results": sorted_log,
    }
    with open(search_path, "w") as f:
        json.dump(search_save, f, indent=4)
    print(f"\n  Search log saved to {search_path}")

    print(f"\n  Running multi-seed experiments with best params...")
    print(f"  Note: Ridge is deterministic, all seeds yield identical results.")
    all_results = []
    for seed in SEEDS:
        try:
            results, test_pred = run_single(seed, dataset, best_params)
            all_results.append({"seed": seed, **results})
            pred_path = os.path.join(output_dir, f"alpha158_{instruments}_seed{seed}_pred.pkl")
            test_pred.to_pickle(pred_path)
            print(f"    pred saved to {pred_path}")
        except Exception as e:
            print(f"  seed={seed}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("  no successful runs!")
        return

    print(f"\n  Summary ({len(all_results)}/{len(SEEDS)} seeds):")
    for seg_name in ["train", "valid", "test"]:
        ics = [r[seg_name]["IC"] for r in all_results]
        rics = [r[seg_name]["RankIC"] for r in all_results]
        print(f"    {seg_name:>5}: IC={np.mean(ics):.4f}({np.std(ics):.4f})  "
              f"RankIC={np.mean(rics):.4f}({np.std(rics):.4f})")

    save_path = os.path.join(output_dir, f"alpha158_{instruments}.json")
    save_data = {
        "instruments": instruments,
        "model": "ridge_linear",
        "features": "alpha158",
        "tuning": "sklearn_ParameterGrid_QlibLinearModel",
        "best_params": best_params_save,
        "time_splits": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "test":  [TEST_START, TEST_END],
        },
        "seeds": list(SEEDS),
        "per_seed": [],
    }

    for r in all_results:
        entry = {"seed": r["seed"]}
        for seg in ["train", "valid", "test"]:
            entry[seg] = r[seg]
        save_data["per_seed"].append(entry)

    for seg_name in ["train", "valid", "test"]:
        ics = [r[seg_name]["IC"] for r in all_results]
        rics = [r[seg_name]["RankIC"] for r in all_results]
        icirs = [r[seg_name]["ICIR"] for r in all_results]
        ricirs = [r[seg_name]["RankICIR"] for r in all_results]
        save_data[f"{seg_name}_summary"] = {
            "IC_mean": float(np.mean(ics)),
            "IC_std": float(np.std(ics)),
            "RankIC_mean": float(np.mean(rics)),
            "RankIC_std": float(np.std(rics)),
            "ICIR_mean": float(np.mean(icirs)),
            "RankICIR_mean": float(np.mean(ricirs)),
            "num_seeds": len(ics),
        }

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"\n  saved to {save_path}")



def print_summary(run_list, output_dir):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (Alpha158 + Ridge Linear)")
    print("=" * 80)
    print(f"{'Instruments':<14} {'Segment':<8} "
          f"{'IC':>16} {'RankIC':>16} {'ICIR':>10}")
    print("-" * 64)

    for instruments in run_list:
        save_path = os.path.join(output_dir, f"alpha158_{instruments}.json")
        if not os.path.exists(save_path):
            continue
        with open(save_path) as f:
            data = json.load(f)

        if "best_params" in data:
            print(f"  Tuned params: {data['best_params']}")

        for seg in ["train", "valid", "test"]:
            s = data[f"{seg}_summary"]
            print(f"{instruments:<14} {seg:<8} "
                  f"{s['IC_mean']:>7.4f}({s['IC_std']:.4f}) "
                  f"{s['RankIC_mean']:>7.4f}({s['RankIC_std']:.4f}) "
                  f"{s['ICIR_mean']:>8.4f}")
        print()



def main():
    parser = argparse.ArgumentParser(
        description="Alpha158 + Ridge Linear Baseline"
    )
    parser.add_argument("--instruments", default="csi300",
                        choices=["csi300", "csi800", "all"])
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--data_path", default="~/.qlib/qlib_data/cn_data_2024h1")
    parser.add_argument("--output_dir", default="./baselines_output/alpha158_baseline")
    args = parser.parse_args()

    init_qlib(args.data_path)

    run_list = ALL_INSTRUMENTS if args.run_all else [args.instruments]
    for instruments in run_list:
        run_experiment(instruments, args.output_dir)

    print_summary(run_list, args.output_dir)


if __name__ == '__main__':
    main()