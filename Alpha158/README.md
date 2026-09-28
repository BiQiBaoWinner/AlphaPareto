# Alpha158 + Ridge Linear Baseline

This method uses Qlib's [Alpha158](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py) feature set (158 price-volume technical factors) with a Ridge regression model as the baseline predictor.

## Method

1. **Features**: Alpha158 handler generates 158 technical indicators from OHLCV data, preprocessed with `Fillna` and `CSRankNorm`.
2. **Label**: 20-day forward return — `Ref($close, -20) / $close - 1`.
3. **Hyperparameter Search**: Grid search over Ridge regularization strength (`alpha`: 7 values from 0.001 to 1000) and intercept fitting (True/False), totaling 14 combinations. Scored by negative MSE on the validation set.
4. **Evaluation**: Train with best hyperparameters across 5 random seeds. Ridge is a deterministic closed-form solution, so all seeds produce identical results (multi-seed structure is retained for format consistency with other methods).

## Usage

```bash
# Single market
python alpha158_new.py --instruments csi300

# All three markets
python alpha158_new.py --run_all

# Custom data path
python alpha158_new.py --run_all --data_path /path/to/qlib_data
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--instruments` | `csi300` | Market universe: `csi300`, `csi800`, or `all` |
| `--run_all` | `False` | Run all three markets sequentially |
| `--data_path` | `~/.qlib/qlib_data/cn_data_2024h1` | Qlib data directory |
| `--output_dir` | `./baselines_output/alpha158_baseline` | Output directory |

## Output

For each market, the script produces:
- `alpha158_{instrument}.json` — Per-seed IC/RankIC/ICIR metrics and summary statistics.
- `alpha158_{instrument}_search.json` — Full hyperparameter search log with rankings.

## Dependencies

```
qlib
scikit-learn
numpy
pandas
scipy
```
