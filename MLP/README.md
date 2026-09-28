# MLP (Alpha158) Baseline

This method uses Qlib's Alpha158 feature set with a Multi-Layer Perceptron (MLP) neural network as the predictor.

## Method

1. **Features**: Same Alpha158 handler as the linear baseline (158 technical indicators, `Fillna` + `CSRankNorm`).
2. **Label**: 20-day forward return — `Ref($close, -20) / $close - 1`.
3. **Hyperparameter Search**: Randomized search (20 iterations) over:
   - Layer architectures: `(256,)`, `(128, 64)`, `(256, 128)`, `(512, 256)`, `(256, 128, 64)`, `(512, 256, 128)`
   - Learning rate: `1e-3`, `5e-4`, `1e-4`
   - Batch size: `2048`, `4096`
   - Weight decay: `0.0`, `1e-5`, `1e-4`

   Search phase trains for 6,000 max steps with early stopping; scored by negative MSE on the validation set.
4. **Final Training**: Best hyperparameters are trained for 20,000 max steps (with early stopping, patience=50) across 5 random seeds.

## Usage

```bash
# Single market
python mlp_158_new.py --instruments csi300 --gpu 0

# All three markets
python mlp_158_new.py --run_all --gpu 0

# CPU mode
python mlp_158_new.py --instruments csi300 --gpu -1
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--instruments` | `csi300` | Market universe: `csi300`, `csi800`, or `all` |
| `--run_all` | `False` | Run all three markets sequentially |
| `--data_path` | `~/.qlib/qlib_data/cn_data_2024h1` | Qlib data directory |
| `--output_dir` | `./baselines_output/mlp_alpha158_baseline` | Output directory |
| `--n_search_iter` | `20` | Number of random search iterations |
| `--gpu` | `3` | GPU device ID (`-1` for CPU) |

## Output

For each market, the script produces:
- `mlp_alpha158_{instrument}.json` — Per-seed IC/RankIC/ICIR metrics and summary statistics.
- `mlp_alpha158_{instrument}_search.json` — Top-20 hyperparameter search results.

## Dependencies

```
qlib
torch
scikit-learn
numpy
pandas
scipy
```
