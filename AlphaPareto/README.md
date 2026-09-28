# AlphaPareto

AlphaPareto is a reinforcement-learning framework for formulaic alpha mining, built on top of AlphaGen with two core modules:

- Module A: LLM-assisted parsing/scoring and LLM-aware policy features.
- Module B: Pareto-style multi-objective optimization with lambda solving.

This is the full model branch (A + B).

## Branch Identity

- Module A (LLM): enabled
- Module B (Pareto/lambda): enabled
- Main control hyperparameter for B: `c0` (first dimension of `c_override`)

## Repository Layout

- `alphagen/`: core expression engine, alpha pool logic, RL utilities
- `alphagen/models/linear_alpha_pool.py`: objective computation and lambda integration
- `alphagen/rl/compute_metrics.py`: multi-objective metric vector
- `alphagen/rl/lambda_solver.py`: lambda optimization
- `alphagen_llm/`: LLM clients and prompts
- `alphagen_qlib/`: Qlib data adapters
- `scripts/rl.py`: main training entry

## Environment Setup

Recommended:

- Linux
- Python 3.10
- CUDA-enabled PyTorch (optional but recommended)

```bash
conda create -n alphapareto python=3.10 -y
conda activate alphapareto
pip install -r requirements.txt
```

## Data Preparation

Prepare Qlib metadata, then collect and dump local market data:

```bash
python -m data_collection.fetch_baostock_data
python -m data_collection.qlib_dump_bin
```

Default runtime path in training scripts is typically:

- `~/.qlib/qlib_data/cn_data_rolling`

If your local path differs, update the initialization path in the script.

## Quick Start

You can start the training using the centralized script located in the root `scripts/` directory:

```bash
# From the Upload/ directory
bash scripts/rl_all.sh
```

Alternatively, run a single experiment manually:

```bash
python scripts/rl.py \
  --instruments csi300 \
  --pool-capacity 20 \
  --leading-days 20 \
  --llm deepseek-r1:7b \
  --c0 0.08
```

## Main Parameters

- `pool-capacity`: alpha pool capacity
- `instruments`: market universe
- `leading-days`: target horizon
- `llm`: LLM model name
- `c0`: first dimension of Pareto threshold/preference vector
- `steps`: total timesteps (in `main`/`run_single_experiment`)

## c0 Flow

`c0` is passed along this path:

1. parser argument in `scripts/rl.py`
2. `run_single_experiment(c0=...)`
3. pool builder with `c_override = [c0, 0.95, 0.90, 0.6]`
4. objective update calls `solve_lambda(..., c_override=self.c_override)`

## Outputs

Typical outputs:

- checkpoints and pool snapshots under `./out/results/...`
- tensorboard logs under `./out/tensorboard`
- summary JSON in each experiment directory

## Ablation Protocol

For comparison with `w/o Lambda` and `w/o LLM`, keep fixed:

- data splits
- seed list
- pool capacity
- total training steps
- leading-days

Compare at least:

- validation/test IC
- validation/test objective
- best-step performance stability across seeds

## Troubleshooting

```bash
CUDA_LAUNCH_BLOCKING=1 python -m scripts.rl --instruments csi300 --pool-capacity 20 --leading-days 20 --llm deepseek-llm-7b-chat --c0 0.08
```

```bash
nvidia-smi
dmesg -T | grep -Ei "NVRM|Xid"
```

## Acknowledgement

This project extends the AlphaGen line of work for practical and ablation-oriented alpha mining experiments.
