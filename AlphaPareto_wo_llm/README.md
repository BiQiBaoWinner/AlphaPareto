# AlphaPareto w/o LLM

This branch is the ablation variant that removes Module A and keeps Module B.

## Branch Identity

- Module A (LLM): disabled
- Module B (Pareto/lambda): enabled
- `c0`: enabled and used by lambda optimization

## What Is Kept

- Multi-objective metric computation
- Pareto-style scalarization via lambda solving
- `c0 -> c_override -> solve_lambda` control path

## Repository Layout

- `alphagen/models/linear_alpha_pool.py`: objective computation and lambda integration
- `alphagen/rl/compute_metrics.py`: multi-objective metric vector
- `alphagen/rl/lambda_solver.py`: lambda optimization
- `alphagen/rl/env/`: no-LLM environment branch
- `scripts/rl.py`: main training entry

## Environment Setup

```bash
conda create -n alphapareto python=3.10 -y
conda activate alphapareto
pip install -r requirements.txt
```

## Data Preparation

```bash
python -m data_collection.fetch_baostock_data
python -m data_collection.qlib_dump_bin
```

Default runtime path is typically:

- `~/.qlib/qlib_data/cn_data_rolling`

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
  --c0 0.08
```

The script entry forces `llm_model = "NoneLLM"`, so LLM components are not used in effective runtime.

## Main Parameters

- `pool-capacity`: alpha pool capacity
- `instruments`: market universe
- `leading-days`: target horizon
- `c0`: first dimension of `c_override`
- `steps`: total timesteps (in `main`/`run_single_experiment`)

## c0 Flow

`c0` is passed along this path:

1. parser argument in `scripts/rl.py`
2. `run_single_experiment(c0=...)`
3. pool builder with `c_override = [c0, 0.95, 0.90, 0.6]`
4. objective update calls `solve_lambda(..., c_override=self.c_override)`

## Outputs

Typical outputs:

- checkpoints and pool snapshots under `./out/results/AlphaPareto_NoneLLM/...`
- tensorboard logs under `./out/tensorboard`
- summary JSON in each experiment directory

## Ablation Protocol

For comparison with full model and `w/o Lambda`, keep fixed:

- data splits
- seed list
- pool capacity
- total training steps
- leading-days

Compare at least:

- validation/test objective
- IC at best objective
- best-step performance stability across seeds

## Troubleshooting

```bash
CUDA_LAUNCH_BLOCKING=1 python -m scripts.rl --instruments csi300 --pool-capacity 20 --leading-days 20 --c0 0.08
```

```bash
nvidia-smi
dmesg -T | grep -Ei "NVRM|Xid"
```

## Acknowledgement

This branch is provided for controlled ablation studies on the contribution of LLM-assisted components.
