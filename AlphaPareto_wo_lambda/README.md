# AlphaPareto w/o Lambda

This branch is the ablation variant that keeps Module A and removes Module B.

## Branch Identity

- Module A (LLM): enabled
- Module B (Pareto/lambda): disabled
- `c0`: not used in this branch

## What Is Removed

- Pareto vector objective and lambda-solver optimization path
- `c0`-controlled threshold flow for multi-objective scalarization

Training follows IC-oriented scalar objective behavior.

This Upload copy is anonymized for review. Local paths and host-specific endpoints are configured via environment variables.

## Repository Layout

- `alphagen/`: core expression engine, alpha pool logic, RL utilities
- `alphagen_llm/`: LLM clients and prompts
- `alphagen_qlib/`: Qlib data adapters
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

LLM endpoints can be overridden with `OLLAMA_EMB_ENDPOINT` and `OLLAMA_GEN_ENDPOINT`.

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
  --llm deepseek-r1:7b
```

## Main Parameters

- `pool-capacity`: alpha pool capacity
- `instruments`: market universe
- `leading-days`: target horizon
- `llm`: LLM model name for LLM-aware extractor path
- `steps`: total timesteps (in `main`/`run_single_experiment`)

## Outputs

Typical outputs:

- checkpoints and pool snapshots under `./out/results/...`
- tensorboard logs under `./out/tensorboard`
- summary JSON in each experiment directory

## Ablation Protocol

For comparison with full model and `w/o LLM`, keep fixed:

- data splits
- seed list
- pool capacity
- total training steps
- leading-days

Compare at least:

- validation/test IC
- best-step performance stability across seeds

## Troubleshooting

```bash
CUDA_LAUNCH_BLOCKING=1 python -m scripts.rl --instruments csi300 --pool-capacity 20 --leading-days 20 --llm deepseek-llm-7b-chat
```

## Acknowledgement

This branch is provided for controlled ablation studies on the contribution of Pareto/lambda optimization.

## Anonymized Release Notes

- `ALPHAPARETO_WO_LAMBDA_QLIB_PATH`: Qlib data path (default `~/.qlib/qlib_data/cn_data_rolling`).
- `ALPHAPARETO_WO_LAMBDA_DATA_PATH`: Baostock download output path (default `./data`).
- `ALPHAPARETO_WO_LAMBDA_QLIB_EXPORT_PATH`: Qlib export path for generated data.
- `ALPHAPARETO_WO_LAMBDA_QLIB_BASE_PATH`: Base Qlib data path used by the exporter.
- `ALPHAPARETO_WO_LAMBDA_OUTPUT_DIR`: Output root (default `out`).
- `ALPHAPARETO_WO_LAMBDA_RESULTS_DIR`: Results root (default `$ALPHAPARETO_WO_LAMBDA_OUTPUT_DIR/results`).
- `ALPHAPARETO_WO_LAMBDA_BACKTEST_DIR`: Backtest output root (default `out/backtests`).
