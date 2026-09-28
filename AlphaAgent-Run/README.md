# AlphaAgent: LLM-Driven Alpha Factor Mining

AlphaAgent is an LLM-driven alpha factor discovery system that iteratively proposes, constructs, evaluates, and refines formulaic alpha factors. The source code is modified from the [original repository](https://github.com/microsoft/RD-Agent) with custom experiment configurations and evaluation scripts for the AlphaPareto benchmark.

## Method Overview

AlphaAgent runs an evolutionary loop (5 iterations by default):

1. **Propose** — LLM generates a market hypothesis based on prior feedback.
2. **Construct** — LLM translates the hypothesis into formulaic factor expressions.
3. **Calculate** — Factor expressions are evaluated on price-volume data via `expr_parser` + `function_lib`.
4. **Backtest** — Factors are backtested using Qlib's LGBModel + TopkDropout strategy.
5. **Feedback** — Backtest results are fed back to the LLM for the next iteration.

After all iterations, `extract_results.py` collects discovered factor expressions and computes IC metrics.

## Directory Structure

```
AlphaAgent-Run/
├── AlphaAgent/                # Source code (pip install -e .)
│   ├── .env.example           # Environment template (fill in API key)
│   ├── alphaagent/            # Core package
│   │   ├── app/cli.py         # CLI: alphaagent mine / backtest
│   │   ├── components/        # Workflow, coder, knowledge management
│   │   ├── scenarios/qlib/    # Qlib integration (configs, proposals, backtest)
│   │   ├── oai/               # LLM integration (DeepSeek)
│   │   └── core/              # Evolving agent framework
│   └── configs/               # Per-instrument experiment configs
│       ├── csi300/            #   conf.yaml, conf_cn_combined_kdd_ver.yaml, generate.py
│       ├── csi800/
│       └── all/
├── extract_results.py         # Factor extraction & IC evaluation
├── patch_config.py            # Config injection for each experiment
└── generate.py                # Price-volume HDF5 data generation
```

## Setup

```bash
# 1. Install the package
cd AlphaAgent-Run/AlphaAgent
pip install -e .

# 2. Configure API key
cp .env.example .env
# Edit .env and fill in your DeepSeek API key
```

### Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | DeepSeek API key |
| `CHAT_MODEL` | LLM model for factor proposal (`deepseek-chat`) |
| `REASONING_MODEL` | Reasoning model (`deepseek-reasoner`) |
| `CHAT_SEED` | Random seed (set automatically by `patch_config.py`) |
| `FACTOR_MINING_TIMEOUT` | Timeout in seconds per experiment |

## Running Experiments

### Step 1: Patch configuration

```bash
cd AlphaAgent-Run
python patch_config.py --instrument csi300 --seed 42
```

This copies the instrument-specific YAML configs and `generate.py` into the AlphaAgent package directory, and sets `CHAT_SEED` in `.env`.

### Step 2: Run factor mining

```bash
cd AlphaAgent
alphaagent mine --direction "Explore alpha factors based on price-volume patterns and market microstructure"
```

The mining process runs 5 iterations, generating ~10 factor expressions per round. Logs are saved to `log/` or `log_AlphaAgent/`.

### Step 3: Extract and evaluate factors

```bash
cd AlphaAgent-Run
python extract_results.py \
    --log_dir ./AlphaAgent/log \
    --output ./results/csi300/seed_42/stepover_test_results.json
```

### Repeat for all experiments

Run the above 3 steps for each combination of instrument (`csi300`, `csi800`, `all`) and seed (`0, 42, 128, 256, 512`), totaling 15 experiments.

## Output Format

`extract_results.py` produces a JSON file:

```json
{
    "exprs": ["expr1", "expr2", ...],
    "ics_ret": [0.045, 0.032, ...],
    "weights": [0.35, 0.25, ...],
    "best_valid_ic": 0.076,
    "test_ic_ret": 0.088
}
```

- `exprs` — Discovered factor expressions.
- `ics_ret` — Per-factor test IC values.
- `weights` — IC-weighted combination weights (based on validation IC).
- `test_ic_ret` — Combined portfolio test IC.

## Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/*/conf.yaml` | Single-factor Qlib backtest config (LGBModel, TopkDropout) |
| `configs/*/conf_cn_combined_kdd_ver.yaml` | Combined-factor backtest config |
| `configs/*/generate.py` | Price-volume data generation for the instrument |
| `AlphaAgent/alphaagent/app/qlib_rd_loop/conf.py` | Iteration count (`evolving_n=5`) and component wiring |

## Dependencies

```
qlib
torch
fire
python-dotenv
pandas
numpy
scipy
```
