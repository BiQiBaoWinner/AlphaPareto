# RD-Agent: Multi-Agent R&D Framework for Factor Mining

RD-Agent is a multi-agent autonomous research and development framework that uses LLMs to iteratively propose, implement, and evaluate alpha factors. The source code is modified from the [original repository](https://github.com/microsoft/RD-Agent) with custom experiment configurations and evaluation scripts for the AlphaPareto benchmark.

## Method Overview

RD-Agent follows a similar evolutionary loop to AlphaAgent but generates Python code (rather than symbolic expressions) for each factor:

1. **Propose** — LLM proposes a research hypothesis and factor idea.
2. **Code** — LLM generates a Python `factor.py` implementing the factor.
3. **Run** — The factor code is executed on price-volume data.
4. **Evaluate** — Factor performance is backtested via Qlib.
5. **Feedback** — Results are summarized and fed back for the next iteration.

After all iterations, `extract_results.py` collects factor code, translates expressions to a unified format via `alphapareto_bridge.py`, and computes IC metrics.

## Directory Structure

```
RD-Agent-Run/
├── rdagent/                   # Source code (pip install -e .)
│   ├── app/cli.py             # CLI: rdagent fin_factor / fin_model / ...
│   ├── components/            # Workflow, coder, knowledge management
│   ├── scenarios/qlib/        # Qlib integration (configs, proposals, backtest)
│   ├── oai/                   # LLM integration (DeepSeek)
│   └── core/                  # Evolving agent framework
├── configs/                   # Per-instrument experiment configs
│   ├── csi300/
│   ├── csi800/
│   └── all/
├── .env.example               # Environment template (fill in API key)
├── extract_results.py         # Factor extraction & IC evaluation
├── patch_config.py            # Config injection for each experiment
└── alphapareto_bridge.py      # Translate Python factors to symbolic expressions
```

## Setup

```bash
# 1. Install the package
cd RD-Agent-Run
pip install -e .

# 2. Configure API key
cp .env.example .env
# Edit .env and fill in your DeepSeek API key
```

### Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `CHAT_MODEL` | LLM model for proposals (`deepseek/deepseek-chat`) |
| `REASONING_MODEL` | Reasoning model (`deepseek/deepseek-reasoner`) |
| `CHAT_SEED` | Random seed (set automatically by `patch_config.py`) |
| `FACTOR_MINING_TIMEOUT` | Timeout in seconds per experiment |

## Running Experiments

### Step 1: Patch configuration

```bash
cd RD-Agent-Run
python patch_config.py --instrument csi300 --seed 42
```

This updates the market/benchmark settings in `rdagent/scenarios/qlib/experiment/factor_template/conf*.yaml` and sets `CHAT_SEED` in `.env`.

### Step 2: Run factor mining

```bash
rdagent fin_factor
```

The mining process runs iteratively, generating Python factor implementations. Logs are saved to `log/`.

### Step 3: Extract and evaluate factors

```bash
python extract_results.py \
    --log_dir ./log \
    --output ./results/csi300/seed_42/stepover_test_results.json
```

`extract_results.py` walks the log directory, finds `FactorTask` objects in pkl files, executes each factor's Python code to compute values, then calculates IC metrics. It also uses `alphapareto_bridge.py` to translate Python factor code into a unified symbolic expression format.

### Repeat for all experiments

Run the above 3 steps for each combination of instrument (`csi300`, `csi800`, `all`) and seed (`0, 42, 128, 256, 512`), totaling 15 experiments.

## Output Format

`extract_results.py` produces a JSON file with the same format as AlphaAgent:

```json
{
    "exprs": ["expr1", "expr2", ...],
    "ics_ret": [0.045, 0.032, ...],
    "weights": [0.35, 0.25, ...],
    "best_valid_ic": 0.076,
    "test_ic_ret": 0.088
}
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `patch_config.py` | Injects instrument/seed into RD-Agent configs |
| `extract_results.py` | Extracts factors from logs, computes IC |
| `alphapareto_bridge.py` | Translates Python factor code to symbolic expressions for unified evaluation |

## Dependencies

```
qlib
torch
typer
python-dotenv
pandas
numpy
scipy
```
