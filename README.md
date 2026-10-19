# AlphaPareto

**Formulaic Alpha Discovery with LLM-Guided Multi-Objective Reinforcement Learning**

Authors: Yingbo Zhao, Zeyu Yang, Zhoufan Zhu

This repository contains the code for reproducing the experiments in the our [Neurip 2026 paper](https://arxiv.org/abs/2609.34188). We evaluate four alpha factor discovery methods on Chinese A-share markets (CSI 300, CSI 800, and All) using the [Qlib](https://github.com/microsoft/qlib) framework.

## Repository Structure

```
.
├── Alpha158/              # Alpha158 + Ridge linear baseline
│   └── alpha158_new.py
├── MLP/                   # Alpha158 + MLP neural network baseline
│   └── mlp_158_new.py
├── AlphaGen/              # AlphaGen RL framework and GP/DSO/LLM baselines
├── AlphaForge/            # AlphaForge (AFF): factor mining and factor combination
├── AlphaPareto/           # Full AlphaPareto branch (LLM + Pareto/lambda)
├── AlphaPareto_wo_lambda/ # AlphaPareto ablation: remove Pareto/lambda module
├── AlphaPareto_wo_llm/    # AlphaPareto ablation: remove LLM module
├── AlphaAgent-Run/        # AlphaAgent: LLM-driven factor mining
│   ├── AlphaAgent/        #   Source code (modified from original repo)
│   ├── extract_results.py #   Factor extraction & IC evaluation
│   ├── patch_config.py    #   Experiment configuration injection
│   └── generate.py        #   Price-volume data generation
├── RD-Agent-Run/          # RD-Agent: Multi-agent R&D framework
    ├── rdagent/            #   Source code (modified from original repo)
    ├── extract_results.py  #   Factor extraction & IC evaluation
    ├── patch_config.py     #   Experiment configuration injection
    └── alphapareto_bridge.py  # Expression format bridge
└── scripts/                # Unified batch scripts for RL and backtesting
```

## Requirements

- Python 3.8+
- [Qlib](https://github.com/microsoft/qlib) (for data and backtesting)
- PyTorch (for MLP baseline)
- scikit-learn, numpy, pandas, scipy
- DeepSeek API key (for AlphaAgent and RD-Agent)
- **Ollama (for AlphaPareto Module A)**: If you use the LLM-guided branch, ensure a local Ollama instance is running. The default configuration uses `qwen-embed:4b` for text embeddings. You can modify its endpoint `_DEF_BASE` or model name `_DEF_MODEL` in `alphagen/rl/llm_features.py`.

```bash
pip install qlib torch scikit-learn numpy pandas scipy requests
```

## Data Preparation

All methods use Qlib's China A-share dataset. Download it once:

```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_2024h1 --region cn --interval 1d
```

For AlphaAgent and RD-Agent, price-volume HDF5 data is auto-generated on first run via `generate.py`.

## Experiment Configuration

### Time Splits

| Period | Start | End | Purpose |
|--------|-------|-----|---------|
| Data | 2012-07-01 | 2025-06-30 | Full data range for factor calculation |
| Train | 2013-07-01 | 2023-06-30 | Model training |
| Valid | 2023-07-01 | 2024-06-30 | Hyperparameter tuning / factor weighting |
| Test | 2024-07-01 | 2025-06-30 | Out-of-sample evaluation |

### Market Universes

| Instrument | Benchmark | Description |
|------------|-----------|-------------|
| csi300 | SH000300 | CSI 300 large-cap index |
| csi800 | SH000906 | CSI 800 broad-based index |
| all | SH000001 | All A-share stocks |


## Quick Start

### Alpha158 Baseline

```bash
cd Alpha158
python alpha158_new.py --instruments csi300
python alpha158_new.py --run_all  # Run all three markets
```

### MLP Baseline

```bash
cd MLP
python mlp_158_new.py --instruments csi300 --gpu 0
python mlp_158_new.py --run_all --gpu 0
```

### GP Baseline

```bash
cd AlphaGen
mkdir -p out/logs
CUDA_VISIBLE_DEVICES=1 python gp.py 2>&1 | tee out/logs/baseline_gp_output_newrange2_debug.log
```

### AlphaAgent

```bash
cd AlphaAgent-Run
cp AlphaAgent/.env.example AlphaAgent/.env  # Fill in your API key
pip install -e AlphaAgent/
python patch_config.py --instrument csi300 --seed 42
cd AlphaAgent && alphaagent mine --direction "Explore alpha factors based on price-volume patterns"
```

### RD-Agent

```bash
cd RD-Agent-Run
cp .env.example .env  # Fill in your API key
pip install -e .       # Install from rdagent/ package
python patch_config.py --instrument csi300 --seed 42
rdagent fin_factor
```

See each method's README for detailed instructions.

### AlphaForge (AFF)

```bash
cd AlphaForge
python train_AFF.py --instruments csi300 --train_end_year 2020 --seeds [0,42,128,256,512] --save_name exp --zoo_size 30
python combine_AFF_flow.py --instruments csi300 --train_end_year 2020 --seeds [0,42,128,256,512] --save_name exp --n_factors 10 --window inf --metrics_chunk_days 50 --metrics_device auto
```

### AlphaGen

```bash
cd AlphaGen
python -m scripts.rl --instruments csi300
```

### AlphaPareto

```bash
cd AlphaPareto
python scripts/rl.py --instruments csi300 --pool-capacity 20 --leading-days 20 --llm deepseek-r1:7b --c0 0.08
```

### AlphaPareto w/o Lambda

```bash
cd AlphaPareto_wo_lambda
python scripts/rl.py --instruments csi300 --pool-capacity 20 --leading-days 20 --llm deepseek-r1:7b
```

### AlphaPareto w/o LLM

```bash
cd AlphaPareto_wo_llm
python scripts/rl.py --instruments csi300 --pool-capacity 20 --leading-days 20 --c0 0.08
```

## Evaluation Metrics

- **IC**: Pearson correlation between predicted and actual returns (cross-sectional, averaged)

For the Pareto family, we additionally report the following objective-related metrics in the Upload branches:

- **RRE**: risk-regularization related objective term
- **PFS**: pool factor synergy objective term
- **DH**: diversity/entropy related objective term

These metrics are used in `AlphaPareto`, `AlphaPareto_wo_lambda`, and `AlphaPareto_wo_llm`, together with the fixed seed protocol and shared train/valid/test splits.

## Acknowledgements

This work builds on [Qlib](https://github.com/microsoft/qlib), [AlphaGen](https://github.com/ICT-FinD-Lab/alphagen), [AlphaForge](https://github.com/DulyHao/AlphaForge), [AlphaQCM](https://github.com/ZhuZhouFan/AlphaQCM), [AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent) and [RD-Agent](https://github.com/microsoft/RD-Agent). We thank the authors and maintainers of these projects for making their code available.

## Contact 

Please feel free to raise an issue in this GitHub repository if you have any questions or encounter any issues.

## Citation
Please cite our paper if you feel this code helps.

@inproceedings{zhao2026Alphapareto,
  title = {AlphaPareto: Formulaic Alpha Discovery with LLM-Guided Multi-Objective Reinforcement Learning},
  author = {Yingbo, Zhao and Zeyu, Yang and Zhu, Zhoufan},
  booktitle = {Proceedings of the 40th Conference on Neural Information Processing Systems},
  year = {2026},
}
