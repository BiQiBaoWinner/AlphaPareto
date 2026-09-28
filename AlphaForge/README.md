# AlphaForge(AFF)


### Data Preparation
Similar to [AlphaGen](https://github.com/RL-MLDM/alphagen), we use [Qlib](https://github.com/microsoft/qlib#data-preparation) as the data storage tool and download data from the free and open-source data source [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5).

Please install Qlib [Qlib](https://github.com/microsoft/qlib) first

Then download stock data through running `data_collection/fetch_baostock_data.py`

Then modify the corresponding `/path/for/qlib_data` in `gan.utils.data.py` to the data you downloaded (the default setting is `~/.qlib/qlib_data/cn_data_rolling`)


### Run Our Model

#### stage1: Mining alpha factors
```shell
python train_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0] --save_name=test --zoo_size=30
```

#### stage2: Combining alpha factors
```shell
python combine_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0] --save_name=test --n_factors=10 --window=inf
```
Alternatively, for the optimized combining flow:
```shell
python combine_AFF_flow.py --instruments=csi300 --train_end_year=2020 --seeds=[0] --save_name=test --n_factors=10 --window=inf --metrics_chunk_days=50 --metrics_device=auto
```

### Automated Training Scripts
We provide batch processing scripts in the `scripts/` directory for convenience:
- `scripts/baseline_aff_inst.sh`: Main entry point to run the AlphaForge training and combination pipeline. By default, it uses `baseline_aff_common_flow.sh` to optimize memory usage.
- `scripts/baseline_aff_evalres.sh`: Script to evaluate and summarize the results after training.

**Note on Memory Optimization:**
We provide two versions of the combination scripts:
1. **Standard Version** (`combine_AFF.py` / `baseline_aff_common.sh`): Recommended for general use if GPU memory is sufficient.
2. **Memory-Optimized Flow Version** (`combine_AFF_flow.py` / `baseline_aff_common_flow.sh`): If you encounter **Out-of-Memory (OOM)** issues during the factor combination stage, please use this version. it processes metrics in chunks to significantly reduce peak memory consumption.

#### stage3: Show the results

You could run the ipython notebook file 

```shell
exp_AFF_calc_result.ipynb
```

to generate and concat experiment result.


### Run baseline experiments

The experiment process of other models is similar to running our AFF model, Except that none of the other models have a combine step.

#### GP:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### RL:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### DSO:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### ML models including XGBoost, LightGBM and MLP:

train & show results: `exp_ML_train_and_result.ipynb`



