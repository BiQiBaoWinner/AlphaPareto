import torch 
from torch import nn
import pandas as pd
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# from alphagen.config import *
# from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.rl.env.core import AlphaEnvCore
import torch.nn.functional as F
from gan.dataset import Collector
from gan.network.generater import NetG_DCGAN
from gan.network.masker import NetM
from gan.network.predictor import NetP, train_regression_model,train_regression_model_with_weight
from alphagen.rl.env.wrapper import SIZE_ACTION,action2token

from alphagen_generic.features import open_
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *

from gan.utils.data import get_data_by_year

from alphagen.utils.correlation import batch_pearsonr,batch_spearmanr


def main(
    instruments: str = "csi300",
    freq: str = 'day',
    save_name: str = 'aff_baseline',
    window: int = float('inf'),
    device: str = 'cuda:0',
    result = [],
    pred_dfs = {},
    root_path = os.path.expanduser('~/out/results/AlphaForge/'),
):

    for n_factors in [10, 20]:
        for seed in [0, 42, 128, 256, 512]:
            cur_seed_valid_ic = []
            cur_seed_test_ic = []
            all_pred_df_list = []
            # for train_end in range(2020,2021):
            train_end = '2023-06-30'
            print(n_factors,seed,train_end)
            
            returned = get_data_by_year(
                train_start = '2013-07-01',train_end = '2023-06-30',
                valid_start = '2023-07-01', valid_end = '2024-06-30',
                test_start = '2024-07-01', test_end = '2025-06-30',
                instruments=instruments, target=target, freq=freq,
            )
            data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
            

            path = f'{root_path}/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final.pkl'
            tensor_save_path = f'{root_path}/{save_name}_{instruments}_{train_end}_{seed}/pred_{train_end}_{n_factors}_{window}_{seed}.pt'
            tensor_valid_save_path = f'{root_path}/{save_name}_{instruments}_{train_end}_{seed}/pred_valid_{train_end}_{n_factors}_{window}_{seed}.pt'
            if not os.path.exists(path) or not os.path.exists(tensor_save_path) or not os.path.exists(tensor_valid_save_path):
                print(f'{path} or {tensor_save_path} or {tensor_valid_save_path} does not exist, skip!')
                
                result.append(dict(
                    seed = seed,
                    num = n_factors,
                    valid_ic = None,
                    test_ic = None,
                ))
                
                continue

            pred_test = torch.load(tensor_save_path).to(device)
            pred_valid = torch.load(tensor_valid_save_path).to(device)
            tgt = target.evaluate(data_all)
            
            
            ones = torch.ones_like(tgt)
            ones = ones * torch.nan
            ones[-data_test.n_days:] = pred_test
            cur_df = data_all.make_dataframe(ones)
            all_pred_df_list.append(
                cur_df.unstack().iloc[-data_test.n_days:].stack(future_stack=True)
            )

            tgt_valid = tgt[-data_test.n_days - data_valid.n_days:-data_test.n_days].to(device)
            tgt_test = tgt[-data_test.n_days:].to(device)

            valid_len = min(pred_valid.shape[0], tgt_valid.shape[0])
            test_len = min(pred_test.shape[0], tgt_test.shape[0])
            pred_valid = pred_valid[-valid_len:]
            tgt_valid = tgt_valid[-valid_len:]
            pred_test = pred_test[-test_len:]
            tgt_test = tgt_test[-test_len:]

            valid_ic_s = torch.nan_to_num(batch_pearsonr(pred_valid, tgt_valid), nan=0)
            test_ic_s = torch.nan_to_num(batch_pearsonr(pred_test, tgt_test), nan=0)

            cur_seed_valid_ic.append(valid_ic_s)
            cur_seed_test_ic.append(test_ic_s)

            pred_dfs[f"{n_factors}_{seed}"] = pd.concat(all_pred_df_list,axis=0)
            valid_ic = torch.cat(cur_seed_valid_ic)
            test_ic = torch.cat(cur_seed_test_ic)

            valid_ic_mean = valid_ic.mean().item()
            test_ic_mean = test_ic.mean().item()
            tmp = dict(
                seed = seed,
                num = n_factors,
                valid_ic = valid_ic_mean,
                test_ic = test_ic_mean,
            )
            result.append(tmp)
    
    with open(f'{root_path}/{save_name}_{instruments}_results.json','w') as f:
        json.dump(result, f)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/Your/Path/To/out/results/AlphaForge/')
    parser.add_argument('--save_name', type=str, default='aff_baseline')
    parser.add_argument('--instruments', type=str, default='csi300')
    args = parser.parse_args()
    
    instruments = args.instruments
    root_path = os.path.expanduser(args.root_path)

    device = 'cuda:0'
    result = []
    pred_dfs = {}
    
    main(
        instruments=instruments,
        device=device,
        result=result,
        pred_dfs=pred_dfs,
        root_path=root_path,
    )
    
    print(f'{instruments} has been done!')