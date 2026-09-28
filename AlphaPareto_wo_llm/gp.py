import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


funcs = [make_function(**func._asdict()) for func in generic_funcs]

# instruments = 'csi300'
# seed = 2
# reseed_everything(seed)

device = torch.device("cuda:0")
initialize_qlib("~/.qlib/qlib_data/cn_data_rolling")

cache = {}
calculator_train = None
calculator_valid = None
calculator_test = None
Metric = None
current_capacity = 0
current_seed = 0
current_instruments = ''
generation = 0


def _metric(x, y, w):
    key = y[0]

    if key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            return cached['train']
        cache[key] = {'train': cached, 'valid': -1}
        return cached
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.

    expr = eval(key)
    try:
        ic_train = calculator_train.calc_single_IC_ret(expr)
    except OutOfDataRangeError:
        ic_train = -1.
    if np.isnan(ic_train):
        ic_train = -1.
        
    try:
        ic_valid = calculator_valid.calc_single_IC_ret(expr)
    except OutOfDataRangeError:
        ic_valid = -1.
    if np.isnan(ic_valid):
        ic_valid = -1.
        
        
    cache[key] = {'train': ic_train, 'valid': ic_valid}
    return ic_train


def configure_experiment(instruments: str, capacity: int, seed: int):
    global cache, calculator_train, calculator_valid, calculator_test
    global Metric, current_capacity, current_seed, current_instruments, generation

    reseed_everything(seed)
    cache = {}
    current_capacity = capacity
    current_seed = seed
    current_instruments = instruments
    generation = 0

    data_train = StockData(instruments, "2013-07-01", "2023-06-30", device=device)
    data_valid = StockData(instruments, "2023-07-01", "2024-06-30", device=device)
    data_test = StockData(instruments, "2024-07-01", "2025-06-30", device=device)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)
    Metric = make_fitness(function=_metric, greater_is_better=True)


def try_single():
    if not cache:
        return {}
    # top_key = Counter(cache).most_common(1)[0][0]
    top_key, metrics = max(cache.items(), key=lambda item: item[1]['valid'])
    
    expr = eval(top_key)
    ic_test = calculator_test.calc_single_all_ret(expr)
    return {
        'ic_train': metrics["train"],
        'ic_valid': metrics["valid"],
        'ic_test': ic_test,
    }


def try_pool(capacity: int, mutual_ic_thres: Optional[float] = None):
    pool = MseAlphaPool(
        capacity=capacity,
        calculator=calculator_train,
        ic_lower_bound=None
    )
    exprs = []

    def acceptable(expr: str) -> bool:
        if mutual_ic_thres is None:
            return True
        return all(abs(pool.calculator.calc_mutual_IC(e, eval(expr))) <= mutual_ic_thres
                   for e in exprs)

    # most_common = dict(Counter(cache).most_common(capacity if mutual_ic_thres is None else None))
    ranked = sorted(cache.items(), key=lambda item: item[1]['valid'], reverse=True)
    # for key in most_common:
    for key, metrics in ranked:
        if acceptable(key):
            exprs.append(eval(key))
            if len(exprs) >= capacity:
                break
    pool.force_load_exprs(exprs)

    ic_train = pool.test_ensemble(calculator_train)
    ic_valid = pool.test_ensemble(calculator_valid)
    ic_test = pool.test_ensemble(calculator_test)
    return {
        "ic_train": ic_train,
        "ic_valid": ic_valid,
        "ic_test": ic_test,
        "pool_state": pool.to_json_dict()
    }


def ev():
    global generation
    generation += 1
    directory = f"out/gp_newrange2/{current_instruments}/{current_capacity}/{current_seed}"
    os.makedirs(directory, exist_ok=True)
    if generation % 4 != 0:
        return
    res = {"pool": current_capacity, "res": try_pool(current_capacity, mutual_ic_thres=0.7)}
    with open(f'{directory}/{generation}.json', 'w') as f:
        json.dump({'res': res, 'cache': cache}, f, indent=4)


def run_experiment(instruments: str, capacity: int, seed: int):
    configure_experiment(instruments, capacity, seed)

    features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    terminals = features + constants

    X_train = np.array([terminals])
    y_train = np.array([[1]])

    est_gp = SymbolicRegressor(
        population_size=1000,
        generations=40,
        init_depth=(2, 6),
        tournament_size=600,
        stopping_criteria=1.,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.,
        random_state=seed,
        function_set=funcs,
        metric=Metric,  # type: ignore
        const_range=None,
        n_jobs=1
    )
    est_gp.fit(X_train, y_train, callback=ev)
    print(est_gp._program.execute(X_train))


if __name__ == '__main__':
    
    # seeds = (0, 42, 128, 256, 512, 1024, 2023, 2024, 2025, 2048)
    seeds = (0, 42, 128, 256, 512)
    
    configs = [
        ('csi300', 10),
        ('csi300', 20),
        ('csi800', 10),
        ('csi800', 20),
        ('all', 10),
        ('all', 20),
    ]
    
    for instruments, capacity in configs:
        for seed in seeds:
            run_experiment(instruments, capacity, seed)
