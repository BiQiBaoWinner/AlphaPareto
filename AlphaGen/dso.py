import numpy as np
from typing import Optional
import os
import json

from alphagen.data.expression import *
from alphagen_qlib.calculator import QLibStockDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *



funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(['open', 'close', 'high', 'low', 'volume', 'vwap']):
    funcs[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)


# instruments = 'csi300'
# import sys
# seed = int(sys.argv[1])
# reseed_everything(seed)

cache = {}
device = torch.device('cuda:0')
# data_train = StockData(instruments, '2011-07-01', '2021-06-30', device=device)
# data_valid = StockData(instruments, '2021-07-01', '2024-06-30', device=device)
# data_test = StockData(instruments, '2024-07-01', '2025-06-30', device=device)
# calculator_train = QLibStockDataCalculator(data_train, target)
# calculator_valid = QLibStockDataCalculator(data_valid, target)
# calculator_test = QLibStockDataCalculator(data_test, target)


def run_experiment(instruments: str, capacity: int, seed: int):
    global cache
    reseed_everything(seed)
    cache = {}

    data_train = StockData(instruments, '2011-07-01', '2021-06-30', device=device)
    data_valid = StockData(instruments, '2021-07-01', '2024-06-30', device=device)
    data_test = StockData(instruments, '2024-07-01', '2025-06-30', device=device)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = MseAlphaPool(capacity=capacity, calculator=calculator_train, ic_lower_bound=None)

    class Ev:
        def __init__(self, pool):
            self.cnt = 0
            self.pool: MseAlphaPool = pool
            self.results = {}
            self.cache = {}
            self.best_key: Optional[str] = None
            self.directory = os.path.join("out", "dso", instruments, str(capacity), str(seed))
            os.makedirs(self.directory, exist_ok=True)

        def alpha_ev_fn(self, key):
            expr = eval(key)
            try:
                ic_train = self.pool.try_new_expr(expr)
            except OutOfDataRangeError:
                ic_train = -1.
            finally:
                ic_valid = -1.
                try:
                    ic_valid = calculator_valid.calc_single_IC_ret(expr)
                except OutOfDataRangeError:
                    pass
                if np.isnan(ic_train):
                    ic_train = -1.
                if np.isnan(ic_valid):
                    ic_valid = -1.
                self.cache[key] = {'train': ic_train, 'valid': ic_valid}
                if self.best_key is None or ic_valid > self.cache[self.best_key]['valid']:
                    self.best_key = key
                self.cnt += 1
                if self.cnt % 100 == 0 and self.best_key:
                    best_expr = eval(self.best_key)
                    ic_test = calculator_test.calc_single_all_ret(best_expr)
                    self.results[self.cnt] = {
                        'best_key': self.best_key,
                        'ic_train': self.cache[self.best_key]['train'],
                        'ic_valid': self.cache[self.best_key]['valid'],
                        'ic_test': ic_test,
                    }
                    self.dump(str(self.cnt))
                    # print(self.cnt, self.results[self.cnt])
                return ic_train
            
        def dump(self, step: str):
            best_metrics = None
            test_metrics = None
            if self.best_key:
                best_metrics = self.cache.get(self.best_key)
                try:
                    expr = eval(self.best_key)
                    ic_test = calculator_test.calc_single_all_ret(expr)
                    test_metrics = {"ic_test": ic_test}
                except OutOfDataRangeError:
                    test_metrics = {"ic_test": -1.}
            payload = {
                "step": step,
                "pool_capacity": self.pool.capacity,
                "best_key": self.best_key,
                "best_metrics": best_metrics,
                "test_metrics": test_metrics,
                "history": self.results,
                "cache": self.cache,
            }
            with open(os.path.join(self.directory, f"{step}.json"), "w") as f:
                json.dump(payload, f, indent=4)

    ev = Ev(pool)
    X = np.array([['open_', 'close', 'high', 'low', 'volume', 'vwap']])
    y = np.array([[1]])
    functions.function_map = funcs
    config = dict(
        task=dict(
            task_type='regression',
            function_set=list(funcs.keys()),
            metric='alphagen',
            metric_params=[lambda key: ev.alpha_ev_fn(key)],
        ),
        training={'n_samples': 20000, 'batch_size': 128, 'epsilon': 0.05},
        prior={'length': {'min_': 2, 'max_': 20, 'on': True}}
    )
    model = DeepSymbolicRegressor(config=config)
    model.fit(X, y)
    ev.dump("final")


if __name__ == '__main__':
    configs = [
        ('csi300', 10),
        ('csi300', 20),
        ('csi800', 10),
        ('csi800', 20),
        ('all', 10),
        ('all', 20),
    ]
    seeds = (0, 42, 128, 256, 512, 1024, 2023, 2024, 2025, 2048)

    for instruments, capacity in configs:
        for seed in seeds:
            run_experiment(instruments, capacity, seed)
