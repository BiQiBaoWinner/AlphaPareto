import gymnasium as gym
import sys
sys.modules['gym'] = gym

from typing import Optional, TypeVar, Callable, Optional, Tuple
import os
import pickle
import warnings
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy

from alphagen.data.expression import *
from alphagen.data.parser import parse_expression
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.utils import load_alpha_pool_by_path


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


@dataclass
class BacktestResult(DataClassJsonMixin):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 50,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: Union[pd.Series, pd.DataFrame],
        output_prefix: Optional[str] = None
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = TopkDropoutStrategy(
                    signal=prediction,
                    topk=self._top_k,
                    n_drop=self._n_drop,
                    only_tradable=True,
                    forbid_all_trade_at_limit=True
                )
                executor = exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result.to_json())
        return report, result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )


if __name__ == "__main__":
    
    import argparse
    
    # initialize_qlib("~/.qlib/qlib_data/cn_data")
    initialize_qlib("~/.qlib/qlib_data/cn_data_rolling")
    qlib_backtest = QlibBacktest(top_k=50, n_drop=5)
    
    parser = argparse.ArgumentParser(description='Different instruments')
    parser.add_argument('--pool-capacity', type=int, default=10, help='Maximum size of the alpha pool')
    parser.add_argument('--instruments', type=str, default='csi300', help='Stock subset name')
    parser.add_argument('--which-project', type=str, default='newdata_rolling_newrange2', help='Which project to run backtest for')
    args = parser.parse_args()
    instruments = args.instruments
    max_p = args.pool_capacity
    project = args.which_project
    
    data = StockData(
        instrument=instruments,
        start_time="2024-07-01",
        end_time="2025-06-30"
    )
    calc = QLibStockDataCalculator(data, None)

    def run_backtest(prefix: str, seed: List, pool_size: int, inst: str, exprs: List[List[Expression]], weights: List[List[float]]):
        df_list = []
        for seed, exprs, weights in zip(seed, exprs, weights):
            if project == "newdata_rolling_newrange2" and seed == 256:
                continue
            if project == "newdata_rolling_newrange2_llm_deepseek-llm-7b-chat" and seed == 128:
                continue
            df = data.make_dataframe(calc.make_ensemble_alpha(exprs, weights))
            df_list.append(df)
        df = pd.concat(df_list, axis=1).mean(axis=1)
        qlib_backtest.run(df, output_prefix=f"/Your/Path/To/out/backtests/{project}/{inst}/{pool_size}/{seeds_list}")

    # seeds_list = []
    # exprs_list = []
    # weights_list = []
    # for p in Path(f"out/gp_newrange2/{instruments}/{max_p}").iterdir():
    #     seed = int(p.name)
    #     with open(p / "40.json") as f:
    #         report = json.load(f)
    #     state = report["res"]["res"]["pool_state"]
    #     seeds_list.append(seed)
    #     exprs_list.append([parse_expression(e) for e in state["exprs"]])
    #     weights_list.append(state["weights"])
    # run_backtest("gp", seeds_list, max_p, instruments, exprs_list, weights_list)
    # print('Finished(GP):', f'{instruments}_{max_p}_{seeds_list}')
    # # run_backtest("gp", seed, [parse_expression(e) for e in state["exprs"]], state["weights"])
    # exit(0)
    
    seeds_list = []
    exprs_list = []
    weights_list = []
    for p in Path(f"/Your/Path/To/out/results/{project}").iterdir():
        inst, size, seed, time, ver = p.name.split('_', 4)
        size, seed = int(size), int(seed)
        if inst != instruments or size != max_p:
            continue
        if project == 'newdata_rolling_newrange2' and seed == 256:
            continue
        if project == 'newdata_rolling_newrange2_llm_deepseek-llm-7b-chat' and seed == 128:
            continue
        exprs, weights = load_alpha_pool_by_path(str(p / "Stepover_test_results.json"))
        seeds_list.append(seed)
        exprs_list.append(exprs)
        weights_list.append(weights)
    run_backtest(ver, seeds_list, max_p, instruments, exprs_list, weights_list)
    print('Finished(RL):', f'{inst}_{max_p}_{seeds_list}')
    # for p in Path("out/llm-tests/interaction").iterdir():
    #     if not p.name.startswith("v1"):
    #         continue
    #     run = int(p.name[3])
    #     with open(p / "report.json") as f:
    #         report = json.load(f)
    #     state = report[-1]["pool_state"]
    #     run_backtest("pure_llm", run, [parse_expression(t[0]) for t in state], [t[1] for t in state])
