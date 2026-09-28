import gymnasium as gym
import sys
sys.modules['gym'] = gym  # alias gymnasium as gym for legacy imports

from typing import Optional, TypeVar, Callable, Optional, Tuple
import os
import glob
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
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.strategy.order_generator import OrderGenWInteract

from alphagen.data.expression import *
from alphagen.data.parser import parse_expression
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.utils import load_alpha_pool_by_path
from alphaagent_expr_adapter import convert_alphaagent_pool


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


class MonthEndTopkDropoutStrategy(TopkDropoutStrategy):
    def _is_month_end_step(self, trade_step: int) -> bool:
        trade_len = self.trade_calendar.get_trade_len()
        if trade_step >= trade_len - 1:
            return True
        cur_start, cur_end = self.trade_calendar.get_step_time(trade_step)
        next_start, _ = self.trade_calendar.get_step_time(trade_step + 1)
        return pd.Timestamp(cur_end).month != pd.Timestamp(next_start).month

    def _should_trade_on_step(self, trade_step: int) -> bool:
        # Execute on the first trading day after month-end signal.
        if trade_step <= 0:
            return False
        return self._is_month_end_step(trade_step - 1)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        if not self._should_trade_on_step(trade_step):
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)

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
        
        if isinstance(prediction, pd.DataFrame):
            prediction = prediction.iloc[:, 0]
            
        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = MonthEndTopkDropoutStrategy(
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
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"], freq='day')["risk"]
        returns = risk_analysis(report["return"] - report["cost"], freq='day')["risk"]

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
    parser = argparse.ArgumentParser(description='Different instruments')
    # parser.add_argument('--candidate-pool-capacity', type=int, nargs='+', default=10, help='candidate size of the alpha pool')
    parser.add_argument('--instruments', type=str, default='csi300', help='Stock subset name')
    parser.add_argument('--which-project', type=str, default='newdata_rolling_newrange2', help='Which project to run backtest for')
    parser.add_argument('--benchmark', type=str, default='SH000300', help='Benchmark for backtest')
    parser.add_argument('--topk', type=int, default=50, help='Top k stocks to select in backtest strategy')
    parser.add_argument('--dropN', type=int, default=5, help='Number of stocks to drop in backtest strategy')
    args = parser.parse_args()
    instruments = args.instruments
    # c_p = args.candidate_pool_capacity
    project = args.which_project
    bm = args.benchmark
    
    initialize_qlib("~/.qlib/qlib_data/cn_data_rolling")
    qlib_backtest = QlibBacktest(top_k=args.topk, benchmark=bm, n_drop=args.dropN)
    
    data = StockData(
        instrument=instruments,
        start_time="2024-07-01",
        end_time="2025-06-30"
    )
    calc = QLibStockDataCalculator(data, None)
    
    with open("/Your/Path/To/out/results/FJSON.json") as f:
        FJSON = json.load(f)

    def run_backtest(project: str, seed: List, pool_size: int, inst: str, exprs: List[List[Expression]], weights: List[List[float]], benchmark: str = "SH000300"):
        df_list = []
        for s, exprs, weights in zip(seed, exprs, weights):
            
            df = data.make_dataframe(calc.make_ensemble_alpha(exprs, weights))
            df_list.append(df)
            
        df = pd.concat(df_list, axis=1).mean(axis=1)
        if '/' in project:
            project = project.replace('/', '')
        qlib_backtest.run(df, output_prefix=f"/Your/Path/To/out/backtests{args.topk}_drop{args.dropN}/{inst}/{project}/{seed}_{pool_size}")

    def run_backtest_alphaagent(project: str, seed: List, pool_size: int, inst: str, exprs: List[List[Expression]], weights: List[List[float]], benchmark: str = "SH000300"):
        # AlphaAgent factors can contain nested long windows; use a larger
        # history window here so this special case does not affect other projects.
        aa_data = StockData(
            instrument=inst,
            start_time="2024-07-01",
            end_time="2025-06-30",
            max_backtrack_days=220
        )
        aa_calc = QLibStockDataCalculator(aa_data, None)

        df_list = []
        for s, exprs, weights in zip(seed, exprs, weights):
            valid_exprs = []
            valid_weights = []
            for e, w in zip(exprs, weights):
                try:
                    aa_calc.evaluate_alpha(e)
                    valid_exprs.append(e)
                    valid_weights.append(w)
                except OutOfDataRangeError:
                    print(f"Skip out-of-range expr for seed {s}: {e}")

            if len(valid_exprs) == 0:
                print(f"Skip seed {s}: no valid AlphaAgent expr after range filtering")
                continue

            den = sum(abs(w) for w in valid_weights)
            if den < 1e-12:
                print(f"Skip seed {s}: zero l1 weight after range filtering")
                continue
            valid_weights = [w / den for w in valid_weights]

            df = aa_data.make_dataframe(aa_calc.make_ensemble_alpha(valid_exprs, valid_weights))
            df_list.append(df)

        if len(df_list) == 0:
            print(f"No valid AlphaAgent predictions for {inst}, skip backtest")
            return

        df = pd.concat(df_list, axis=1).mean(axis=1)
        qlib_backtest.run(df, output_prefix=f"/Your/Path/To/out/backtests{args.topk}_drop{args.dropN}/{inst}/{project}/{seed}_{pool_size}")

    def run_backtest_ml(project: str, seed: List, inst: str, meta_alphas: List[pd.DataFrame], benchmark: str = "SH000300"):

        df = pd.concat(meta_alphas, axis=1).mean(axis=1)
        qlib_backtest.run(df, output_prefix=f"/Your/Path/To/out/backtests{args.topk}_drop{args.dropN}/{inst}/{project}/{seed}_ML")

    if project.startswith('GP') or project.startswith('gp'):
        seeds_list = []
        exprs_list = []
        weights_list = []
        opt_p_list = []
        p_seed_dict = FJSON['GP'][instruments]
        
        for p_size in ['10', '20']:
            if p_size not in p_seed_dict.keys():
                continue
            seeds = p_seed_dict[p_size]
            
            if len(seeds) == 0:
                print(f"No seeds for GP with pool size {p_size} on {instruments}, skipping")
                continue
            
            for s in seeds:
                p = os.path.join(f"/Your/Path/To/out/gp_newrange2/{instruments}", p_size + '/' + s)

                with open(os.path.join(p, "40.json")) as f:
                    report = json.load(f)
                    
                state = report["res"]["res"]["pool_state"]
                seeds_list.append(int(s))
                
                exprs_list_p = []
                weights_list_p = []
                for e, w in zip(state["exprs"], state["weights"]):
                    try:
                        exprs_list_p.append(parse_expression(e))
                        weights_list_p.append(w)
                    except Exception as ex:
                        print(f"Error parsing expression {e} for seed {s} with pool size {p_size}: {ex}")
                        continue
                den = sum(abs(w) for w in weights_list_p)
                weights_list_p = [w / den for w in weights_list_p]
                exprs_list.append(exprs_list_p)
                weights_list.append(weights_list_p)
                opt_p_list.append(int(p_size))
                
        run_backtest(project, seeds_list, opt_p_list, instruments, exprs_list, weights_list, bm)
        print('Finished(GP):', f'GP_{instruments}_{opt_p_list}_{seeds_list}')
    
    if project in ['Alpha158', 'LGBM', 'XGB', 'MLP']:
        
        seeds_list = []
        meta_alphas = []
        p_seed_dict = FJSON[project][instruments]
        
        p_size = 'N/A'
        seeds = p_seed_dict[p_size] if p_size in p_seed_dict.keys() else []
        
        if len(seeds) == 0:
            raise ValueError(f"No seeds for {project} on {instruments}")
        
        for s in seeds:
            
            if not os.path.exists(os.path.join(f"/Your/Path/To/out/results/baselines_lgbm_xgb_mlp/{project.lower()}_baseline/", f"{project.lower()}_{instruments}_seed{s}_pred.pkl")):
                print(f"Meta-alpha file not found for seed {s} at expected path, skipping: /Your/Path/To/out/results/baselines_lgbm_xgb_mlp/{project.lower()}_baseline/{project.lower()}_{instruments}_seed{s}_pred.pkl")
                if not os.path.exists(os.path.join(f"/Your/Path/To/out/results/baselines_lgbm_xgb_mlp/{project.lower()}_alpha158_baseline/", f"{project.lower()}_alpha158_{instruments}_seed{s}_pred.pkl")):
                    print(f"Meta-alpha file not found for seed {s} at expected path, skipping: /Your/Path/To/out/results/baselines_lgbm_xgb_mlp/{project.lower()}_alpha158_baseline/{project.lower()}_alpha158_{instruments}_seed{s}_pred.pkl")
                    continue
            
            if project == 'Alpha158':
                p = os.path.join(f"/Your/Path/To/out/results/baselines_lgbm_xgb_mlp/{project.lower()}_baseline/", f"{project.lower()}_{instruments}_seed{s}_pred.pkl")
            else:
                p = os.path.join(f"/Your/Path/To/out/results/baselines_lgbm_xgb_mlp/{project.lower()}_alpha158_baseline/", f"{project.lower()}_alpha158_{instruments}_seed{s}_pred.pkl")
                
            with open(p, 'rb') as f:
                meta_alpha = pickle.load(f)
            meta_alphas.append(meta_alpha)
            seeds_list.append(int(s))
            print(f"Loaded meta-alpha for seed {s} from {p}")
            
        if len(seeds_list) > 0:
            run_backtest_ml(project.lower(), seeds_list, instruments, meta_alphas, bm)
            print('Finished(ML):', f'{project}_{instruments}_{seeds_list}')
    
    if project.startswith('AlphaPareto') or project.startswith('AlphaGen'):
        seeds_list = []
        exprs_list = []
        weights_list = []
        opt_p_list = []
        
        if project == 'AlphaPareto':
            
            model = 'AlphaPareto'
            
            for p_size in ['10', '20']:
                seeds = [str(s) for s in FJSON[model][instruments][p_size]]
                s_submodel_dict = FJSON[model][instruments][f"{p_size}:seed:M"]
                for s in seeds:
                    sub_model = s_submodel_dict[str(s)]
                    
                    llm = sub_model.split(')')[0].split('(')[1]
                    normal_sub_model = f"AlphaPareto_deepseek-r1:{llm[2:]}" if 'ds' in llm else f"AlphaPareto_qwen-embed:{llm[2:]}"
                    normal_sub_model_path = f"/Your/Path/To/out/results/{normal_sub_model}"
                    normal_sub_model_path = list(Path(normal_sub_model_path).glob(f"{instruments}_{p_size}_{s}_*"))[0]
                    
                    exprs, weights = load_alpha_pool_by_path(os.path.join(normal_sub_model_path, "Stepover_test_results_BaseIC.json"))
                    seeds_list.append((int(s)))
                    exprs_list.append(exprs)
                    weights_list.append(weights)
                    opt_p_list.append(int(p_size))
                
        elif project == 'AlphaPareto(w/o lambda)':
            
            model = 'AlphaPareto(w/o lambda)'
            
            for p_size in ['10', '20']:
                seeds = [str(s) for s in FJSON[model][instruments][p_size]]
                s_submodel_dict = FJSON[model][instruments][f"{p_size}:seed:M"]
                for s in seeds:
                    sub_model = s_submodel_dict[str(s)]
                    
                    llm = sub_model.split('(')[1].split(' ')[0]
                    normal_sub_model = f"AlphaPareto_wo_lambda_deepseek-r1:{llm[2:]}" if 'ds' in llm else f"AlphaPareto_wo_lambda_qwen-embed:{llm[2:]}"
                    normal_sub_model_path = f"/Your/Path/To/out/results/{normal_sub_model}"
                    normal_sub_model_path = list(Path(normal_sub_model_path).glob(f"{instruments}_{p_size}_{s}_*"))[0]
                    
                    exprs, weights = load_alpha_pool_by_path(os.path.join(normal_sub_model_path, "Stepover_test_results.json"))
                    seeds_list.append(int(s))
                    exprs_list.append(exprs)
                    weights_list.append(weights)
                    opt_p_list.append(int(p_size))
        
        else:
            for p in Path(f"/Your/Path/To/out/results/{project}").iterdir():
                inst, size, seed, horizon, ver = p.name.split('_', 4)
                
                if inst != instruments:
                    continue
                
                size, seed = str(size), str(seed)
                
                if project.split('_')[0] == 'AlphaGen':
                    model = "AlphaGen"
                    
                elif 'wo' in project.split('_'):
                    if '_' in project and project.split('_')[-1] == 'llm':
                        model = 'AlphaPareto(w/o LLM)'
                    elif ':' in project and project.split(':')[-1] == '4b':
                        model = f"{project.split('_')[0]}(qw{project.split(':')[-1]} w/o lambda)"
                    elif ':' in project and project.split(':')[-1] != '4b':
                        model = f"{project.split('_')[0]}(ds{project.split(':')[-1]} w/o lambda)"

                else:
                    if project.split(':')[-1] == '4b':
                        model = f"{project.split('_')[0]}(qw{project.split(':')[-1]})"
                    else:
                        model = f"{project.split('_')[0]}(ds{project.split(':')[-1]})"
                
                target_seeds = [str(s) for s in FJSON[model][inst][size]]
                if seed not in target_seeds:
                    continue
                
                print(f"======Processing {model} with {inst} and pool size {size}, seed {seed}======")
                if model in ['AlphaGen', 'AlphaPareto(ds1.5b w/o lambda)', 'AlphaPareto(qw4b w/o lambda)', 'AlphaPareto(ds7b w/o lambda)', 'AlphaPareto(ds14b w/o lambda)', 'AlphaPareto(ds32b w/o lambda)']:
                    exprs, weights = load_alpha_pool_by_path(str(p / "Stepover_test_results.json"))
                elif model in ['AlphaPareto(w/o LLM)', 'AlphaPareto(ds1.5b)', 'AlphaPareto(qw4b)', 'AlphaPareto(ds7b)', 'AlphaPareto(ds14b)', 'AlphaPareto(ds32b)']:
                    exprs, weights = load_alpha_pool_by_path(str(p / "Stepover_test_results_BaseIC.json"))
                seeds_list.append(int(seed))
                exprs_list.append(exprs)
                weights_list.append(weights)
                opt_p_list.append(int(size))
        
        if len(seeds_list) > 0:
            run_backtest(project, seeds_list, opt_p_list, instruments, exprs_list, weights_list, bm)

    if project.startswith('AlphaForge'):
        seeds_list = []
        meta_alphas = []
        for p in Path(f"/Your/Path/To/out/results/{project}").iterdir():
            
            if str(p).endswith('.json'):
                continue
            _, _, inst, _, seed = p.name.split('_', 4)
            seed = str(seed)
            
            if inst != instruments:
                continue
            
            for p_size in ['10', '20']:
                seeds = [str(s) for s in FJSON[project][inst][p_size]]
                if seed not in seeds:
                    continue
                
                pkl_p = os.path.join(p, f"pred_2023-06-30_{p_size}_inf_{seed}_df.pkl")
                with open(pkl_p, 'rb') as f:
                    meta_alpha = pickle.load(f)
                meta_alphas.append(meta_alpha)
                seeds_list.append(int(seed))
                print(f"Loaded meta-alpha for seed {seed} from {p}")
            
        if len(seeds_list) > 0:
            run_backtest_ml(project, seeds_list, instruments, meta_alphas, bm)
            print('Finished(ML):', f'{project}_{instruments}_{seeds_list}')
                
    if project.startswith('AlphaAgent') or project.startswith('RDAgent'):
        seeds_list = []
        exprs_list = []
        weights_list = []
        opt_p_list = []
        for p in Path(f"/Your/Path/To/out/results/{project}").iterdir():
            inst = p.name

            if inst != instruments:
                continue
            
            for p2 in Path(f'/Your/Path/To/out/results/{project}/{p.name}').iterdir():
                _, _, _, _, seedNum = p2.name.split('_', 4)
                seed = str(seedNum[4:-5])
                p_size = "10"
                model = project
                
                seeds = [str(s) for s in FJSON[model][inst][p_size]]
                if seed not in seeds:
                    continue
                
                print(f"======Processing {model} with {inst} and pool size {p_size}, seed {seed}======")
                with open(str(p2)) as f:
                    report = json.load(f)
                exprs_raw = report['exprs']
                weights_raw = report['weights']
                
                valid_exprs = []
                valid_weights = []
                for e, w in zip(exprs_raw, weights_raw):
                    try:
                        valid_exprs.append(parse_expression(e))
                        valid_weights.append(w)
                    except Exception as ex:
                        # print(f"Error parsing expression {e} for seed {seed} and instrument {inst}: {ex}")
                        continue
                if not valid_exprs:
                    raise ValueError(f"No valid expressions found for seed {seed} and instrument {inst}. Check the JSON file for issues.")
                
                den = sum(abs(w) for w in valid_weights)
                valid_weights = [w / den for w in valid_weights]
                exprs, weights = valid_exprs, valid_weights
                
                if len(exprs) == 0:
                    print(f"Skip seed {seed}: no valid AlphaAgent expr after conversion")
                    continue
                seeds_list.append(int(seed))
                exprs_list.append(exprs)
                weights_list.append(weights)
                opt_p_list.append(int(p_size))
        
        if len(seeds_list) > 0:
            run_backtest_alphaagent(project, seeds_list, opt_p_list, instruments, exprs_list, weights_list, bm)

    if project.startswith('AlphaQCM'):
        seeds_list = []
        exprs_list = []
        weights_list = []
        opt_p_list = []
        for p in Path(f"/Your/Path/To/out/results/{project}").iterdir():
            if p.suffix == '.csv':
                continue
            inst, seed = p.name.split('_', 4)
            seed = str(seed[4:-5])
            
            if inst != instruments:
                continue
            
            model = 'AlphaQCM'
            p_seed_dict = FJSON[project][instruments]
            p_size = 'N/A'
            seeds = p_seed_dict[p_size] if p_size in p_seed_dict.keys() else []
            
            if len(seeds) == 0:
                raise ValueError(f"No seeds for {project} on {instruments}")
            
            if seed not in seeds:
                continue
            
            print(f"======Processing {model} with {inst} and pool size {p_size}, seed {seed}======")
            exprs, weights = load_alpha_pool_by_path(str(p))
            seeds_list.append(int(seed))
            exprs_list.append(exprs)
            weights_list.append(weights)
            opt_p_list.append(len(exprs))
        
        if len(seeds_list) > 0:
            run_backtest(project, seeds_list, opt_p_list, instruments, exprs_list, weights_list, bm)
            print('Finished(AlphaQCM):', f'AlphaQCM_{inst}_{opt_p_list}_{seeds_list}')