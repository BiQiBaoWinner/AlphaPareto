import json
import os
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet, Enhanced_LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        valid_calculators: List[QLibStockDataCalculator],
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.valid_calculators = valid_calculators
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n
        
        self.best_pool = None
        self.best_valid_obj = -np.inf
        self.best_model_step = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.chat_session is not None:
            self._try_use_llm()

        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/train_ic', self.pool.current_train_ic)
        self.logger.record('pool/train_obj', self.pool.current_train_obj)
        self.logger.record('pool/best_obj(Trainset-Updating)', self.pool.best_obj)
        self.logger.record('pool/ic_at_best_obj(Trainset-Updating)', self.pool.ic_at_best_obj)
        self.logger.record('pool/last_lambda', self.pool._pool_lambda.tolist() if self.pool._pool_lambda is not None else None)
        self.logger.record('pool/last_reward_vec', self.pool._pool_reward_vec.tolist() if self.pool._pool_reward_vec is not None else None)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        n_days = sum(calculator.data.n_days for calculator in self.valid_calculators)

        for i, valid_calculator in enumerate(self.valid_calculators, start=1):
            ic_valid = self.pool.test_ensemble(valid_calculator)
            self.logger.record(f'Valid/ic_', ic_valid)
            obj_valid = self.pool.test_ensemble_obj(valid_calculator)
            self.logger.record(f'Valid/obj_', obj_valid)
        
        ic_test = self.pool.test_ensemble(self.test_calculators[0])
        obj_test = self.pool.test_ensemble_obj(self.test_calculators[0])
        self.logger.record(f'Test/ic_', ic_test)
        self.logger.record(f'Test/obj_', obj_test)

        if obj_valid > self.best_valid_obj:
            self.best_valid_obj = obj_valid
            self.best_pool = self.pool.clone()
            self.best_model_step = self.num_timesteps
            
        best_pool_test_ic = self.best_pool.test_ensemble(self.test_calculators[0])
        best_pool_test_obj = self.best_pool.test_ensemble_obj(self.test_calculators[0])
            
        self.save_checkpoint(add_info = {
            'valid_ic': ic_valid,
            "valid_obj": obj_valid,
            'test_ic': ic_test,
            "test_obj": obj_test,
            "best_pool_test_ic": best_pool_test_ic,
            "best_pool_test_obj": best_pool_test_obj,
        })

    def save_checkpoint(self, add_info):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            raw_json = self.pool.to_json_dict()
            new_json = {**raw_json, **add_info}
            json.dump(new_json, f)

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["train_ic"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.ic_at_best_obj:.4f}, obj={self.pool.ic_at_best_obj:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    @property
    def pool(self) -> LinearAlphaPool:
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    leading_days: int = 20,
    llm_model: str = "deepseek-llm-7b-chat",
    c0: float = 0,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3
):
    reseed_everything(seed)
    initialize_qlib("~/.qlib/qlib_data/cn_data_rolling")

    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"AlphaPareto_{llm_model}/{instruments}_{pool_capacity}_{seed}_{leading_days}_{timestamp}_{tag}"
    save_path = os.path.join("./out/A100/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -leading_days) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )
    
    segments = [
        ('2013-07-01', '2023-06-30'),
        ('2023-07-01', '2024-06-30'), 
        ('2024-07-01', '2025-06-30')
    ]
    
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]
    train_calculator = calculators[0]
    valid_calculator = calculators[1]
    test_calculator = calculators[2]

    def build_pool(exprs: List[Expression], c0: float = 0.0) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=train_calculator,
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
            c_override = [c0, 0.95, 0.93, 0.80]
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([], c0=c0)
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=False,
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        valid_calculators=[valid_calculator], 
        test_calculators=[test_calculator],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=Enhanced_LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
                llm_model=llm_model,
                llm_endpoint=os.environ.get("DEEPSEEK_ENDPOINT"),
                llm_backend=os.environ.get("DEEPSEEK_BACKEND"),
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log="./out/A100/tensorboard",
        device=device,
        verbose=1,
    )

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
    )

    valid_best_obj = checkpoint_callback.best_valid_obj
    valid_best_pool = checkpoint_callback.best_pool
    valid_ic_ret = valid_best_pool.test_ensemble(valid_calculator)
    test_ic_ret = valid_best_pool.test_ensemble(test_calculator)
    test_obj = valid_best_pool.test_ensemble_obj(test_calculator)
    state = valid_best_pool.state
    exprs = [ str(state['exprs'][i]) for i in range(valid_best_pool.size) ]
    ics_ret = [ state['ics_ret'][i] for i in range(valid_best_pool.size) ]
    weights = [ state['weights'][i] for i in range(valid_best_pool.size) ]
    results_path = os.path.join(save_path, "Stepover_test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_model_step": checkpoint_callback.best_model_step,
            'exprs': exprs,
            'ics_ret': ics_ret,
            'weights': weights,
            'ic_at_best_obj': valid_best_pool.ic_at_best_obj,
            'valid_ic': valid_ic_ret,
            'test_ic': test_ic_ret,
            'best_valid_obj': valid_best_obj,
            'test_obj': test_obj
            }, f, indent=4)
    
def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: str = "csi300",
    leading_days: int = 20,
    llm_model: str = "deepseek-llm-7b-chat",
    c0: float = 0,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param leading_days: Leading days for target calculation
    :param llm_model: LLM model to use in Enhanced_LSTMSharedNet, pass "NoneLLM" to use LSTMSharedNet without LLM
    :param c0: The first dimension of thresholds in Pareto front selection
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds, )
    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=default_steps[int(pool_capacity)] if steps is None else int(steps),
            leading_days=leading_days,
            llm_model=llm_model,
            c0=c0,
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps
        )


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Different seeds&p')
    parser.add_argument('--pool-capacity', type=int, default=20, help='Maximum size of the alpha pool')
    parser.add_argument('--instruments', type=str, default='csi300', help='Stock subset name')
    parser.add_argument('--leading-days', type=int, default=20, help='Leading days for target calculation')
    parser.add_argument('--llm', type=str, default='deepseek-llm-7b-chat', help='LLM model to use in Enhanced_LSTMSharedNet')
    parser.add_argument('--c0', type=float, default=0.0, help='The first dimension of thresholds in Pareto front selection')
    args = parser.parse_args()
    p = args.pool_capacity
    instruments = args.instruments
    leading_days = args.leading_days
    llm_model = args.llm
    c0 = args.c0
    
    fire.Fire(main(
        random_seeds = (0, 42, 128, 256, 512),
        instruments = instruments,
        pool_capacity = p,
        leading_days = leading_days,
        llm_model = llm_model,
        c0 = c0
    ))
