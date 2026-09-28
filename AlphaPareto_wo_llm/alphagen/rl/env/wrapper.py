from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np
import json
from gymnasium.spaces import Dict as SpaceDict, Box

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore

SIZE_NULL = 1
SIZE_OP = len(OPERATORS) # from config.py
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP

MAX_POOL_STATE_BYTES = 8192

class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: AlphaEnvCore,
        subexprs: Optional[List[Expression]] = None
    ):
        super().__init__(env)
        self.subexprs = subexprs or []
        self.size_action = SIZE_ACTION + len(self.subexprs)
        self.action_space = gym.spaces.Discrete(self.size_action)
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=self.size_action + SIZE_NULL - 1,
        #     shape=(MAX_EXPR_LENGTH, ),
        #     dtype=np.uint8
        # )
        
        self.observation_space = SpaceDict({
            "original_obs": Box(
                low=0,
                high=self.size_action + SIZE_NULL - 1,
                shape=(MAX_EXPR_LENGTH,),
                dtype=np.uint8
            ),
            "pool_state": Box(
                low=0,
                high=255,
                shape=(MAX_POOL_STATE_BYTES,),
                dtype=np.uint8
            ),
        })

    # def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
    #     self.counter = 0
    #     self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
    #     self.env.reset()
    #     return self.state, {}

    # def step(self, action: int):
    #     _, reward, done, truncated, info = self.env.step(self.action(action))
    #     if not done:
    #         self.state[self.counter] = action
    #         self.counter += 1
    #     return self.state, self.reward(reward), done, truncated, info
    
    def _pool_state_array(self) -> np.ndarray:
        pool_dict = self.env.pool.to_json_dict()
        minimal = {
            "exprs": pool_dict.get("exprs", []),
            "weights": pool_dict.get("weights", []),
        }
        payload = json.dumps(minimal, ensure_ascii=False).encode("utf-8")
        payload = payload[:MAX_POOL_STATE_BYTES - 4]
        buf = np.zeros(MAX_POOL_STATE_BYTES, dtype=np.uint8)
        length_bytes = len(payload).to_bytes(4, byteorder="little")
        buf[:4] = np.frombuffer(length_bytes, dtype=np.uint8)
        buf[4: (4 + len(payload)) ] = np.frombuffer(payload, dtype=np.uint8)
        return buf
    
    def reset(self, **kwargs) -> Tuple[dict, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        return {"original_obs": self.state.copy(),
                "pool_state": self._pool_state_array()}, {}
    
    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return {"original_obs": self.state.copy(),
                "pool_state": self._pool_state_array()}, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0              # Operators
        for i in range(offset, offset + SIZE_OP):
            if valid['op'][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        if action < len(self.subexprs):
            return ExpressionToken(self.subexprs[action])
        action -= len(self.subexprs)
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        assert False

class AlphaEnvWrapper_wo_llm(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: AlphaEnvCore,
        subexprs: Optional[List[Expression]] = None
    ):
        super().__init__(env)
        self.subexprs = subexprs or []
        self.size_action = SIZE_ACTION + len(self.subexprs)
        self.action_space = gym.spaces.Discrete(self.size_action)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.size_action + SIZE_NULL - 1,
            shape=(MAX_EXPR_LENGTH, ),
            dtype=np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0              # Operators
        for i in range(offset, offset + SIZE_OP):
            if valid['op'][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        if action < len(self.subexprs):
            return ExpressionToken(self.subexprs[action])
        action -= len(self.subexprs)
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        assert False

def AlphaEnv(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, **kwargs):
    return AlphaEnvWrapper(AlphaEnvCore(pool=pool, **kwargs), subexprs=subexprs)

def AlphaEnv_wo_llm(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, **kwargs):
    return AlphaEnvWrapper_wo_llm(AlphaEnvCore(pool=pool, **kwargs), subexprs=subexprs)
