from typing import Type
from alphagen.data.expression import *


MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = [
    Abs,  # Sign,
    Log,
    Add, Sub, Mul, Div, Greater, Less,
    Ref, Mean, Sum, Std, Var,  # Skew, Kurt,
    Max, Min,
    Med, Mad,  # Rank,
    Delta, WMA, EMA,
    Cov, Corr
]

DELTA_TIMES = [1, 5, 10, 20, 40]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
