"""AlphaAgent expression adapter.

This module converts AlphaAgent-style formula strings into Expression objects
that can be evaluated by the current AlphaPareto framework.

Design goals:
1) Cover operators observed from all JSON files in out/results/AlphaAgent.
2) Keep conversion deterministic and explicit.
3) Fail loudly with clear errors for unsupported syntax.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from alphagen.data.expression import (
    Abs,
    Add,
    BinaryOperator,
    CSRank,
    Constant,
    Corr,
    Cov,
    Delta,
    Div,
    EMA,
    Expression,
    Feature,
    Greater,
    Less,
    Log,
    Mad,
    Max,
    Mean,
    Med,
    Min,
    Mul,
    Operator,
    PairRollingOperator,
    Pow,
    Rank,
    Ref,
    RollingOperator,
    Sign,
    Std,
    Sub,
    Sum,
    UnaryOperator,
    Var,
    WMA,
)
from alphagen_qlib.stock_data import FeatureType, StockData
from alphagen.utils.maybe import Maybe, none, some


class AlphaAgentExprError(ValueError):
    """Raised when a formula cannot be converted."""


# -----------------------------
# Extra operators not provided by the base expression set
# -----------------------------


class CSZScore(UnaryOperator):
    """Cross-sectional z-score for each date across stocks."""

    def __init__(self, operand: Expression) -> None:
        self._operand = operand

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls):
        return cls

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        if len(args) != 1:
            return some(f"{cls.__name__} expects 1 operand, got {len(args)}")
        if not isinstance(args[0], Expression):
            return some(f"{cls.__name__} expects an expression operand")
        return none(str)

    def _apply(self, x: Tensor) -> Tensor:
        mean = torch.nanmean(x, dim=1, keepdim=True)
        std = torch.sqrt(torch.nanmean((x - mean) ** 2, dim=1, keepdim=True))
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return (x - mean) / std

    @property
    def operands(self):
        return (self._operand,)

    @property
    def is_featured(self):
        return self._operand.is_featured

    def __str__(self) -> str:
        return f"CSZScore({self._operand})"


class CSStd(UnaryOperator):
    """Cross-sectional standard deviation for each date across stocks."""

    def __init__(self, operand: Expression) -> None:
        self._operand = operand

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls):
        return cls

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        if len(args) != 1:
            return some(f"{cls.__name__} expects 1 operand, got {len(args)}")
        if not isinstance(args[0], Expression):
            return some(f"{cls.__name__} expects an expression operand")
        return none(str)

    def _apply(self, x: Tensor) -> Tensor:
        mean = torch.nanmean(x, dim=1, keepdim=True)
        std = torch.sqrt(torch.nanmean((x - mean) ** 2, dim=1, keepdim=True))
        return std.expand_as(x)

    @property
    def operands(self):
        return (self._operand,)

    @property
    def is_featured(self):
        return self._operand.is_featured

    def __str__(self) -> str:
        return f"CSStd({self._operand})"


class CompareBinary(BinaryOperator):
    """Binary comparison output as float mask (0 or 1)."""

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        raise NotImplementedError


class Gt(CompareBinary):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs > rhs).to(lhs.dtype)


class Ge(CompareBinary):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs >= rhs).to(lhs.dtype)


class Lt(CompareBinary):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs < rhs).to(lhs.dtype)


class Le(CompareBinary):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs <= rhs).to(lhs.dtype)


class LogicBinary(BinaryOperator):
    """Logical binary op over numeric masks."""

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        raise NotImplementedError


class And(LogicBinary):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return ((lhs > 0) & (rhs > 0)).to(lhs.dtype)


class Or(LogicBinary):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return ((lhs > 0) | (rhs > 0)).to(lhs.dtype)


class IfElse(Operator):
    """Ternary operator: cond ? true_expr : false_expr."""

    def __init__(self, cond: Expression, when_true: Expression, when_false: Expression) -> None:
        self._cond = cond
        self._true = when_true
        self._false = when_false

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls):
        return cls

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        if len(args) != 3:
            return some(f"{cls.__name__} expects 3 operands, got {len(args)}")
        if any(not isinstance(a, Expression) for a in args):
            return some(f"{cls.__name__} expects expression operands")
        return none(str)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        cond = self._cond.evaluate(data, period)
        tv = self._true.evaluate(data, period)
        fv = self._false.evaluate(data, period)
        return torch.where(cond > 0, tv, fv)

    @property
    def operands(self):
        return self._cond, self._true, self._false

    @property
    def is_featured(self):
        return self._cond.is_featured or self._true.is_featured or self._false.is_featured


class TsQuantile(RollingOperator):
    """Rolling quantile."""

    def __init__(self, operand: Expression, delta_time: int, quantile: float) -> None:
        super().__init__(operand, delta_time)
        self._q = float(quantile)

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        if len(args) != 3:
            return some(f"{cls.__name__} expects 3 operands")
        if not isinstance(args[0], Expression):
            return some(f"{cls.__name__} expects an expression as first operand")
        if not isinstance(args[1], (int, float)):
            return some(f"{cls.__name__} expects window as second operand")
        if not isinstance(args[2], (int, float)):
            return some(f"{cls.__name__} expects quantile as third operand")
        return none(str)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop)).unfold(0, self._delta_time, 1)
        return torch.quantile(values, q=self._q, dim=-1)

    def _apply(self, operand: Tensor) -> Tensor:
        # Kept only to satisfy RollingOperator abstract interface.
        raise NotImplementedError

    def __str__(self) -> str:
        return f"TsQuantile({self._operand},{self._delta_time},{self._q})"


class RegrBeta(PairRollingOperator):
    """Rolling regression beta of lhs on rhs."""

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        rhs_c = rhs - rhs.mean(dim=-1, keepdim=True)
        lhs_c = lhs - lhs.mean(dim=-1, keepdim=True)
        den = (rhs_c ** 2).sum(dim=-1)
        den = torch.where(den.abs() < 1e-8, torch.ones_like(den), den)
        num = (lhs_c * rhs_c).sum(dim=-1)
        return num / den


class RegrResi(PairRollingOperator):
    """Rolling regression residual at the latest point."""

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        rhs_mean = rhs.mean(dim=-1, keepdim=True)
        lhs_mean = lhs.mean(dim=-1, keepdim=True)
        rhs_c = rhs - rhs_mean
        lhs_c = lhs - lhs_mean
        den = (rhs_c ** 2).sum(dim=-1)
        den = torch.where(den.abs() < 1e-8, torch.ones_like(den), den)
        beta = (lhs_c * rhs_c).sum(dim=-1) / den
        alpha = lhs_mean.squeeze(-1) - beta * rhs_mean.squeeze(-1)
        y_hat_last = alpha + beta * rhs[:, :, -1]
        return lhs[:, :, -1] - y_hat_last


class RSI(RollingOperator):
    """Standard RSI over a rolling window."""

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        diffs = values[1:] - values[:-1]
        diffs = diffs.unfold(0, self._delta_time, 1)
        up = torch.relu(diffs).mean(dim=-1)
        down = torch.relu(-diffs).mean(dim=-1)
        rs = up / (down + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))

    def _apply(self, operand: Tensor) -> Tensor:
        raise NotImplementedError


class SMA(RollingOperator):
    """Approximate Chinese-style SMA with EMA-like smoothing using m/n."""

    def __init__(self, operand: Expression, delta_time: int, m: int) -> None:
        super().__init__(operand, delta_time)
        self._m = int(m)

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        if len(args) != 3:
            return some(f"{cls.__name__} expects 3 operands")
        if not isinstance(args[0], Expression):
            return some(f"{cls.__name__} expects an expression as first operand")
        if not isinstance(args[1], (int, float)):
            return some(f"{cls.__name__} expects window as second operand")
        if not isinstance(args[2], (int, float)):
            return some(f"{cls.__name__} expects m as third operand")
        return none(str)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop)).unfold(0, self._delta_time, 1)
        n = self._delta_time
        alpha = float(self._m) / float(max(n, 1))
        out = values[:, :, 0]
        for i in range(1, n):
            out = alpha * values[:, :, i] + (1.0 - alpha) * out
        return out

    def _apply(self, operand: Tensor) -> Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"SMA({self._operand},{self._delta_time},{self._m})"


# -----------------------------
# Build expression object from AlphaAgent DSL via safe eval
# -----------------------------


class _Node:
    def __init__(self, expr: Expression | float) -> None:
        self.expr = expr if isinstance(expr, Expression) else Constant(float(expr))

    @staticmethod
    def _coerce(other: "_Node | Expression | float | int") -> Expression:
        if isinstance(other, _Node):
            return other.expr
        if isinstance(other, Expression):
            return other
        return Constant(float(other))

    def __add__(self, other):
        return _Node(Add(self.expr, self._coerce(other)))

    def __radd__(self, other):
        return _Node(Add(self._coerce(other), self.expr))

    def __sub__(self, other):
        return _Node(Sub(self.expr, self._coerce(other)))

    def __rsub__(self, other):
        return _Node(Sub(self._coerce(other), self.expr))

    def __mul__(self, other):
        return _Node(Mul(self.expr, self._coerce(other)))

    def __rmul__(self, other):
        return _Node(Mul(self._coerce(other), self.expr))

    def __truediv__(self, other):
        return _Node(Div(self.expr, self._coerce(other)))

    def __rtruediv__(self, other):
        return _Node(Div(self._coerce(other), self.expr))

    def __pow__(self, other):
        return _Node(Pow(self.expr, self._coerce(other)))

    def __rpow__(self, other):
        return _Node(Pow(self._coerce(other), self.expr))

    def __neg__(self):
        return _Node(Sub(0.0, self.expr))

    def __abs__(self):
        return _Node(Abs(self.expr))

    def __gt__(self, other):
        return _Node(Gt(self.expr, self._coerce(other)))

    def __ge__(self, other):
        return _Node(Ge(self.expr, self._coerce(other)))

    def __lt__(self, other):
        return _Node(Lt(self.expr, self._coerce(other)))

    def __le__(self, other):
        return _Node(Le(self.expr, self._coerce(other)))

    def __and__(self, other):
        return _Node(And(self.expr, self._coerce(other)))

    def __rand__(self, other):
        return _Node(And(self._coerce(other), self.expr))

    def __or__(self, other):
        return _Node(Or(self.expr, self._coerce(other)))

    def __ror__(self, other):
        return _Node(Or(self._coerce(other), self.expr))


def _n(x) -> _Node:
    if isinstance(x, _Node):
        return x
    if isinstance(x, Expression):
        return _Node(x)
    return _Node(float(x))


def _ternary_to_if(expr: str) -> str:
    """Convert C-style ternary `a ? b : c` into `IF(a,b,c)` recursively."""

    def prev_non_space(s: str, i: int) -> int:
        while i >= 0 and s[i].isspace():
            i -= 1
        return i

    def find_matching_lpar(s: str, rpar_idx: int) -> int:
        bal = 1
        for i in range(rpar_idx - 1, -1, -1):
            if s[i] == ')':
                bal += 1
            elif s[i] == '(':
                bal -= 1
                if bal == 0:
                    return i
        raise AlphaAgentExprError(f"Unmatched parenthesis in ternary: {s}")

    def find_colon(s: str, qpos: int) -> int:
        depth = 0
        nest = 0
        for i in range(qpos + 1, len(s)):
            ch = s[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '?' and depth == 0:
                nest += 1
            elif ch == ':' and depth == 0:
                if nest == 0:
                    return i
                nest -= 1
        raise AlphaAgentExprError(f"Missing ':' in ternary: {s}")

    def find_false_end(s: str, colon_idx: int) -> int:
        depth = 0
        for i in range(colon_idx + 1, len(s)):
            ch = s[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth == 0:
                    return i
                depth -= 1
        return len(s)

    s = expr
    while '?' in s:
        qpos = s.find('?')
        rpar = prev_non_space(s, qpos - 1)
        if rpar < 0 or s[rpar] != ')':
            raise AlphaAgentExprError(f"Ternary condition must be parenthesized: {expr}")
        lpar = find_matching_lpar(s, rpar)
        colon = find_colon(s, qpos)
        false_end = find_false_end(s, colon)

        cond = s[lpar + 1:rpar].strip()
        tv = s[qpos + 1:colon].strip()
        fv = s[colon + 1:false_end].strip()
        replacement = f"IF({cond},{tv},{fv})"
        s = s[:lpar] + replacement + s[false_end:]
    return s


def _preprocess(expr: str) -> str:
    # Convert ternary and boolean operators into python-evaluable form.
    s = _ternary_to_if(expr)
    s = s.replace("&&", "&")
    s = s.replace("||", "|")

    # Map feature references from $close to v_close.
    s = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"v_\1", s)
    return s


def _make_env() -> Dict[str, object]:
    env: Dict[str, object] = {}

    # Features
    for f in FeatureType:
        env[f"v_{f.name.lower()}"] = _Node(Feature(f))
    # AlphaAgent uses $return; this framework does not expose it as a base feature.
    # Map it to 1-day close-to-close return.
    env["v_return"] = _Node(Div(Sub(_n(env["v_close"]).expr, Ref(_n(env["v_close"]).expr, 1)), Add(Ref(_n(env["v_close"]).expr, 1), 1e-8)))

    # Basic ops
    env["ABS"] = lambda x: _Node(Abs(_n(x).expr))
    env["LOG"] = lambda x: _Node(Log(_n(x).expr))
    env["SIGN"] = lambda x: _Node(Sign(_n(x).expr))
    env["POW"] = lambda x, y: _Node(Pow(_n(x).expr, _n(y).expr))
    env["MAX"] = lambda x, y: _Node(Greater(_n(x).expr, _n(y).expr))
    env["MIN"] = lambda x, y: _Node(Less(_n(x).expr, _n(y).expr))

    # Rolling aliases
    env["DELAY"] = lambda x, n: _Node(Ref(_n(x).expr, int(n)))
    env["DELTA"] = lambda x, n: _Node(Delta(_n(x).expr, int(n)))
    env["TS_MEAN"] = lambda x, n: _Node(Mean(_n(x).expr, int(n)))
    env["TS_SUM"] = lambda x, n: _Node(Sum(_n(x).expr, int(n)))
    env["TS_STD"] = lambda x, n: _Node(Std(_n(x).expr, int(n)))
    env["TS_VAR"] = lambda x, n: _Node(Var(_n(x).expr, int(n)))
    env["TS_MAX"] = lambda x, n: _Node(Max(_n(x).expr, int(n)))
    env["TS_MEDIAN"] = lambda x, n: _Node(Med(_n(x).expr, int(n)))
    env["TS_MAD"] = lambda x, n: _Node(Mad(_n(x).expr, int(n)))
    env["TS_RANK"] = lambda x, n: _Node(Rank(_n(x).expr, int(n)))
    env["TS_CORR"] = lambda x, y, n: _Node(Corr(_n(x).expr, _n(y).expr, int(n)))
    env["TS_COVARIANCE"] = lambda x, y, n: _Node(Cov(_n(x).expr, _n(y).expr, int(n)))
    env["TS_QUANTILE"] = lambda x, n, q: _Node(TsQuantile(_n(x).expr, int(n), float(q)))

    # Common tech indicators and aliases
    env["EMA"] = lambda x, n: _Node(EMA(_n(x).expr, int(n)))
    env["WMA"] = lambda x, n: _Node(WMA(_n(x).expr, int(n)))
    env["DECAYLINEAR"] = lambda x, n: _Node(WMA(_n(x).expr, int(n)))
    env["SMA"] = lambda x, n, m: _Node(SMA(_n(x).expr, int(n), int(m)))
    env["RSI"] = lambda x, n: _Node(RSI(_n(x).expr, int(n)))
    env["MACD"] = lambda x, f, s: _Node(Sub(EMA(_n(x).expr, int(f)), EMA(_n(x).expr, int(s))))

    # Cross-sectional operators
    env["RANK"] = lambda x: _Node(CSRank(_n(x).expr))
    env["ZSCORE"] = lambda x: _Node(CSZScore(_n(x).expr))
    env["STD"] = lambda x: _Node(CSStd(_n(x).expr))

    # TS z-score and pct-change
    env["TS_ZSCORE"] = lambda x, n: _Node(
        Div(
            Sub(_n(x).expr, Mean(_n(x).expr, int(n))),
            Add(Std(_n(x).expr, int(n)), 1e-8),
        )
    )
    env["TS_PCTCHANGE"] = lambda x, n: _Node(
        Div(
            Sub(_n(x).expr, Ref(_n(x).expr, int(n))),
            Add(Ref(_n(x).expr, int(n)), 1e-8),
        )
    )

    # Bollinger aliases
    env["BB_MIDDLE"] = lambda x, n: _Node(Mean(_n(x).expr, int(n)))
    env["BB_UPPER"] = lambda x, n: _Node(
        Add(Mean(_n(x).expr, int(n)), Mul(2.0, Std(_n(x).expr, int(n))))
    )
    env["BB_LOWER"] = lambda x, n: _Node(
        Sub(Mean(_n(x).expr, int(n)), Mul(2.0, Std(_n(x).expr, int(n))))
    )

    # Conditional and condition-based rolling aliases
    env["IF"] = lambda cond, tv, fv: _Node(IfElse(_n(cond).expr, _n(tv).expr, _n(fv).expr))
    env["COUNT"] = lambda cond, n: _Node(Sum(_n(cond).expr, int(n)))
    env["SUMIF"] = lambda x, n, cond: _Node(Sum(Mul(_n(x).expr, _n(cond).expr), int(n)))

    # Regression aliases
    env["REGBETA"] = lambda x, y, n: _Node(RegrBeta(_n(x).expr, _n(y).expr, int(n)))
    env["REGRESI"] = lambda x, y, n: _Node(RegrResi(_n(x).expr, _n(y).expr, int(n)))

    # SEQUENCE is mostly used as REGBETA's second argument in current results.
    # We map it to a smoothed ramp proxy from Ref over close to stay featured.
    env["SEQUENCE"] = lambda n: _Node(Mean(_n(env["v_close"]).expr, int(n)))

    # SUMAC currently behaves as a rolling sum alias in current pipeline.
    env["SUMAC"] = lambda x, n: _Node(Sum(_n(x).expr, int(n)))

    return env


def convert_alphaagent_expr(expr: str) -> Expression:
    """Convert one AlphaAgent expression string into framework Expression.

    Raises:
        AlphaAgentExprError: if conversion or parsing fails.
    """
    env = _make_env()
    text = _preprocess(expr)
    try:
        value = eval(text, {"__builtins__": {}}, env)
    except Exception as ex:
        raise AlphaAgentExprError(f"Failed to convert expression: {expr}\nReason: {ex}") from ex

    if isinstance(value, _Node):
        return value.expr
    if isinstance(value, (int, float)):
        return Constant(float(value))
    if isinstance(value, Expression):
        return value
    raise AlphaAgentExprError(f"Converted result is not an Expression: {type(value).__name__}")


def convert_alphaagent_pool(exprs: List[str], weights: List[float]) -> Tuple[List[Expression], List[float]]:
    """Convert and filter a pool of AlphaAgent expressions.

    - Expressions failing conversion are skipped.
    - Weights are re-normalized by L1 norm after filtering.
    """
    converted_exprs: List[Expression] = []
    converted_weights: List[float] = []

    for e, w in zip(exprs, weights):
        try:
            converted_exprs.append(convert_alphaagent_expr(e))
            converted_weights.append(float(w))
        except AlphaAgentExprError as ex:
            print(f"[AlphaAgent adapter] Skip invalid expr: {e} | {ex}")

    if not converted_weights:
        return [], []

    den = sum(abs(w) for w in converted_weights)
    if den < 1e-12:
        den = 1.0
    converted_weights = [w / den for w in converted_weights]
    return converted_exprs, converted_weights
