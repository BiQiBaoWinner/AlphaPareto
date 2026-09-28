"""Utility functions for evaluating alpha expressions with multi-dimensional metrics.

This module allows callers to evaluate one or multiple quantitative factors (expressed in
qlib-compatible string syntax) directly against a ``TensorAlphaCalculator`` instance and obtain
the Pareto dimensions used by the current reward vector: PPS(IC), RRE, PFS, and DH.

The main entry point ``compute_metrics`` only requires the factor strings and their
associated weights. All dataset-specific details (target, time range, instruments, etc.) are
encapsulated inside the provided ``TensorAlphaCalculator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
torch.set_num_threads(1)

from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression, Operator
from alphagen.data.parser import parse_expression
from alphagen.utils.correlation import batch_pearsonr


@dataclass(frozen=True)
class MetricResult:
    """Container for the evaluation dimensions."""

    pps: float
    rre: float
    pfs: float
    dh: float
    reward_vec: np.ndarray

    @property
    def vector(self) -> np.ndarray:
        """Return the metrics stacked in the canonical order as a NumPy array."""

        return self.reward_vec

    def to_dict(self) -> Dict[str, float]:
        """Expose the metrics as a plain dictionary."""

        return {
            "pps": float(self.pps),
            "rre": float(self.rre),
            "pfs": float(self.pfs),
            "dh": float(self.dh),
        }


def compute_batch_metrics(
    factor_formulas: Sequence[str],
    weights: Sequence[float],
    calculator: TensorAlphaCalculator,
    *,
    window_size: Optional[int] = 252,
    step_size: int = 20,
    random_state: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
) -> MetricResult:
    """
    Return PPS, RRE, PFS, DH for the given formulas and weights.

    Metrics (rolling windows):
    - PPS (IC): Mean over time of cross-sectional Pearson correlation between factor and target (higher is better).
    - RRE: Stability of rank distributions across days via KL; mapped to 1/(1+KL) and averaged (higher is more stable).
    - PFS: Robustness under Gaussian/t noise on factors; mean row correlation vs perturbed values (higher is more robust).
    - DH: Normalized spectral entropy of the factor covariance (diversity; higher is more diverse).
    """

    if len(factor_formulas) == 0:
        raise ValueError("At least one factor expression must be provided.")
    if len(factor_formulas) != len(weights):
        raise ValueError("The number of formulas and weights must match.")

    exprs = [parse_expression(expr) for expr in factor_formulas]
    weight_arr = np.asarray(weights, dtype=float)
    if weight_arr.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array-like.")
    if np.allclose(weight_arr, 0.0):
        weight_arr = np.full_like(weight_arr, 1.0 / len(weight_arr))

    target_device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    with torch.no_grad():
        ensemble_tensor = calculator.make_ensemble_alpha(exprs, weight_arr.tolist())
        target_tensor = calculator.target
        ensemble_tensor = ensemble_tensor.to(target_device, non_blocking=True)
        target_tensor = target_tensor.to(target_device, non_blocking=True)
        device = ensemble_tensor.device
        torch_rng = torch.Generator(device=device)
        if random_state is not None:
            torch_rng.manual_seed(int(random_state.integers(0, 2**31 - 1)))
            
        list_tensor = calculator.make_concatenated_alpha(exprs)
        concatenated_tensor = torch.stack(list_tensor) # (num_factors, days, stocks)
        concatenated_tensor = concatenated_tensor.to(target_device, non_blocking=True)

    mask = torch.isfinite(target_tensor)
    cnt = mask.sum()
    if cnt == 0:
        inferred_noise_var = 0.0
    else:
        mean = (target_tensor * mask).sum() / cnt
        var = ((target_tensor - mean) ** 2 * mask).sum() / cnt
        inferred_noise_var = float(var.item())

    if window_size is None or window_size <= 0:
        raise NotImplementedError("Window size must be a positive integer.")

    tot_days = ensemble_tensor.shape[0]
    reward_vecs: List[List[float]] = []
    for start in range(max(0, tot_days - window_size), -1, -step_size):
        end = min(start + window_size, tot_days)
        if end - start >= 5:
            end = start + window_size
            sub_tensor = ensemble_tensor[start:end]   # (T, N) on device
            sub_target = target_tensor[start:end]
            
            sub_dh_tensor = concatenated_tensor[:, start:end] # (num_factors, T, N)
            sub_dh_tensor = sub_dh_tensor.permute(1, 2, 0).reshape(-1, concatenated_tensor.shape[0]) # (T*N, num_factors)

            ic_t = batch_pearsonr(sub_tensor, sub_target)  # (T, N) on device
            sub_ic = _pps_ic(ic_t)

            sub_rre = _rre_stability(sub_tensor)
            sub_pfs = _pfs_robustness(sub_tensor, inferred_noise_var, torch_rng)
            sub_dh = _dh_entropy(sub_dh_tensor)

            reward_vecs.append([sub_ic, sub_rre, sub_pfs, sub_dh])

    reward_vec = torch.tensor(reward_vecs, device=device, dtype=torch.float32)
    return MetricResult(
        reward_vec=reward_vec.cpu().numpy(),
        pps=float(torch.nanmean(reward_vec[:, 0]).item()),
        rre=float(torch.nanmean(reward_vec[:, 1]).item()),
        pfs=float(torch.nanmean(reward_vec[:, 2]).item()),
        dh=float(torch.nanmean(reward_vec[:, 3]).item()),
    )

def compute_testset_metrics(
    factor_formulas: Sequence[str],
    weights: Sequence[float],
    calculator: TensorAlphaCalculator,
    *,
    random_state: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
) -> MetricResult:
    """
    Return PPS, RRE, PFS, DH for the given formulas and weights.

    Full test-set evaluation using all available data (no rolling window). Use for final evaluation, not training feedback.
    """

    if len(factor_formulas) == 0:
        raise ValueError("At least one factor expression must be provided.")
    if len(factor_formulas) != len(weights):
        raise ValueError("The number of formulas and weights must match.")

    exprs = [parse_expression(expr) for expr in factor_formulas]
    weight_arr = np.asarray(weights, dtype=float)
    if weight_arr.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array-like.")
    if np.allclose(weight_arr, 0.0):
        weight_arr = np.full_like(weight_arr, 1.0 / len(weight_arr))

    target_device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    with torch.no_grad():
        ensemble_tensor = calculator.make_ensemble_alpha(exprs, weight_arr.tolist())
        target_tensor = calculator.target
        ensemble_tensor = ensemble_tensor.to(target_device, non_blocking=True)
        target_tensor = target_tensor.to(target_device, non_blocking=True)
        device = ensemble_tensor.device
        torch_rng = torch.Generator(device=device)
        if random_state is not None:
            torch_rng.manual_seed(int(random_state.integers(0, 2**31 - 1)))
            
        list_tensor = calculator.make_concatenated_alpha(exprs)
        concatenated_tensor = torch.stack(list_tensor) # (num_factors, days, stocks)
        concatenated_tensor = concatenated_tensor.to(target_device, non_blocking=True)

    mask = torch.isfinite(target_tensor)
    cnt = mask.sum()
    if cnt == 0:
        inferred_noise_var = 0.0
    else:
        mean = (target_tensor * mask).sum() / cnt
        var = ((target_tensor - mean) ** 2 * mask).sum() / cnt
        inferred_noise_var = float(var.item())
        
    reward_vecs: List[List[float]] = []
    ic_t = batch_pearsonr(ensemble_tensor, target_tensor)
    ic_ = _pps_ic(ic_t)
    rre_ = _rre_stability(ensemble_tensor)
    pfs_ = _pfs_robustness(ensemble_tensor, inferred_noise_var, torch_rng)
    
    concatenated_tensor = concatenated_tensor.permute(1, 2, 0).reshape(-1, concatenated_tensor.shape[0]) # (T*N, num_factors)
    dh_ = _dh_entropy(concatenated_tensor)

    reward_vecs.append([ic_, rre_, pfs_, dh_])
    reward_vec = torch.tensor(reward_vecs, device=device, dtype=torch.float32)
    
    return MetricResult(
        reward_vec=reward_vec.cpu().numpy(),
        pps=float(torch.nanmean(reward_vec[:, 0]).item()),
        rre=float(torch.nanmean(reward_vec[:, 1]).item()),
        pfs=float(torch.nanmean(reward_vec[:, 2]).item()),
        dh=float(torch.nanmean(reward_vec[:, 3]).item()),
    )

def _pps_ic(ic_tensor: torch.Tensor) -> float:
    """Predictive power score: average IC after omitting NaNs (torch)."""
    finite = ic_tensor[torch.isfinite(ic_tensor)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())


def _rre_stability(panel: torch.Tensor) -> float:
    """Relative ranking efficiency via KL of rank distributions (torch)."""
    # panel: (T, N)
    eps = 1e-8
    mask = torch.isfinite(panel)
    x = torch.where(mask, panel, torch.full_like(panel, -1e30))
    order = torch.argsort(x, dim=1)
    ranks = torch.argsort(order, dim=1).float() + 1.0
    ranks = ranks * mask.float()

    denom = ranks.sum(dim=1, keepdim=True).clamp_min(eps)
    probs = torch.where(mask, ranks / denom, torch.zeros_like(ranks))
    probs_prev = torch.roll(probs, shifts=1, dims=0)
    probs_prev[0] = probs[0]

    kl = (probs * (probs.clamp_min(eps).log() - probs_prev.clamp_min(eps).log())).sum(dim=1)
    rre = torch.nanmean(1.0 / (1.0 + kl))
    return float(rre.item())


def _pfs_robustness(
    panel: torch.Tensor,
    noise_var: float,
    rng: Optional[torch.Generator],
) -> float:
    """Perturbation feature stability with noisy replicas (torch)."""
    eps = 1e-8
    noise_dof = 3
    sigma = math.sqrt(max(noise_var, 1e-3))

    def _row_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x) & torch.isfinite(y)
        cnt = mask.sum(dim=1)
        mean_x = (x * mask).sum(dim=1) / cnt.clamp_min(1)
        mean_y = (y * mask).sum(dim=1) / cnt.clamp_min(1)
        x_c = (x - mean_x.unsqueeze(1)) * mask
        y_c = (y - mean_y.unsqueeze(1)) * mask
        denom = (cnt - 1).clamp_min(1)
        cov = (x_c * y_c).sum(dim=1) / denom
        var_x = (x_c * x_c).sum(dim=1) / denom
        var_y = (y_c * y_c).sum(dim=1) / denom
        std_x = var_x.clamp_min(eps).sqrt()
        std_y = var_y.clamp_min(eps).sqrt()
        corr = cov / (std_x * std_y)
        corr = torch.where(cnt >= 2, corr, torch.full_like(corr, float("nan")))
        return corr
    
    def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def get_ranks(tensor):
            mask = torch.isfinite(tensor)
            tmp = torch.where(mask, tensor, torch.full_like(tensor, -1e30))
            ranks = torch.argsort(torch.argsort(tmp, dim=1), dim=1).float() + 1.0
            return torch.where(mask, ranks, torch.full_like(ranks, float("nan")))
        
        rx = get_ranks(x)
        ry = get_ranks(y)
        return _row_corr(rx, ry) 

    normal_noise = torch.randn(panel.shape, device=panel.device) * sigma
    noisy_normal = panel * (1.0 + normal_noise)

    if noise_dof > 2:
        t_dist = torch.distributions.StudentT(df=noise_dof)
        t_noise = t_dist.sample(panel.shape).to(panel.device)
        t_noise = t_noise * sigma * math.sqrt((noise_dof - 2.0) / noise_dof)
    else:
        t_noise = torch.full_like(panel, sigma)
    noisy_t = panel * (1.0 + t_noise)

    corr1 = _spearman_corr(panel, noisy_normal)
    corr2 = _spearman_corr(panel, noisy_t)
    pfs = torch.nanmean(torch.stack([corr1, corr2], dim=0))
    return float(pfs.item())


def _dh_entropy(panel: torch.Tensor) -> float:
    """Diversity via covariance entropy (alpha-level diversity).
    
    Args:
        panel: Shape (T*N, m), where m is the number of alpha factors.
    """
    if panel.shape[0] <= 1:
        return 0.0

    mask = torch.isfinite(panel)
    cnt = mask.sum(dim=0)
    mean = (panel * mask).sum(dim=0) / cnt.clamp_min(1)
    x_c = (panel - mean) * mask
    denom = (cnt.clamp_min(2) - 1).clamp_min(1)
    cov = (x_c.transpose(0, 1) @ x_c) / denom.unsqueeze(0)
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.transpose(0, 1))

    diag_mean = torch.diagonal(cov).mean().abs()
    jitter = (diag_mean + 1e-6) * 1e-4
    cov = cov + jitter * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

    try:
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    except RuntimeError:
        return 0.0

    total = eigvals.sum()
    if total <= 1e-12:
        return 0.0
    
    # p_i = lambda_i / sum(lambda_j)
    probs = eigvals / total
    probs = probs[probs > 1e-12]
    
    entropy = -(probs * probs.log()).sum()
    m = panel.shape[1]
    max_entropy = math.log(m)
    
    return float((entropy / max_entropy).item() if max_entropy > 0 else entropy.item())
