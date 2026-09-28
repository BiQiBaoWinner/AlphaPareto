from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch


def solve_lambda(
    rewards: np.ndarray,
    q: float = 0.8,
    lr: float = 0.05,
    steps: int = 50,
    retries: int = 2,
    min_q: float = 0.5,
    device: torch.device = torch.device('cuda:0'),
    c_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve lambda >= 0 that maximizes
    g(λ) = -log( mean_j exp(λ^T r_j) ) + λ^T c,
    where c is per-dimension quantile of rewards at level q.

    Returns (lambda_star, c, used_q).
    """
    if rewards.ndim != 2:
        rewards = rewards.reshape(len(rewards), -1)
    # rewards = rewards.astype(np.float32)
    B, dim = rewards.shape
    if B == 1:
        raise ValueError("solve_lambda requires at least 2 reward vectors.")
    device = device or torch.device("cpu")

    current_q = q
    for _ in range(retries + 1):
        if c_override is not None:
            c = np.asarray(c_override, dtype=np.float32)
        else:
            c = np.quantile(rewards, current_q, axis=0)
        c_t = torch.as_tensor(c, device=device, dtype=torch.float32)
        r_t = torch.as_tensor(rewards, device=device, dtype=torch.float32)

        tau = torch.zeros(dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([tau], lr=lr)

        ok = False
        for _ in range(steps):
            lam = torch.nn.functional.softplus(tau)  # ensure >=0
            scores = (r_t * lam).sum(dim=1)
            g = -torch.logsumexp(scores, dim=0) + torch.log(torch.tensor(B, device=device, dtype=torch.float32))
            g = g + (lam * c_t).sum()
            optimizer.zero_grad()
            (-g).backward()  # maximize g => minimize -g
            optimizer.step()
            if torch.isfinite(g):
                ok = True
        if ok:
            lam_star = torch.nn.functional.softplus(tau).detach().cpu().numpy()
            return lam_star, c, current_q
        if c_override is None:
            current_q = max(min_q, current_q - 0.05)
        else:
            break

    # fallback
    lam_star = np.zeros(dim, dtype=np.float32)
    if c_override is not None:
        c = np.asarray(c_override, dtype=np.float32)
    else:
        c = np.quantile(rewards, current_q, axis=0)
    return lam_star, c, current_q
