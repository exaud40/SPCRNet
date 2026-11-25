
from __future__ import annotations

import numpy as np


def dual_update(mu: float, e_hat_norm: float, lr: float) -> float:
    """normalized units: mu <- max(0, mu + lr (E_hat / E_bar - 1))."""
    if lr <= 0:
        raise ValueError("dual step size must be positive")
    return max(0.0, float(mu) + lr * (float(e_hat_norm) - 1.0))


def threshold_update(
    comm_deltas: np.ndarray, current: float, margin: float, lo: float, hi: float, quantile: float = 0.98
) -> float:

    d = np.asarray(comm_deltas, dtype=np.float64).ravel()
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    if d.size == 0:
        return float(current)
    level = float(np.max(d)) if quantile == 1.0 else float(np.quantile(d, quantile))
    return float(np.clip(level + margin, lo, hi))
