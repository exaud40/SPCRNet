
from __future__ import annotations
import numpy as np


def grant_bandwidth(beta: np.ndarray, active: np.ndarray, B_tot: float) -> np.ndarray:

    beta = np.asarray(beta, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if np.any(active & ((beta <= 0) | (beta > 1))):
        raise ValueError("active followers must request a share in (0, 1]")
    demand = float(np.sum(beta[active]))
    scale = max(1.0, demand)
    return np.where(active, beta * B_tot / scale, 0.0)


def scale_power(p: np.ndarray, P_max: float) -> np.ndarray:
    """P = p * P_max with p clipped to [0, 1]."""
    return np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0) * P_max
