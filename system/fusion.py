
from __future__ import annotations
import numpy as np


def log_linear_pool(
    b_own: np.ndarray,
    delta_own: float,
    b_nb: np.ndarray,
    w_nb: np.ndarray,
    T_F: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:

    if T_F <= 0:
        raise ValueError("T_F must be positive")
    b_own = np.clip(np.asarray(b_own, dtype=np.float64), eps, 1.0)
    b_nb = np.clip(np.asarray(b_nb, dtype=np.float64).reshape(-1, b_own.shape[0]), eps, 1.0)
    w_nb = np.asarray(w_nb, dtype=np.float64).reshape(-1)
    if w_nb.shape[0] != b_nb.shape[0]:
        raise ValueError("one weight per neighbor belief is required")

    logits = delta_own * np.log(b_own) + (w_nb[:, None] * np.log(b_nb)).sum(axis=0)
    logits = logits / T_F
    logits -= logits.max()
    b_plus = np.exp(logits)
    b_plus /= b_plus.sum()
    return b_plus, float(b_plus.max())
