
from __future__ import annotations

import numpy as np


def td_residuals(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    """td_error: e(t) = r(t) + gamma (1 - done_t) V(s_{t+1}) - V(s_t)."""
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    if v.shape[0] != r.shape[0] + 1 or d.shape != r.shape:
        raise ValueError("need values of length T+1 and dones of length T")
    return r + gamma * (1.0 - d) * v[1:] - v[:-1]


def reliability_weight(delta: np.ndarray, kappa: float) -> np.ndarray:
    """wgae_weight: w = 1 / (1 + kappa * delta * (1 - delta)), in [4/(4+kappa), 1]."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    d = np.clip(np.asarray(delta, dtype=np.float64), 0.0, 1.0)
    return 1.0 / (1.0 + kappa * d * (1.0 - d))


def gae(e: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Standard GAE: A(t) = e(t) + gamma lam (1 - done_t) A(t+1). Returns (T,)."""
    e = np.asarray(e, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    adv = np.zeros_like(e)
    acc = 0.0
    for t in range(e.shape[0] - 1, -1, -1):
        acc = e[t] + gamma * lam * (1.0 - d[t]) * acc
        adv[t] = acc
    return adv


def wgae(e: np.ndarray, dones: np.ndarray, delta: np.ndarray, gamma: float, lam: float, kappa: float) -> np.ndarray:

    e = np.asarray(e, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    T = e.shape[0]
    if delta.ndim != 2 or delta.shape[0] != T + 1:
        raise ValueError("delta must have shape (T+1, M)")
    w_next = reliability_weight(delta[1:], kappa)            # (T, M): w_u(t+1)
    adv = np.zeros((T, delta.shape[1]))
    acc = np.zeros(delta.shape[1])
    for t in range(T - 1, -1, -1):
        acc = e[t] + gamma * lam * (1.0 - d[t]) * w_next[t] * acc
        adv[t] = acc
    return adv

def wgae_explicit(e: np.ndarray, delta: np.ndarray, gamma: float, lam: float, kappa: float) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    w = reliability_weight(np.asarray(delta, dtype=np.float64), kappa)   # (T+1, M)
    T, M = e.shape[0], w.shape[1]
    adv = np.zeros((T, M))
    for t in range(T):
        for u in range(M):
            coef, total = 1.0, 0.0
            for l in range(T - t):
                if l > 0:
                    coef *= gamma * lam * w[t + l, u]
                total += coef * e[t + l]
            adv[t, u] = total
    return adv


def critic_targets(values: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """V_tar(t) = V(s_t) + mean_u A_u(t)  Returns (T,)."""
    v = np.asarray(values, dtype=np.float64)
    a = np.asarray(adv, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return v[: a.shape[0]] + a.mean(axis=1)


def kappa_schedule(kappa: float, iteration: int, anneal_iters: int) -> float:
    if anneal_iters <= 0:
        return float(kappa)
    return float(kappa) * max(0.0, 1.0 - (iteration - 1) / anneal_iters)
