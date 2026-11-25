
from __future__ import annotations
import numpy as np

LOCAL, A2A, A2G = 0, 1, 2


def _safe(B: np.ndarray, mode: np.ndarray) -> np.ndarray:
    """Bandwidth with LOCAL rows replaced by 1 ."""
    B = np.asarray(B, dtype=np.float64)
    if np.any((mode != LOCAL) & (B <= 0)):
        raise ValueError("communicating followers must have positive bandwidth")
    return np.where(mode == LOCAL, 1.0, B)


def tx_latency(mode: np.ndarray, B: np.ndarray, n_resp: np.ndarray, k_b: int, k_s: int) -> np.ndarray:
    """tx_latency: 0 | n_resp * k_b / B | 2 k_s / B for LOCAL | A2A | A2G."""
    mode = np.asarray(mode)
    Bs = _safe(B, mode)
    n = np.asarray(n_resp, dtype=np.float64)
    return np.select([mode == A2A, mode == A2G], [n * k_b / Bs, 2.0 * k_s / Bs], default=0.0)


def decision_latency(
    mode: np.ndarray, d_tx: np.ndarray, chi_L: float, chi_F: float, chi_G: float, f_U: float, f_G: float
) -> np.ndarray:
    """decision_latency: chi_L/f_U + D_tx + {0, chi_F/f_U, chi_G/f_G}[mode]."""
    mode = np.asarray(mode)
    extra = np.select([mode == A2A, mode == A2G], [chi_F / f_U, chi_G / f_G], default=0.0)
    return chi_L / f_U + np.asarray(d_tx, dtype=np.float64) + extra


def n_links(mode: np.ndarray, n_resp: np.ndarray) -> np.ndarray:
    """Number of activated radio links: 0 | n_resp + 1 (request + responses) | 2 (two A2G hops)."""
    mode = np.asarray(mode)
    n = np.asarray(n_resp, dtype=np.float64)
    return np.select([mode == A2A, mode == A2G], [n + 1.0, 2.0], default=0.0)


def comm_energy(
    mode: np.ndarray, P: np.ndarray, B: np.ndarray, n_resp: np.ndarray, P_l: float, k_b: int, k_s: int,
    E0: float = 0.0,
) -> np.ndarray:
    """energy [Joules]:
        transmit part   0 | n_resp^2 k_b P / B | k_s (P + P_l) / B
      + overhead part   E0 * n_links(mode)     (circuit operation and link setup, E0 = P_c T_oh)
    """
    mode = np.asarray(mode)
    Bs = _safe(B, mode)
    P = np.asarray(P, dtype=np.float64)
    n = np.asarray(n_resp, dtype=np.float64)
    tx = np.select([mode == A2A, mode == A2G], [n**2 * k_b * P / Bs, k_s * (P + P_l) / Bs], default=0.0)
    return tx + E0 * n_links(mode, n_resp)


def timeliness(D: np.ndarray, D_th: float, zeta_D: float) -> np.ndarray:
    """timeliness: 1 / (1 + exp(zeta_D (D - D_th))), computed stably via tanh."""
    x = zeta_D * (np.asarray(D, dtype=np.float64) - D_th)
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def qoe(q: np.ndarray, D: np.ndarray, D_th: float, zeta_D: float) -> np.ndarray:
    """QoE: q * timeliness(D)."""
    return np.asarray(q, dtype=np.float64) * timeliness(D, D_th, zeta_D)
