# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:53:42 2026

@author: Exaud
"""

from __future__ import annotations
import numpy as np
from ..comm.fidelity import logistic_fidelity
from ..system.latency_energy import timeliness
from ..eval.metrics import comm_prob_by_confidence


def threshold_a2g(psi_s, gamma_db, D0, D2, E2_norm, mu, D_th, zeta_D):
    """vectorized over gamma_db / D2 / E2_norm / mu."""
    q2 = np.clip(logistic_fidelity(gamma_db, psi_s), 0.0, 1.0)
    num = q2 * timeliness(D2, D_th, zeta_D) - mu * np.asarray(E2_norm, float)
    return np.clip(num / timeliness(D0, D_th, zeta_D), 0.0, 1.0)


def empirical_threshold(delta, mode, n_bins: int = 20, level: float = 0.5, min_count: int = 30) -> float:
    """Smallest confidence at which the binned probability of communicating falls below ``level``
    and stays there (NaN if it never does). Bins with fewer than min_count samples are ignored."""
    centers, prob, counts = comm_prob_by_confidence(delta, mode, n_bins)
    ok = counts >= min_count
    centers, prob = centers[ok], prob[ok]
    below = prob < level
    for i in range(len(below)):
        if below[i:].all():
            return float(centers[i])
    return float("nan")
