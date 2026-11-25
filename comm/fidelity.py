# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:26 2026

@author: Exaud
"""

from __future__ import annotations
import numpy as np


def logistic_fidelity(snr_db: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Eq. (fidelity): psi1 / (1 + exp(-psi2 (snr_db - psi3))) + psi4 (stable via tanh)."""
    psi = np.asarray(psi, dtype=np.float64)
    x = psi[1] * (np.asarray(snr_db, dtype=np.float64) - psi[2])
    return psi[0] * 0.5 * (1.0 + np.tanh(0.5 * x)) + psi[3]


def interp_on_grid(table: np.ndarray, grid_db: np.ndarray, snr_db: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    grid = np.asarray(grid_db, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be 1-D, strictly increasing, with >= 2 points")
    if table.shape[-1] != grid.size:
        raise ValueError("last axis of table must match the grid")
    x = np.broadcast_to(np.clip(np.asarray(snr_db, dtype=np.float64), grid[0], grid[-1]), table.shape[:-1])
    idx = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    lo = np.take_along_axis(table, idx[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(table, (idx + 1)[..., None], axis=-1)[..., 0]
    frac = (x - grid[idx]) / (grid[idx + 1] - grid[idx])
    return lo + frac * (hi - lo)


def fit_fidelity(snr_db: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit of eq. (fidelity) to (snr_db, acc). Returns (psi, R^2).
    """
    from scipy.optimize import curve_fit

    x, y = np.asarray(snr_db, float), np.asarray(acc, float)
    f = lambda xx, a, k, c, d: logistic_fidelity(xx, np.array([a, k, c, d]))
    lo, hi = float(y.min()), float(y.max())
    p0 = [max(hi - lo, 1e-3), 0.5, float(x[np.argmin(np.abs(y - 0.5 * (lo + hi)))]), lo]
    bounds = ([0.0, 0.01, x.min() - 20.0, 0.0], [1.0, 5.0, x.max() + 20.0, 1.0])
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # covariance warnings on short grids; only psi is used
            psi, _ = curve_fit(f, x, y, p0=np.clip(p0, bounds[0], bounds[1]), bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        psi = np.array(p0, float)
    psi[0] = min(psi[0], 1.0 - psi[3])
    res = y - f(x, *psi)
    tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / tot if tot > 0 else 1.0
    return np.asarray(psi, float), r2
