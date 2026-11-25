# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:15:47 2026

@author: Exaud
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0


def a2a_gain(dist: np.ndarray, f_c: float) -> np.ndarray:
    """Free-space A2A gain, from eq. (a2a_gain): h = beta0 / d^2, beta0 = (c / (4 pi f_c))^2."""
    d = np.asarray(dist, dtype=np.float64)
    if np.any(d <= 0):
        raise ValueError("A2A distance must be positive")
    beta0 = (C_LIGHT / (4.0 * np.pi * f_c)) ** 2
    return beta0 / d**2


def a2a_snr(P: np.ndarray, h: np.ndarray, B: np.ndarray, n_resp: np.ndarray, N0: float) -> np.ndarray:
    """Per-responder A2A SNR, eq. (a2a_snr): gamma = |N_u| P h / (N0 B).

    Each of the n_resp responders transmits with power P on a sub-band B / n_resp.
    """
    P, h, B, n = (np.asarray(x, dtype=np.float64) for x in (P, h, B, n_resp))
    return n * P * h / (N0 * B)


def elevation_deg(H: float, r: np.ndarray) -> np.ndarray:
    """Elevation angle in degrees: (180/pi) * arctan(H / r), 90 deg directly below (r = 0)."""
    r = np.asarray(r, dtype=np.float64)
    return np.degrees(np.arctan2(H, r))


def los_probability(theta_deg: np.ndarray, a: float, b: float) -> np.ndarray:
    """Al-Hourani LoS probability, eq. (plos): 1 / (1 + a exp(-b (theta - a)))."""
    theta = np.asarray(theta_deg, dtype=np.float64)
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def a2g_mean_pathloss_db(
    r: np.ndarray, H: float, f_c: float, a: float, b: float, eta_los_db: float, eta_nlos_db: float
) -> np.ndarray:
    """Mean A2G path loss [dB], eq. (a2g_pl): FSPL(d) + P_LoS eta_LoS + (1 - P_LoS) eta_NLoS,
    with d = sqrt(r^2 + H^2)."""
    r = np.asarray(r, dtype=np.float64)
    d = np.sqrt(r**2 + H**2)
    fspl = 20.0 * np.log10(4.0 * np.pi * f_c * d / C_LIGHT)
    p_los = los_probability(elevation_deg(H, r), a, b)
    return fspl + p_los * eta_los_db + (1.0 - p_los) * eta_nlos_db


def nakagami_power(rng: np.random.Generator, m: float, size) -> np.ndarray:
    """Small-scale power gain |g|^2 ~ Gamma(shape=m, scale=1/m) (unit mean), eq. (a2g_gain)."""
    if m < 0.5:
        raise ValueError("Nakagami m must be >= 0.5")
    return rng.gamma(shape=m, scale=1.0 / m, size=size)


def a2g_gain(pathloss_db: np.ndarray, fading_power: np.ndarray) -> np.ndarray:
    """Instantaneous A2G gain, eq. (a2g_gain): H = |g|^2 10^(-PL/10)."""
    return np.asarray(fading_power, dtype=np.float64) * 10.0 ** (-np.asarray(pathloss_db, dtype=np.float64) / 10.0)


def snr(P: np.ndarray, gain: np.ndarray, B: np.ndarray, N0: float) -> np.ndarray:
    """Generic SNR: P * gain / (N0 * B). Used for the two A2G hops."""
    P, gain, B = (np.asarray(x, dtype=np.float64) for x in (P, gain, B))
    return P * gain / (N0 * B)


def af_end_to_end_snr(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Variable-gain AF relay SNR, eq. (af_snr): g1 g2 / (g1 + g2 + 1)."""
    g1, g2 = np.asarray(g1, dtype=np.float64), np.asarray(g2, dtype=np.float64)
    return g1 * g2 / (g1 + g2 + 1.0)
