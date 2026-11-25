# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:00:19 2026

@author: Exaud
"""

from __future__ import annotations

import numpy as np

KEYS = ("mode", "correct", "qoe", "qoe_real", "E", "D", "sigma", "delta", "gamma_a2g_db", "gamma_a2a_db", "B", "P",
        "a2g_snr_ref_db", "n_conf_nb")


def stack_infos(infos: list[dict]) -> dict[str, np.ndarray]:
    """Concatenate per-step info dicts into flat per-(slot, follower) arrays."""
    if not infos:
        raise ValueError("no steps logged")
    required = ("mode", "correct", "E", "sigma")
    missing = [k for k in required if k not in infos[0]]
    if missing:
        raise KeyError(f"info logs lack required keys {missing}")
    # optional keys 
    return {k: np.concatenate([np.asarray(i[k]) for i in infos]) for k in KEYS if k in infos[0]}


def summarize(infos: list[dict], E_bar: float, D_bar_sigma: float = 0.5) -> dict[str, float]:
    s = stack_infos(infos)
    comm = s["mode"] != 0
    out = {
        "accuracy": float(s["correct"].mean()),
        # QoE = Pr{correct and timely}, estimated by the empirical mean of 1{correct} * sigma_D(D).
        # "qoe_model" uses the modeled correctness q, it is biased for policies that
        # offload selectively, because Psi_s is the edge accuracy averaged over ALL samples.
        "qoe": float(s["qoe_real"].mean()) if "qoe_real" in s else float("nan"),
        "qoe_model": float(s["qoe"].mean()) if "qoe" in s else float("nan"),
        "energy_J": float(s["E"].mean()),
        "energy_norm": float(s["E"].mean() / E_bar),
        "comm_rate": float(comm.mean()),
        "local_share": float((s["mode"] == 0).mean()),
        "a2a_share": float((s["mode"] == 1).mean()),
        "a2g_share": float((s["mode"] == 2).mean()),
        "deadline_ratio": float((s["sigma"] >= D_bar_sigma).mean()),
        "n_samples": int(s["mode"].size),
    }
    out["mean_P_comm_W"] = float(s["P"][comm].mean()) if comm.any() else 0.0
    out["mean_B_comm_MHz"] = float(s["B"][comm].mean() / 1e6) if comm.any() else 0.0
    return out


def comm_prob_by_confidence(delta: np.ndarray, mode: np.ndarray, n_bins: int = 20):
    """Binned P(communicate | delta): returns (bin_centers, prob, counts). Empty bins give NaN."""
    delta, comm = np.asarray(delta, float), np.asarray(mode) != 0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(delta, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    hits = np.bincount(idx, weights=comm.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(counts > 0, hits / np.maximum(counts, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), prob, counts
