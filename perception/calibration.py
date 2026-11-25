# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:07:42 2026

@author: Exaud
"""

from __future__ import annotations
import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _golden(f, lo: float, hi: float, iters: int = 80) -> float:
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    p = _softmax(np.asarray(logits, np.float64) / T)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature scaling: argmin_T NLL(softmax(logits / T), labels), T in [lo, hi]."""
    logits, labels = np.asarray(logits, np.float64), np.asarray(labels, int)
    return float(np.exp(_golden(lambda lt: nll(logits, labels, np.exp(lt)), np.log(lo), np.log(hi))))


def bce(conf_logits: np.ndarray, target: np.ndarray, T: float = 1.0) -> float:
    z = np.asarray(conf_logits, np.float64) / T
    t = np.asarray(target, np.float64)
    # stable: log(1 + e^z) - t z
    return float(np.mean(np.logaddexp(0.0, z) - t * z))


def fit_binary_temperature(conf_logits: np.ndarray, correct: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Same for the confidence head: argmin_T BCE(sigmoid(conf_logits / T), correct)."""
    return float(np.exp(_golden(lambda lt: bce(conf_logits, correct, np.exp(lt)), np.log(lo), np.log(hi))))


def reliability_curve(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    """(bin_centers, accuracy_per_bin, confidence_per_bin, count_per_bin); empty bins give NaN."""
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.bincount(idx, weights=correct, minlength=n_bins) / cnt
        cf = np.bincount(idx, weights=conf, minlength=n_bins) / cnt
    acc[cnt == 0] = np.nan
    cf[cnt == 0] = np.nan
    return 0.5 * (edges[:-1] + edges[1:]), acc, cf, cnt


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Eq. (ece) with equal-width bins on [0, 1]: sum_b |B_b|/n * |acc_b - conf_b|."""
    _, acc, cf, cnt = reliability_curve(conf, correct, n_bins)
    ok = cnt > 0
    return float(np.sum(cnt[ok] / cnt.sum() * np.abs(acc[ok] - cf[ok])))


def failure_auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of confidence for separating correct (positive) from wrong decisions (Mann-Whitney, ties = 1/2)."""
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct).astype(bool)
    pos, neg = conf[correct], conf[~correct]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < len(sv):                                    # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((np.asarray(conf, np.float64) - np.asarray(correct, np.float64)) ** 2))
