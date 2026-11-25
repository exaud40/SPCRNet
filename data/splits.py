# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:35:39 2026

@author: Exaud
"""

from __future__ import annotations
import numpy as np
from .base import SPLITS


def split_sequences(
    seq_ids: list[str], ratios: dict[str, float], seed: int = 0
) -> dict[str, list[str]]:
    """Partition unique sequence ids into the four splits according to ``ratios``.
    """
    if set(ratios) != set(SPLITS):
        raise KeyError(f"ratios must have keys {SPLITS}, got {sorted(ratios)}")
    total = sum(ratios.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1, got {total}")

    unique = sorted(set(seq_ids))
    order = np.random.default_rng(seed).permutation(len(unique))
    shuffled = [unique[i] for i in order]

    n = len(shuffled)
    out: dict[str, list[str]] = {}
    start = 0
    for k, name in enumerate(SPLITS):
        count = n - start if k == len(SPLITS) - 1 else int(round(ratios[name] * n))
        count = max(0, min(count, n - start))
        out[name] = sorted(shuffled[start:start + count])
        start += count
    return out
