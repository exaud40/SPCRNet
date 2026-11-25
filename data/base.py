# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:21:19 2026

@author: Exaud
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

SPLITS = ("train", "heldout", "cal", "test")


@dataclass
class MultiViewSample:
    scene_id: str
    seq_id: str
    view_id: int
    frame: int
    label: int
    image: np.ndarray | None = None


PERCEPTION_KEYS = {
    "seq_id": "(S,) str",
    "scene_id": "(S,) str",
    "view_id": "(S,) int",
    "frame": "(S,) int",
    "label": "(S,) int  ground truth y",
    "logits": "(S,C) float  pre-temperature task logits",
    "belief": "(S,C) float  calibrated b_u, eq. after (gss)",
    "delta": "(S,) float  calibrated confidence, eq. (confidence_head)",
    "tokens": "(S,L_s,d) float16  semantic tokens Z_u (optional; needed only for codec stage)",
}

FIDELITY_KEYS = {
    "snr_grid_db": "(G,) float",
    "edge_correct_prob": "(S,G) float  P(edge decision correct | sample, SNR), over noise draws",
    "belief_hat": "(S,G,C) float  decoded neighbor belief b_hat at each SNR (mean over draws)",
    "delta_hat": "(S,G) float  decoded neighbor confidence",
    "psi_b": "(4,) float  fitted Psi_{k_b} parameters",
    "psi_s": "(4,) float  fitted Psi_{k_s} parameters",
}


def save_cache(path: str | Path, arrays: dict[str, np.ndarray], schema: dict[str, str]) -> None:
    missing = [k for k in schema if k not in arrays and k != "tokens"]
    if missing:
        raise KeyError(f"Cache is missing required keys: {missing}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_cache(path: str | Path, schema: dict[str, str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    missing = [k for k in schema if k not in data and k != "tokens"]
    if missing:
        raise KeyError(f"Cache {path} is missing keys: {missing}")
    return data


def validate_perception_cache(c: dict[str, np.ndarray]) -> None:
    """Cheap invariants...call after every export and every load in tests."""
    S, C = c["belief"].shape
    assert c["logits"].shape == (S, C)
    assert np.allclose(c["belief"].sum(1), 1.0, atol=1e-4), "beliefs must sum to 1"
    assert ((c["delta"] >= 0) & (c["delta"] <= 1)).all(), "delta must lie in [0,1]"
    assert ((c["label"] >= 0) & (c["label"] < C)).all()
    assert len(c["scene_id"]) == S
