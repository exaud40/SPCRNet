# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:36:51 2026

@author: Exaud
"""

"""build and debug the environment and RL without any trained model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .base import FIDELITY_KEYS, PERCEPTION_KEYS, SPLITS, save_cache, validate_perception_cache

CLEAR, OCCLUDED, HARD = 0, 1, 2

# Beta parameters of the local confidence per view type (mean in comments).
_BETA = {CLEAR: (8.0, 2.0),     # ~0.80
         OCCLUDED: (2.0, 4.0),  # ~0.33
         HARD: (3.0, 3.0)}      # ~0.50
# Factor applied to edge accuracy per view type (occluded tokens lack the target).
EDGE_FACTOR = {CLEAR: 1.0, OCCLUDED: 0.55, HARD: 0.8}


def _logistic(snr_db: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return psi[0] / (1.0 + np.exp(-psi[1] * (snr_db - psi[2]))) + psi[3]


def _scene_types(rng: np.random.Generator, frames: int, p_occluded: float, p_hard: float,
                 p_switch: float = 0.1) -> np.ndarray:
    """Per-frame scene type with temporal persistence (Markov switching)."""
    probs = np.array([1.0 - p_occluded - p_hard, p_occluded, p_hard])
    if probs.min() < 0:
        raise ValueError("p_occluded + p_hard must be <= 1")
    types = np.empty(frames, dtype=np.int64)
    types[0] = rng.choice(3, p=probs)
    for f in range(1, frames):
        types[f] = rng.choice(3, p=probs) if rng.random() < p_switch else types[f - 1]
    return types


def _belief(decision: int, delta: float, num_classes: int) -> np.ndarray:
    """Belief with argmax == decision and max == delta (delta > 1/C guaranteed by caller)."""
    b = np.full(num_classes, (1.0 - delta) / (num_classes - 1))
    b[decision] = delta
    return b


def make_split(
    rng: np.random.Generator,
    n_seq: int,
    frames_per_seq: int,
    views: int,
    num_classes: int,
    p_occluded: float = 0.3,
    p_hard: float = 0.15,
    split_name: str = "s",
) -> dict[str, np.ndarray]:
    if num_classes < 2 or views < 1:
        raise ValueError("need num_classes >= 2 and views >= 1")
    floor = 1.0 / num_classes + 0.02  # keeps delta the strict argmax of the belief

    rows: dict[str, list] = {k: [] for k in
                             ("seq_id", "scene_id", "view_id", "frame", "label", "belief", "delta", "scene_type")}
    for s in range(n_seq):
        seq = f"{split_name}{s:04d}"
        types = _scene_types(rng, frames_per_seq, p_occluded, p_hard)
        for f in range(frames_per_seq):
            y = int(rng.integers(num_classes))
            occluded_view = int(rng.integers(views))
            for v in range(views):
                if types[f] == OCCLUDED:
                    vtype = OCCLUDED if v == occluded_view else CLEAR
                else:
                    vtype = int(types[f])
                delta = float(np.clip(rng.beta(*_BETA[vtype]), floor, 0.995))
                if rng.random() < delta:
                    decision = y
                else:
                    decision = int(rng.choice([c for c in range(num_classes) if c != y]))
                rows["seq_id"].append(seq)
                rows["scene_id"].append(f"{seq}_f{f:05d}")
                rows["view_id"].append(v)
                rows["frame"].append(f)
                rows["label"].append(y)
                rows["belief"].append(_belief(decision, delta, num_classes))
                rows["delta"].append(delta)
                rows["scene_type"].append(vtype)

    belief = np.asarray(rows["belief"], dtype=np.float64)
    out = {
        "seq_id": np.asarray(rows["seq_id"]),
        "scene_id": np.asarray(rows["scene_id"]),
        "view_id": np.asarray(rows["view_id"], dtype=np.int64),
        "frame": np.asarray(rows["frame"], dtype=np.int64),
        "label": np.asarray(rows["label"], dtype=np.int64),
        "logits": np.log(belief),
        "belief": belief,
        "delta": np.asarray(rows["delta"], dtype=np.float64),
        "scene_type": np.asarray(rows["scene_type"], dtype=np.int64),
    }
    validate_perception_cache(out)
    return out


def make_fidelity(
    rng: np.random.Generator,
    perception: dict[str, np.ndarray],
    snr_grid_db: np.ndarray,
    psi_b: np.ndarray,
    psi_s: np.ndarray,
) -> dict[str, np.ndarray]:
    del rng
    grid = np.asarray(snr_grid_db, dtype=np.float64)
    belief = perception["belief"]
    S, C = belief.shape
    stype = perception.get("scene_type", np.zeros(S, dtype=np.int64))

    psi_b_curve = np.clip(_logistic(grid, psi_b), 0.0, 1.0)               # (G,)
    psi_s_curve = np.clip(_logistic(grid, psi_s), 0.0, 1.0)               # (G,)
    factor = np.vectorize(EDGE_FACTOR.get)(stype).astype(np.float64)      # (S,)

    edge = np.clip(factor[:, None] * psi_s_curve[None, :], 0.0, 1.0)      # (S,G)
    uni = np.full(C, 1.0 / C)
    w = psi_b_curve[None, :, None]                                        # (1,G,1)
    belief_hat = w * belief[:, None, :] + (1.0 - w) * uni[None, None, :]  # (S,G,C)
    delta_hat = psi_b_curve[None, :] * perception["delta"][:, None] + (1.0 - psi_b_curve[None, :]) / C

    return {
        "snr_grid_db": grid,
        "edge_correct_prob": edge,
        "belief_hat": belief_hat.astype(np.float32),
        "delta_hat": delta_hat.astype(np.float32),
        "psi_b": np.asarray(psi_b, dtype=np.float64),
        "psi_s": np.asarray(psi_s, dtype=np.float64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="caches/synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    sizes = {"train": 60, "heldout": 15, "cal": 10, "test": 15}  # sequences per split
    grid = np.arange(-10, 20.1, 2.5)
    psi_b = np.array([0.95, 0.6, -2.0, 0.03])
    psi_s = np.array([0.90, 0.4, 3.0, 0.05])
    for split in SPLITS:
        p = make_split(rng, sizes[split], frames_per_seq=200, views=2,
                       num_classes=args.classes, split_name=split[:2])
        save_cache(out / f"perception_{split}.npz", p, PERCEPTION_KEYS)
        f = make_fidelity(rng, p, grid, psi_b, psi_s)
        save_cache(out / f"fidelity_{split}.npz", f, FIDELITY_KEYS)
    print(f"wrote synthetic caches to {out}")


if __name__ == "__main__":
    main()
