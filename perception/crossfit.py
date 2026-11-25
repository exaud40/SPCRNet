
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..data.base import PERCEPTION_KEYS, save_cache, validate_perception_cache
from ..data.mdmt import NUM_CLASSES, SCENE_EDGES
from .calibration import ece, failure_auroc, fit_temperature

TAUS = np.round(np.arange(0.05, 0.91, 0.05), 2)
SIM_SPLITS = {"train": ("train", "heldout"), "cal": ("cal",), "test": ("test",)}   # our_split groups -> simulator


def fit_edge_map(n_hat: np.ndarray, n_union: np.ndarray) -> tuple[float, float, float]:
    """log(1 + n_union) ~ N(a log(1 + n_hat) + b, s^2), least squares."""
    X = np.stack([np.log1p(n_hat), np.ones_like(n_hat)], 1)
    y = np.log1p(n_union)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    s = float(np.std(y - X @ np.array([a, b])))
    return float(a), float(b), max(s, 0.05)


def det_level_probs(n_hat: np.ndarray, a: float, b: float, s: float) -> np.ndarray:
    """P(level) from a detection count through the edge map (numpy twin of train_split.edge_logits)."""
    from math import erf

    mu = a * np.log1p(np.asarray(n_hat, float)) + b
    bounds = np.array([math.log1p(e - 0.5) for e in SCENE_EDGES])
    cdf = 0.5 * (1 + np.vectorize(erf)((bounds[None, :] - mu[:, None]) / (s * math.sqrt(2))))
    p = np.diff(np.concatenate([np.zeros((len(mu), 1)), cdf, np.ones((len(mu), 1))], 1), axis=1)
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)

#  folds
def make_folds(rows: list[dict], k: int, seed: int = 0) -> dict[str, int]:
    """Sequence -> fold. Sequences sorted by mean scene count and dealt out in shuffled blocks of k, so every
    fold spans the whole density range (the label is a density level)."""
    seqs = sorted({r["seq"] for r in rows}, key=lambda s: (int(s) if s.isdigit() else s))
    mean = {s: np.mean([r["n_union"] for r in rows if r["seq"] == s]) for s in seqs}
    order = sorted(seqs, key=lambda s: mean[s])
    rng = np.random.default_rng(seed)
    fold = {}
    for i in range(0, len(order), k):
        block = order[i:i + k]
        for s, f in zip(block, rng.permutation(k)[:len(block)]):
            fold[s] = int(f)
    return fold


def oof_rows(rows: list[dict], stride: int) -> list[dict]:
    """Every stride-th time instant, both views kept (the simulator pairs the views of a time instant)."""
    if stride <= 1:
        return list(rows)
    first = {}
    for r in rows:
        first[r["seq"]] = min(first.get(r["seq"], r["frame"]), r["frame"])
    return [r for r in rows if (r["frame"] - first[r["seq"]]) % stride == 0]


# - calibration and metrics
def fit_cal(P: dict, m: np.ndarray) -> dict:
    """Calibration from the out-of-fold predictions selected by mask m (clean channel = index 0)."""
    nv, nu, y = P["n_view"][m], P["n_union"][m], P["label"][m]
    errs = np.abs(P["counts"][m, 0, :] - nv[:, None]).mean(0)
    ti = int(np.argmin(errs))
    a, b, s = fit_edge_map(P["counts"][m, 0, ti], nu)
    return {"tau_i": ti, "tau": float(TAUS[ti]), "edge_a": a, "edge_b": b, "edge_s": s,
            "T_b": fit_temperature(P["local_logits"][m], y), "T_e": fit_temperature(P["edge_logits"][m, 0], y),
            "count_mae_view": float(errs[ti])}


def softmax_T(lg: np.ndarray, T: float) -> np.ndarray:
    z = lg / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def apply_cal(P: dict, folds: np.ndarray, K: int) -> tuple[dict, list[dict]]:
    """Nested calibration: fold k gets the calibration fitted on the other folds."""
    N, S = P["edge_logits"].shape[:2]
    out = {"local_belief": np.zeros((N, NUM_CLASSES)), "edge_belief": np.zeros((N, S, NUM_CLASSES)),
           "edge_count": np.zeros((N, S)), "det_belief": np.zeros((N, S, NUM_CLASSES))}
    cals = []
    for k in range(K):
        mk = folds == k
        if not mk.any():
            cals.append({})
            continue
        c = fit_cal(P, ~mk & (folds >= 0))
        cals.append(c)
        out["local_belief"][mk] = softmax_T(P["local_logits"][mk], c["T_b"])
        out["edge_belief"][mk] = softmax_T(P["edge_logits"][mk], c["T_e"])
        out["edge_count"][mk] = P["counts"][mk, :, c["tau_i"]]
        for j in range(S):
            out["det_belief"][mk, j] = det_level_probs(out["edge_count"][mk, j], c["edge_a"], c["edge_b"], c["edge_s"])
    return out, cals


def scene_bootstrap(correct: np.ndarray, seq: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of the frame accuracy, resampling sequences (frames within a sequence are not independent)."""
    seqs, inv = np.unique(seq, return_inverse=True)
    tot = np.bincount(inv, weights=np.asarray(correct, float), minlength=len(seqs))
    cnt = np.bincount(inv, minlength=len(seqs)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(seqs), (n_boot, len(seqs)))
    acc = tot[pick].sum(1) / cnt[pick].sum(1)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return float(lo), float(hi)


def metrics(P: dict, C: dict, m: np.ndarray, snr_names: list[str]) -> dict:
    y, seq = P["label"][m], P["seq"][m]
    lb = C["local_belief"][m]
    lc = (lb.argmax(1) == y).astype(float)
    out = {"n_frames": int(m.sum()), "n_seqs": int(len(np.unique(seq))),
           "local": {"acc": float(lc.mean()), "acc_ci95": scene_bootstrap(lc, seq), "ece": ece(lb.max(1), lc),
                     "auroc": failure_auroc(lb.max(1), lc),
                     "count_mae": float(np.mean(np.abs(np.expm1(P["local_mu"][m]) - P["n_union"][m])))},
           "edge": {}}
    for j, name in enumerate(snr_names):
        eb = C["edge_belief"][m, j]
        ec = (eb.argmax(1) == y).astype(float)
        out["edge"][name] = {"acc": float(ec.mean()), "acc_ci95": scene_bootstrap(ec, seq), "ece": ece(eb.max(1), ec),
                             "auroc": failure_auroc(eb.max(1), ec),
                             "acc_from_detections": float((C["det_belief"][m, j].argmax(1) == y).mean()),
                             "view_count_mae": float(np.mean(np.abs(C["edge_count"][m, j] - P["n_view"][m])))}
    # paired difference edge(clean) - local, CI over sequences
    d = (C["edge_belief"][m, 0].argmax(1) == y).astype(float) - lc
    out["edge_minus_local_clean"] = {"diff": float(d.mean()), "ci95": scene_bootstrap(d, seq)}
    return out


#  export
def export(P: dict, C: dict, rows: list[dict], snr_grid: list[float], cache: Path) -> None:
    sp = np.array([r["our_split"] for r in rows])
    for split in ("train", "heldout", "cal", "test"):
        m = sp == split
        if not m.any():
            continue
        p_raw = softmax_T(P["local_logits"][m], 1.0)
        p = C["local_belief"][m]
        a = {"seq_id": P["seq"][m], "scene_id": np.array([r["scene_id"] for r, k in zip(rows, m) if k]),
             "view_id": P["view"][m], "frame": P["frame"][m], "label": P["label"][m],
             "logits": P["local_logits"][m].astype(np.float32), "belief": p.astype(np.float32),
             "delta": p.max(1).astype(np.float32), "delta_raw": p_raw.max(1).astype(np.float32),
             "msp_raw": p_raw.max(1).astype(np.float32), "n_view": P["n_view"][m].astype(np.int64),
             "n_union": P["n_union"][m].astype(np.int64), "count_pred": np.expm1(P["local_mu"][m]).astype(np.float32),
             "count_logsig": P["local_ls"][m].astype(np.float32), "fold": P["fold"][m].astype(np.int64)}
        if "local_view" in P:
            a["count_view_pred"] = P["local_view"][m].astype(np.float32)
        order = np.lexsort((a["view_id"], a["frame"], a["seq_id"]))
        a = {k: v[order] for k, v in a.items()}
        validate_perception_cache(a)
        save_cache(cache / "perception_raw" / f"perception_{split}.npz", a, PERCEPTION_KEYS)
        eb = C["edge_belief"][m][order]
        ec = (eb.argmax(-1) == a["label"][:, None]).astype(np.float32)
        (cache / "edge_raw").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / "edge_raw" / f"edge_{split}.npz", snr_grid_db=np.array(snr_grid, float),
                            edge_correct_clean=ec[:, 0], edge_correct=ec[:, 1:], edge_belief=eb[:, 1:].astype(np.float32),
                            edge_count=C["edge_count"][m][order][:, 1:].astype(np.float32),
                            seq_id=a["seq_id"], view_id=a["view_id"], frame=a["frame"],
                            **({"edge_mu": P["edge_mu"][m][order][:, 1:].astype(np.float32),
                                "edge_view_count": P["edge_view"][m][order][:, 1:].astype(np.float32)}
                               if "edge_mu" in P else {}))
    for rl, srcs in SIM_SPLITS.items():
        parts = [np.load(cache / "perception_raw" / f"perception_{s}.npz") for s in srcs
                 if (cache / "perception_raw" / f"perception_{s}.npz").exists()]
        if parts:
            a = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files}
            save_cache(cache / f"perception_{rl}.npz", a, PERCEPTION_KEYS)


#  pooling
def pool(out: Path, rows: list[dict], K: int) -> tuple[dict, list[dict]]:
    """Concatenate the finished folds' predictions with the row metadata."""
    parts, R = [], []
    for k in range(K):
        f = out / f"fold{k}" / "oof.npz"
        if f.exists():
            z = np.load(f)
            parts.append({n: z[n] for n in z.files})
            R.extend(rows[i] for i in z["row"])
    if not parts:
        raise SystemExit(f"no finished folds in {out}")
    keys = set.intersection(*(set(p) for p in parts))           # folds re-predicted with newer outputs or not
    P = {n: np.concatenate([p[n] for p in parts]) for n in parts[0] if n in keys}
    P.update({"label": np.array([r["label"] for r in R]), "n_view": np.array([r["n_view"] for r in R], float),
              "n_union": np.array([r["n_union"] for r in R], float), "seq": np.array([r["seq"] for r in R]),
              "view": np.array([r["view"] for r in R], np.int64), "frame": np.array([r["frame"] for r in R], np.int64),
              "split": np.array([r["split"] for r in R])})
    return P, R
