
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.base import PERCEPTION_KEYS, save_cache, validate_perception_cache
from .calibration import ece, failure_auroc

RL_SPLITS = {"train": ("heldout", "cal"), "cal": ("cal",), "test": ("test",)}


def _softmax(z, T=1.0):
    z = z / T - (z / T).max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def arrays_for(rows: list[dict], res: dict, T_b: float, T_d: float) -> dict[str, np.ndarray]:
    lg = res["logits"].astype(np.float64)
    belief = _softmax(lg, T_b)
    c = res["conf_logit"].astype(np.float64)
    a = {
        "seq_id": np.array([r["seq"] for r in rows]),
        "scene_id": np.array([r["scene_id"] for r in rows]),
        "view_id": np.array([r["view"] for r in rows], dtype=np.int64),
        "frame": np.array([r["frame"] for r in rows], dtype=np.int64),
        "label": np.array([r["label"] for r in rows], dtype=np.int64),
        "logits": lg.astype(np.float32),
        "belief": belief.astype(np.float32),
        "delta": (1.0 / (1.0 + np.exp(-c / T_d))).astype(np.float32),
        "delta_raw": (1.0 / (1.0 + np.exp(-c))).astype(np.float32),
        "msp_raw": _softmax(lg).max(1).astype(np.float32),
        "n_view": np.array([r["n_view"] for r in rows], dtype=np.int64),
        "n_union": np.array([r["n_union"] for r in rows], dtype=np.int64),
    }
    if "log_count" in res:
        a["count_pred"] = np.expm1(res["log_count"]).astype(np.float32)
    if "log_sigma" in res:
        a["count_logsig"] = res["log_sigma"].astype(np.float32)
    if "tokens" in res:
        a["tokens"] = res["tokens"].astype(np.float16)
    a["belief"] = (a["belief"] / a["belief"].sum(1, keepdims=True)).astype(np.float32)
    return a


def sort_scenes(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.lexsort((a["view_id"], a["frame"], a["seq_id"]))
    return {k: v[order] for k, v in a.items()}


def concat(parts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summary(a: dict) -> dict:
    correct = (a["belief"].argmax(1) == a["label"]).astype(float)
    return {"n": int(len(correct)), "accuracy": float(correct.mean()),
            "ece_delta": ece(a["delta"], correct), "ece_delta_raw": ece(a["delta_raw"], correct),
            "ece_msp_raw": ece(a["msp_raw"], correct), "ece_msp_cal": ece(a["belief"].max(1), correct),
            "auroc_delta": failure_auroc(a["delta"], correct), "auroc_msp": failure_auroc(a["msp_raw"], correct)}


def write_caches(per_split: dict[str, dict], cache_out: str | Path) -> dict:
    cache_out = Path(cache_out)
    for split, a in per_split.items():
        validate_perception_cache(a)
        save_cache(cache_out / "perception_raw" / f"perception_{split}.npz", sort_scenes(a), PERCEPTION_KEYS)
    for rl_split, srcs in RL_SPLITS.items():
        if all(s in per_split for s in srcs):
            a = sort_scenes(concat([per_split[s] for s in srcs]))
            validate_perception_cache(a)
            save_cache(cache_out / f"perception_{rl_split}.npz", a, PERCEPTION_KEYS)
    return {s: summary(a) for s, a in per_split.items()}


def export_all(model, rows: dict[str, list], args, device) -> dict:
    from .train_perception import infer, loader

    T_b, T_d = float(model.T_b), float(model.T_delta)
    per_split = {}
    for split, rs in rows.items():
        if not rs:
            continue
        res = infer(model, loader(rs, args.img_root, False, args.batch * 2, args.workers), device, keep_tokens=True)
        per_split[split] = arrays_for(rs, res, T_b, T_d)
        print(f"[export] {split}: {len(rs)} rows", flush=True)
    rep = write_caches(per_split, args.cache_out)
    for s, r in rep.items():
        print(f"[export] {s:8s} acc {r['accuracy']:.3f}  ECE delta {r['ece_delta']:.3f} (raw {r['ece_delta_raw']:.3f})  "
              f"ECE MSP {r['ece_msp_raw']:.3f}  AUROC delta {r['auroc_delta']:.3f} vs MSP {r['auroc_msp']:.3f}")
    return rep
