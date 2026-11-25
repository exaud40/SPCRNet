# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:02:01 2026

@author: Exaud
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..data.mdmt import SCENE_EDGES, count_bin
from ..perception.calibration import ece
from ..perception.crossfit import det_level_probs, scene_bootstrap

SPLITS = ("train", "heldout", "cal", "test")


def norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, None)
    return p / p.sum(-1, keepdims=True)


def load(cache: Path, run: Path, snr_db: float) -> dict:
    L, E = [], []
    for s in SPLITS:
        f = cache / "perception_raw" / f"perception_{s}.npz"
        if f.exists():
            L.append(dict(np.load(f)))
            E.append(dict(np.load(cache / "edge_raw" / f"edge_{s}.npz")))
            L[-1]["mdmt_test"] = np.full(len(L[-1]["label"]), s == "test")
    D = {k: np.concatenate([d[k] for d in L]) for k in ("seq_id", "view_id", "frame", "label", "belief", "delta",
                                                         "n_view", "n_union", "fold", "mdmt_test")}
    grid = E[0]["snr_grid_db"]
    j = int(np.argmin(np.abs(grid - snr_db)))
    D["snr_used"] = float(grid[j])
    D["edge_head"] = np.concatenate([e["edge_belief"][:, j] for e in E])
    cnt = np.concatenate([e["edge_count"][:, j] for e in E])
    for e, d in zip(E, L):                                  # same order as the perception cache (export)
        assert np.array_equal(e["frame"], d["frame"]) and np.array_equal(e["view_id"], d["view_id"])
    cals = json.loads((run / "metrics.json").read_text())["calibration_per_fold"]
    det = np.zeros_like(D["edge_head"])
    for k, c in enumerate(cals):
        m = D["fold"] == k
        if m.any():
            det[m] = det_level_probs(cnt[m], c["edge_a"], c["edge_b"], c["edge_s"])
    D["edge_det"] = det
    D["count_local"] = np.concatenate([d["count_pred"] for d in L]).astype(float)      # union-scale estimate
    D["count_det"] = cnt.astype(float)                                                 # view-scale detections
    if all("count_view_pred" in d for d in L):                                         
        D["count_view"] = np.concatenate([d["count_view_pred"] for d in L]).astype(float)
        D["edge_view"] = np.concatenate([e["edge_view_count"][:, j] for e in E]).astype(float)
        D["edge_mu"] = np.expm1(np.concatenate([e["edge_mu"][:, j] for e in E]).astype(float))
    D["edge_both"] = norm(D["edge_head"] * det)
    # partner view of the same time instant
    key = {(s, f, v): i for i, (s, f, v) in enumerate(zip(D["seq_id"], D["frame"], D["view_id"]))}
    D["partner"] = np.array([key.get((s, f, 1 - v), -1) for s, f, v in zip(D["seq_id"], D["frame"], D["view_id"])])
    return D


def levels_from_logcount(mu: np.ndarray, s: float) -> np.ndarray:
    """P(level) for log(1 + n_union) ~ N(mu, s^2)."""
    from math import erf, log1p, sqrt

    bounds = np.array([log1p(e - 0.5) for e in SCENE_EDGES])
    cdf = 0.5 * (1 + np.vectorize(erf)((bounds[None, :] - mu[:, None]) / (s * sqrt(2))))
    return norm(np.diff(np.concatenate([np.zeros((len(mu), 1)), cdf, np.ones((len(mu), 1))], 1), axis=1))


def count_fusion(D: dict, cols: list[np.ndarray]) -> np.ndarray:
    X = np.stack([np.log1p(np.maximum(c, 0)) for c in cols] + [np.ones(len(cols[0]))], 1)
    t = np.log1p(D["n_union"].astype(float))
    ok = D["partner"] >= 0
    out = np.zeros((len(t), len(SCENE_EDGES) + 1))
    for k in np.unique(D["fold"]):
        tr, te = (D["fold"] != k) & ok, D["fold"] == k
        w, *_ = np.linalg.lstsq(X[tr], t[tr], rcond=None)
        s = max(float(np.std(t[tr] - X[tr] @ w)), 0.05)
        out[te] = levels_from_logcount(X[te] @ w, s)
    return out


def ceilings(D: dict, m: np.ndarray) -> dict:
    p = D["partner"]
    a1, a2 = np.log1p(D["n_view"].astype(float)), np.log1p(D["n_view"][np.maximum(p, 0)].astype(float))
    t, y = np.log1p(D["n_union"].astype(float)), D["label"]
    out = {}
    for name, X in {"own count": np.stack([a1, np.ones_like(a1)], 1),
                    "both views": np.stack([a1, a2, np.ones_like(a1)], 1)}.items():
        pred = np.zeros(len(y), int)
        for k in np.unique(D["fold"]):
            tr, te = (D["fold"] != k) & (p >= 0), D["fold"] == k
            w, *_ = np.linalg.lstsq(X[tr], t[tr], rcond=None)
            pred[te] = count_bin(np.expm1(X[te] @ w), SCENE_EDGES)
        out[name] = float(np.mean(pred[m] == y[m]))
    return out


def analyse(D: dict, m: np.ndarray, gate_q: float) -> dict:
    y, seq = D["label"], D["seq_id"]
    p = D["partner"]
    ok = m & (p >= 0)
    b_loc = D["belief"]
    modes = {"LOCAL": b_loc, "A2G head": D["edge_head"], "A2G detections": D["edge_det"],
             "A2G head x detections": D["edge_both"],
             "A2A (fuse partner's local belief)": norm(b_loc * b_loc[np.maximum(p, 0)]),
             "A2A + A2G (partner's local x own edge)": norm(D["edge_both"] * b_loc[np.maximum(p, 0)])}
    q = np.maximum(p, 0)
    cl, cd = D["count_local"], D["count_det"]
    modes.update({
        "LOCAL count (refit)": count_fusion(D, [cl]),
        "A2G count fusion (own local + own detections)": count_fusion(D, [cl, cd]),
        "A2A count fusion (own + partner local counts)": count_fusion(D, [cl, cl[q]]),
        "A2A+A2G count fusion (+ own detections)": count_fusion(D, [cl, cl[q], cd]),
    })
    if "count_view" in D:                     # exchange what each UAV SEES (view count), as the ceiling does
        lv, ev, em = D["count_view"], D["edge_view"], D["edge_mu"]
        modes.update({
            "A2A view counts (own + partner)": count_fusion(D, [lv, lv[q]]),
            "A2A view + union counts (own + partner)": count_fusion(D, [cl, lv, cl[q], lv[q]]),
            "A2G all edge counts (local, det., edge head)": count_fusion(D, [cl, lv, cd, ev, em]),
            "A2A+A2G all counts (+ partner local)": count_fusion(D, [cl, lv, cd, ev, em, cl[q], lv[q]]),
        })
    corr = {k: (v.argmax(1) == y).astype(float) for k, v in modes.items()}
    lw = corr["LOCAL"] == 0
    idx = np.flatnonzero(ok)
    low = idx[np.argsort(D["delta"][idx], kind="stable")[:max(1, int(round(gate_q * len(idx))))]]
    gated = np.zeros(len(y), bool)
    gated[low] = True                                     # rank-based: ties in delta cannot inflate the share
    thr = float(D["delta"][low].max())
    res = {"n_frames": int(ok.sum()), "n_seqs": int(len(np.unique(seq[ok]))), "ceilings": ceilings(D, ok),
           "gate": {"quantile": gate_q, "delta_threshold": float(thr), "share": float(gated.sum() / ok.sum())},
           "modes": {}}
    for k, c in corr.items():
        d = c - corr["LOCAL"]
        conf = modes[k].max(1)
        res["modes"][k] = {"acc": float(c[ok].mean()), "ece": ece(conf[ok], c[ok]), "acc_ci95": scene_bootstrap(c[ok], seq[ok]),
                           "acc_on_gated": float(c[gated].mean()),
                           "p_right_given_local_wrong": float(c[ok & lw].mean()),
                           "gain_vs_local": float(d[ok].mean()), "gain_ci95": scene_bootstrap(d[ok], seq[ok]),
                           "gain_on_gated": float(d[gated].mean()),
                           "gain_on_gated_ci95": scene_bootstrap(d[gated], seq[gated])}
    res["oracle_local_or_A2G"] = float(np.maximum(corr["LOCAL"], corr["A2G head x detections"])[ok].mean())
    res["oracle_local_or_A2A"] = float(np.maximum(corr["LOCAL"], corr["A2A (fuse partner's local belief)"])[ok].mean())
    res["oracle_local_or_A2A_count"] = float(np.maximum(corr["LOCAL"], corr["A2A count fusion (own + partner local counts)"])[ok].mean())
    res["oracle_any"] = float(np.max(np.stack([corr[k] for k in corr]), 0)[ok].mean())
    return res


def table(name: str, r: dict) -> str:
    c = r["ceilings"]
    lines = [f"== {name}: {r['n_seqs']} sequences, {r['n_frames']} frames; ceilings (perfect counts): "
             f"own count {c['own count']:.3f} | both views {c['both views']:.3f}",
             f"   gate: lowest {r['gate']['quantile']:.0%} local confidence (delta <= {r['gate']['delta_threshold']:.3f})",
             f"   {'mode':48s} {'acc':>6s} {'ECE':>5s} {'95% CI':>15s} {'gain':>7s} {'95% CI':>17s} {'acc|gated':>9s} "
             f"{'gain|gated':>10s} {'95% CI':>17s} {'P(ok|loc wrong)':>15s}"]
    for k, v in r["modes"].items():
        lines.append(f"   {k:48s} {v['acc']:6.3f} {v['ece']:5.3f} [{v['acc_ci95'][0]:.3f},{v['acc_ci95'][1]:.3f}] "
                     f"{v['gain_vs_local']:+7.3f} [{v['gain_ci95'][0]:+.3f},{v['gain_ci95'][1]:+.3f}] "
                     f"{v['acc_on_gated']:9.3f} {v['gain_on_gated']:+10.3f} "
                     f"[{v['gain_on_gated_ci95'][0]:+.3f},{v['gain_on_gated_ci95'][1]:+.3f}] "
                     f"{v['p_right_given_local_wrong']:15.3f}")
    lines.append(f"   oracle: local|A2G {r['oracle_local_or_A2G']:.3f}  local|A2A {r['oracle_local_or_A2A']:.3f}  "
                 f"any {r['oracle_any']:.3f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="caches/mdmt_cf")
    ap.add_argument("--run", default="runs_perception/crossfit")
    ap.add_argument("--snr_db", type=float, default=20.0, help="edge SNR used (nearest grid point)")
    ap.add_argument("--gate_q", type=float, default=0.3)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    D = load(Path(args.cache), Path(args.run), args.snr_db)
    rep = {"snr_db": D["snr_used"], "all_scenes": analyse(D, np.ones(len(D["label"]), bool), args.gate_q),
           "mdmt_test": analyse(D, D["mdmt_test"], args.gate_q)}
    text = f"edge at {D['snr_used']:g} dB\n" + table("all scenes", rep["all_scenes"]) + "\n" + table("MDMT test", rep["mdmt_test"])
    print(text)
    out = Path(args.out or (Path(args.run) / "mode_value.json"))
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"saved {out} and {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
