# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:28:55 2026

@author: Exaud
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..comm.fidelity import fit_fidelity, logistic_fidelity
from ..data.base import FIDELITY_KEYS, PERCEPTION_KEYS, load_cache, save_cache
from ..data.mdmt import NUM_CLASSES, read_frames
from ..perception.calibration import ece, fit_temperature
from ..perception.crossfit import det_level_probs, fit_edge_map, pool, softmax_T

SIM = ("train", "cal", "test")


# A2G
def _fit_edge_at(P: dict, m: np.ndarray, j: int) -> dict:
    nv, nu, y = P["n_view"][m], P["n_union"][m], P["label"][m]
    errs = np.abs(P["counts"][m, j, :] - nv[:, None]).mean(0)
    ti = int(np.argmin(errs))
    a, b, s = fit_edge_map(P["counts"][m, j, ti], nu)
    T_e = fit_temperature(P["edge_logits"][m, j], y)
    lp = np.log(np.clip(softmax_T(P["edge_logits"][m, j], T_e) *
                        det_level_probs(P["counts"][m, j, ti], a, b, s), 1e-12, None))
    T_p = fit_temperature(lp, y)
    return {"tau_i": ti, "a": a, "b": b, "s": s, "T_e": T_e, "T_p": T_p}


def edge_beliefs(P: dict, K: int) -> tuple[np.ndarray, list]:
    """(N, G, C) calibrated A2G beliefs at every grid SNR (index 1.. of the prediction axis), nested over folds."""
    N, S = P["edge_logits"].shape[:2]
    out = np.zeros((N, S - 1, NUM_CLASSES))
    cals = []
    for k in range(K):
        mk = P["fold"] == k
        if not mk.any():
            continue
        for j in range(1, S):
            c = _fit_edge_at(P, ~mk, j)
            cals.append({"fold": k, "grid_index": j - 1, **c})
            prod = softmax_T(P["edge_logits"][mk, j], c["T_e"]) * det_level_probs(P["counts"][mk, j, c["tau_i"]],
                                                                                 c["a"], c["b"], c["s"])
            out[mk, j - 1] = softmax_T(np.log(np.clip(prod, 1e-12, None)), c["T_p"])
    return out, cals


#  A2A
def belief_channel(b: np.ndarray, delta: np.ndarray, grid_db: np.ndarray, k_b: int, draws: int, scale: dict,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analog transmission of (centred log b, delta) over k_b complex symbols at each grid SNR.
    Returns belief_hat (S,G,C), delta_hat (S,G), fidelity 1 - TV (S,G), means over draws."""
    rng = np.random.default_rng(seed)
    S, C = b.shape
    lb = np.log(np.clip(b, 1e-6, None))
    lb = lb - lb.mean(1, keepdims=True)
    x = np.concatenate([lb / scale["logit"], (delta[:, None] - scale["delta_mean"]) / scale["delta"]], 1)  # unit power
    reps = max(1, (2 * k_b) // (C + 1))                    # repetitions of each real over the 2 k_b reals
    bh, dh, fid = (np.zeros((S, len(grid_db), C)), np.zeros((S, len(grid_db))), np.zeros((S, len(grid_db))))
    for g, snr_db in enumerate(grid_db):
        std = 1.0 / np.sqrt(reps * 10 ** (snr_db / 10))      # per-real noise after combining the repetitions
        for _ in range(draws):
            xr = x + rng.normal(0, std, x.shape)
            l = xr[:, :C] * scale["logit"]
            p = np.exp(l - l.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            d = np.clip(xr[:, C] * scale["delta"] + scale["delta_mean"], 1.0 / C, 1.0)
            bh[:, g] += p / draws
            dh[:, g] += d / draws
            fid[:, g] += (1.0 - 0.5 * np.abs(p - b).sum(1)) / draws
    return bh, dh, fid


#  main
def build(run: Path, cache: Path, frames_csv: Path, K: int, k_b: int, draws: int, seed: int = 0,
          psi_fit: str = "train+cal") -> dict:
    frames = read_frames(frames_csv)
    frames = [dict(r, split={"train": "train", "heldout": "train", "cal": "val", "test": "test"}[r["our_split"]])
              for r in frames]
    P, R = pool(run, frames, K)
    grid = np.load(cache / "edge_raw" / "edge_test.npz")["snr_grid_db"].astype(float)   # prediction SNRs 1..
    if P["edge_logits"].shape[1] != len(grid) + 1:
        raise SystemExit("SNR grid of the edge caches does not match the fold predictions")
    eb, cals = edge_beliefs(P, K)
    ec = (eb.argmax(-1) == P["label"][:, None]).astype(np.float32)          # (N, G)
    key = {(s, int(f), int(v)): i for i, (s, f, v) in enumerate(zip(P["seq"], P["frame"], P["view"]))}

    pcs = {s: load_cache(cache / f"perception_{s}.npz", PERCEPTION_KEYS) for s in SIM}
    cal_b = pcs["cal"]["belief"].astype(float)
    lb = np.log(np.clip(cal_b, 1e-6, None))
    lb -= lb.mean(1, keepdims=True)
    dcal = pcs["cal"]["delta"].astype(float)
    scale = {"logit": float(np.sqrt((lb ** 2).mean())), "delta_mean": float(dcal.mean()),
             "delta": float(max(dcal.std(), 1e-3))}

    out, curves = {}, {}
    for s in SIM:
        pc = pcs[s]
        idx = np.array([key[(q, int(f), int(v))] for q, f, v in zip(pc["seq_id"], pc["frame"], pc["view_id"])])
        bh, dh, fid = belief_channel(pc["belief"].astype(float), pc["delta"].astype(float), grid, k_b, draws, scale,
                                     seed + SIM.index(s))
        out[s] = {"snr_grid_db": grid, "edge_correct_prob": ec[idx], "edge_belief": eb[idx].astype(np.float32),
                  "belief_hat": bh.astype(np.float32), "delta_hat": dh.astype(np.float32),
                  "belief_fidelity": fid.astype(np.float32)}
        curves[s] = {"edge_acc": ec[idx].mean(0).tolist(), "belief_fidelity": fid.mean(0).tolist(),
                     "local_acc": float((pc["belief"].argmax(1) == pc["label"]).mean()),
                     "edge_ece": [ece(eb[idx, g].max(1), ec[idx, g]) for g in range(len(grid))]}
    fit_on = psi_fit.split("+")                    # all splits are out of fold; never fit on 'test'
    n = {s: len(out[s]["edge_correct_prob"]) for s in fit_on}
    pooled = lambda k: sum(np.array(curves[s][k]) * n[s] for s in fit_on) / sum(n.values())
    psi_s, r2_s = fit_fidelity(grid, pooled("edge_acc"))
    psi_b, r2_b = fit_fidelity(grid, pooled("belief_fidelity"))
    for s in SIM:
        out[s]["psi_s"], out[s]["psi_b"] = psi_s, psi_b
        save_cache(cache / f"fidelity_{s}.npz", out[s], FIDELITY_KEYS)
    rep = {"psi_fit_on": psi_fit, "snr_grid_db": grid.tolist(), "k_b": k_b, "draws": draws, "belief_scale": scale, "curves": curves,
           "psi_s": psi_s.tolist(), "psi_s_r2": r2_s, "psi_b": psi_b.tolist(), "psi_b_r2": r2_b,
           "psi_s_on_grid": logistic_fidelity(grid, psi_s).tolist(), "psi_b_on_grid": logistic_fidelity(grid, psi_b).tolist(),
           "edge_calibration": cals}
    return rep


def text(rep: dict) -> str:
    g = rep["snr_grid_db"]
    lines = [f"SNR grid [dB]          " + " ".join(f"{x:7.1f}" for x in g)]
    for s, c in rep["curves"].items():
        lines.append(f"{s:5s} A2G edge acc     " + " ".join(f"{x:7.3f}" for x in c["edge_acc"])
                     + f"   (LOCAL {c['local_acc']:.3f})")
        lines.append(f"{s:5s} A2G edge ECE     " + " ".join(f"{x:7.3f}" for x in c["edge_ece"]))
        lines.append(f"{s:5s} A2A 1-TV(b^,b)   " + " ".join(f"{x:7.3f}" for x in c["belief_fidelity"]))
    lines.append(f"Psi fitted on: {rep['psi_fit_on']}")
    lines.append(f"psi_s = {np.round(rep['psi_s'], 3).tolist()}  R^2 {rep['psi_s_r2']:.3f}  -> "
                 + " ".join(f"{x:.3f}" for x in rep["psi_s_on_grid"]))
    lines.append(f"psi_b = {np.round(rep['psi_b'], 3).tolist()}  R^2 {rep['psi_b_r2']:.3f}  -> "
                 + " ".join(f"{x:.3f}" for x in rep["psi_b_on_grid"]))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs_perception/crossfit")
    ap.add_argument("--cache", default="caches/mdmt_cf")
    ap.add_argument("--frames", default="caches/mdmt/frames.csv")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--k_b", type=int, default=8, help="A2A belief payload [complex symbols] (configs/default.yaml)")
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psi_fit", default="train+cal", choices=["train+cal", "train", "cal"],
                    help="splits whose mean curves the Psi fits use (out of fold; 5-scene 'cal' alone is noisy)")
    args = ap.parse_args(argv)
    rep = build(Path(args.run), Path(args.cache), Path(args.frames), args.folds, args.k_b, args.draws, args.seed,
                args.psi_fit)
    t = text(rep)
    print(t)
    (Path(args.cache) / "fidelity_report.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    (Path(args.cache) / "fidelity_report.txt").write_text(t, encoding="utf-8")
    print(f"wrote fidelity_{{train,cal,test}}.npz and fidelity_report.txt to {args.cache}")


if __name__ == "__main__":
    main()
