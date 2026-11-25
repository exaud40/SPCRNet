# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:55:01 2026

@author: Exaud
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..rl.baselines import AlwaysA2A, AlwaysA2G, AlwaysLocal, FixedThreshold, GreedyGate
from .evaluate import evaluate, load_split
from .experiments.common import base_parser, experiment_cfg, proposed_job

TAUS = (0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.7)
SPLITS = ("train", "cal", "test")


def cache_gain(pc: dict, fc: dict, q: float = 0.3, snr_db: float = 20.0) -> dict:
    loc = (pc["belief"].argmax(1) == pc["label"]).astype(float)
    g = int(np.argmin(np.abs(fc["snr_grid_db"] - snr_db)))
    edge = fc["edge_correct_prob"][:, g].astype(float)
    low = np.argsort(pc["delta"], kind="stable")[: int(round(q * len(loc)))]
    return {"local": float(loc.mean()), "a2g": float(edge.mean()), "gain_all": float((edge - loc).mean()),
            "gain_low_conf": float((edge[low] - loc[low]).mean()), "n_seqs": int(len(np.unique(pc["seq_id"])))}


def main(argv=None) -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--n_eval", type=int, default=30, help="episodes per policy and split")
    ap.add_argument("--runs_glob", default="", help="trained run dirs")
    args = ap.parse_args(argv)
    cfg = experiment_cfg(args)
    seeds = [int(x) for x in args.seeds.split(",")]
    runs = [proposed_job(args, s).run_dir for s in seeds]
    runs = [r for r in runs if (Path(r) / "checkpoint.pt").exists()]
    if not runs:                                     # config edited after training (e.g. the SNR grid lines)
        tag = Path(proposed_job(args, seeds[0]).run_dir).name.rsplit("_s", 1)[0]
        cands = sorted(Path(args.runs_root, Path(args.cache).name).glob(f"*/proposed/{tag}_s*/checkpoint.pt"))
        runs = [c.parent for c in cands]
        if runs:
            print(f"(config hash changed since training; using {runs[0].parent.parent})")
    if args.runs_glob:
        runs = [p.parent for p in sorted(Path().glob(args.runs_glob + "/checkpoint.pt"))]
    print(f"trained proposed runs found: {len(runs)}")
    rep = {}
    for split in SPLITS:
        pc, fc = load_split(args.cache, split)
        r = {"cache": cache_gain(pc, fc), "policies": {}}
        pols = [AlwaysLocal(), AlwaysA2A(), AlwaysA2G()] + [GreedyGate(t) for t in TAUS] + [FixedThreshold(t) for t in TAUS]
        for p in pols:
            s, _ = evaluate(p, cfg, pc, fc, args.n_eval)
            name = p.name + (f" tau={p.tau:.2f}" if hasattr(p, "tau") else "")
            r["policies"][name] = {k: float(s[k]) for k in ("qoe", "energy_norm", "a2a_share", "a2g_share")}
        if runs:
            from .experiments.common import eval_trained

            ss = [eval_trained(rd, args.cache, args.n_eval, split=split)[0] for rd in runs]
            r["policies"]["Proposed (5 seeds)"] = {k: float(np.mean([s[k] for s in ss]))
                                                   for k in ("qoe", "energy_norm", "a2a_share", "a2g_share")}
            r["policies"]["Proposed (5 seeds)"]["qoe_std"] = float(np.std([s["qoe"] for s in ss]))
        g45 = GreedyGate(0.45)
        r["eval_noise_greedy045"] = [float(evaluate(g45, cfg, pc, fc, args.n_eval, seed)[0]["qoe"]) for seed in (1234, 5678)]
        r["delta_quantiles"] = {str(q): float(np.quantile(pc["delta"], q)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)}
        rep[split] = r
        c = r["cache"]
        print(f"\n== {split} ({c['n_seqs']} seqs): cache LOCAL {c['local']:.3f}  A2G {c['a2g']:.3f}  "
              f"A2G gain all {c['gain_all']:+.3f}  on 30% least confident {c['gain_low_conf']:+.3f}")
        print(f"   {'policy':34s} {'QoE':>6s} {'E/Ebar':>7s} {'A2A':>6s} {'A2G':>6s}")
        print(f"   delta quantiles 10/30/50/70/90%: " + " ".join(f"{v:.3f}" for v in r["delta_quantiles"].values())
              + f"   | Greedy-Gate 0.45 QoE with two evaluation seeds: {r['eval_noise_greedy045'][0]:.3f} / "
              f"{r['eval_noise_greedy045'][1]:.3f}")
        for n, v in r["policies"].items():
            print(f"   {n:34s} {v['qoe']:6.3f} {v['energy_norm']:7.2f} {v['a2a_share']:6.3f} {v['a2g_share']:6.3f}")
    out = Path(args.out_dir) / "diag_shift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
