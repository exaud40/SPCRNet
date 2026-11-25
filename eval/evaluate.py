# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:56:45 2026

@author: Exaud
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import Config, apply_overrides, load_config
from ..data.base import FIDELITY_KEYS, PERCEPTION_KEYS, load_cache
from ..env.uav_env import UAVEnv
from ..rl.baselines import BASELINES, TUNABLE, Policy
from .metrics import stack_infos, summarize


SCENE_ORDER = "cycle"


def run_episodes(policy: Policy, cfg: Config, pc: dict, fc: dict, n_episodes: int, seed: int = 1234) -> list[dict]:
    """Roll out ``policy`` for n_episodes full episodes, return the per-step info dicts."""
    infos: list[dict] = []
    for k in range(n_episodes):
        env = UAVEnv(cfg, pc, fc, seed=seed + k)
        env.scene_order = SCENE_ORDER
        env.set_price_and_threshold(policy.mu, policy.delta_th)
        out = env.reset()
        done = False
        while not done:
            out = env.step(policy.act(out, env))
            infos.append(out.info)
            done = out.done
    return infos


def evaluate(policy: Policy, cfg: Config, pc: dict, fc: dict, n_episodes: int = 50, seed: int = 1234):
    """Return (summary dict, stacked per-(slot, follower) arrays)."""
    infos = run_episodes(policy, cfg, pc, fc, n_episodes, seed)
    return summarize(infos, cfg.constraint.E_bar), stack_infos(infos)


def tune_threshold(cls, cfg: Config, pc: dict, fc: dict, budget_norm: float = 1.0,
                   grid=None, n_episodes: int = 10, seed: int = 99):
    """ Returns tau, qoe, energy_norm)."""
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2) if grid is None else grid
    table = []
    for tau in grid:
        s, _ = evaluate(cls(float(tau)), cfg, pc, fc, n_episodes, seed)
        table.append((float(tau), s["qoe"], s["energy_norm"]))
    feasible = [r for r in table if r[2] <= budget_norm]
    best = max(feasible, key=lambda r: r[1]) if feasible else min(table, key=lambda r: r[2])
    return best[0], table


def load_split(cache: str | Path, split: str):
    cache = Path(cache)
    return (load_cache(cache / f"perception_{split}.npz", PERCEPTION_KEYS),
            load_cache(cache / f"fidelity_{split}.npz", FIDELITY_KEYS))


def build_policies(spec: str, cfg: Config, cache: str, tune_split: str = "cal") -> list[Policy]:
    """spec: comma list of local, a2a, a2g, random, fixed, greedy, trained:<run_dir>[:deterministic].
    Trained policies are evaluated stochastically (the policy the energy constraint applies to)."""
    out: list[Policy] = []
    tune_data = None
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        key = item.split(":")[0]
        if key in BASELINES:
            out.append(BASELINES[key]())
        elif key in TUNABLE:
            tune_data = tune_data or load_split(cache, tune_split)
            tau, _ = tune_threshold(TUNABLE[key], cfg, *tune_data)
            pol = TUNABLE[key](tau)
            pol.name = f"{pol.name} (tau={tau:.2f})"
            out.append(pol)
        elif key == "trained":
            from ..rl.policy import TrainedPolicy

            parts = item.split(":")
            run_dir = ":".join(parts[1:]).replace(":deterministic", "").replace(":stochastic", "")
            env0 = UAVEnv(cfg, *load_split(cache, "test"), seed=0)
            out.append(TrainedPolicy(run_dir, env0.obs_dim, cfg.rl.hidden,
                                     deterministic=item.endswith(":deterministic")))
        else:
            raise KeyError(f"unknown policy '{item}'")
    return out


def format_table(rows: list[tuple[str, dict]]) -> str:
    head = f"{'policy':38s} {'acc':>6s} {'QoE':>6s} {'E/Ebar':>7s} {'comm':>6s} {'A2A':>6s} {'A2G':>6s}"
    lines = [head, "-" * len(head)]
    for name, s in rows:
        lines.append(f"{name[:38]:38s} {s['accuracy']:6.3f} {s['qoe']:6.3f} {s['energy_norm']:7.2f} "
                     f"{s['comm_rate']:6.1%} {s['a2a_share']:6.1%} {s['a2g_share']:6.1%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default="caches/synth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--policies", default="local,a2a,a2g,random,fixed,greedy")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--e_bar", type=float, default=None)
    ap.add_argument("--out", default=None, help="write summaries to this JSON file")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.e_bar is not None:
        cfg.constraint.E_bar = args.e_bar
    pc, fc = load_split(args.cache, args.split)
    rows = []
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass
    for pol in build_policies(args.policies, cfg, args.cache):
        s, _ = evaluate(pol, cfg, pc, fc, args.episodes, args.seed)
        rows.append((pol.name, s))
        print(f"evaluated {pol.name}", flush=True)
    print(format_table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({name: s for name, s in rows}, fh, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
