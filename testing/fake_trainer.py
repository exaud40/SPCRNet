
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ..config import apply_overrides, load_config
from ..rl.log_fields import LOG_FIELDS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    for a in ("--config", "--cache", "--split", "--reward", "--device", "--log_dir"):
        ap.add_argument(a, default=None)
    for a in ("--seed", "--iterations", "--n_envs", "--threads", "--rollout_len"):
        ap.add_argument(a, type=int, default=None)
    for a in ("--kappa", "--lam", "--e_bar", "--mu_init", "--threshold_quantile"):
        ap.add_argument(a, type=float, default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--keep_iters", default="")
    ap.add_argument("--no_threshold_adapt", action="store_true")
    ap.add_argument("--fail", action="store_true")
    args = ap.parse_args(argv)
    if args.fail:
        raise SystemExit(3)
    cfg = apply_overrides(load_config(args.config or "configs/default.yaml"), args.set)
    if args.kappa is not None:
        cfg.rl.kappa = args.kappa
    if args.lam is not None:
        cfg.rl.gae_lambda = args.lam
    if args.e_bar is not None:
        cfg.constraint.E_bar = args.e_bar
    out = Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"cfg": asdict(cfg)}, fh, default=str)
    rng = np.random.default_rng(args.seed or 0)
    n = args.iterations or cfg.rl.iterations
    it = np.arange(1, n + 1)
    rise = 1 - np.exp(-it / (20 + 3 * cfg.rl.kappa))
    mu_fixed = cfg.rl.lr_dual == 0
    with open(out / "progress.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        w.writeheader()
        for k in range(n):
            row = {f: 0.0 for f in LOG_FIELDS}
            row.update(iter=int(it[k]), qoe=0.6 + 0.2 * rise[k] + 0.005 * rng.standard_normal(),
                       accuracy=0.65 + 0.2 * rise[k], energy_norm=1.0 + 0.3 * np.exp(-it[k] / 30),
                       mu=(args.mu_init or 0.0) if mu_fixed else 0.8 * rise[k],
                       delta_th=cfg.constraint.delta_max - (0 if args.no_threshold_adapt else 0.07 * rise[k]),
                       grad_var=(1.0 / (1 + 0.15 * cfg.rl.kappa)) * (1 + 0.2 * rng.random()))
            w.writerow(row)
    (out / "checkpoint.pt").write_bytes(b"fake")
    for k in (int(x) for x in args.keep_iters.split(",") if x.strip()):
        if k <= n:
            (out / f"ckpt_iter{k}.pt").write_bytes(b"fake")


if __name__ == "__main__":
    main()
