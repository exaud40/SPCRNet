
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from ...config import Config, _fill, apply_overrides, load_config
from ..launcher import Job, is_complete, launch
from ..plotting import smooth


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default="caches/synth")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated training seeds")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=4, help="training processes run at the same time")
    ap.add_argument("--episodes", type=int, default=30, help="evaluation episodes per policy")
    ap.add_argument("--skip_train", action="store_true", help="only aggregate existing runs")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="config override applied to every job of this experiment")
    ap.add_argument("--fig_dir", default="figures")
    return ap


def run_root(args) -> Path:
 
    import json
    from dataclasses import asdict

    cfg = asdict(experiment_cfg(args))
    tag = "cfg_" + hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return Path(args.runs_root) / Path(args.cache).name / tag


def fmt_num(v: float) -> str:
    return f"{v:g}"


def proposed_job(args, seed: int, ebar_mJ: float | None = None, overrides: list[str] | None = None,
                 subdir: str = "proposed", name: str | None = None, extra: list[str] | None = None) -> Job:
    
    cfg = experiment_cfg(args)
    ebar_mJ = cfg.constraint.E_bar * 1e3 if ebar_mJ is None else ebar_mJ
    tag = f"ebar{fmt_num(ebar_mJ)}"
    run_dir = run_root(args) / subdir / f"{tag}_s{seed}"
    return train_job(name or f"{subdir}/{tag}_s{seed}", run_dir, args.cache, seed, args.iterations,
                     extra=["--e_bar", f"{ebar_mJ * 1e-3:.6g}", *(extra or [])],
                     overrides=list(args.set) + list(overrides or []), config=args.config)


def check_inputs(args) -> None:
    
    import importlib.util
    import os

    problems = []
    if not Path(args.config).exists():
        problems.append(f"config file not found: {Path(args.config).resolve()}")
    cache = Path(args.cache)
    missing = [f"{k}_{s}.npz" for k in ("perception", "fidelity") for s in ("train", "cal", "test")
               if not (cache / f"{k}_{s}.npz").exists()]
    if missing:
        problems.append(f"cache folder {cache.resolve()} lacks {missing}.\n"
                        f"  Copy your caches/ folder here, or rebuild: python -m uavsc.data.synthetic --out {args.cache}")
    if "UAVSC_TRAINER_MODULE" not in os.environ and importlib.util.find_spec("torch") is None:
        problems.append("PyTorch is not importable in this Python (activate torch_env)")
    if problems:
        raise SystemExit("Cannot start training:\n- " + "\n- ".join(problems) +
                         f"\n(working directory: {Path.cwd()}; run from the project root)")


def maybe_launch(args, jobs: list[Job]) -> None:
    check_inputs(args)
    if args.skip_train:
        missing = [j.name for j in jobs if not is_complete(j.run_dir, j.iterations)]
        if missing:
            print(f"[warn] --skip_train: {len(missing)} run(s) incomplete and ignored: {missing[:5]}...")
        return
    launch(jobs, max_parallel=args.parallel)


def complete(jobs: list[Job]) -> list[Job]:
    return [j for j in jobs if is_complete(j.run_dir, j.iterations)]


def seeds_of(args) -> list[int]:
    return [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]


def train_job(name: str, run_dir: Path | str, cache: str, seed: int, iterations: int,
              extra: list[str] | None = None, overrides: list[str] | None = None, config: str = "configs/default.yaml") -> Job:
    args = ["--config", config, "--cache", cache, "--seed", str(seed)]
    for o in overrides or []:
        args += ["--set", o]
    return Job(name=name, run_dir=str(run_dir), args=args + list(extra or []), iterations=iterations)


#  logs
def load_progress(run_dir: str | Path) -> dict[str, np.ndarray]:
    with open(Path(run_dir) / "progress.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty log in {run_dir}")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def seeds_mean_std(logs: list[dict], key: str, window: int = 10):

    n = min(len(l[key]) for l in logs)
    curves = np.stack([smooth(l[key][:n], window) for l in logs])
    return logs[0]["iter"][:n], curves.mean(0), curves.std(0)


def final_value(log: dict, key: str, frac: float = 0.1) -> float:
 
    x = log[key]
    return float(np.mean(x[-max(1, int(len(x) * frac)):]))


def iters_to_fraction(log: dict, key: str = "qoe", frac: float = 0.95, window: int = 10) -> float:
    
    y = smooth(log[key], window)
    start, final = y[0], final_value({key: y}, key)
    if final <= start:
        return float("nan")
    hit = np.flatnonzero(y >= start + frac * (final - start))
    return float(log["iter"][hit[0]]) if hit.size else float("nan")


def mean_std(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0) if v.size else (float("nan"), float("nan"))


#  configs and evaluation
def load_run_cfg(run_dir: str | Path) -> Config:

    with open(Path(run_dir) / "config.json", encoding="utf-8") as fh:
        raw = json.load(fh)["cfg"]
    return _fill(Config, raw)


def experiment_cfg(args) -> Config:
    return apply_overrides(load_config(args.config), args.set)


def ebar_list(args) -> list[float]:

    base = experiment_cfg(args).constraint.E_bar * 1e3
    out = []
    for t in str(args.ebar_mJ).split(","):
        t = t.strip()
        if t:
            out.append(round(float(t[:-1]) * base, 6) if t.endswith("x") else float(t))
    return out


def eval_trained(run_dir: str | Path, cache: str, episodes: int, seed: int = 1234, split: str = "test",
                 deterministic: bool = False):
  
    import torch

    torch.manual_seed(seed)
    from ...env.uav_env import UAVEnv
    from ...rl.policy import TrainedPolicy
    from ..evaluate import evaluate, load_split

    cfg = load_run_cfg(run_dir)
    pc, fc = load_split(cache, split)
    obs_dim = UAVEnv(cfg, pc, fc, seed=0).obs_dim
    pol = TrainedPolicy(run_dir, obs_dim, cfg.rl.hidden, deterministic=deterministic)
    summary, arrays = evaluate(pol, cfg, pc, fc, episodes, seed)
    summary["mu_final"], summary["delta_th_final"] = pol.mu, pol.delta_th
    return summary, arrays, cfg


def evaluate_runs(run_dirs: list, cache: str, episodes: int, evaluator=None):
  
    evaluator = evaluator or eval_trained
    sums, arrs, cfg = [], [], None
    for rd in run_dirs:
        s, a, c = evaluator(rd, cache, episodes)
        sums.append(s)
        arrs.append(a)
        cfg = cfg or c
    keys = set.intersection(*(set(a) for a in arrs)) if arrs else set()
    cat = {k: np.concatenate([np.ravel(a[k]) for a in arrs]) for k in keys}
    return sums, cat, cfg


def agg(summaries: list[dict], keys=("qoe", "accuracy", "energy_norm", "energy_J", "comm_rate",
                                     "a2a_share", "a2g_share", "mu_final", "delta_th_final")) -> dict:
    
    out = {}
    for k in keys:
        if summaries and all(k in s for s in summaries):
            out[k] = mean_std([s[k] for s in summaries])
    out["n_seeds"] = len(summaries)
    return out


#  outputs
def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
    return path


def latex_table(rows: list[dict], columns: list[tuple[str, str, str]], caption: str, label: str) -> str:

    def cell(v, fmt):
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return fmt.format(v[0]) + r"$\pm$" + fmt.format(v[1])
        return fmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v)

    head = " & ".join(["Method"] + [h for _, h, _ in columns]) + r" \\"
    body = [" & ".join([r["name"]] + [cell(r[k], f) for k, _, f in columns]) + r" \\" for r in rows]
    return "\n".join([r"\begin{table}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\footnotesize", r"\begin{tabular}{@{}l" + "c" * len(columns) + r"@{}}", r"\toprule", head,
                      r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
