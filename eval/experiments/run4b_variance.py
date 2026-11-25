from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np

from ..plotting import COL_W, DBL_W, save, set_style
from .common import (base_parser, complete, latex_table, load_run_cfg, maybe_launch, mean_std, proposed_job,
                     seeds_of, write_json)

import torch
from ...rl.advantage import td_residuals, wgae
from ...rl.wgae_ppo import Trainer, make_env

KAPPAS = (2.0, 4.0, 8.0)
PROBE_ITERS = (10, 25, 50, 100)
COLORS = {"MC": "#7d3c98", "GAE": "#2471a3", "WGAE k=2": "#e59866", "WGAE k=4": "#c0392b", "WGAE k=8": "#78281f"}


def estimators(lam: float) -> list[tuple[str, float, float]]:
    return [("MC", 1.0, 0.0), ("GAE", lam, 0.0)] + [(f"WGAE k={k:g}", lam, k) for k in KAPPAS]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _snr(g: np.ndarray) -> tuple[float, float]:
    """(bias-corrected SNR, variance) of the mean of K gradient samples g (K, P)."""
    K, mean = g.shape[0], g.mean(0)
    var = float(((g - mean) ** 2).sum(1).mean())
    return (max(float(mean @ mean) - var / K, 0.0) / var if var > 0 else float("nan")), var


def split_half_cos(g: np.ndarray, g_ref: np.ndarray) -> tuple[float, float, float]:
    """Direction agreement free of shared rollout noise: cos between the estimator's mean over one half of
    the rollouts and the reference mean over the OTHER half (averaged over both pairings), plus each one's
    self-agreement between halves. Returns (cross, self_est, self_ref)."""
    e, o = slice(0, None, 2), slice(1, None, 2)
    cross = 0.5 * (_cos(g[e].mean(0), g_ref[o].mean(0)) + _cos(g[o].mean(0), g_ref[e].mean(0)))
    return cross, _cos(g[e].mean(0), g[o].mean(0)), _cos(g_ref[e].mean(0), g_ref[o].mean(0))


def disattenuated(cross: float, self_a: float, self_b: float) -> float:
    """Noise-corrected direction agreement (Spearman's correction): cross / sqrt(self_a * self_b).
    NaN when either self-agreement is not positive (the direction itself is not resolved)."""
    if self_a <= 0 or self_b <= 0:
        return float("nan")
    return float(np.clip(cross / np.sqrt(self_a * self_b), -1.0, 1.0))


def grad_stats(grads: dict[str, np.ndarray], grads_norm: dict[str, np.ndarray] | None = None,
               ref: str = "GAE", mc: str = "MC") -> dict:

    gd = grads_norm if grads_norm is not None else grads
    out = {}
    m_mc = grads[mc].mean(0)
    for name, g in grads.items():
        _, var = _snr(g)
        snr, _ = _snr(gd[name])
        cross, s_est, s_mc = split_half_cos(gd[name], gd[mc])
        out[name] = {"raw_var": var, "mean_norm": float(np.linalg.norm(g.mean(0))), "snr": snr,
                     "cos_to_mc": _cos(g.mean(0), m_mc), "cos_split": cross, "self_cos": s_est,
                     "cos_disatt": disattenuated(cross, s_est, s_mc)}
        if grads_norm is not None:
            out[name]["norm_var"] = _snr(grads_norm[name])[1]
    for name in out:
        out[name]["raw_var_reduction"] = 1.0 - out[name]["raw_var"] / out[ref]["raw_var"]
        out[name]["snr_gain"] = out[name]["snr"] / out[ref]["snr"] if out[ref]["snr"] > 0 else float("nan")
        if grads_norm is not None:
            out[name]["norm_var_reduction"] = 1.0 - out[name]["norm_var"] / out[ref]["norm_var"]
    K = grads[mc].shape[0]
    out["_mc_ref_snr"] = float(m_mc @ m_mc / (out[mc]["raw_var"] / K))
    out["_mc_self_cos"] = out[mc]["self_cos"]
    return out


def probe(run_dir: str | Path, cache: str, n_rollouts: int = 64, seed: int = 0, ckpt: str = "checkpoint.pt",
          split: str = "train") -> dict:
    
    cfg = load_run_cfg(run_dir)
    envs = [make_env(cfg, cache, split, seed=50_000 + 100 * seed + i) for i in range(cfg.rl.n_envs)]
    tr = Trainer(cfg, envs, tempfile.mkdtemp(prefix="run4b_"), seed=seed, threads=1)
    state = torch.load(Path(run_dir) / ckpt, map_location="cpu")
    tr.actor.load_state_dict(state["actor"])
    tr.critic.load_state_dict(state["critic"])
    tr.mu, tr.delta_th = float(state["mu"]), float(state["delta_th"])
    params = [p for p in tr.actor.parameters() if p.requires_grad]
    ests = estimators(cfg.rl.gae_lambda)
    raw: dict[str, list] = {n: [] for n, _, _ in ests}
    nrm: dict[str, list] = {n: [] for n, _, _ in ests}
    gamma = cfg.rl.gamma_disc

    def flat_grad(loss):
        gr = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten()
                          for x, p in zip(gr, params)]).cpu().numpy()

    for _ in range(n_rollouts):
        tr._outs = None
        tr.collect()
        b = tr.buf
        O = b.obs.shape[-1]
        logp, _, _ = tr.actor.evaluate(tr._t(b.obs.reshape(-1, O)), tr._t(b.mask.reshape(-1, 3), torch.bool),
                                       tr._t(b.mode.reshape(-1), torch.long), tr._t(b.z.reshape(-1, 2)))
        for name, lam, kappa in ests:
            adv = np.zeros((b.T, b.N, b.M))
            for e in range(b.N):
                res = td_residuals(b.reward[:, e], b.values_ext[:, e], b.done[:, e], gamma)
                adv[:, e] = wgae(res, b.done[:, e], b.delta_ext[:, e], gamma, lam, kappa)
            a = adv.reshape(-1)
            raw[name].append(flat_grad(-(logp * tr._t(a)).mean()))
            nrm[name].append(flat_grad(-(logp * tr._t((a - a.mean()) / (a.std() + 1e-8))).mean()))
    return grad_stats({k: np.stack(v) for k, v in raw.items()}, {k: np.stack(v) for k, v in nrm.items()})


def aggregate(per_ckpt: list[dict]) -> dict:
    names = [n for n in per_ckpt[0] if not n.startswith("_")]
    keys = [k for k in ("raw_var_reduction", "norm_var_reduction", "snr", "snr_gain", "cos_to_mc", "cos_split",
                        "self_cos", "cos_disatt") if k in per_ckpt[0][names[0]]]
    out = {n: {k: mean_std([p[n][k] for p in per_ckpt]) for k in keys} for n in names}
    for k in ("_mc_ref_snr", "_mc_self_cos"):
        if k in per_ckpt[0]:
            out[k] = mean_std([p[k] for p in per_ckpt])
    return out


EARLY, LATE = (10, 25), (50, 100)


def plot(by_iter: dict[int, dict], fig_dir) -> Path:
  
    plt = set_style()
    its = sorted(by_iter)
    names = [n for n in by_iter[its[0]] if not n.startswith("_")]
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.1), constrained_layout=True)
    panels = [("raw_var_reduction", "Grad. variance reduction\nvs GAE (raw advantages)"),
              ("self_cos", "Direction consistency\n(split-half self-cosine)"),
              ("cos_split", "Agreement with MC direction\n(split-half cross-cosine)")]
    for ax, (key, ylabel) in zip(axes, panels):
        for n in names:
            if n == "MC" and key != "self_cos":
                continue
            if key not in by_iter[its[0]][n]:
                continue
            m = np.array([by_iter[i][n][key][0] for i in its])
            sd = np.array([by_iter[i][n][key][1] for i in its])
            lab = n.replace("WGAE k=", r"WGAE $\kappa$=")
            ax.errorbar(its, m, yerr=sd, color=COLORS.get(n, "k"), marker="o", ms=3, capsize=2, label=lab)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(its, [str(i) for i in its])
        ax.set_xlabel("Checkpoint (training iteration)")
        ax.set_ylabel(ylabel)
        if key == "cos_split":
            ax.axhline(0.0, color="k", lw=0.6)
    axes[1].legend(fontsize=6)
    path = save(fig, Path(fig_dir) / "run4b_variance")
    plt.close(fig)
    return path


def table(per_iter: dict[int, list], out_dir) -> str:
    
    def pool(its):
        lst = [p for i in its if i in per_iter for p in per_iter[i]]
        return aggregate(lst) if lst else None

    early, late = pool(EARLY), pool(LATE)
    ref = early or late
    rows = []
    for n in [k for k in ref if not k.startswith("_")]:
        r = {"name": n.replace("WGAE k=", r"WGAE ($\kappa$=") + (")" if n.startswith("WGAE") else "")}
        for tag, agg_ in (("e", early), ("l", late)):
            r["rv" + tag] = tuple(100 * x for x in agg_[n]["raw_var_reduction"]) if agg_ else "--"
            r["sc" + tag] = agg_[n]["self_cos"] if agg_ else "--"
            r["cs" + tag] = agg_[n]["cos_split"] if agg_ and n != "MC" else "--"
        rows.append(r)
    tex = latex_table(rows, [("rve", r"Var.\ red.\ [\%] (E)", "{:.0f}"), ("sce", "Consist. (E)", "{:.2f}"),
                             ("cse", r"Agree.\ MC (E)", "{:.2f}"), ("scl", "Consist. (L)", "{:.2f}"),
                             ("csl", r"Agree.\ MC (L)", "{:.2f}")],
                      "Policy-gradient statistics of the advantage estimators")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "run4b_table.tex").write_text(tex, encoding="utf-8")
    return tex


def report(per_iter: dict[int, list], args) -> dict:
    by_iter = {it: aggregate(per) for it, per in sorted(per_iter.items()) if per}
    plot(by_iter, args.fig_dir)
    table(per_iter, args.out_dir)
    for it, a in by_iter.items():
        print(f"\ncheckpoint iter {it}  (MC self-consistency {a['_mc_self_cos'][0]:+.2f})")
        for n, v in a.items():
            if n.startswith("_"):
                continue
            print(f"  {n:10s} var red raw {v['raw_var_reduction'][0]:+6.0%}  "
                  f"norm {v.get('norm_var_reduction', (float('nan'),))[0]:+6.0%}  "
                  f"consistency {v['self_cos'][0]:+.2f}  agreement with MC {v['cos_split'][0]:+.2f}")
    return by_iter


def main(argv=None, prober=None) -> dict:
    ap = base_parser(__doc__)
    ap.add_argument("--rollouts", type=int, default=64, help="independent rollouts per checkpoint")
    ap.add_argument("--probe_iters", default=",".join(map(str, PROBE_ITERS)),
                    help="checkpoints (training iterations) to probe")
    ap.add_argument("--run_dirs", default="", help="probe these runs' final checkpoint.pt instead of training")
    ap.add_argument("--from_json", default="", help="only re-plot / re-tabulate an existing run4b.json")
    ap.set_defaults(seeds="0,1,2", iterations=max(PROBE_ITERS))
    args = ap.parse_args(argv)
    if args.from_json:
        import json

        with open(args.from_json, encoding="utf-8") as fh:
            per_iter = {int(k): v for k, v in json.load(fh)["per_checkpoint"].items()}
        return report(per_iter, args)
    its = sorted(int(x) for x in args.probe_iters.split(",") if x.strip())
    args.iterations = max(its)                     # the probe runs only need to reach the last checkpoint
    if args.run_dirs:
        targets = {0: [(d, "checkpoint.pt") for d in args.run_dirs.split(",") if d]}
    else:
        # short trainings that keep intermediate checkpoints (ckpt_iter<k>.pt)
        jobs = [proposed_job(args, s, subdir="run4b", extra=["--keep_iters", ",".join(map(str, its))])
                for s in seeds_of(args)]
        maybe_launch(args, jobs)
        done = complete(jobs)
        targets = {i: [(j.run_dir, f"ckpt_iter{i}.pt") for j in done if (Path(j.run_dir) / f"ckpt_iter{i}.pt").exists()]
                   for i in its}
    prober = prober or (lambda d, ck, s: probe(d, args.cache, args.rollouts, seed=s, ckpt=ck))
    per_iter = {}
    for it, lst in targets.items():
        per = []
        for s, (d, ck) in enumerate(lst):
            print(f"[run4b] {d} {ck}: {args.rollouts} rollouts", flush=True)
            per.append(prober(d, ck, s))
        if per:
            per_iter[it] = per
    if not per_iter:
        raise SystemExit("no checkpoints to probe")
    write_json({"rollouts": args.rollouts, "targets": {str(k): v for k, v in targets.items()},
                "per_checkpoint": {str(k): v for k, v in per_iter.items()},
                "aggregate": {str(k): aggregate(v) for k, v in per_iter.items()}}, Path(args.out_dir) / "run4b.json")
    return report(per_iter, args)


if __name__ == "__main__":
    main()
