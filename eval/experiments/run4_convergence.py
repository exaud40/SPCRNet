
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..plotting import COL_W, DBL_W, line_with_band, save, set_style
from .common import (base_parser, complete, final_value, iters_to_fraction, latex_table, load_progress,
                     maybe_launch, mean_std, proposed_job, run_root, seeds_mean_std, seeds_of, train_job,
                     write_json)

_CONST = ["--set", "rl.kappa_anneal_iters=0"]
VARIANTS = [
    ("ppo", "PPO", ["--lam", "1.0", "--kappa", "0"], "#7d3c98", "^"),
    ("gae", "GAE-PPO", ["--kappa", "0"], "#2471a3", "s"),
    ("wgae_k2", r"WGAE-PPO ($\kappa$=2)", _CONST, "#e59866", "v"),
    ("wgae_k4", r"WGAE-PPO ($\kappa$=4)", ["--kappa", "4", *_CONST], "#f0b27a", "<"),
    ("wgae_k8", r"WGAE-PPO ($\kappa$=8)", ["--kappa", "8", *_CONST], "#78281f", "D"),
    ("wgae_k2_a50", r"WGAE-PPO ($\kappa$: 2$\to$0, 50 it.)", ["--set", "rl.kappa_anneal_iters=50"], "#117a65", "P"),
    ("wgae_k2_a100", r"WGAE-PPO ($\kappa$: 2$\to$0, 100 it.)", ["--set", "rl.kappa_anneal_iters=100"], "#1f618d", "X"),
    ("proposed", r"WGAE-PPO ($\kappa$: 2$\to$0, 200 it.)", None, "#c0392b", "o"),
]
DEFAULT_VARIANTS = "ppo,gae,wgae_k2,wgae_k4,wgae_k8,proposed"
ANNEAL_VARIANTS = "gae,wgae_k2,wgae_k2_a50,wgae_k2_a100,proposed"


def selected(args) -> list:
    want = [k.strip() for k in getattr(args, "variants", DEFAULT_VARIANTS).split(",") if k.strip()]
    return [v for v in VARIANTS if v[0] in want]


def jobs_for(args) -> dict[str, list]:
    out = {}
    for key, _, extra, _, _ in selected(args):
        js = []
        for s in seeds_of(args):
            if extra is None:
                js.append(proposed_job(args, s))
            else:
                rd = run_root(args) / "run4" / f"{key}_s{s}"
                js.append(train_job(f"run4/{key}_s{s}", rd, args.cache, s, args.iterations, extra=extra,
                                    overrides=args.set, config=args.config))
        out[key] = js
    return out


def analyze(jobs: dict[str, list]) -> dict:
    res = {}
    for key, label, *_ in VARIANTS:
        if key not in jobs:
            continue
        logs = [load_progress(j.run_dir) for j in complete(jobs[key])]
        if not logs:
            continue
        res[key] = {
            "label": label, "n_seeds": len(logs),
            "qoe_final": mean_std([final_value(l, "qoe") for l in logs]),
            "accuracy_final": mean_std([final_value(l, "accuracy") for l in logs]),
            "energy_norm_final": mean_std([final_value(l, "energy_norm") for l in logs]),
            "mu_final": mean_std([final_value(l, "mu") for l in logs]),
            "delta_th_final": mean_std([final_value(l, "delta_th") for l in logs]),
            "iters_to_95": mean_std([iters_to_fraction(l, "qoe", 0.95) for l in logs]),
            "grad_var_mean": mean_std([float(np.mean(l["grad_var"])) for l in logs]),
            "_logs": logs,
        }
    if "gae" in res:
        base = res["gae"]["grad_var_mean"][0]
        for r in res.values():
            r["grad_var_reduction_vs_gae"] = float(1.0 - r["grad_var_mean"][0] / base) if base > 0 else float("nan")
    return res


def recommend(clean: dict) -> str:
  
    cands = {k: v for k, v in clean.items() if k.startswith("wgae_k2") or k == "proposed"}
    if len(cands) < 2 or "wgae_k2" not in cands:
        return ""
    best = max(cands, key=lambda k: cands[k]["qoe_final"][0])
    base = cands["wgae_k2"]
    lines = ["Annealing comparison (final QoE, iterations to 95%):"]
    for k, v in clean.items():
        lines.append(f"  {k:14s} {v['qoe_final'][0]:.4f} +- {v['qoe_final'][1]:.4f}   {v['iters_to_95'][0]:.0f} it.")
    d = cands[best]["qoe_final"][0] - base["qoe_final"][0]
    noise = (cands[best]["qoe_final"][1] ** 2 + base["qoe_final"][1] ** 2) ** 0.5
    if best == "wgae_k2" or d <= 0:
        lines.append("RECOMMENDATION: keep constant kappa = 2 (no annealed schedule beats it).")
    elif d < noise:
        lines.append(f"RECOMMENDATION: {best} is better by {d:+.4f}, within noise ({noise:.4f}); keep constant "
                     f"kappa = 2 unless more seeds confirm it.")
    else:
        lines.append(f"RECOMMENDATION: adopt {best} (final QoE {d:+.4f} over constant kappa = 2, noise {noise:.4f}).")
    return "\n".join(lines)


def plot(res: dict, fig_dir: str | Path, window: int = 10, tag: str = "run4") -> list[Path]:
    plt = set_style()
    styles = {k: (lab, c, m) for k, lab, _, c, m in VARIANTS}
    paths = []
  
    for key_log, ylabel, fname, logy in [("qoe", "QoE", f"{tag}_qoe", False)]:
        fig, ax = plt.subplots(figsize=(COL_W, 2.4))
        for k, r in res.items():
            it, m, s = seeds_mean_std(r["_logs"], key_log, window)
            lab, c, mk = styles[k]
            line_with_band(ax, it, m, s, lab, color=c, marker=mk)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(ncol=1, loc="lower right", fontsize=6)
        paths.append(save(fig, Path(fig_dir) / fname))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.1), constrained_layout=True)
    for ax, (key_log, ylabel) in zip(axes, [("energy_norm", r"$\bar{E}_u/\bar{E}$"), ("mu", r"Energy price $\mu$"),
                                            ("delta_th", r"Threshold $\delta_{\rm th}$")]):
        for k, r in res.items():
            it, m, s = seeds_mean_std(r["_logs"], key_log, window)
            lab, c, mk = styles[k]
            line_with_band(ax, it, m, s, lab, color=c, marker=mk)
        if key_log == "energy_norm":
            ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=6)
    paths.append(save(fig, Path(fig_dir) / f"{tag}_constraint"))
    plt.close(fig)
    return paths


def main(argv=None) -> dict:
    ap = base_parser(__doc__)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"comma list of variant keys; 'anneal' = {ANNEAL_VARIANTS}")
    ap.add_argument("--tag", default="run4", help="prefix of the output files")
    args = ap.parse_args(argv)
    if args.variants == "anneal":
        args.variants = ANNEAL_VARIANTS
    jobs = jobs_for(args)
    maybe_launch(args, [j for js in jobs.values() for j in js])
    res = analyze(jobs)
    if not res:
        raise SystemExit("no complete runs to analyze")
    plot(res, args.fig_dir, tag=args.tag)
    clean = {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in res.items()}
    write_json(clean, Path(args.out_dir) / f"{args.tag}.json")
    rows = [{"name": r["label"], "q": r["qoe_final"], "it": r["iters_to_95"], "e": r["energy_norm_final"]}
            for r in clean.values()]
    tex = latex_table(rows, [("q", "QoE", "{:.3f}"), ("it", "Iter. to 95\\%", "{:.0f}"),
                             ("e", r"$\bar E_u/\bar E$", "{:.2f}")],
                      "Convergence of the advantage estimators")
    Path(args.out_dir, f"{args.tag}_table.tex").write_text(tex, encoding="utf-8")
    rec = recommend(clean)
    if rec:
        print("\n" + rec)
    for r in clean.values():
        print(f"{r['label']:24s} QoE {r['qoe_final'][0]:.3f}+-{r['qoe_final'][1]:.3f}  "
              f"iters95 {r['iters_to_95'][0]:.0f}  E/Ebar {r['energy_norm_final'][0]:.2f}")
    return clean


if __name__ == "__main__":
    main()
