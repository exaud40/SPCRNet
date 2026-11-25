
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..plotting import COL_W, save, set_style
from .common import (agg, base_parser, complete, evaluate_runs, final_value, latex_table, load_progress,
                     maybe_launch, mean_std, proposed_job, seeds_of, write_json)

# key -> (table label, trainer extra args, config overrides)
VARIANTS = {
    "full": ("Proposed (full)", [], []),
    "no_gate": ("w/o confidence gate", ["--no_threshold_adapt"], ["constraint.delta_max=1.01"]),
    "fixed_th": (r"Fixed $\delta_{\rm th}$", ["--no_threshold_adapt"], []),
    "conf_msp": ("MSP confidence", [], ["env.confidence_source=msp"]),
    "conf_raw": ("Uncalibrated confidence", [], ["env.confidence_source=raw"]),
    "a2a_only": ("A2A only", [], ["env.allowed_modes=0,1"]),
    "a2g_only": ("A2G only", [], ["env.allowed_modes=0,2"]),
    "model_reward": ("Model-based reward", ["--reward", "model"], []),
    "gae": (r"GAE ($\kappa$=0)", ["--kappa", "0"], []),
    "const_kappa": (r"Constant $\kappa$ (no annealing)", [], ["rl.kappa_anneal_iters=0"]),
}


def msp_equals_confidence(cache: str) -> bool:
    from ...data.base import PERCEPTION_KEYS, load_cache

    pc = load_cache(Path(cache) / "perception_test.npz", PERCEPTION_KEYS)
    return bool(np.allclose(pc["belief"].max(axis=1), pc["delta"], atol=1e-6))


def variant_jobs(args, key: str) -> list:
    label, extra, ov = VARIANTS[key]
    if key == "full":
        return [proposed_job(args, s) for s in seeds_of(args)]
    return [proposed_job(args, s, overrides=ov, extra=extra, subdir=f"run7/{key}") for s in seeds_of(args)]


def fixed_mu_jobs(args, mu_star: float) -> dict[str, list]:
    out = {}
    for f in [float(x) for x in args.fixed_mu.split(",") if x.strip()]:
        key = f"fixed_mu_x{f:g}"
        out[key] = [proposed_job(args, s, overrides=["rl.lr_dual=0"], extra=["--mu_init", f"{f * mu_star:.6g}"],
                                 subdir=f"run7/{key}") for s in seeds_of(args)]
    return out


def mu_star_of(jobs: list) -> float:
    logs = [load_progress(j.run_dir) for j in complete(jobs)]
    return mean_std([final_value(l, "mu") for l in logs])[0] if logs else float("nan")


def analyze(args, jobs: dict[str, list], evaluator=None) -> dict:
    res = {}
    for key, js in jobs.items():
        done = complete(js)
        if not done:
            continue
        sums, _, _ = evaluate_runs([j.run_dir for j in done], args.cache, args.episodes, evaluator)
        a = agg(sums)
        a["violation"] = mean_std([max(0.0, s["energy_norm"] - 1.0) for s in sums])
        label = VARIANTS[key][0] if key in VARIANTS else rf"Fixed $\mu$ = {key.split('_x')[1]}$\mu^\star$"
        res[key] = {"label": label, **a}
    if "full" in res:
        q0 = res["full"]["qoe"][0]
        for r in res.values():
            r["delta_qoe_vs_full"] = r["qoe"][0] - q0
    return res


def plot(res: dict, fig_dir) -> Path:
    plt = set_style()
    keys = list(res)
    q = np.array([res[k]["qoe"][0] for k in keys])
    qs = np.array([res[k]["qoe"][1] for k in keys])
    e = np.array([res[k]["energy_norm"][0] for k in keys])
    fig, ax = plt.subplots(figsize=(COL_W, 0.28 * len(keys) + 0.6))
    y = np.arange(len(keys))[::-1]
    colors = ["#c0392b" if k == "full" else "#85929e" for k in keys]
    ax.barh(y, q, xerr=qs, color=colors, height=0.65, capsize=2, error_kw={"lw": 0.8})
    for yi, qi, ei in zip(y, q, e):
        ax.text(qi + 0.005, yi, rf"{ei:.2f}$\bar E$", va="center", fontsize=6,
                color="#b03a2e" if ei > 1.02 else "#333333")
    ax.set_yticks(y, [res[k]["label"] for k in keys])
    lo = max(0.0, float(np.nanmin(q - qs)) - 0.05)
    ax.set_xlim(lo, min(1.0, float(np.nanmax(q + qs)) + 0.08))
    ax.set_xlabel("QoE (label: average energy)")
    ax.grid(axis="y", visible=False)
    path = save(fig, Path(fig_dir) / "run7_ablation")
    plt.close(fig)
    return path


def main(argv=None, evaluator=None) -> dict:
    ap = base_parser(__doc__)
    ap.add_argument("--variants", default=",".join(VARIANTS), help="subset of variant keys")
    ap.add_argument("--fixed_mu", default="0.5,2", help="fixed-penalty factors f (mu = f * mu*); '' to skip")
    args = ap.parse_args(argv)
    keys = [k for k in args.variants.split(",") if k.strip()]
    if "full" not in keys:
        keys.insert(0, "full")
    if "conf_msp" in keys and msp_equals_confidence(args.cache):
        keys.remove("conf_msp")
        print("[run7] skipping conf_msp: in this cache the max softmax probability equals the calibrated "
              "confidence (true for synthetic data by construction), so the variant would be identical to 'full'")
    jobs = {k: variant_jobs(args, k) for k in keys}
    maybe_launch(args, [j for js in jobs.values() for j in js])          # stage 1
    mu_star = mu_star_of(jobs["full"])
    if args.fixed_mu.strip() and np.isfinite(mu_star):
        fm = fixed_mu_jobs(args, mu_star)
        maybe_launch(args, [j for js in fm.values() for j in js])        # stage 2 (needs mu*)
        jobs.update(fm)
    res = analyze(args, jobs, evaluator)
    if not res:
        raise SystemExit("no complete runs to analyze")
    plot(res, args.fig_dir)
    write_json({"mu_star": mu_star, "variants": res}, Path(args.out_dir) / "run7.json")
    rows = [{"name": r["label"], "q": r["qoe"], "acc": r["accuracy"], "e": r["energy_norm"],
             "c": r["comm_rate"], "d": f"{r.get('delta_qoe_vs_full', float('nan')):+.3f}"} for r in res.values()]
    tex = latex_table(rows, [("q", "QoE", "{:.3f}"), ("acc", "Accuracy", "{:.3f}"), ("e", r"$\bar E_u/\bar E$", "{:.2f}"),
                             ("c", "Comm. rate", "{:.2f}"), ("d", r"$\Delta$QoE", "{}")],
                      "Ablation study at the default budget (mean $\\pm$ std over seeds).", "tab:run7")
    Path(args.out_dir, "run7_table.tex").write_text(tex, encoding="utf-8")
    for r in res.values():
        print(f"{r['label']:28s} QoE {r['qoe'][0]:.3f}  E/Ebar {r['energy_norm'][0]:.2f}  "
              f"dQoE {r.get('delta_qoe_vs_full', float('nan')):+.3f}")
    return res


if __name__ == "__main__":
    main()
