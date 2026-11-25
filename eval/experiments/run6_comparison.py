
from __future__ import annotations
from pathlib import Path
import numpy as np
from ...config import apply_overrides
from ...rl.baselines import AlwaysA2A, AlwaysA2G, AlwaysLocal, FixedThreshold, GreedyGate, RandomPolicy
from ..evaluate import evaluate, load_split, tune_threshold
from ..plotting import COL_W, save, set_style, style_for
from .common import (agg, ebar_list, base_parser, complete, evaluate_runs, experiment_cfg, latex_table, maybe_launch,
                     proposed_job, seeds_of, write_json)

# name -> (config key, values, axis label, value -> override string, value of the default config)
SWEEPS = {
    "B_tot": ("channel.B_tot", [1.0, 2.0, 4.0], "Total bandwidth $B_{\\rm tot}$ [MHz]",
              lambda v: f"channel.B_tot={v * 1e6:g}", lambda c: c.channel.B_tot / 1e6),
    "P_max": ("channel.P_max_dbm", [14.0, 20.0, 26.0], "Max. follower power $P_{\\max}$ [dBm]",
              lambda v: f"channel.P_max_dbm={v:g}", lambda c: c.channel.P_max_dbm),
    "D_th": ("qoe.D_th", [50.0, 100.0, 200.0], "Deadline $D_{\\rm th}$ [ms]",
             lambda v: f"qoe.D_th={v * 1e-3:g}", lambda c: c.qoe.D_th * 1e3),
    "M": ("network.M", [4.0, 6.0, 8.0], "Followers per cluster $M$",
          lambda v: f"network.M={int(v)}", lambda c: float(c.network.M)),
}
TAU_GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def sweep_names(args) -> list[str]:
    return [] if args.sweeps.strip().lower() == "none" else [s.strip() for s in args.sweeps.split(",") if s.strip()]


def jobs_for(args) -> tuple[dict, dict]:
    cfg = experiment_cfg(args)
    seeds = seeds_of(args)
    pareto = {e: [proposed_job(args, s, e) for s in seeds] for e in ebar_list(args)}
    sweeps = {}
    for name in sweep_names(args):
        _, values, _, to_override, default_of = SWEEPS[name]
        sweeps[name] = {}
        for v in values:
            if np.isclose(v, default_of(cfg)):        # the default point is the canonical run
                sweeps[name][v] = [proposed_job(args, s) for s in seeds]
            else:
                sweeps[name][v] = [proposed_job(args, s, overrides=[to_override(v)], subdir=f"sweep_{name}{v:g}")
                                   for s in seeds]
    return pareto, sweeps


# baselines
def baseline_points(cfg, pc, fc, episodes: int, taus=TAU_GRID) -> dict:
    """Energy [mJ] and QoE of every baseline (threshold baselines over the tau grid)."""
    out = {}
    for name, cls in [("Fixed-Threshold", FixedThreshold), ("Greedy-Gate", GreedyGate)]:
        rows = []
        for t in taus:
            s, _ = evaluate(cls(float(t)), cfg, pc, fc, episodes)
            rows.append((float(t), s["energy_J"] * 1e3, s["qoe"], s["accuracy"]))
        out[name] = rows
    for name, pol in [("Always-Local", AlwaysLocal()), ("Always-A2A", AlwaysA2A()),
                      ("Always-A2G", AlwaysA2G()), ("Random", RandomPolicy(0))]:
        s, _ = evaluate(pol, cfg, pc, fc, episodes)
        out[name] = [(float("nan"), s["energy_J"] * 1e3, s["qoe"], s["accuracy"])]
    return out


def frontier(rows) -> tuple[np.ndarray, np.ndarray]:
    """Upper QoE-energy frontier of a swept baseline: (energy sorted, running max of QoE)."""
    r = sorted(rows, key=lambda x: x[1])
    e = np.array([x[1] for x in r])
    q = np.maximum.accumulate(np.array([x[2] for x in r]))
    return e, q


def energy_to_reach(rows, qoe: float) -> float:
    """Energy [mJ] the baseline needs to reach ``qoe`` (linear interpolation on its frontier;
    NaN if it never reaches it)."""
    e, q = frontier(rows)
    if qoe > q.max():
        return float("nan")
    if qoe <= q[0]:
        return float(e[0])
    i = int(np.flatnonzero(q >= qoe)[0])
    if q[i] == q[i - 1]:
        return float(e[i])
    return float(e[i - 1] + (qoe - q[i - 1]) * (e[i] - e[i - 1]) / (q[i] - q[i - 1]))


# analysis
def analyze_pareto(args, pareto: dict, evaluator=None) -> dict:
    cfg = experiment_cfg(args)
    pc, fc = load_split(args.cache, "test")
    base = baseline_points(cfg, pc, fc, args.episodes)
    prop = {}
    for e, js in sorted(pareto.items()):
        done = complete(js)
        if not done:
            continue
        sums, _, _ = evaluate_runs([j.run_dir for j in done], args.cache, args.episodes, evaluator)
        a = agg(sums)
        e_mJ, q = a["energy_J"][0] * 1e3, a["qoe"][0]
        e_g = energy_to_reach(base["Greedy-Gate"], q)
        prop[f"{e:g}"] = {**a, "energy_mJ": (e_mJ, a["energy_J"][1] * 1e3), "budget_mJ": e,
                          "greedy_energy_mJ_same_qoe": e_g,
                          "energy_saving_vs_greedy": float(1 - e_mJ / e_g) if np.isfinite(e_g) and e_g > 0 else float("nan")}
    return {"proposed": prop, "baselines": base}


def analyze_sweeps(args, sweeps: dict, evaluator=None) -> dict:
    out = {}
    for name, per_value in sweeps.items():
        _, _, _, to_override, _ = SWEEPS[name]
        out[name] = {}
        for v, js in sorted(per_value.items()):
            cfg = apply_overrides(experiment_cfg(args), [to_override(v)])
            pc, fc = load_split(args.cache, "test")
            cal = load_split(args.cache, args.tune_split)
            row = {}
            for bname, cls in [("Fixed-Threshold", FixedThreshold), ("Greedy-Gate", GreedyGate)]:
                tau, _ = tune_threshold(cls, cfg, *cal, n_episodes=max(2, args.episodes // 3))
                s, _ = evaluate(cls(tau), cfg, pc, fc, args.episodes)
                row[bname] = {"qoe": s["qoe"], "energy_norm": s["energy_norm"], "tau": tau}
            for bname, pol in [("Always-Local", AlwaysLocal()), ("Always-A2A", AlwaysA2A())]:
                s, _ = evaluate(pol, cfg, pc, fc, args.episodes)
                row[bname] = {"qoe": s["qoe"], "energy_norm": s["energy_norm"]}
            done = complete(js)
            if done:
                sums, _, _ = evaluate_runs([j.run_dir for j in done], args.cache, args.episodes, evaluator)
                a = agg(sums)
                row["Proposed"] = {"qoe": a["qoe"][0], "qoe_std": a["qoe"][1], "energy_norm": a["energy_norm"][0]}
            out[name][f"{v:g}"] = row
    return out


# plot figures
def plot_pareto(res: dict, fig_dir) -> Path:
    plt = set_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    for i, (name, rows) in enumerate(res["baselines"].items()):
        c, m = style_for(name, i)
        e = [r[1] for r in rows]
        q = [r[2] for r in rows]
        if len(rows) > 1:      # every tau as a light marker, its upper frontier as a line
            ax.scatter(e, q, color=c, marker=m, s=8, alpha=0.35, linewidths=0)
            fe, fq = frontier(rows)
            ax.plot(fe, fq, color=c, marker=m, ms=3, label=f"{name} ($\\tau$ sweep)", lw=0.9)
        else:   # zero energy (Always-Local) cannot sit on a log axis: pin it to the left edge
            pos = [r[1] for rr in res["baselines"].values() for r in rr if r[1] > 0]
            e0 = [x if x > 0 else 0.5 * min(pos or [1e-3]) for x in e]
            ax.scatter(e0, q, color=c, marker=m, s=22, label=name + (" (0 mJ)" if e[0] <= 0 else ""), zorder=3)
    p = res["proposed"]
    if p:
        e = np.array([v["energy_mJ"][0] for v in p.values()])
        es = np.array([v["energy_mJ"][1] for v in p.values()])
        q = np.array([v["qoe"][0] for v in p.values()])
        qs = np.array([v["qoe"][1] for v in p.values()])
        c, m = style_for("Proposed")
        order = np.argsort(e)
        ax.errorbar(e[order], q[order], xerr=es[order], yerr=qs[order], color=c, marker=m, ms=5, lw=1.4,
                    capsize=2, label="Proposed (WGAE-PPO)", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Average energy per follower and slot [mJ]")
    ax.set_ylabel("QoE")
    ax.legend(fontsize=6, ncol=1, loc="upper left")
    path = save(fig, Path(fig_dir) / "run6_pareto")
    plt.close(fig)
    return path


def plot_sweeps(res: dict, fig_dir) -> list[Path]:
    plt = set_style()
    paths = []
    for name, per_value in res.items():
        xs = [float(v) for v in per_value]
        methods = sorted({m for row in per_value.values() for m in row}, key=lambda m: m != "Proposed")
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        for i, m in enumerate(methods):
            ys = [per_value[v].get(m, {}).get("qoe", np.nan) for v in per_value]
            c, mk = style_for(m, i)
            ax.plot(xs, ys, color=c, marker=mk, label=m, lw=1.4 if m == "Proposed" else 0.9)
        ax.set_xlabel(SWEEPS[name][2])
        ax.set_ylabel("QoE")
        ax.legend(fontsize=6, ncol=2)
        paths.append(save(fig, Path(fig_dir) / f"run6_sweep_{name}"))
        plt.close(fig)
    return paths


def main(argv=None, evaluator=None) -> dict:
    ap = base_parser(__doc__)
    ap.add_argument("--ebar_mJ", default="0.5x,0.75x,1x,1.5x,2x", help="budgets for the trade-off curve [mJ]; 'x' = multiple of E_bar")
    ap.add_argument("--sweeps", default="B_tot,P_max,D_th,M", help="comma list, or 'none'")
    ap.add_argument("--tune_split", default="cal", help="split the threshold baselines are tuned on (train on MDMT)")
    args = ap.parse_args(argv)
    pareto, sweeps = jobs_for(args)
    maybe_launch(args, [j for js in pareto.values() for j in js] +
                 [j for per in sweeps.values() for js in per.values() for j in js])
    res = {"pareto": analyze_pareto(args, pareto, evaluator), "sweeps": analyze_sweeps(args, sweeps, evaluator)}
    plot_pareto(res["pareto"], args.fig_dir)
    if res["sweeps"]:
        plot_sweeps(res["sweeps"], args.fig_dir)
    write_json(res, Path(args.out_dir) / "run6.json")

    # Table: every method at the default budget (threshold baselines tuned on cal to meet it).
    cfg = experiment_cfg(args)
    cal, test = load_split(args.cache, args.tune_split), load_split(args.cache, "test")
    rows = []
    for name, pol in [("Always-Local", AlwaysLocal()), ("Always-A2A", AlwaysA2A()), ("Always-A2G", AlwaysA2G()),
                      ("Random", RandomPolicy(0))]:
        s, _ = evaluate(pol, cfg, *test, args.episodes)
        rows.append({"name": name, "acc": s["accuracy"], "q": s["qoe"], "e": s["energy_norm"], "c": s["comm_rate"]})
    for name, cls in [("Fixed-Threshold", FixedThreshold), ("Greedy-Gate", GreedyGate)]:
        tau, _ = tune_threshold(cls, cfg, *cal)
        s, _ = evaluate(cls(tau), cfg, *test, args.episodes)
        rows.append({"name": f"{name} ($\\tau$={tau:.2f})", "acc": s["accuracy"], "q": s["qoe"],
                     "e": s["energy_norm"], "c": s["comm_rate"]})
    p = res["pareto"]["proposed"].get(f"{cfg.constraint.E_bar * 1e3:g}")
    if p:
        rows.append({"name": "Proposed", "acc": p["accuracy"], "q": p["qoe"], "e": p["energy_norm"], "c": p["comm_rate"]})
    tex = latex_table(rows, [("acc", "Accuracy", "{:.3f}"), ("q", "QoE", "{:.3f}"), ("e", r"$\bar E_u/\bar E$", "{:.2f}"),
                             ("c", "Comm. rate", "{:.2f}")],
                      "Comparison at the default budget (proposed: mean $\\pm$ std over seeds).", "tab:run6")
    Path(args.out_dir, "run6_table.tex").write_text(tex, encoding="utf-8")
    for e, v in res["pareto"]["proposed"].items():
        print(f"E_bar {e} mJ: QoE {v['qoe'][0]:.3f}  E {v['energy_mJ'][0]:.3f} mJ  "
              f"saving vs Greedy-Gate at equal QoE {v['energy_saving_vs_greedy']:+.0%}")
    return res


if __name__ == "__main__":
    main()
