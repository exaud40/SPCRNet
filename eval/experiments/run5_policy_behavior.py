
from __future__ import annotations
from pathlib import Path
import numpy as np
from ...data.base import FIDELITY_KEYS, load_cache
from ..analysis import empirical_threshold, threshold_a2g
from ..metrics import comm_prob_by_confidence
from ..plotting import COL_W, save, set_style
from .common import (agg, ebar_list, base_parser, complete, evaluate_runs, maybe_launch, proposed_job, seeds_of,
                     write_json)

BUDGET_COLORS = ["#2471a3", "#c0392b", "#239b56", "#d68910", "#7d3c98"]


def parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def jobs_for(args) -> dict[float, list]:
    return {e: [proposed_job(args, s, e) for s in seeds_of(args)] for e in ebar_list(args)}


def analytic_threshold(arrays: dict, cfg, psi_s, mu: float) -> float:
    """delta*_A2G at the median operating point of the A2G decisions actually taken."""
    a2g = np.asarray(arrays["mode"]) == 2
    if not a2g.any() or "a2g_snr_ref_db" not in arrays:
        return float("nan")
    gamma = float(np.median(arrays["a2g_snr_ref_db"]))
    D2 = float(np.median(arrays["D"][a2g]))
    E2n = float(np.median(arrays["E"][a2g])) / cfg.constraint.E_bar
    D0 = cfg.compute.chi_L / cfg.compute.f_U
    return float(threshold_a2g(psi_s, gamma, D0, D2, E2n, mu, cfg.qoe.D_th, cfg.qoe.zeta_D))


def min_count(n_total: int, n_bins: int = 20) -> int:
    
    return int(min(30, max(3, n_total // (5 * n_bins))))


def binned_mode_share(x: np.ndarray, mode: np.ndarray, edges: np.ndarray, min_count: int = 30):
    
    comm = mode != 0
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    p1, p2, n = [], [], []
    for b in range(len(edges) - 1):
        m = comm & (idx == b)
        n.append(int(m.sum()))
        p1.append(float((mode[m] == 1).mean()) if m.sum() >= min_count else np.nan)
        p2.append(float((mode[m] == 2).mean()) if m.sum() >= min_count else np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), np.array(p1), np.array(p2), np.array(n)


def analyze(args, jobs: dict[float, list], evaluator=None) -> dict:
    psi_s = load_cache(Path(args.cache) / "fidelity_test.npz", FIDELITY_KEYS)["psi_s"]
    res = {}
    for e, js in jobs.items():
        done = complete(js)
        if not done:
            continue
        sums, arr, cfg = evaluate_runs([j.run_dir for j in done], args.cache, args.episodes, evaluator)
        a = agg(sums)
        mu, dth = a["mu_final"][0], a["delta_th_final"][0]
        res[e] = {"summary": a, "delta_th_learned": dth, "mu": mu,
                  "delta_star_a2g": analytic_threshold(arr, cfg, psi_s, mu),
                  "delta_empirical": empirical_threshold(arr["delta"], arr["mode"]),
                  "_arr": arr}
    return res


def plot(res: dict, fig_dir: str | Path, focus: float | None = None) -> list[Path]:
    plt = set_style()
    paths = []
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for i, (e, r) in enumerate(sorted(res.items())):
        c = BUDGET_COLORS[i % len(BUDGET_COLORS)]
        x, p, n = comm_prob_by_confidence(r["_arr"]["delta"], r["_arr"]["mode"])
        ok = n >= min_count(n.sum())
        ax.plot(x[ok], p[ok], color=c, marker="o", label=rf"$\bar E$={e:g} mJ")
        ax.axvline(r["delta_th_learned"], color=c, ls="--", lw=0.8)
        if np.isfinite(r["delta_star_a2g"]):
            ax.axvline(r["delta_star_a2g"], color=c, ls=":", lw=1.0)
    ax.set_xlabel(r"Local confidence $\delta_u$")
    ax.set_ylabel(r"$\Pr\{a_u \neq$ LOCAL $\mid \delta_u\}$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title=r"-- learned $\delta_{\rm th}$,  $\cdots$ $\delta^\star_{\rm A2G}$", title_fontsize=6)
    paths.append(save(fig, Path(fig_dir) / "run5_gate"))
    plt.close(fig)

    e0 = focus if focus in res else sorted(res)[len(res) // 2]
    arr = res[e0]["_arr"]
    mode = np.asarray(arr["mode"])
    panels = []
    if "a2g_snr_ref_db" in arr:
        lo, hi = np.percentile(arr["a2g_snr_ref_db"], [2, 98])
        panels.append(("a2g_snr_ref_db", np.linspace(lo, hi, 9), "Reference A2G SNR [dB]", "run5_mode_snr"))
    if "n_conf_nb" in arr:
        k = int(np.max(arr["n_conf_nb"]))
        panels.append(("n_conf_nb", np.arange(-0.5, k + 1.0), "Confident co-observing neighbors", "run5_mode_nb"))
    for key, edges, xlabel, fname in panels:
        x, p1, p2, _ = binned_mode_share(np.asarray(arr[key], float), mode, edges,
                                         min_count(int((mode != 0).sum()), len(edges) - 1))
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        if key == "n_conf_nb":       # integer counts (0/1 with two views): grouped bars, not lines
            ok = np.isfinite(p1)
            xb, w = np.round(x[ok]).astype(int), 0.38
            ax.bar(xb - w / 2, p1[ok], w, color="#5d6d7e", label="A2A")
            ax.bar(xb + w / 2, p2[ok], w, color="#212f3d", label="A2G")
            ax.set_xticks(xb)
            ax.grid(axis="x", visible=False)
        else:
            ax.plot(x, p1, color="#5d6d7e", marker="<", label="A2A")
            ax.plot(x, p2, color="#212f3d", marker=">", label="A2G")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Share of communicating decisions")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(title=rf"$\bar E$={e0:g} mJ", title_fontsize=6)
        paths.append(save(fig, Path(fig_dir) / fname))
        plt.close(fig)
    return paths


def main(argv=None, evaluator=None) -> dict:
    ap = base_parser(__doc__)
    ap.add_argument("--ebar_mJ", default="0.5x,1x,2x", help="energy budgets [mJ]; 'x' = multiple of the config E_bar")
    args = ap.parse_args(argv)
    jobs = jobs_for(args)
    maybe_launch(args, [j for js in jobs.values() for j in js])
    res = analyze(args, jobs, evaluator)
    if not res:
        raise SystemExit("no complete runs to analyze")
    plot(res, args.fig_dir)
    clean = {f"{e:g}": {k: v for k, v in r.items() if not k.startswith("_")} for e, r in res.items()}
    write_json(clean, Path(args.out_dir) / "run5.json")
    for e, r in clean.items():
        print(f"E_bar {e} mJ: learned delta_th {r['delta_th_learned']:.3f}  delta*_A2G {r['delta_star_a2g']:.3f}  "
              f"empirical {r['delta_empirical']:.3f}  QoE {r['summary']['qoe'][0]:.3f}")
    return clean


if __name__ == "__main__":
    main()
