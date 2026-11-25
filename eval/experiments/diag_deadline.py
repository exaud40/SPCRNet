from __future__ import annotations
from pathlib import Path
from ...config import apply_overrides
from ...rl.baselines import AlwaysA2A, AlwaysLocal
from ..evaluate import evaluate, load_split
from .common import (agg, base_parser, complete, evaluate_runs, experiment_cfg, maybe_launch, proposed_job,
                     seeds_of, write_json)

SETTINGS = {
    "base": ([], 1),
    "long": ([], 2),
    "ent03": (["rl.entropy_coef=0.03"], 1),
    "ent05": (["rl.entropy_coef=0.05"], 1),
}


def jobs_for(args) -> dict:
    out = {}
    for key, (ov, mult) in SETTINGS.items():
        js = []
        for s in seeds_of(args):
            j = proposed_job(args, s, overrides=[f"qoe.D_th={args.d_th_ms * 1e-3:g}", *ov], subdir=f"diag_dth/{key}")
            if mult != 1:
                j.iterations = args.iterations * mult
            js.append(j)
        out[key] = js
    return out


def main(argv=None, evaluator=None) -> dict:
    ap = base_parser(__doc__)
    ap.add_argument("--d_th_ms", type=float, default=50.0)
    ap.set_defaults(seeds="0,1")
    args = ap.parse_args(argv)
    jobs = jobs_for(args)
    maybe_launch(args, [j for js in jobs.values() for j in js])
    cfg = apply_overrides(experiment_cfg(args), [f"qoe.D_th={args.d_th_ms * 1e-3:g}"])
    test = load_split(args.cache, "test")
    res = {}
    for name, pol in [("Always-Local", AlwaysLocal()), ("Always-A2A", AlwaysA2A())]:
        s, _ = evaluate(pol, cfg, *test, args.episodes)
        res[name] = {"qoe": (s["qoe"], 0.0), "energy_norm": (s["energy_norm"], 0.0), "a2a_share": (s["a2a_share"], 0.0),
                     "a2g_share": (s["a2g_share"], 0.0)}
    for key, js in jobs.items():
        done = complete(js)
        if done:
            sums, _, _ = evaluate_runs([j.run_dir for j in done], args.cache, args.episodes, evaluator)
            res[key] = agg(sums)
    write_json(res, Path(args.out_dir) / "diag_deadline.json")
    print(f"\nD_th = {args.d_th_ms:g} ms, E_bar = {cfg.constraint.E_bar * 1e3:g} mJ")
    print(f"{'setting':14s} {'QoE':>14s} {'E/Ebar':>7s} {'A2A':>6s} {'A2G':>6s}")
    for k, r in res.items():
        print(f"{k:14s} {r['qoe'][0]:.3f} +- {r['qoe'][1]:.3f} {r['energy_norm'][0]:7.2f} "
              f"{r['a2a_share'][0]:6.1%} {r['a2g_share'][0]:6.1%}")
    return res


if __name__ == "__main__":
    main()
