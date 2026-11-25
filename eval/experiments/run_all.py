from __future__ import annotations
import sys
from . import run4_convergence as r4
from . import run5_policy_behavior as r5
from . import run6_comparison as r6
from . import run7_ablation as r7
from .common import base_parser, maybe_launch

MODULES = {"run4": r4, "run5": r5, "run6": r6, "run7": r7}


def collect_jobs(argv: list[str], only: list[str]):
    jobs = []
    if "run4" in only:
        a = base_parser("").parse_args(argv)
        jobs += [j for js in r4.jobs_for(a).values() for j in js]
    if "run5" in only:
        ap = base_parser("")
        ap.add_argument("--ebar_mJ", default="0.5x,1x,2x")
        jobs += [j for js in r5.jobs_for(ap.parse_args(argv)).values() for j in js]
    if "run6" in only:
        ap = base_parser("")
        ap.add_argument("--ebar_mJ", default="0.5x,0.75x,1x,1.5x,2x")
        ap.add_argument("--sweeps", default="B_tot,P_max,D_th,M")
        pareto, sweeps = r6.jobs_for(ap.parse_args(argv))
        jobs += [j for js in pareto.values() for j in js] + [j for p in sweeps.values() for js in p.values() for j in js]
    if "run7" in only:
        a = base_parser("").parse_args(argv)
        jobs += [j for k in r7.VARIANTS for j in r7.variant_jobs(a, k)]
    unique = {}
    for j in jobs:                                   # identical run dirs are trained once
        unique.setdefault(j.run_dir, j)
    return list(unique.values())


def main(argv=None, evaluator=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = ["run4", "run5", "run6", "run7"]
    if "--only" in argv:
        i = argv.index("--only")
        only = [s.strip() for s in argv[i + 1].split(",")]
        argv = argv[:i] + argv[i + 2:]
    common = base_parser("").parse_args(argv)
    jobs = collect_jobs(argv, only)
    print(f"[run_all] {len(jobs)} distinct training run(s) for {only}")
    maybe_launch(common, jobs)
    out = {}
    for name in only:
        print(f"\n===== {name} =====")
        mod = MODULES[name]
        # run7 may still train its fixed-penalty stage (it needs mu* of the finished full runs)
        extra = [] if (name == "run7" or "--skip_train" in argv) else ["--skip_train"]
        kw = {} if name == "run4" else {"evaluator": evaluator}
        out[name] = mod.main(argv + extra, **kw)
    return out


if __name__ == "__main__":
    main()
