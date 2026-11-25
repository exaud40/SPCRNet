
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

CONFIGS = {
    "finetune512": ["--img_root", "caches/mdmt/img", "--grids", "caches/mdmt/grids_36x64.npz"],
    "frozen512": ["--img_root", "caches/mdmt/img", "--grids", "caches/mdmt/grids_36x64.npz", "--freeze_backbone",
                  "--lr", "1e-3"],
    "frozen1024": ["--img_root", "caches/mdmt/img1024", "--grids", "caches/mdmt/grids_36x64.npz",
                   "--freeze_backbone", "--lr", "1e-3", "--batch", "16"],
    "finetune1024": ["--img_root", "caches/mdmt/img1024", "--grids", "caches/mdmt/grids_36x64.npz",
                     "--backbone_lr_mult", "0.1", "--batch", "16"],
}


def summarize(run_dir: Path) -> dict:
    with open(run_dir / "stage1_log.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    best = max(rows, key=lambda r: float(r["select_acc"]))
    return {"epochs": len(rows), "best_epoch": int(best["epoch"]), "select_acc": float(best["select_acc"]),
            "acc_from_count": float(best["select_acc_from_count"]), "count_mae": float(best["select_count_mae"]),
            "train_acc": float(best["train_acc"]), "minutes_per_epoch": float(rows[-1]["minutes"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default="runs_perception/sweep")
    ap.add_argument("--extra", default="", help="extra arguments passed to every run, e.g. '--workers 6'")
    ap.add_argument("--then_train", type=int, default=0, metavar="EPOCHS",
                    help="afterwards train the best setting fully (stages 1-3 + export) for this many epochs")
    args = ap.parse_args(argv)
    names = [c for c in args.configs.split(",") if c]
    results = {}
    for name in names:
        cfg = CONFIGS[name]
        img_root = Path(cfg[cfg.index("--img_root") + 1])
        if not img_root.exists():
            print(f"[sweep] skip {name}: {img_root} missing (see the docstring for the prepare command)")
            continue
        run_dir = Path(args.out) / name
        if not (run_dir / "stage1_log.csv").exists() or summarize(run_dir)["epochs"] < args.epochs:
            cmd = [sys.executable, "-m", "uavsc.perception.train_perception", "--stage1_only", "--epochs",
                   str(args.epochs), "--out", str(run_dir), *cfg, *args.extra.split()]
            print(f"[sweep] {name}: {' '.join(cmd[2:])}", flush=True)
            subprocess.run(cmd, check=True)
        results[name] = summarize(run_dir)
    print(f"\n{'setting':14s} {'acc (10 unseen scenes)':>22s} {'from count':>10s} {'count MAE':>9s} "
          f"{'train acc':>9s} {'best ep':>7s} {'min/ep':>6s}")
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["select_acc"]):
        print(f"{n:14s} {r['select_acc']:22.3f} {r['acc_from_count']:10.3f} {r['count_mae']:9.1f} "
              f"{r['train_acc']:9.3f} {r['best_epoch']:7d} {r['minutes_per_epoch']:6.1f}")
    if args.then_train and results:
        best = max(results, key=lambda k: results[k]["select_acc"])
        cmd = [sys.executable, "-m", "uavsc.perception.train_perception", "--epochs", str(args.then_train),
               "--out", "runs_perception/spcrnet", *CONFIGS[best], *args.extra.split()]
        print(f"\n[sweep] best setting: {best}; full training: {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, "-m", "uavsc.eval.experiments.run1_calibration", "--cache", "caches/mdmt"],
                       check=True)


if __name__ == "__main__":
    main()
