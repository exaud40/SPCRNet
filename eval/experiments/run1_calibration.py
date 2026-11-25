from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from ...data.base import PERCEPTION_KEYS, load_cache
from ...perception.calibration import brier, ece, failure_auroc, reliability_curve
from ..plotting import COL_W, DBL_W, save, set_style
from .common import latex_table, write_json

SOURCES = [("MSP", "msp_raw", "#d68910"), ("Head", "delta_raw", "#2471a3"), ("Head + T", "delta", "#c0392b")]


def analyze(c: dict) -> dict:
    correct = (c["belief"].argmax(1) == c["label"]).astype(float)
    res = {"n": int(len(correct)), "accuracy": float(correct.mean()),
           "per_class_acc": [float(correct[c["label"] == k].mean()) if (c["label"] == k).any() else float("nan")
                             for k in range(c["belief"].shape[1])]}
    if "count_pred" in c and "n_union" in c:
        from ...data.mdmt import SCENE_EDGES

        res["count_mae"] = float(np.mean(np.abs(c["count_pred"] - c["n_union"])))
        res["count_rel_err"] = float(np.median(np.abs(c["count_pred"] - c["n_union"]) / np.maximum(c["n_union"], 1)))
        res["accuracy_from_count"] = float(np.mean(np.searchsorted(SCENE_EDGES, c["count_pred"], side="right") == c["label"]))
    for name, key, _ in SOURCES:
        if key in c:
            conf = c[key].astype(float)
            res[name] = {"ece": ece(conf, correct), "auroc": failure_auroc(conf, correct), "brier": brier(conf, correct),
                         "mean_conf": float(conf.mean())}
    if "n_view" in c and "n_union" in c:
        share = c["n_view"] / np.maximum(c["n_union"], 1)
        edges = np.quantile(share, np.linspace(0, 1, 6))
        idx = np.clip(np.searchsorted(edges, share, side="right") - 1, 0, 4)
        res["by_visible_share"] = [{"share": float(share[idx == b].mean()), "accuracy": float(correct[idx == b].mean()),
                                    "delta": float(c["delta"][idx == b].mean()), "n": int((idx == b).sum())}
                                   for b in range(5) if (idx == b).any()]
    return res


def plot(c: dict, res: dict, fig_dir) -> list[Path]:
    plt = set_style()
    correct = (c["belief"].argmax(1) == c["label"]).astype(float)
    srcs = [s for s in SOURCES if s[1] in c]
    fig, axes = plt.subplots(1, len(srcs), figsize=(DBL_W, 2.2), constrained_layout=True, sharey=True)
    for ax, (name, key, col) in zip(np.atleast_1d(axes), srcs):
        x, acc, cf, cnt = reliability_curve(c[key], correct, 10)
        ok = cnt > 0
        ax.bar(x[ok], acc[ok], width=0.09, color=col, alpha=0.85, label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{name}: ECE {res[name]['ece']:.3f}, AUROC {res[name]['auroc']:.2f}")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy")
    paths = [save(fig, Path(fig_dir) / "run1_reliability")]
    plt.close(fig)
    if "by_visible_share" in res:
        b = res["by_visible_share"]
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        xs = [r["share"] for r in b]
        ax.plot(xs, [r["accuracy"] for r in b], "o-", color="#212f3d", label="Accuracy")
        ax.plot(xs, [r["delta"] for r in b], "s--", color="#c0392b", label=r"Mean confidence $\delta_u$")
        ax.set_xlabel("Share of scene targets visible in this view")
        ax.set_ylabel("Probability")
        ax.legend()
        paths.append(save(fig, Path(fig_dir) / "run1_occlusion"))
        plt.close(fig)
    return paths


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="caches/mdmt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--fig_dir", default="figures")
    args = ap.parse_args(argv)
    path = Path(args.cache) / "perception_raw" / f"perception_{args.split}.npz"
    if not path.exists():
        path = Path(args.cache) / f"perception_{args.split}.npz"
    c = load_cache(path, PERCEPTION_KEYS)
    res = analyze(c)
    plot(c, res, args.fig_dir)
    write_json(res, Path(args.out_dir) / "run1.json")
    rows = [{"name": n, **{k: res[n][k] for k in ("ece", "auroc", "brier", "mean_conf")}} for n, _, _ in SOURCES if n in res]
    tex = latex_table(rows, [("ece", "ECE", "{:.3f}"), ("auroc", "AUROC", "{:.3f}"), ("brier", "Brier", "{:.3f}"),
                             ("mean_conf", "Mean conf.", "{:.3f}")],
                      f"Confidence quality on the MDMT test split (accuracy {res['accuracy']:.3f}, "
                      f"{res['n']} views).", "tab:run1")
    Path(args.out_dir, "run1_table.tex").write_text(tex, encoding="utf-8")
    print(f"accuracy {res['accuracy']:.3f} on {res['n']} test views; per class {np.round(res['per_class_acc'], 3).tolist()}")
    if "count_mae" in res:
        print(f"scene count: MAE {res['count_mae']:.1f} targets, median relative error {res['count_rel_err']:.1%}, "
              f"accuracy of the binned count {res['accuracy_from_count']:.3f}")
    for n, _, _ in SOURCES:
        if n in res:
            print(f"  {n:9s} ECE {res[n]['ece']:.3f}  AUROC {res[n]['auroc']:.3f}  Brier {res[n]['brier']:.3f}")
    return res


if __name__ == "__main__":
    main()
