
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.mdmt import load_boxes, load_grids, read_frames
from .crossfit import TAUS, apply_cal, export, make_folds, metrics, oof_rows, pool
from .train_split import DetFrames, make_loader, run_eval, train



def predict_fold(model, rows, args, device, dsets, snrs) -> dict:
    """Raw out-of-fold outputs; detection counts are stored at every tau so calibration can be done later."""
    dl = make_loader(DetFrames(rows, args.img_root, dsets["boxes"], dsets["grids"], False), args.batch, False,
                     args.workers)
    res = run_eval(model, dl, device, snrs=snrs, gen_seed=args.seed)
    counts = np.stack([np.stack([[(sc >= t).sum() for t in TAUS] for sc in res["scores"][s]]) for s in snrs], 1)
    return {"local_logits": res["local_logits"], "local_mu": res["local_mu"], "local_ls": res["local_ls"],
            "local_view": res["local_view"],
            "edge_mu": np.stack([res["edge_mu"][s] for s in snrs], 1),
            "edge_view": np.stack([res["edge_view"][s] for s in snrs], 1),
            "edge_logits": np.stack([res["edge_logits"][s] for s in snrs], 1),          # (N, S, C)
            "counts": counts.astype(np.float32)}                                         # (N, S, len(TAUS))



def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="caches/mdmt/frames.csv")
    ap.add_argument("--img_root", default="caches/mdmt/img1024")
    ap.add_argument("--boxes", default="caches/mdmt/boxes.npz")
    ap.add_argument("--grids", default="caches/mdmt/grids_36x64.npz")
    ap.add_argument("--out", default="runs_perception/crossfit")
    ap.add_argument("--cache_out", default="caches/mdmt_cf")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold", type=int, default=-1, help="train/predict only this fold (default: all unfinished)")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--repredict", action="store_true",
                    help="redo the out-of-fold predictions from the saved fold models (no training)")
    ap.add_argument("--local_head", default="payload", choices=["payload", "rich"],
                    help="UAV local head: on the payload (v0.15) or a counting back-end on layer2 features (v0.16)")
    ap.add_argument("--c_b", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2, help="fixed; split run 2 peaked at epoch 1-2 on unseen scenes")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_local", type=float, default=0.5)
    ap.add_argument("--lambda_edge", type=float, default=0.5)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--p_clean", type=float, default=0.3)
    ap.add_argument("--train_stride", type=int, default=3, help="every k-th training frame")
    ap.add_argument("--oof_stride", type=int, default=3, help="every k-th time instant predicted out of fold")
    ap.add_argument("--snr_grid", default="-5,0,5,10,20")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--init_mmdet", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.init_mmdet:
        raise SystemExit("--init_mmdet is not allowed with cross-fitting: that detector has seen out-of-fold scenes")
    args.select_on = "none"

    frames = read_frames(args.frames)
    frames = [dict(r, split={"train": "train", "heldout": "train", "cal": "val", "test": "test"}[r["our_split"]])
              for r in frames]
    if args.smoke:
        seqs = sorted({r["seq"] for r in frames})[:8]
        frames = [r for r in frames if r["seq"] in seqs]
        frames = [r for i, r in enumerate(frames) if i % max(1, len(frames) // 64) == 0]
        args.folds, args.epochs, args.train_stride, args.oof_stride = 2, 1, 1, 1
        args.out, args.cache_out, args.snr_grid = args.out + "_smoke", str(Path(args.cache_out) / "smoke"), "0,10"
    K = args.folds
    out, cache = Path(args.out), Path(args.cache_out)
    out.mkdir(parents=True, exist_ok=True)
    fold_of = make_folds(frames, K, args.seed)
    (out / "folds.json").write_text(json.dumps(fold_of, indent=2), encoding="utf-8")
    snr_grid = [float(s) for s in args.snr_grid.split(",") if s.strip()]
    snrs = (None, *snr_grid)
    snr_names = ["clean"] + [f"{s:g}dB" for s in snr_grid]

    if not args.report_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dsets = {"boxes": load_boxes(args.boxes) if Path(args.boxes).exists() else None,
                 "grids": load_grids(args.grids) if Path(args.grids).exists() else None}
        if dsets["boxes"] is None:
            raise SystemExit(f"{args.boxes} not found: python -m uavsc.data.mdmt boxes --root <MDMT folder>")
        from PIL import Image

        from .splitnet import SplitDetector

        with Image.open(Path(args.img_root) / frames[0]["image"]) as im:
            W, H = im.size
        todo = [args.fold] if args.fold >= 0 else list(range(K))
        for k in todo:
            fo = out / f"fold{k}"
            if (fo / "oof.npz").exists() and not args.repredict:
                print(f"[fold {k}] finished earlier, skipped")
                continue
            fo.mkdir(parents=True, exist_ok=True)
            tr = [r for r in frames if fold_of[r["seq"]] != k]
            te_idx = [i for i, r in enumerate(frames) if fold_of[r["seq"]] == k]
            sub = oof_rows([frames[i] for i in te_idx], args.oof_stride)
            keep = {id(r) for r in sub}
            te_idx = [i for i in te_idx if id(frames[i]) in keep]
            print(f"[fold {k}] train on {len({r['seq'] for r in tr})} sequences ({len(tr) // args.train_stride} "
                  f"frames/epoch), predict {len(te_idx)} frames of {len({frames[i]['seq'] for i in te_idx})} "
                  f"held-out sequences", flush=True)
            torch.manual_seed(args.seed + k)
            model = SplitDetector(args.c_b, pretrained=not args.no_pretrained, min_size=H, max_size=W,
                          local_head=args.local_head).to(device)
            if not (out / "uav_cost.json").exists():
                cost = {**model.uav_cost(H, W), "local_head": args.local_head, "image": [W, H],
                        "k_s": model.payload_symbols(H, W)}
                (out / "uav_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
                print(f"UAV side: {cost['uav_gmacs']:.1f} GMACs/frame, {cost['uav_params_M']:.1f} M params "
                      f"(local head '{args.local_head}'); payload k_s = {cost['k_s']}", flush=True)
            a = argparse.Namespace(**{**vars(args), "out": str(fo), "seed": args.seed + k})
            if not (fo / "best.pt").exists():
                train(model, {"train": tr, "heldout": [], "cal": []}, a, device, dsets, fo)
            model.load_state_dict(torch.load(fo / "best.pt", map_location=device)["state_dict"])
            pred = predict_fold(model, [frames[i] for i in te_idx], a, device, dsets, snrs)
            np.savez_compressed(fo / "oof.npz", row=np.array(te_idx, np.int64),
                                fold=np.full(len(te_idx), k, np.int64), **pred)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    P, R = pool(out, frames, K)
    done = sorted(set(P["fold"].tolist()))
    if len(done) < 2:
        print(f"folds finished: {done}; at least 2 are needed for nested calibration")
        return
    C, cals = apply_cal(P, P["fold"], K)
    cost_f = out / "uav_cost.json"
    report = {"uav_cost": json.loads(cost_f.read_text()) if cost_f.exists() else None, "folds_finished": done, "fold_of_seq": fold_of, "calibration_per_fold": cals,
              "all_scenes": metrics(P, C, np.ones(len(R), bool), snr_names),
              "mdmt_test": metrics(P, C, P["split"] == "test", snr_names) if (P["split"] == "test").any() else None,
              "per_fold": {int(k): metrics(P, C, P["fold"] == k, snr_names) for k in done}}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for name in ("all_scenes", "mdmt_test"):
        r = report[name]
        if r is None:
            continue
        e = r["edge"]["clean"]
        d = r["edge_minus_local_clean"]
        print(f"[{name}] {r['n_seqs']} seqs: local {r['local']['acc']:.3f} [{r['local']['acc_ci95'][0]:.3f}, "
              f"{r['local']['acc_ci95'][1]:.3f}]  edge clean {e['acc']:.3f} [{e['acc_ci95'][0]:.3f}, {e['acc_ci95'][1]:.3f}]  "
              f"edge-local {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
              + "  ".join(f"{n} {v['acc']:.3f}" for n, v in r["edge"].items() if n != "clean"), flush=True)
    print("T_b per fold " + " ".join(f"{c.get('T_b', float('nan')):.2f}" for c in cals)
          + " | T_e per fold " + " ".join(f"{c.get('T_e', float('nan')):.2f}" for c in cals))
    if len(done) == K:
        export(P, C, R, snr_grid, cache)
        print(f"exported out-of-fold caches to {cache}")
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
