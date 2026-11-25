
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.base import PERCEPTION_KEYS, save_cache, validate_perception_cache
from ..data.mdmt import NUM_CLASSES, SCENE_EDGES, load_boxes, load_grids, read_frames
from .calibration import ece, failure_auroc, fit_temperature
from .splitnet import SplitDetector, awgn, level_logits


#  data
class DetFrames(Dataset):
    """(image [0,1] (3,H,W), target dict for the detector, local targets dict, row index)."""

    def __init__(self, rows, img_root, boxes, grids, train: bool, seed: int = 0, grid_shape=(36, 64)):
        self.rows, self.root, self.boxes, self.grids, self.train = rows, Path(img_root), boxes, grids, train
        self.rng = np.random.default_rng(seed)
        self.grid_shape = grid_shape

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        r = self.rows[i]
        with Image.open(self.root / r["image"]) as im:
            x = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        H, W = x.shape[:2]
        key = (r["seq"], r["view"], r["frame"])
        b, c = self.boxes.get(key, (np.zeros((0, 4), np.float32), np.zeros(0, np.uint8))) if self.boxes else \
            (np.zeros((0, 4), np.float32), np.zeros(0, np.uint8))
        b = b.astype(np.float32).copy()
        g = self.grids.get(key, np.zeros(self.grid_shape)).astype(np.float32) if self.grids else \
            np.zeros(self.grid_shape, np.float32)
        if self.train:
            rng = np.random.default_rng((self.rng.integers(1 << 31) + i) % (1 << 31))
            if rng.random() < 0.5:
                x, g = x[:, ::-1], g[:, ::-1]
                b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
            x = np.clip((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        bp = b * np.array([W, H, W, H], np.float32)
        keep = (bp[:, 2] - bp[:, 0] >= 1.0) & (bp[:, 3] - bp[:, 1] >= 1.0)
        tgt = {"boxes": torch.from_numpy(bp[keep]).reshape(-1, 4), "labels": torch.from_numpy(c[keep].astype(np.int64))}
        loc = {"label": int(r["label"]), "n_union": float(r["n_union"]), "n_view": float(r["n_view"]),
               "grid": torch.from_numpy(np.ascontiguousarray(g))}
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), tgt, loc, i


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    loc = {"label": torch.tensor([b[2]["label"] for b in batch]),
           "n_union": torch.tensor([b[2]["n_union"] for b in batch]),
           "n_view": torch.tensor([b[2]["n_view"] for b in batch]),
           "grid": torch.stack([b[2]["grid"] for b in batch])}
    return imgs, tgts, loc, torch.tensor([b[3] for b in batch])


def make_loader(ds, batch, train, workers):
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers, collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), drop_last=train, persistent_workers=workers > 0)


#  losses
def local_losses(loc_out: dict, loc_t: dict) -> dict:
    dens, grid = loc_out["density"], loc_t["grid"]
    if dens.shape[-2:] != grid.shape[-2:]:
        gh, gw = grid.shape[-2:]
        h, w = dens.shape[-2:]
        if h >= gh and w >= gw:                       # prediction finer than the target: sum-pool the prediction
            dens = F.adaptive_avg_pool2d(dens[:, None], (gh, gw))[:, 0] * (h * w) / (gh * gw)
        else:                                         # target finer: sum-pool the target
            grid = F.adaptive_avg_pool2d(grid[:, None], (h, w))[:, 0] * (gh * gw) / (h * w)
    k = torch.tensor([1.0, 2.0, 1.0], device=dens.device)
    k = (k[:, None] * k[None, :] / 16.0).view(1, 1, 3, 3)
    blur = lambda m: F.conv2d(m[:, None], k, padding=1)[:, 0]
    l_dens = ((blur(dens) - blur(grid)) ** 2).mean() + F.smooth_l1_loss(
        torch.log1p(dens.sum((-2, -1))), torch.log1p(grid.sum((-2, -1))))
    t = torch.log1p(loc_t["n_union"])
    l_cnt = (0.5 * ((t - loc_out["mu"]) / torch.exp(loc_out["log_sigma"])) ** 2 + loc_out["log_sigma"]).mean()
    l_lvl = F.nll_loss(loc_out["logits"], loc_t["label"])
    return {"local_density": l_dens, "local_count": l_cnt, "local_level": l_lvl}


#  evaluation
@torch.no_grad()
def run_eval(model, dl, device, snrs=(None,), gen_seed: int = 0) -> dict:
    """Local outputs once, and edge detections at each SNR. Returns numpy arrays in dataset order."""
    model.eval()
    g = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(gen_seed)
    out: dict[str, list] = {"idx": [], "local_logits": [], "local_mu": [], "local_ls": [], "local_view": []}
    scores = {s: [] for s in snrs}                 # per sample: array of detection scores
    elog = {s: [] for s in snrs}                   # edge count-head logits per SNR
    emu = {s: [] for s in snrs}                    # edge union-count estimate log(1 + n) per SNR
    eview = {s: [] for s in snrs}                  # edge density total = estimate of the VIEW count
    for x, _, _, idx in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            z, loc = model.uav_forward(x)
        out["idx"].append(idx.numpy())
        out["local_logits"].append(loc["logits"].float().cpu().numpy())
        out["local_mu"].append(loc["mu"].float().cpu().numpy())
        out["local_ls"].append(loc["log_sigma"].float().cpu().numpy())
        out["local_view"].append(loc["density"].float().sum((-2, -1)).cpu().numpy())   # density ~ view grid
        for s in snrs:
            zz = z.float()
            if s is not None:
                zz = zz + torch.randn(zz.shape, generator=g, device=zz.device) * math.sqrt(10 ** (-s / 10))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                dets = model.server_forward(x, zz.to(z.dtype))
                eo = model.edge_belief()
            scores[s].extend([d["scores"].float().cpu().numpy() for d in dets])
            elog[s].append(eo["logits"].float().cpu().numpy())
            emu[s].append(eo["mu"].float().cpu().numpy())
            eview[s].append(eo["density"].float().sum((-2, -1)).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    order = np.argsort(res["idx"])
    res = {k: v[order] for k, v in res.items()}
    res["scores"] = {s: [v[i] for i in order] for s, v in scores.items()}
    res["edge_logits"] = {s: np.concatenate(v)[order] for s, v in elog.items()}
    res["edge_mu"] = {s: np.concatenate(v)[order] for s, v in emu.items()}
    res["edge_view"] = {s: np.concatenate(v)[order] for s, v in eview.items()}
    return res


def counts_at(scores: list[np.ndarray], tau: float) -> np.ndarray:
    return np.array([(s >= tau).sum() for s in scores], dtype=np.float64)


def edge_logits(n_hat: np.ndarray, a: float, b: float, s: float) -> np.ndarray:
    mu = torch.from_numpy(a * np.log1p(n_hat) + b).float()
    bounds = torch.tensor([math.log1p(e - 0.5) for e in SCENE_EDGES], dtype=torch.float32)
    return level_logits(mu, torch.full_like(mu, math.log(max(s, 1e-3))), bounds).numpy()


def fit_edge_map(n_hat: np.ndarray, n_union: np.ndarray) -> tuple[float, float, float]:
    X = np.stack([np.log1p(n_hat), np.ones_like(n_hat)], 1)
    y = np.log1p(n_union)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    s = float(np.std(y - X @ np.array([a, b])))
    return float(a), float(b), max(s, 0.05)


def calibrate(res_cal: dict, rows_cal: list[dict], clean=None) -> dict:
    nv = np.array([r["n_view"] for r in rows_cal], float)
    nu = np.array([r["n_union"] for r in rows_cal], float)
    y = np.array([r["label"] for r in rows_cal])
    taus = np.round(np.arange(0.05, 0.91, 0.05), 2)
    errs = [np.mean(np.abs(counts_at(res_cal["scores"][clean], t) - nv)) for t in taus]
    tau = float(taus[int(np.argmin(errs))])
    a, b, s = fit_edge_map(counts_at(res_cal["scores"][clean], tau), nu)
    T_b = fit_temperature(res_cal["local_logits"], y)
    T_e = fit_temperature(res_cal["edge_logits"][clean], y)
    return {"T_e": T_e, "tau": tau, "edge_a": a, "edge_b": b, "edge_s": s, "T_b": T_b, "cal_count_mae_view": float(min(errs))}


def summarize(res: dict, rows: list[dict], cal: dict) -> dict:
    y = np.array([r["label"] for r in rows])
    nv = np.array([r["n_view"] for r in rows], float)
    nu = np.array([r["n_union"] for r in rows], float)
    loc_p = torch.softmax(torch.from_numpy(res["local_logits"]) / cal.get("T_b", 1.0), -1).numpy()
    loc_c = (loc_p.argmax(1) == y).astype(float)
    out = {"n": int(len(y)), "local_acc": float(loc_c.mean()),
           "local_count_mae": float(np.mean(np.abs(np.expm1(res["local_mu"]) - nu))),
           "local_ece_msp": ece(loc_p.max(1), loc_c), "local_auroc_msp": failure_auroc(loc_p.max(1), loc_c),
           "edge": {}}
    for s, sc in res["scores"].items():
        n_hat = counts_at(sc, cal["tau"])
        d_p = np.exp(edge_logits(n_hat, cal["edge_a"], cal["edge_b"], cal["edge_s"]))
        e_p = torch.softmax(torch.from_numpy(res["edge_logits"][s]) / cal.get("T_e", 1.0), -1).numpy()
        e_c = (e_p.argmax(1) == y).astype(float)
        out["edge"]["clean" if s is None else f"{s:g}dB"] = {
            "acc": float(e_c.mean()), "acc_from_detections": float((d_p.argmax(1) == y).mean()),
            "view_count_mae": float(np.mean(np.abs(n_hat - nv))),
            "ece": ece(e_p.max(1), e_c), "auroc": failure_auroc(e_p.max(1), e_c),
            "acc_per_class": [float(e_c[y == k].mean()) if (y == k).any() else float("nan") for k in range(NUM_CLASSES)]}
    return out


#  training
def _write_log(out: Path, hist: list[dict]) -> None:
    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[-1]))
        w.writeheader()
        w.writerows(hist)


def train(model, rows, args, device, dsets, out: Path) -> dict:
    tr = rows["train"][::args.train_stride]
    dl = make_loader(DetFrames(tr, args.img_root, dsets["boxes"], dsets["grids"], True, args.seed), args.batch, True,
                     args.workers)
    fixed = args.select_on == "none"                  # cross-fitting: fixed epochs, the last one is kept
    if not fixed:
        sel_rows = rows["cal"] if args.select_on == "cal" else rows["heldout"] + rows["cal"]
        sel = sel_rows[::args.select_stride]
        dl_sel = make_loader(DetFrames(sel, args.img_root, dsets["boxes"], dsets["grids"], False), args.batch, False,
                             args.workers)
    groups = model.parameter_groups()
    opt = torch.optim.AdamW([{"params": groups["pretrained"], "lr": args.lr},
                             {"params": groups["new"], "lr": args.lr * 3}], weight_decay=1e-4)
    steps, warm = args.epochs * len(dl), min(1000, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / max(1, steps)))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist, best, rng = [], -1.0, np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, agg, n = time.time(), {}, 0
        for b, (x, tgts, loc, _) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loc = {k: v.to(device) for k, v in loc.items()}
            snr = torch.from_numpy(rng.uniform(args.snr_min, args.snr_max, len(x))).float().to(device)
            clean = torch.from_numpy(rng.random(len(x)) < args.p_clean).to(device)
            snr = torch.where(clean, torch.full_like(snr, 60.0), snr)          # 60 dB ~ noiseless
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                z, loc_out = model.uav_forward(x)
                det_losses = model.server_forward(x, awgn(z.float(), snr).to(z.dtype), tgts)
                edge_out = model.edge_belief()
            losses = {k: v.float() for k, v in det_losses.items()}
            losses.update({k: args.lambda_local * v for k, v in local_losses(loc_out, loc).items()})
            losses.update({k.replace("local_", "edge_"): args.lambda_edge * v
                           for k, v in local_losses(edge_out, loc).items()})
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("[train] non-finite loss, batch skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
            if b % 200 == 0:
                print(f"[train] epoch {ep} batch {b}/{len(dl)}  loss {float(loss):.3f}  "
                      + "  ".join(f"{k} {float(v):.3f}" for k, v in losses.items()), flush=True)
            if args.smoke and b >= 2:
                break
        if fixed:
            row = {"epoch": ep, **{k: v / max(n, 1) for k, v in agg.items()}, "minutes": (time.time() - t0) / 60}
            hist.append(row)
            print(f"[epoch {ep}] done ({row['minutes']:.1f} min)", flush=True)
            _write_log(out, hist)
            if ep == args.epochs:
                torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": ep}, out / "best.pt")
            continue
        res = run_eval(model, dl_sel, device, snrs=(None, 5.0))
        cal = calibrate(res, sel, clean=None)
        sm = summarize(res, sel, cal)
        row = {"epoch": ep, **{k: v / max(n, 1) for k, v in agg.items()}, "sel_local_acc": sm["local_acc"],
               "sel_edge_acc_clean": sm["edge"]["clean"]["acc"], "sel_edge_acc_5dB": sm["edge"]["5dB"]["acc"],
               "sel_view_count_mae": sm["edge"]["clean"]["view_count_mae"], "minutes": (time.time() - t0) / 60}
        hist.append(row)
        print(f"[epoch {ep}] {args.select_on}: local acc {row['sel_local_acc']:.3f}  edge acc clean "
              f"{row['sel_edge_acc_clean']:.3f} / 5 dB {row['sel_edge_acc_5dB']:.3f}  edge view-count MAE "
              f"{row['sel_view_count_mae']:.1f}  ({row['minutes']:.1f} min)", flush=True)
        _write_log(out, hist)
        score = 0.5 * (row["sel_edge_acc_clean"] + row["sel_edge_acc_5dB"]) + 0.5 * row["sel_local_acc"]
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": ep}, out / "best.pt")
    return {"epochs": hist, "best_score": best}


# export
def export(model, rows, args, device, dsets, cal: dict, out_cache: Path, snr_grid) -> dict:
    """Perception caches (local outputs, simulator format) and edge results per SNR for every split."""
    rep = {}
    for split, rs in rows.items():
        if not rs:
            continue
        dl = make_loader(DetFrames(rs, args.img_root, dsets["boxes"], dsets["grids"], False), args.batch, False,
                         args.workers)
        res = run_eval(model, dl, device, snrs=(None, *snr_grid))
        y = np.array([r["label"] for r in rs])
        lg = res["local_logits"].astype(np.float64)
        p_raw = torch.softmax(torch.from_numpy(lg), -1).numpy()
        p = torch.softmax(torch.from_numpy(lg) / cal["T_b"], -1).numpy()
        a = {"seq_id": np.array([r["seq"] for r in rs]), "scene_id": np.array([r["scene_id"] for r in rs]),
             "view_id": np.array([r["view"] for r in rs], np.int64), "frame": np.array([r["frame"] for r in rs], np.int64),
             "label": y.astype(np.int64), "logits": lg.astype(np.float32), "belief": p.astype(np.float32),
             "delta": p.max(1).astype(np.float32), "delta_raw": p_raw.max(1).astype(np.float32),
             "msp_raw": p_raw.max(1).astype(np.float32), "n_view": np.array([r["n_view"] for r in rs], np.int64),
             "n_union": np.array([r["n_union"] for r in rs], np.int64),
             "count_pred": np.expm1(res["local_mu"]).astype(np.float32), "count_logsig": res["local_ls"].astype(np.float32)}
        order = np.lexsort((a["view_id"], a["frame"], a["seq_id"]))
        a = {k: v[order] for k, v in a.items()}
        validate_perception_cache(a)
        save_cache(out_cache / "perception_raw" / f"perception_{split}.npz", a, PERCEPTION_KEYS)
        grid = [s for s in snr_grid]
        e_correct, e_belief, e_count = [], [], []
        for s in [None, *grid]:
            n_hat = counts_at(res["scores"][s], cal["tau"])
            eb = torch.softmax(torch.from_numpy(res["edge_logits"][s]) / cal["T_e"], -1).numpy()
            e_belief.append(eb)
            e_correct.append((eb.argmax(1) == y).astype(np.float32))
            e_count.append(n_hat)
        (out_cache / "edge_raw").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_cache / "edge_raw" / f"edge_{split}.npz", snr_grid_db=np.array(grid, float), edge_correct_clean=e_correct[0][order],
            edge_correct=np.stack(e_correct[1:], 1)[order], edge_belief=np.stack(e_belief[1:], 1)[order].astype(np.float32),
            edge_count=np.stack(e_count[1:], 1)[order].astype(np.float32))
        rep[split] = summarize(res, rs, cal)
        print(f"[export] {split:8s} local acc {rep[split]['local_acc']:.3f}  edge acc clean "
              f"{rep[split]['edge']['clean']['acc']:.3f}  " + "  ".join(
                  f"{k} {v['acc']:.3f}" for k, v in rep[split]["edge"].items() if k != "clean"), flush=True)
    # simulator files: train = heldout + cal (frames the model was not trained on), cal, test
    for rl, srcs in {"train": ("heldout", "cal"), "cal": ("cal",), "test": ("test",)}.items():
        parts = [np.load(out_cache / "perception_raw" / f"perception_{s}.npz") for s in srcs
                 if (out_cache / "perception_raw" / f"perception_{s}.npz").exists()]
        if parts:
            a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
            save_cache(out_cache / f"perception_{rl}.npz", a, PERCEPTION_KEYS)
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="caches/mdmt/frames.csv")
    ap.add_argument("--img_root", default="caches/mdmt/img1024")
    ap.add_argument("--boxes", default="caches/mdmt/boxes.npz")
    ap.add_argument("--grids", default="caches/mdmt/grids_36x64.npz")
    ap.add_argument("--out", default="runs_perception/split")
    ap.add_argument("--cache_out", default="caches/mdmt")
    ap.add_argument("--local_head", default="payload", choices=["payload", "rich"],
                    help="UAV local head: on the payload (v0.15) or a counting back-end on layer2 features (v0.16)")
    ap.add_argument("--c_b", type=int, default=8, help="payload channels (k_s = c_b * H/16 * W/16 / 2 symbols)")
    ap.add_argument("--epochs", type=int, default=4, help="the first run peaked at epoch 2 on unseen scenes")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_local", type=float, default=0.5)
    ap.add_argument("--lambda_edge", type=float, default=0.5)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--p_clean", type=float, default=0.3, help="share of training samples sent without noise")
    ap.add_argument("--train_stride", type=int, default=2, help="every k-th training frame (consecutive frames are near-identical)")
    ap.add_argument("--select_stride", type=int, default=4, help="every k-th selection frame for epoch selection")
    ap.add_argument("--select_on", default="auto", choices=["auto", "heldout+cal", "cal", "none"],
                    help="frames used to pick the best epoch; auto = cal only when --init_mmdet is used (the MDMT "
                         "detector was trained on the MDMT train split, which contains our heldout sequences)")
    ap.add_argument("--snr_grid", default="-5,0,5,10,20", help="SNRs [dB] for the exported edge results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--init_mmdet", default="", help="ResNet-50 weights from an MDMT mmdetection checkpoint "
                    "(only if it was trained without our cal/test sequences)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--no_export", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = read_frames(args.frames)
    rows = {s: [r for r in frames if r["our_split"] == s] for s in ("train", "heldout", "cal", "test")}
    dsets = {"boxes": load_boxes(args.boxes) if Path(args.boxes).exists() else None,
             "grids": load_grids(args.grids) if Path(args.grids).exists() else None}
    if dsets["boxes"] is None:
        raise SystemExit(f"{args.boxes} not found: python -m uavsc.data.mdmt boxes --root <MDMT folder>")
    if args.smoke:
        rows = {s: v[:: max(1, len(v) // 24)][:24] for s, v in rows.items()}
        args.epochs, args.train_stride, args.select_stride = 1, 1, 1
        args.out, args.cache_out = args.out + "_smoke", str(Path(args.cache_out) / "smoke_split")
        args.snr_grid = "0,10"
    out, cache = Path(args.out), Path(args.cache_out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    with Image.open(Path(args.img_root) / rows["train"][0]["image"]) as im:
        W, H = im.size
    model = SplitDetector(args.c_b, pretrained=not args.no_pretrained, min_size=H, max_size=W,
                          local_head=args.local_head).to(device)
    if args.select_on == "auto":
        args.select_on = "cal" if args.init_mmdet else "heldout+cal"
    if args.init_mmdet:
        from .splitnet import load_mmdet_backbone

        print(f"initialised {load_mmdet_backbone(model, args.init_mmdet)} backbone tensors from {args.init_mmdet}")
    n_uav = sum(p.numel() for p in model.uav.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"device {device}; image {W}x{H}; UAV head {n_uav / 1e6:.1f} M params, edge {(n_all - n_uav) / 1e6:.1f} M; "
          f"payload k_s = {model.payload_symbols(H, W)} complex symbols", flush=True)
    report = {"k_s": model.payload_symbols(H, W), "uav_params_M": n_uav / 1e6, "edge_params_M": (n_all - n_uav) / 1e6}
    if not args.eval_only:
        report["train"] = train(model, rows, args, device, dsets, out)
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    report["best_epoch"] = ckpt["epoch"]
    dl_cal = make_loader(DetFrames(rows["cal"], args.img_root, dsets["boxes"], dsets["grids"], False), args.batch,
                         False, args.workers)
    cal = calibrate(run_eval(model, dl_cal, device, snrs=(None,)), rows["cal"], clean=None)
    model.uav.T_b.fill_(cal["T_b"])
    model.edge_count.T_e.fill_(cal["T_e"])
    report["calibration"] = cal
    print(f"[calibration] edge T_e {cal['T_e']:.3f}  tau {cal['tau']:.2f}  edge map a {cal['edge_a']:.3f} b {cal['edge_b']:.3f} "
          f"s {cal['edge_s']:.3f}  local T_b {cal['T_b']:.3f}", flush=True)
    if not args.no_export:
        snr_grid = [float(s) for s in args.snr_grid.split(",") if s.strip()]
        report["export"] = export(model, rows, args, device, dsets, cal, cache, snr_grid)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
