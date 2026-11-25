
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import apply_overrides, load_config
from ..data.mdmt import NUM_CLASSES, SCENE_EDGES, load_grids, read_frames
from .calibration import bce, ece, failure_auroc, fit_binary_temperature, fit_temperature
from .dataset import FramesDataset
from .spcrnet import build

GRIDS: dict | None = None
GRID_SHAPE: tuple[int, int] = (9, 16)


def loader(rows, img_root, train: bool, batch: int, workers: int, seed: int = 0) -> DataLoader:
    return DataLoader(FramesDataset(rows, img_root, train=train, seed=seed, grids=GRIDS, grid_shape=GRID_SHAPE),
                      batch_size=batch,
                      shuffle=train, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=workers > 0)


@torch.no_grad()
def infer(model, dl, device, keep_tokens: bool = False) -> dict[str, np.ndarray]:
    """Eval-mode pass over a loader; returns numpy arrays in dataset order."""
    model.eval()
    out: dict[str, list] = {}
    for x, y, idx, _, _ in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            o = model(x.to(device, non_blocking=True))
        keys = ["z_bar", "logits", "conf_logit", "log_count", "log_sigma"] + (["tokens"] if keep_tokens else [])
        for k in keys:
            out.setdefault(k, []).append(o[k].float().cpu().numpy())
        out.setdefault("label", []).append(y.numpy())
        out.setdefault("idx", []).append(idx.numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    order = np.argsort(res["idx"])
    return {k: v[order] for k, v in res.items()}


def _blur(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([1.0, 2.0, 1.0], device=x.device)
    k = (k[:, None] * k[None, :]) / 16.0
    return F.conv2d(x[:, None], k.view(1, 1, 3, 3), padding=1)[:, 0]


def density_loss(pred: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Squared error between blurred density maps (a 3x3 binomial blur tolerates one-cell localisation errors),
    plus a count-consistency term. The target is sum-pooled to the prediction's grid if the sizes differ."""
    if pred.shape[-2:] != grid.shape[-2:]:
        gh, gw = grid.shape[-2:]
        h, w = pred.shape[-2:]
        grid = F.adaptive_avg_pool2d(grid[:, None], (h, w))[:, 0] * (gh * gw) / (h * w)
    cell = ((_blur(pred) - _blur(grid)) ** 2).mean()
    tot = F.smooth_l1_loss(torch.log1p(pred.sum((-2, -1))), torch.log1p(grid.sum((-2, -1))))
    return cell + tot


def gaussian_nll(mu: torch.Tensor, log_sig: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (0.5 * ((target - mu) / torch.exp(log_sig)) ** 2 + log_sig).mean()


#stage 1
def train_stage1(model, rows, args, device, log_path: Path) -> dict:
    groups = model.parameter_groups()["stage1"]
    bb = {id(p) for p in model.backbone.parameters()}
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        groups = [p for p in groups if id(p) not in bb]
    params = [{"params": [p for p in groups if id(p) not in bb], "lr": args.lr}]
    if not args.freeze_backbone:
        params.append({"params": [p for p in groups if id(p) in bb], "lr": args.lr * args.backbone_lr_mult})
    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    tr = rows["train"][::args.train_stride]
    dl_tr = loader(tr, args.img_root, True, args.batch, args.workers, args.seed)
    sel_rows = rows["heldout"] + (rows["cal"] if args.select_on == "heldout+cal" else [])
    dl_ho = loader(sel_rows, args.img_root, False, args.batch * 2, args.workers)
    steps = args.epochs * len(dl_tr)
    warm = max(1, len(dl_tr) // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps)))))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    lam_d = args.lambda_density if GRIDS is not None else 0.0
    best, best_state, hist = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()                        # frozen BN statistics too
        t0 = time.time()
        acc = {"ce": 0.0, "dens": 0.0, "cnt": 0.0, "ok": 0, "n": 0}
        for b, (x, y, _, g, nu) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            g, nu = g.to(device, non_blocking=True), nu.to(device, non_blocking=True).float()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                Z, dens = model.encode_full(x)
                _, logits, lc, ls = model.head(Z, dens)
            ce = F.nll_loss(logits.float(), y)                     # logits = log b (count-derived belief)
            ld = density_loss(dens.float(), g)
            lcnt = gaussian_nll(lc.float(), ls.float(), torch.log1p(nu))
            loss = args.lambda_ce * ce + lam_d * ld + args.lambda_count * lcnt
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(groups, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = len(y)
            acc["ce"] += float(ce) * n
            acc["dens"] += float(ld) * n
            acc["cnt"] += float(lcnt) * n
            acc["ok"] += int((logits.argmax(-1) == y).sum())
            acc["n"] += n
            if args.smoke and b >= 2:
                break
        ho = infer(model, dl_ho, device)
        acc_ho = float((ho["logits"].argmax(1) == ho["label"]).mean())
        cnt_ho = np.expm1(ho["log_count"])
        nu_ho = np.array([r["n_union"] for r in sel_rows], float)
        N = max(acc["n"], 1)
        row = {"epoch": ep, "train_ce": acc["ce"] / N, "train_density": acc["dens"] / N, "train_count": acc["cnt"] / N,
               "train_acc": acc["ok"] / N, "select_acc": acc_ho,
               "select_count_mae": float(np.mean(np.abs(cnt_ho - nu_ho))),
               "select_acc_from_count": float(np.mean(np.searchsorted(SCENE_EDGES, cnt_ho, side="right") == ho["label"])),
               "minutes": (time.time() - t0) / 60}
        hist.append(row)
        print(f"[stage1] epoch {ep:2d}  CE {row['train_ce']:.3f}  dens {row['train_density']:.3f}  "
              f"train acc {row['train_acc']:.3f}  {args.select_on} acc {acc_ho:.3f} (from count "
              f"{row['select_acc_from_count']:.3f}, count MAE {row['select_count_mae']:.1f})  ({row['minutes']:.1f} min)",
              flush=True)
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(hist)
        if acc_ho > best:
            best = acc_ho
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_select_acc": best, "select_on": args.select_on, "epochs": hist}


# stage 2
HEADS = {"logistic": (1.0, 0.0), "mlp_scalar": (0.0, 0.0), "mlp_zbar": (0.0, 1.0)}   # (logistic flag, zbar gate)


def _set_head(model, name: str) -> None:
    lg, gate = HEADS[name]
    model.logistic_head.fill_(lg)
    model.zbar_gate.fill_(gate)


def _train_head(model, f: dict, epochs: int, device, seed: int = 0) -> None:
    """Fit the active confidence head on cached features f (BCE), fresh initialisation, fixed epochs."""
    for m in list(model.conf_head.modules()) + [model.conf_linear]:
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
    params = list(model.conf_linear.parameters()) if float(model.logistic_head) > 0.5 else list(model.conf_head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2 if len(params) == 2 else 1e-3, weight_decay=1e-2)
    z, lg, lc, ls, c = (torch.from_numpy(f[k]).to(device) for k in ("z_bar", "logits", "log_count", "log_sigma", "c"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(c), generator=g).to(device)
        for i in range(0, len(c), 512):
            j = perm[i:i + 512]
            loss = F.binary_cross_entropy_with_logits(model.confidence(z[j], lg[j], lc[j], ls[j]), c[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def _head_prob(model, f: dict, device) -> np.ndarray:
    return torch.sigmoid(model.confidence(*(torch.from_numpy(f[k]).to(device)
                                            for k in ("z_bar", "logits", "log_count", "log_sigma")))).cpu().numpy()


def _subset(f: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in f.items()}


def train_stage2(model, rows, args, device) -> dict:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameter_groups()["stage2"]:
        p.requires_grad_(True)

    def feats_of(rs):
        r = infer(model, loader(rs, args.img_root, False, args.batch * 2, args.workers), device)
        r["c"] = (r["logits"].argmax(1) == r["label"]).astype(np.float32)
        r["seq"] = np.array([x["seq"] for x in rs])
        return r

    ho, cal = feats_of(rows["heldout"]), feats_of(rows["cal"])
    pool = {k: np.concatenate([ho[k], cal[k]]) for k in ho}
    seqs = sorted(set(pool["seq"].tolist()), key=int)
    report = {"heldout_frac_wrong": float(1 - ho["c"].mean()), "pool_frac_wrong": float(1 - pool["c"].mean()),
              "n_scenes": len(seqs)}
    print(f"[stage2] heldout + cal: {len(pool['c'])} samples, {len(seqs)} sequences, "
          f"fraction wrong {report['pool_frac_wrong']:.2f}", flush=True)
    msp = torch.softmax(torch.from_numpy(pool["logits"]), -1).max(-1).values.numpy()
    report["loso_auroc_msp"] = failure_auroc(msp, pool["c"])
    for name in HEADS:
        _set_head(model, name)
        oof = np.zeros(len(pool["c"]))
        for s in seqs:                                    # leave one sequence (scene) out
            te = pool["seq"] == s
            _train_head(model, _subset(pool, ~te), args.head_epochs, device, args.seed)
            oof[te] = _head_prob(model, _subset(pool, te), device)
        report[name] = {"loso_auroc": failure_auroc(oof, pool["c"]), "loso_ece": ece(oof, pool["c"])}
        print(f"[stage2] head {name:10s}: leave-one-scene-out AUROC {report[name]['loso_auroc']:.3f}  "
              f"ECE {report[name]['loso_ece']:.3f}", flush=True)
    print(f"[stage2] (MSP for reference: AUROC {report['loso_auroc_msp']:.3f})", flush=True)
    choice = max(HEADS, key=lambda k: report[k]["loso_auroc"])
    _set_head(model, choice)
    _train_head(model, ho, args.head_epochs, device, args.seed)            # final fit on heldout only
    report["chosen"] = choice
    print(f"[stage2] chosen head: {choice}", flush=True)
    for p in model.parameters():
        p.requires_grad_(True)
    return report


#stage 3
def calibrate_stage3(model, rows, args, device) -> dict:
    cal = infer(model, loader(rows["cal"], args.img_root, False, args.batch * 2, args.workers), device)
    correct = (cal["logits"].argmax(1) == cal["label"]).astype(float)
    T_b = fit_temperature(cal["logits"], cal["label"])
    T_d = fit_binary_temperature(cal["conf_logit"], correct)
    model.T_b.fill_(T_b)
    model.T_delta.fill_(T_d)
    print(f"[stage3] T_b = {T_b:.3f}  T_delta = {T_d:.3f}  (cal accuracy {correct.mean():.3f}; "
          f"cal BCE of delta {bce(cal['conf_logit'], correct, T_d):.3f})", flush=True)
    return {"T_b": T_b, "T_delta": T_d, "cal_acc": float(correct.mean())}


def main(argv=None) -> None:
    global GRIDS, GRID_SHAPE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--frames", default="caches/mdmt/frames.csv")
    ap.add_argument("--grids", default="", help="default: grids.npz next to frames.csv (density supervision)")
    ap.add_argument("--img_root", default="caches/mdmt/img")
    ap.add_argument("--out", default="runs_perception/spcrnet")
    ap.add_argument("--cache_out", default="caches/mdmt", help="where the perception caches are written")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone_lr_mult", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lambda_density", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--lambda_count", type=float, default=1.0)
    ap.add_argument("--head_epochs", type=int, default=30)
    ap.add_argument("--freeze_backbone", action="store_true", help="train only the layers after the backbone")
    ap.add_argument("--select_on", default="heldout+cal", choices=["heldout", "heldout+cal"],
                    help="sequences used to pick the best stage-1 epoch (10 scenes are less noisy than 5)")
    ap.add_argument("--stage1_only", action="store_true", help="diagnostics: stop after stage 1 (see sweep.py)")
    ap.add_argument("--train_stride", type=int, default=1, help="use every k-th training frame (speed)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--export_only", action="store_true", help="skip training; load <out>/spcrnet.pt and export")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (few batches) to catch errors fast")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = apply_overrides(load_config(args.config), args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = read_frames(args.frames)
    if args.grids:
        gpath = Path(args.grids)
    else:                                                  # finest available grid
        cands = [Path(args.frames).with_name(n) for n in ("grids_36x64.npz", "grids.npz")]
        gpath = next((c for c in cands if c.exists()), cands[0])
    GRIDS = load_grids(gpath) if gpath.exists() else None
    if GRIDS is None:
        print(f"[warn] {gpath} not found: training WITHOUT density supervision "
              f"(run: python -m uavsc.data.mdmt grids --root <MDMT folder> --grid_h 36 --grid_w 64)", flush=True)
    else:
        GRID_SHAPE = tuple(next(iter(GRIDS.values())).shape)
        print(f"density supervision from {gpath} ({GRID_SHAPE[0]} x {GRID_SHAPE[1]} cells)", flush=True)
    rows = {s: [r for r in frames if r["our_split"] == s] for s in ("train", "heldout", "cal", "test")}
    if args.smoke:
        rows = {s: v[:: max(1, len(v) // 96)][:96] for s, v in rows.items()}
        args.epochs, args.head_epochs, args.out = 1, 3, str(Path(args.out).with_name(Path(args.out).name + "_smoke"))
        args.cache_out = str(Path(args.cache_out) / "smoke")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {device}; rows per split { {s: len(v) for s, v in rows.items()} }", flush=True)

    model = build(cfg.perception, NUM_CLASSES, pretrained=not args.no_pretrained,
                  count_edges=tuple(int(e) for e in SCENE_EDGES)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"SPCRNet ({cfg.perception.backbone}): {n_par / 1e6:.2f} M parameters", flush=True)
    meta = {"backbone": cfg.perception.backbone, "num_classes": NUM_CLASSES, "perception": vars(cfg.perception),
            "params_M": n_par / 1e6, "args": vars(args)}
    if args.export_only:
        model.load_state_dict(torch.load(out / "spcrnet.pt", map_location=device)["state_dict"])
        report = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
    else:
        report = {"stage1": train_stage1(model, rows, args, device, out / "stage1_log.csv")}
        if args.stage1_only:
            (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
            print(f"stage 1 only: {out / 'metrics.json'}")
            return
        report["stage2"] = train_stage2(model, rows, args, device)
        report["stage3"] = calibrate_stage3(model, rows, args, device)
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out / "spcrnet.pt")

    from .export_cache import export_all

    report["export"] = export_all(model, rows, args, device)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"saved {out / 'spcrnet.pt'} and {out / 'metrics.json'}; caches in {args.cache_out}")


if __name__ == "__main__":
    main()
