# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:09:19 2026

@author: Exaud
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from .splitnet import SplitDetector, module_cost


def measure(c_b: int = 8, h: int = 576, w: int = 1024, local_head: str = "payload") -> dict:
    m = SplitDetector(c_b, pretrained=False, min_size=h, max_size=w, local_head=local_head).eval()
    x = torch.rand(1, 3, h, w)
    uav_g, uav_p = module_cost(m.uav, lambda: m.uav_forward(x))
    with torch.no_grad():
        z, _ = m.uav_forward(x)

    def edge():
        m.server_forward(x, z)
        m.edge_belief()

    edge_mods = torch.nn.ModuleList([m.server_backbone, m.detector.rpn, m.detector.roi_heads, m.edge_count])
    edge_g, edge_p = module_cost(edge_mods, edge)
    return {"chi_L": 2e9 * uav_g, "chi_G": 2e9 * edge_g, "k_s": m.payload_symbols(h, w), "uav_gmacs": uav_g,
            "edge_gmacs": edge_g, "uav_params_M": uav_p, "edge_params_M": edge_p, "image": [w, h], "c_b": c_b,
            "local_head": local_head}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c_b", type=int, default=8)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--local_head", default="payload", choices=["payload", "rich"])
    ap.add_argument("--f_U", type=float, default=5e12, help="effective onboard FLOPS (assumption)")
    ap.add_argument("--f_G", type=float, default=3e13, help="effective edge FLOPS (assumption)")
    ap.add_argument("--base", default="configs/default.yaml")
    ap.add_argument("--fidelity", default="caches/mdmt_cf/fidelity_test.npz", help="for the SNR grid")
    ap.add_argument("--out", default="configs/mdmt.yaml")
    ap.add_argument("--ebar_frac", type=float, default=0.062,
                    help="E_bar as a share of the Always-A2G energy (synthetic default: 0.1 / 1.609 mJ = 6.2 %%)")
    args = ap.parse_args(argv)
    r = measure(args.c_b, args.height, args.width, args.local_head)
    print(f"UAV  side: {r['uav_gmacs']:.1f} GMACs ({r['chi_L'] / 1e9:.1f} GFLOPs), {r['uav_params_M']:.1f} M params "
          f"-> local latency {r['chi_L'] / args.f_U * 1e3:.1f} ms at f_U = {args.f_U:.1e}")
    print(f"edge side: {r['edge_gmacs']:.1f} GMACs ({r['chi_G'] / 1e9:.1f} GFLOPs), {r['edge_params_M']:.1f} M params "
          f"-> edge latency {r['chi_G'] / args.f_G * 1e3:.1f} ms at f_G = {args.f_G:.1e}")
    print(f"payload k_s = {r['k_s']} complex symbols")
    with open(args.base, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["semantic"]["k_s"] = int(r["k_s"])
    cfg["compute"].update({"chi_L": float(r["chi_L"]), "chi_G": float(r["chi_G"]), "f_U": float(args.f_U),
                           "f_G": float(args.f_G)})
    if Path(args.fidelity).exists():
        cfg["semantic"]["snr_grid_db"] = [float(v) for v in np.load(args.fidelity)["snr_grid_db"]]
    from ..config import Config, _fill
    from ..eval.operating_point import analyze

    e_a2g = analyze(_fill(Config, cfg), 500)["E_always_a2g_mJ"] * 1e-3
    cfg["constraint"]["E_bar"] = float(f"{args.ebar_frac * e_a2g:.3g}")
    print(f"Always-A2G energy {e_a2g * 1e3:.3f} mJ/follower/slot -> E_bar = {cfg['constraint']['E_bar'] * 1e3:.3f} mJ "
          f"({args.ebar_frac:.1%}, same tightness as the synthetic study)")
    hdr = (f"# MDMT configuration written by uavsc.perception.cost from {args.base}.\n"
           f"# Measured: chi_L, chi_G ({r['uav_gmacs']:.1f} / {r['edge_gmacs']:.1f} GMACs at {args.width}x{args.height}), "
           f"k_s (c_b = {args.c_b}). Assumed: f_U = {args.f_U:.1e}, f_G = {args.f_G:.1e} FLOPS.\n"
           f"# E_bar = {args.ebar_frac:.3f} x Always-A2G energy.\n")
    Path(args.out).write_text(hdr + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out}; next: python -m uavsc.eval.operating_point --config {args.out}")


if __name__ == "__main__":
    main()
