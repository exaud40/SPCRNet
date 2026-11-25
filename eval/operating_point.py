# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:03:29 2026

@author: Exaud
"""

from __future__ import annotations
import argparse
import numpy as np
from ..comm import channel as ch
from ..config import Config, dbm_to_w, load_config
from ..system.latency_energy import A2A, A2G, comm_energy, decision_latency, timeliness, tx_latency
from ..system.mobility import edge_positions, init_positions


def analyze(cfg: Config, n_trials: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    C, S, K, Q, net = cfg.channel, cfg.semantic, cfg.compute, cfg.qoe, cfg.network
    N0 = dbm_to_w(C.N0_dbm_hz + C.NF_db)
    P, P_l = dbm_to_w(C.P_max_dbm), dbm_to_w(C.P_l_dbm)
    grid = np.asarray(S.snr_grid_db, dtype=np.float64)

    out: dict[str, list] = {k: [] for k in ("a2a_db", "a2g_db_one", "a2g_db_all")}
    for _ in range(n_trials):
        leader, fol = init_positions(rng, net.M, net.area_radius, net.d_min, net.d_max)
        edges = edge_positions(rng, net.J, net.edge_radius)
        r = np.linalg.norm(edges - leader, axis=1)
        pl = ch.a2g_mean_pathloss_db(r, net.H, C.f_c, C.a, C.b, C.eta_los_db, C.eta_nlos_db)
        Hg = ch.a2g_gain(pl, ch.nakagami_power(rng, C.nakagami_m, net.J)).max()   # best server
        h_ul = ch.a2a_gain(np.linalg.norm(fol - leader, axis=1), C.f_c)
        for B, key in ((C.B_tot, "a2g_db_one"), (C.B_tot / net.M, "a2g_db_all")):
            g1 = ch.snr(P, h_ul, B, N0)
            g2 = ch.snr(P_l, Hg, B, N0)
            out[key].extend(10 * np.log10(ch.af_end_to_end_snr(g1, g2)))
        # A2A between a pair of followers, one responder, equal band share
        d_pair = np.linalg.norm(fol[0] - fol[1]) or 1.0
        out["a2a_db"].append(10 * np.log10(ch.a2a_snr(P, ch.a2a_gain(d_pair, C.f_c), C.B_tot / net.M, 1, N0)))

    res: dict = {}
    for k, v in out.items():
        v = np.asarray(v)
        res[k] = {"p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)),
                  "in_grid": float(np.mean((v >= grid[0]) & (v <= grid[-1])))}

    for B, tag in ((C.B_tot, "one_active"), (C.B_tot / net.M, "all_active")):
        for mode, name in ((A2A, "A2A"), (A2G, "A2G")):
            m = np.array([mode])
            dtx = tx_latency(m, np.array([B]), np.array([1]), S.k_b, S.k_s)
            D = decision_latency(m, dtx, K.chi_L, K.chi_F, K.chi_G, K.f_U, K.f_G)
            E = comm_energy(m, np.array([P]), np.array([B]), np.array([1]), P_l, S.k_b, S.k_s, cfg.channel.E0)
            res[f"{name}_{tag}"] = {"D_ms": float(D[0] * 1e3), "timeliness": float(timeliness(D, Q.D_th, Q.zeta_D)[0]),
                                    "E_mJ": float(E[0] * 1e3)}
    res["local_D_ms"] = K.chi_L / K.f_U * 1e3
    res["E_always_a2g_mJ"] = res["A2G_all_active"]["E_mJ"]
    res["E_bar_mJ"] = cfg.constraint.E_bar * 1e3
    return res


def report(res: dict, cfg: Config) -> list[str]:
    """Print the analysis and return a list of warnings (empty = operating point is informative)."""
    warn, notes = [], []
    grid_hi = float(np.max(cfg.semantic.snr_grid_db))
    print("SNR [dB] (p10 / median / p90, fraction inside codec grid):")
    for k in ("a2a_db", "a2g_db_one", "a2g_db_all"):
        r = res[k]
        print(f"  {k:11s} {r['p10']:6.1f} / {r['median']:6.1f} / {r['p90']:6.1f}   in-grid {r['in_grid']:.0%}")
        if k == "a2a_db" and r["in_grid"] < 0.5 and r["median"] > grid_hi:
            # Short LoS links between nearby UAVs: high SNR is physical, not a config error.
            notes.append("A2A SNR exceeds the codec grid (short LoS links): A2A fidelity saturates, "
                         "so the policy should learn low A2A power")
        elif r["in_grid"] < 0.5:
            warn.append(f"{k}: only {r['in_grid']:.0%} of SNRs inside the codec grid -> fidelity saturates")
    print(f"Latency [ms] (D_th = {cfg.qoe.D_th*1e3:.0f} ms): local {res['local_D_ms']:.1f}")
    for k in ("A2A_one_active", "A2A_all_active", "A2G_one_active", "A2G_all_active"):
        r = res[k]
        print(f"  {k:15s} D = {r['D_ms']:7.2f}   sigma_D = {r['timeliness']:.3f}   E = {r['E_mJ']:.3f} mJ")
    if res["A2G_all_active"]["timeliness"] > 0.99:
        warn.append("deadline never binds: A2G meets D_th even when every follower offloads")
    ratio = res["E_bar_mJ"] / res["E_always_a2g_mJ"]
    print(f"Energy: Always-A2G = {res['E_always_a2g_mJ']:.3f} mJ/follower/slot, "
          f"E_bar = {res['E_bar_mJ']:.3f} mJ ({ratio:.0%} of Always-A2G)")
    if ratio >= 1.0:
        warn.append("energy budget never binds: E_bar >= Always-A2G energy")
    for n in notes:
        print("NOTE:", n)
    print("WARNINGS:" if warn else "OK: all trade-offs bind.")
    for w in warn:
        print("  -", w)
    return warn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. channel.B_tot=5e6 constraint.E_bar=2e-4")
    args = ap.parse_args()
    from ..config import apply_overrides

    cfg = apply_overrides(load_config(args.config), args.set)
    report(analyze(cfg, args.trials), cfg)


if __name__ == "__main__":
    main()
