# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:39:30 2026

@author: Exaud
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..comm import channel as ch
from ..comm.fidelity import interp_on_grid, logistic_fidelity
from ..config import Config, dbm_to_w
from ..system.fusion import log_linear_pool
from ..system.latency_energy import A2A, A2G, LOCAL, comm_energy, decision_latency, timeliness, tx_latency
from ..system.mobility import (edge_positions, init_positions, neighbor_sets, random_waypoints,
                               step_followers)
from ..system.resources import grant_bandwidth, scale_power

# observation normalization constants 
_GAIN_DB_REF, _GAIN_DB_SCALE = -72.0, 12.0     # A2A gains
_PL_DB_REF, _PL_DB_SCALE = 105.0, 20.0         # A2G mean path loss


@dataclass
class StepOutput:
    obs: np.ndarray          # (M, obs_dim)
    state: np.ndarray        # (state_dim,)
    mask: np.ndarray         # (M, 3) feasible modes for the NEXT step
    delta: np.ndarray        # (M,) confidence for the NEXT step 
    reward: float            # Lagrangian reward with q for evaluation
    reward_train: float      # Lagrangian reward with realized outcomes for training
    done: bool
    info: dict


class UAVEnv:
    def __init__(self, cfg: Config, perception_cache: dict, fidelity_cache: dict, seed: int = 0):
        self.cfg = cfg
        self.pc = perception_cache
        self.fc = fidelity_cache
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        # "random" (training): sequences and start frames drawn from self.rng.
        # "cycle" (evaluation): deterministic -- group g of the episode with seed s starts at frame 0 of sequence
        # (s * n_groups + g) mod S and moves on by n_groups sequences, every policy then sees exactly the same
        # frames (paired comparison) and consecutive seeds cover all sequences evenly.
        self.scene_order = "random"
        self.mu = 0.0
        self.delta_th = cfg.constraint.delta_max

        net, chn = cfg.network, cfg.channel
        self.M, self.V = net.M, net.views_per_scene
        if self.M % self.V:
            raise ValueError("M must be a multiple of views_per_scene")
        self.n_groups = self.M // self.V
        self.group = np.arange(self.M) // self.V
        self.view = np.arange(self.M) % self.V
        self.C = self.pc["belief"].shape[1]

        self.N0 = dbm_to_w(chn.N0_dbm_hz + chn.NF_db)
        self.P_max = dbm_to_w(chn.P_max_dbm)
        self.P_l = dbm_to_w(chn.P_l_dbm)
        self.grid = np.asarray(self.fc["snr_grid_db"], dtype=np.float64)
        self.psi_b = np.asarray(self.fc["psi_b"], dtype=np.float64)
        self.psi_s = np.asarray(self.fc["psi_s"], dtype=np.float64)
        self._index_sequences()

    # indexing
    def _index_sequences(self) -> None:
        """self._seq_rows[k] is an (F_k, V) array of cache rows: frame x view."""
        seq = self.pc["seq_id"]
        frame = self.pc["frame"]
        view = self.pc["view_id"]
        order = np.lexsort((view, frame, seq))
        seq_sorted = seq[order]
        bounds = np.flatnonzero(np.r_[True, seq_sorted[1:] != seq_sorted[:-1], True])
        self._seq_rows = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            rows = order[a:b]
            if len(rows) % self.V:
                raise ValueError(f"sequence {seq_sorted[a]} has frames with missing views")
            rows = rows.reshape(-1, self.V)
            if not np.all(view[rows] == np.arange(self.V)):
                raise ValueError(f"sequence {seq_sorted[a]}: views are not 0..V-1 per frame")
            self._seq_rows.append(rows)
        if not self._seq_rows:
            raise ValueError("empty perception cache")

    # interface
    @property
    def obs_dim(self) -> int:
        return 7 + 3 * (self.V - 1)

    @property
    def state_dim(self) -> int:
        return self.M * self.obs_dim + 2 * self.M + 1

    def set_price_and_threshold(self, mu: float, delta_th: float) -> None:
        self.mu, self.delta_th = float(mu), float(delta_th)

    def feasible_modes(self, delta: np.ndarray, neighbors: list[np.ndarray] | None = None) -> np.ndarray:
        """ LOCAL always; A2A/A2G only if delta < delta_th.
        A2A additionally requires at least one co-observing neighbor in range."""
        low = np.asarray(delta) < self.delta_th
        mask = np.zeros((len(low), 3), dtype=bool)
        mask[:, LOCAL] = True
        mask[:, A2G] = low
        has_nb = np.ones(len(low), dtype=bool) if neighbors is None else np.array([n.size > 0 for n in neighbors])
        mask[:, A2A] = low & has_nb
        allowed = np.zeros(3, dtype=bool)
        allowed[np.asarray(self.cfg.env.allowed_modes, dtype=np.int64)] = True
        mask[:, 1:] &= allowed[1:]
        return mask

    def reset(self) -> StepOutput:
        net = self.cfg.network
        self.t = 0
        self.leader, self.fol = init_positions(self.rng, self.M, net.area_radius, net.d_min, net.d_max)
        self.waypoints = random_waypoints(self.rng, self.M, self.leader, net.d_min, net.d_max)
        self.edges = edge_positions(self.rng, net.J, net.edge_radius)
        chn = self.cfg.channel
        r = np.linalg.norm(self.edges - self.leader, axis=1)
        self.pl_edge_db = ch.a2g_mean_pathloss_db(r, net.H, chn.f_c, chn.a, chn.b, chn.eta_los_db, chn.eta_nlos_db)
        S = len(self._seq_rows)
        if self.scene_order == "cycle":
            self.grp_seq = (self.seed * self.n_groups + np.arange(self.n_groups)) % S
            self.grp_frame = np.zeros(self.n_groups, dtype=np.int64)
        else:
            self.grp_seq = self.rng.integers(S, size=self.n_groups)
            self.grp_frame = np.array([self.rng.integers(len(self._seq_rows[k])) for k in self.grp_seq])
        self.n_active_prev = 0
        self._load_frame()
        return self._output(0.0, 0.0, False, {})

    def current(self) -> StepOutput:
        """Observation, state, and mask of the current slot without stepping. Rewards are zero and info is empty."""
        return self._output(0.0, 0.0, False, {})

    def step(self, action: dict) -> StepOutput:
        cfg, M = self.cfg, self.M
        mode = np.asarray(action["mode"], dtype=np.int64).copy()
        p = np.asarray(action["p"], dtype=np.float64)
        beta = np.asarray(action["beta"], dtype=np.float64)
        b, delta, y, rows = self.b, self.delta, self.y, self.rows

        #  gate and A2A fallback
        if np.any((mode != LOCAL) & (delta >= self.delta_th)):
            raise ValueError("gate violated: a confident follower was asked to communicate")
        fallback = (mode == A2A) & np.array([n.size == 0 for n in self.nbrs])
        mode[fallback] = LOCAL

        #  resources
        active = mode != LOCAL
        B = grant_bandwidth(np.where(active, np.clip(beta, 1e-3, 1.0), 1.0), active, cfg.channel.B_tot)
        P = np.where(active, scale_power(p, self.P_max), 0.0)
        n_resp = np.where(mode == A2A, [n.size for n in self.nbrs], 0)

        #  channels: A2G hop 2 is shared by the cluster in this slot which is block fading
        fading = ch.nakagami_power(self.rng, cfg.channel.nakagami_m, cfg.network.J)
        H_best = ch.a2g_gain(self.pl_edge_db, fading).max()
        h_ul = ch.a2a_gain(np.linalg.norm(self.fol - self.leader, axis=1), cfg.channel.f_c)

        # reference channel state for analysis: A2G end-to-end SNR at full power and an equal band share
        B_ref = cfg.channel.B_tot / M
        g_ref = ch.af_end_to_end_snr(ch.snr(self.P_max, h_ul, B_ref, self.N0), ch.snr(self.P_l, H_best, B_ref, self.N0))
        n_conf_nb = np.array([int((delta[n] > delta[u]).sum()) for u, n in enumerate(self.nbrs)])

        #  outcomes
        q = delta.copy()
        y_hat = b.argmax(1)
        correct = (y_hat == y).astype(np.float64)
        gamma_a2g_db = np.full(M, np.nan)
        gamma_a2a_db = np.full(M, np.nan)
        for u in np.flatnonzero(mode == A2A):
            nb = self.nbrs[u]
            h = ch.a2a_gain(np.linalg.norm(self.fol[nb] - self.fol[u], axis=1), cfg.channel.f_c)
            g_db = 10.0 * np.log10(ch.a2a_snr(P[u], h, B[u], nb.size, self.N0))
            b_hat = np.stack([interp_on_grid(self.fc["belief_hat"][rows[v]].T, self.grid, g_db[i])
                              for i, v in enumerate(nb)])
            b_hat = np.clip(b_hat, 1e-8, None)
            b_hat /= b_hat.sum(1, keepdims=True)
            d_hat = np.array([interp_on_grid(self.fc["delta_hat"][rows[v]], self.grid, g_db[i])
                              for i, v in enumerate(nb)])
            w = np.clip(d_hat * logistic_fidelity(g_db, self.psi_b), 0.0, 1.0)
            b_plus, q[u] = log_linear_pool(b[u], delta[u], b_hat, w)
            y_hat[u] = int(b_plus.argmax())
            correct[u] = float(y_hat[u] == y[u])
            gamma_a2a_db[u] = float(np.mean(g_db))
        for u in np.flatnonzero(mode == A2G):
            g1 = ch.snr(P[u], h_ul[u], B[u], self.N0)
            g2 = ch.snr(self.P_l, H_best, B[u], self.N0)
            g_db = float(10.0 * np.log10(ch.af_end_to_end_snr(g1, g2)))
            p_edge = float(np.clip(interp_on_grid(self.fc["edge_correct_prob"][rows[u]], self.grid, g_db), 0, 1))
            correct[u] = float(self.rng.random() < p_edge)
            y_hat[u] = y[u] if correct[u] else (y[u] + 1 + self.rng.integers(self.C - 1)) % self.C
            q[u] = float(np.clip(logistic_fidelity(g_db, self.psi_s), 0.0, 1.0))
            gamma_a2g_db[u] = g_db

        #  latency, energy, QoE
        S, K, Q = cfg.semantic, cfg.compute, cfg.qoe
        d_tx = tx_latency(mode, B, n_resp, S.k_b, S.k_s)
        D = decision_latency(mode, d_tx, K.chi_L, K.chi_F, K.chi_G, K.f_U, K.f_G)
        E = comm_energy(mode, P, B, n_resp, self.P_l, S.k_b, S.k_s, cfg.channel.E0)
        sig = timeliness(D, Q.D_th, Q.zeta_D)
        qoe = q * sig

        #  rewards 
        e_norm = E / cfg.constraint.E_bar
        reward = float(np.sum(qoe - self.mu * (e_norm - 1.0)))
        reward_train = float(np.sum(correct * sig - self.mu * (e_norm - 1.0)))

        info = {"mode": mode, "fallback": fallback, "B": B, "P": P, "n_resp": n_resp, "delta": delta,
                "label": y, "y_hat": y_hat, "correct": correct, "q": q, "D": D, "sigma": sig, "qoe": qoe,
                "a2g_snr_ref_db": 10.0 * np.log10(g_ref), "n_conf_nb": n_conf_nb,
                "qoe_real": correct * sig,   # unbiased per-sample estimate of eq. (QoE): realized correctness x timeliness
                "E": E, "gamma_a2a_db": gamma_a2a_db, "gamma_a2g_db": gamma_a2g_db,
                "delta_th": self.delta_th, "mu": self.mu, "rows": rows.copy()}

        #  mobility
        net = cfg.network
        advance = correct.astype(bool) if cfg.env.progress_on_correct else np.ones(M, dtype=bool)
        self.fol = step_followers(self.fol, self.leader, self.waypoints, advance, net.l_max, net.d_min, net.d_max)
        reached = np.linalg.norm(self.fol - self.waypoints, axis=1) < 1.0
        if reached.any():
            self.waypoints[reached] = random_waypoints(self.rng, int(reached.sum()), self.leader, net.d_min, net.d_max)

        #next slot
        self.n_active_prev = int(active.sum())
        self.t += 1
        self.grp_frame += 1
        for g in range(self.n_groups):
            if self.grp_frame[g] >= len(self._seq_rows[self.grp_seq[g]]):
                self.grp_seq[g] = ((self.grp_seq[g] + self.n_groups) % len(self._seq_rows) if self.scene_order == "cycle"
                                   else self.rng.integers(len(self._seq_rows)))
                self.grp_frame[g] = 0
        self._load_frame()
        done = self.t >= cfg.env.episode_len
        return self._output(reward, reward_train, done, info)

    
    def _load_frame(self) -> None:
        """Cache rows, beliefs, confidences, labels, and neighbor sets of the current slot."""
        self.rows = np.array([self._seq_rows[self.grp_seq[g]][self.grp_frame[g], v]
                              for g, v in zip(self.group, self.view)])
        self.b = self.pc["belief"][self.rows].astype(np.float64)
        self.delta = self._confidence(self.rows)
        self.y = self.pc["label"][self.rows].astype(np.int64)
        self.nbrs = neighbor_sets(self.fol, self.group, self.cfg.network.d_max)

    def _confidence(self, rows: np.ndarray) -> np.ndarray:
        """Confidence signal used for the gate, the observation, fusion, and q."""
        src = self.cfg.env.confidence_source
        if src == "calibrated":
            return self.pc["delta"][rows].astype(np.float64)
        if src == "msp":     # max softmax probability of the UNcalibrated classifier when available
            key = "msp_raw" if "msp_raw" in self.pc else None
            if key:
                return self.pc[key][rows].astype(np.float64)
            return self.pc["belief"][rows].max(axis=1).astype(np.float64)
        if src == "raw":
            if "delta_raw" in self.pc:
                return self.pc["delta_raw"][rows].astype(np.float64)
            #  emulate overconfidence 
            return np.sqrt(self.pc["delta"][rows].astype(np.float64))
        raise ValueError(f"unknown confidence_source '{src}'")

    def _observe(self) -> np.ndarray:
        """Eq. (observation): one fixed-size row per follower."""
        f_c = self.cfg.channel.f_c
        ent = -(self.b * np.log(np.clip(self.b, 1e-12, None))).sum(1) / np.log(self.C)
        h_ul_db = 10 * np.log10(ch.a2a_gain(np.linalg.norm(self.fol - self.leader, axis=1), f_c))
        best_pl = self.pl_edge_db.min()
        obs = np.zeros((self.M, self.obs_dim))
        for u in range(self.M):
            row = [self.delta[u], ent[u]]
            peers = [v for v in np.flatnonzero(self.group == self.group[u]) if v != u]
            for v in peers:                      # fixed slots: every co-observer, available or not
                available = v in self.nbrs[u]
                d = max(np.linalg.norm(self.fol[v] - self.fol[u]), 1.0)
                h_db = 10 * np.log10(ch.a2a_gain(d, f_c))
                row += [self.delta[v] if available else 0.0,
                        (h_db - _GAIN_DB_REF) / _GAIN_DB_SCALE if available else 0.0,
                        float(available)]
            row += [(h_ul_db[u] - _GAIN_DB_REF) / _GAIN_DB_SCALE,
                    (best_pl - _PL_DB_REF) / _PL_DB_SCALE,
                    self.n_active_prev / self.M,
                    self.mu,
                    self.delta_th]
            obs[u] = row
        return obs

    def _output(self, reward: float, reward_train: float, done: bool, info: dict) -> StepOutput:
        obs = self._observe()
        d_max = self.cfg.network.d_max
        state = np.concatenate([obs.ravel(), ((self.fol - self.leader) / d_max).ravel(),
                                [self.t / self.cfg.env.episode_len]])
        return StepOutput(obs=obs, state=state, mask=self.feasible_modes(self.delta, self.nbrs),
                          delta=self.delta.copy(), reward=reward, reward_train=reward_train,
                          done=done, info=info)
