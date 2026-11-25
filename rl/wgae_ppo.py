
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import Config, apply_overrides, load_config
from ..data.base import FIDELITY_KEYS, PERCEPTION_KEYS, load_cache
from ..env.uav_env import UAVEnv
from .advantage import critic_targets, kappa_schedule, td_residuals, wgae
from .buffer import RolloutBuffer
from .constraints import dual_update, threshold_update
from .networks import Critic, HybridActor

from .log_fields import LOG_FIELDS  # noqa: E402  (torch-free, shared with the M8 tools)


def make_env(cfg: Config, cache_dir: str | Path, split: str, seed: int) -> UAVEnv:
    cache_dir = Path(cache_dir)
    pc = load_cache(cache_dir / f"perception_{split}.npz", PERCEPTION_KEYS)
    fc = load_cache(cache_dir / f"fidelity_{split}.npz", FIDELITY_KEYS)
    return UAVEnv(cfg, pc, fc, seed=seed)


class Trainer:


    def __init__(self, cfg: Config, envs: list[UAVEnv] | UAVEnv, log_dir: str | Path, seed: int = 0,
                 device: str = "cpu", threshold_quantile: float = 0.98, threads: int | None = 1,
                 threshold_adapt: bool = True, mu_init: float = 0.0, reward_type: str = "outcome"):
  
        if reward_type not in ("outcome", "model"):
            raise ValueError("reward_type must be 'outcome' or 'model'")
        self.threshold_adapt, self.reward_type = threshold_adapt, reward_type
        self.envs = envs if isinstance(envs, list) else [envs]
        self.env = self.envs[0]
        self.cfg, self.rl = cfg, cfg.rl
        if threads:
            torch.set_num_threads(threads)      # small MLPs run fastest single-threaded on CPU
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = self.env
        self.actor = HybridActor(env.obs_dim, self.rl.hidden).to(self.device)
        self.critic = Critic(env.state_dim, self.rl.hidden).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.rl.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.rl.lr_critic)
        self.N = len(self.envs)
        T = max(1, self.rl.rollout_len // self.N)
        self.buf = RolloutBuffer(T, self.N, env.M, env.obs_dim, env.state_dim)
        self.mu = float(mu_init)
        self.delta_th = cfg.constraint.delta_max
        self.q_th = threshold_quantile
        self.kappa_now = float(self.rl.kappa)
        self.env_steps = 0
        self._outs = None

   #helpers
    def _t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @torch.no_grad()
    def _values(self, states: np.ndarray) -> np.ndarray:
        """Batched critic evaluation; states (..., S) -> values (...)."""
        flat = self._t(states.reshape(-1, states.shape[-1]))
        return self.critic(flat).cpu().numpy().reshape(states.shape[:-1])

    # ------------------------------------------------------------------ rollout
    def collect(self) -> dict:
        buf, N = self.buf, self.N
        for env in self.envs:
            env.set_price_and_threshold(self.mu, self.delta_th)
        # mu and delta_th may have changed since the last observation: rebuild obs, state, and mask
        outs = [env.reset() for env in self.envs] if self._outs is None else [env.current() for env in self.envs]
        logs: dict[str, list] = {k: [] for k in ("E", "qoe", "correct", "mode", "sigma", "delta_comm")}
        buf.clear()
        M = self.env.M
        while not buf.full:
            obs = np.stack([o.obs for o in outs])                   # (N, M, O)
            mask = np.stack([o.mask for o in outs])                 # (N, M, 3)
            a = self.actor.act(self._t(obs.reshape(N * M, -1)), self._t(mask.reshape(N * M, 3), torch.bool))
            mode = a["mode"].cpu().numpy().reshape(N, M)
            p = a["p"].cpu().numpy().reshape(N, M)
            beta = a["beta"].cpu().numpy().reshape(N, M)
            nxts = [env.step({"mode": mode[e], "p": p[e], "beta": beta[e]}) for e, env in enumerate(self.envs)]
            buf.add(state=np.stack([o.state for o in outs]), obs=obs, mask=mask,
                    delta=np.stack([o.delta for o in outs]), mode=mode,
                    z=a["z"].cpu().numpy().reshape(N, M, 2), logp=a["logp"].cpu().numpy().reshape(N, M),
                    reward=np.array([(n.reward_train if self.reward_type == "outcome" else n.reward) / M
                                     for n in nxts]), done=np.array([n.done for n in nxts]))
            for n in nxts:
                i = n.info
                logs["E"].append(i["E"]); logs["qoe"].append(i["qoe_real"]); logs["correct"].append(i["correct"])
                logs["mode"].append(i["mode"]); logs["sigma"].append(i["sigma"])
                logs["delta_comm"].append(i["delta"][i["mode"] != 0])
            outs = [env.reset() if n.done else n for env, n in zip(self.envs, nxts)]
            self.env_steps += N
        buf.finish(self._values(buf.state), self._values(np.stack([o.state for o in outs])),
                   np.stack([o.delta for o in outs]))
        self._outs = outs
        return {k: np.concatenate(v) for k, v in logs.items()}

    #  update
    def update(self) -> dict:
        rl, buf = self.rl, self.buf
        adv = np.zeros((buf.T, buf.N, buf.M))
        v_tar = np.zeros((buf.T, buf.N))
        for e in range(buf.N):                                   # advantages per environment stream
            res = td_residuals(buf.reward[:, e], buf.values_ext[:, e], buf.done[:, e], rl.gamma_disc)
            adv[:, e] = wgae(res, buf.done[:, e], buf.delta_ext[:, e], rl.gamma_disc, rl.gae_lambda, self.kappa_now)
            v_tar[:, e] = critic_targets(buf.values_ext[:, e], adv[:, e])

        obs, mask = self._t(buf.obs), self._t(buf.mask, torch.bool)
        mode, z = self._t(buf.mode, torch.long), self._t(buf.z)
        logp_old, adv_t = self._t(buf.logp), self._t(adv)
        states, v_tar_t = self._t(buf.state), self._t(v_tar)
        n_actor_batches = int(np.ceil(buf.T * buf.N * buf.M / rl.minibatch))

        stats = {"entropy": [], "approx_kl": [], "clip_frac": [], "value_loss": []}
        grads: list[np.ndarray] = []
        for epoch in range(rl.ppo_epochs):
            for ti, ei, ui in buf.actor_minibatches(rl.minibatch, self.rng):
                idx = (self._t(ti, torch.long), self._t(ei, torch.long), self._t(ui, torch.long))
                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                logp, ent, _ = self.actor.evaluate(obs[idx], mask[idx], mode[idx], z[idx])
                ratio = torch.exp(logp - logp_old[idx])
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - rl.clip_eps, 1 + rl.clip_eps) * a)
                loss = -(surr.mean() + rl.entropy_coef * ent.mean())
                self.opt_a.zero_grad()
                loss.backward()
                if epoch == 0:   # gradient-variance diagnostic (Run 4b): before clipping, first epoch only
                    grads.append(torch.cat([p.grad.flatten() for p in self.actor.parameters()
                                            if p.grad is not None]).cpu().numpy())
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.opt_a.step()
                with torch.no_grad():
                    stats["entropy"].append(float(ent.mean()))
                    stats["approx_kl"].append(float((logp_old[idx] - logp).mean()))
                    stats["clip_frac"].append(float(((ratio - 1).abs() > rl.clip_eps).float().mean()))
            for ti, ei in buf.critic_minibatches(n_actor_batches, self.rng):
                cidx = (self._t(ti, torch.long), self._t(ei, torch.long))
                v_loss = ((self.critic(states[cidx]) - v_tar_t[cidx]) ** 2).mean()
                self.opt_c.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_c.step()
                stats["value_loss"].append(float(v_loss))
        g = np.stack(grads)
        grad_var = float(((g - g.mean(0)) ** 2).sum(1).mean())     # E||g - E g||^2 across mini-batches
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["grad_var"] = grad_var
        with torch.no_grad():
            from .networks import LOG_STD_MAX, LOG_STD_MIN
            out["res_std"] = float(self.actor.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean())
            cat, _ = self.actor(obs.reshape(-1, obs.shape[-1]), mask.reshape(-1, 3))
            out["p_local"] = float(cat.probs[:, 0].mean())
        return out

    #  constraints
    def dual_and_threshold_update(self, roll: dict) -> None:
        c = self.cfg.constraint
        e_hat_norm = float(roll["E"].mean() / c.E_bar)
        if self.rl.lr_dual > 0:
            self.mu = dual_update(self.mu, e_hat_norm, self.rl.lr_dual)
        if self.threshold_adapt:
            self.delta_th = threshold_update(roll["delta_comm"], self.delta_th, c.delta_margin,
                                             c.delta_min, c.delta_max, self.q_th)

    #  main loop
    def train(self, iterations: int | None = None, checkpoint_every: int = 50,
              keep_iters: tuple[int, ...] = ()) -> Path:
        """keep_iters: also save ckpt_iter<k>.pt at these iterations (Run 4b probes intermediate policies)."""
        iterations = iterations or self.rl.iterations
        csv_path = self.log_dir / "progress.csv"
        with open(self.log_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump({"cfg": asdict(self.cfg), "threshold_quantile": self.q_th, "threshold_adapt": self.threshold_adapt,
                       "reward_type": self.reward_type, "n_envs": self.N}, fh, indent=2, default=str)
        t0 = time.time()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            writer.writeheader()
            for k in range(1, iterations + 1):
                mu_used, th_used = self.mu, self.delta_th
                self.kappa_now = kappa_schedule(self.rl.kappa, k, self.rl.kappa_anneal_iters)
                roll = self.collect()
                upd = self.update()
                self.dual_and_threshold_update(roll)
                mode = roll["mode"]
                row = {"iter": k, "env_steps": self.env_steps,
                       "return_train": float(self.buf.reward.mean()),
                       "qoe": float(roll["qoe"].mean()), "accuracy": float(roll["correct"].mean()),
                       "energy_norm": float(roll["E"].mean() / self.cfg.constraint.E_bar),
                       "mu": mu_used, "delta_th": th_used, "kappa": self.kappa_now,
                       "comm_rate": float((mode != 0).mean()), "a2a_share": float((mode == 1).mean()),
                       "a2g_share": float((mode == 2).mean()),
                       "deadline_ratio": float((roll["sigma"] >= 0.5).mean()),
                       "wall_s": time.time() - t0, **upd}
                writer.writerow(row)
                fh.flush()
                if k % 10 == 0 or k == 1:
                    print(f"[{k:5d}] ret {row['return_train']:+.3f}  QoE {row['qoe']:.3f}  acc {row['accuracy']:.3f}  "
                          f"E/Ebar {row['energy_norm']:.2f}  mu {mu_used:.3f}  dth {th_used:.3f}  "
                          f"comm {row['comm_rate']:.0%} (A2A {row['a2a_share']:.0%}, A2G {row['a2g_share']:.0%})  "
                          f"H {upd['entropy']:.2f}  kl {upd['approx_kl']:.4f}", flush=True)
                if k in keep_iters:
                    self.save(self.log_dir / f"ckpt_iter{k}.pt")
                if k % checkpoint_every == 0 or k == iterations:
                    self.save(self.log_dir / "checkpoint.pt")
        return csv_path

    def save(self, path: str | Path) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "mu": self.mu, "delta_th": self.delta_th, "env_steps": self.env_steps}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default="caches/synth")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=None, help="override; 0 -> GAE-PPO")
    ap.add_argument("--lam", type=float, default=None, help="override GAE lambda; 1.0 with kappa 0 -> plain PPO")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--rollout_len", type=int, default=None)
    ap.add_argument("--threshold_quantile", type=float, default=0.98)
    ap.add_argument("--n_envs", type=int, default=None, help="parallel environments (default: rl.n_envs)")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 is fastest for small MLPs)")
    ap.add_argument("--e_bar", type=float, default=None, help="override energy budget E_bar [J]")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="config override, repeatable (e.g. --set channel.B_tot=1e6)")
    ap.add_argument("--no_threshold_adapt", action="store_true", help="ablation: keep delta_th = delta_max")
    ap.add_argument("--mu_init", type=float, default=0.0, help="initial energy price (fixed if rl.lr_dual=0)")
    ap.add_argument("--reward", default="outcome", choices=["outcome", "model"])
    ap.add_argument("--keep_iters", default="", help="comma list: also save ckpt_iter<k>.pt (Run 4b)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_dir", default="runs/wgae_ppo")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.kappa is not None:
        cfg.rl.kappa = args.kappa
    if args.lam is not None:
        cfg.rl.gae_lambda = args.lam
    if args.rollout_len is not None:
        cfg.rl.rollout_len = args.rollout_len
    if args.e_bar is not None:
        cfg.constraint.E_bar = args.e_bar
    n_envs = args.n_envs or cfg.rl.n_envs
    envs = [make_env(cfg, args.cache, args.split, seed=args.seed * 1000 + i) for i in range(n_envs)]
    trainer = Trainer(cfg, envs, args.log_dir, seed=args.seed, device=args.device,
                      threshold_quantile=args.threshold_quantile, threads=args.threads,
                      threshold_adapt=not args.no_threshold_adapt, mu_init=args.mu_init, reward_type=args.reward)
    keep = tuple(int(x) for x in args.keep_iters.split(",") if x.strip())
    path = trainer.train(args.iterations, keep_iters=keep)
    print(f"done; log at {path}")


if __name__ == "__main__":
    main()
