
from __future__ import annotations

from pathlib import Path

import numpy as np

from .baselines import Policy


class TrainedPolicy(Policy):


    def __init__(self, run_dir: str | Path, obs_dim: int, hidden: list[int], deterministic: bool = True,
                 name: str | None = None):
        import torch

        from .networks import HybridActor

        self._torch = torch
        ckpt = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu")
        self.actor = HybridActor(obs_dim, hidden)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.delta_th = float(ckpt["delta_th"])
        self.mu = float(ckpt["mu"])
        self.deterministic = deterministic
        self.name = name or f"Proposed ({Path(run_dir).name})"

    def act(self, out, env):
        t = self._torch
        a = self.actor.act(t.as_tensor(out.obs, dtype=t.float32), t.as_tensor(out.mask, dtype=t.bool),
                           deterministic=self.deterministic)
        return {"mode": a["mode"].numpy().astype(np.int64), "p": a["p"].numpy(), "beta": a["beta"].numpy()}
