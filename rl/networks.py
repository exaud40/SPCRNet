
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

LOCAL = 0
_MASK_FILL = -1e9   # finite, so log-softmax never produces NaN
LOG_STD_MIN, LOG_STD_MAX = -5.0, 0.5


def mlp(in_dim: int, hidden: list[int], out_dim: int | None = None, act=nn.Tanh) -> nn.Sequential:
    """Fully connected network with orthogonal initialization."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = h
    if out_dim is not None:
        lin = nn.Linear(d, out_dim)
        nn.init.orthogonal_(lin.weight, gain=1.0)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
    return nn.Sequential(*layers)


def _squash_log_det(z: torch.Tensor) -> torch.Tensor:
    """sum_k log sigmoid'(z_k) = sum_k [log sigmoid(z_k) + log sigmoid(-z_k)]."""
    return (F.logsigmoid(z) + F.logsigmoid(-z)).sum(-1)


class HybridActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: list[int], init_log_std: float = -0.5):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden)
        self.mode_head = nn.Linear(hidden[-1], 3)
        self.res_head = nn.Linear(hidden[-1], 2)
        for head in (self.mode_head, self.res_head):        # small initial outputs -> near-uniform policy
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        self.log_std = nn.Parameter(torch.full((2,), float(init_log_std)))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, Normal]:
        h = self.trunk(obs)
        logits = self.mode_head(h).masked_fill(~mask, _MASK_FILL)
        mean = self.res_head(h)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return Categorical(logits=logits), Normal(mean, std)

    def _log_prob(self, cat: Categorical, gauss: Normal, mode: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        comm = (mode != LOCAL).float()
        lp_res = gauss.log_prob(z).sum(-1) - _squash_log_det(z)
        return cat.log_prob(mode) + comm * lp_res

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False) -> dict:
        cat, gauss = self(obs, mask)
        if deterministic:
            mode = cat.probs.argmax(-1)
            z = gauss.mean
        else:
            mode = cat.sample()
            z = gauss.sample()
        logp = self._log_prob(cat, gauss, mode, z)
        res = torch.sigmoid(z)
        return {"mode": mode, "z": z, "logp": logp, "p": res[..., 0], "beta": res[..., 1]}

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, mode: torch.Tensor, z: torch.Tensor):
        """(log-prob, mode entropy, resource entropy) of stored actions under the current parameters."""
        cat, gauss = self(obs, mask)
        logp = self._log_prob(cat, gauss, mode, z)
        return logp, cat.entropy(), gauss.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(state_dim, hidden, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
