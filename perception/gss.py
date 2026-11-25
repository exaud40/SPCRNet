
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSSM(nn.Module):
    def __init__(self, d: int, state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w_in = nn.Linear(d, d)
        self.w_delta = nn.Linear(d, 1)
        self.w_bc = nn.Linear(d, 2 * state)
        self.w_g = nn.Linear(d, d)
        self.w_out = nn.Linear(d, d)
        # S4D-real initialization: A_n = -(n + 1), shared over channels
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32)).repeat(d, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, L, d) -> (B, L, d)
        x = self.norm(h)
        v = self.w_in(x)                                   # (B, L, d)
        delta = F.softplus(self.w_delta(x))                # (B, L, 1)
        Bc, Cc = self.w_bc(x).chunk(2, dim=-1)             # (B, L, N) each
        A = -torch.exp(self.A_log.float())                 # (d, N)
        s = torch.zeros(h.shape[0], h.shape[2], A.shape[1], device=h.device, dtype=torch.float32)
        ys = []
        for l in range(h.shape[1]):                        # selective scan over tokens
            dl = delta[:, l].float()[:, :, None]           # (B, 1, 1)
            s = torch.exp(dl * A) * s + dl * v[:, l].float()[:, :, None] * Bc[:, l].float()[:, None, :]
            ys.append((s * Cc[:, l].float()[:, None, :]).sum(-1))   # (B, d)
        y = torch.stack(ys, dim=1).to(h.dtype)             # (B, L, d)
        return h + self.w_out(y * F.silu(self.w_g(x)))
