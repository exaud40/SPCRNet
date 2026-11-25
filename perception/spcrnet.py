
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gss import GatedSSM

N_SCALAR_CONF = 6


class TinyBackbone(nn.Module):

    def __init__(self, chans=(32, 64, 128)):
        super().__init__()

        def blk(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1), nn.BatchNorm2d(co), nn.GELU())

        self.stem = nn.Sequential(blk(3, 16, 2), blk(16, 16, 2), blk(16, chans[0], 2))   # stride 8
        self.s3 = blk(chans[0], chans[1], 2)                                               # stride 16
        self.s4 = blk(chans[1], chans[2], 2)                                               # stride 32
        self.channels = list(chans)

    def forward(self, x):
        f2 = self.stem(x)
        f3 = self.s3(f2)
        return [f2, f3, self.s4(f3)]


def make_backbone(name: str, pretrained: bool):
    if name == "tiny":
        bb = TinyBackbone()
        return bb, bb.channels
    import timm

    bb = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
    return bb, list(bb.feature_info.channels())


class SPCRNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, d: int, d_hidden: int, attn_heads: int, ssm_state: int,
                 pretrained: bool = True, grid: int = 7, conf_hidden: int = 64, count_edges=(45, 76, 108)):
        super().__init__()
        self.register_buffer("T_b", torch.tensor(1.0))
        self.register_buffer("T_delta", torch.tensor(1.0))
        self.register_buffer("zbar_gate", torch.tensor(1.0))
        self.register_buffer("logistic_head", torch.tensor(0.0))       # 1: use conf_linear (scalar features)
        self.backbone, (c2, c3, c4) = make_backbone(backbone, pretrained)
        self.pool2 = nn.Conv2d(c2, c2, 4, 4, groups=c2)        # stride 8 -> 32
        self.pool3 = nn.Conv2d(c3, c3, 2, 2, groups=c3)        # stride 16 -> 32
        cin = c2 + c3 + c4
        self.fuse = nn.Sequential(nn.Conv2d(cin, cin, 3, 1, 1, groups=cin), nn.Conv2d(cin, d_hidden, 1),
                                  nn.BatchNorm2d(d_hidden), nn.GELU())
        dd = max(32, d_hidden // 4)                                       # stride-16 density decoder
        self.dens_f3 = nn.Conv2d(c3, dd, 1)
        self.dens_f4 = nn.Conv2d(c4, dd, 1)
        self.dens_out = nn.Sequential(nn.Conv2d(dd, dd, 3, 1, 1), nn.BatchNorm2d(dd), nn.GELU(), nn.Conv2d(dd, 1, 1))
        nn.init.constant_(self.dens_out[-1].bias, -3.0)                   # start near zero density
        self.grid = grid
        self.compress = nn.Sequential(nn.Linear(d_hidden + 1, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden),
                                      nn.GELU(), nn.Linear(d_hidden, d))
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, attn_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.gss = GatedSSM(d, ssm_state)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d, attn_heads, batch_first=True)
        self.count_offset = nn.Parameter(torch.tensor(0.31))             # log(1 / 0.73): union vs own view
        self.count_corr = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 1))
        self.count_logsig = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 1))
        nn.init.zeros_(self.count_corr[-1].weight)
        nn.init.zeros_(self.count_corr[-1].bias)
        nn.init.constant_(self.count_logsig[-1].bias, math.log(0.3))
        bounds = [math.log1p(e - 0.5) for e in count_edges]
        self.register_buffer("cls_bounds", torch.tensor(bounds, dtype=torch.float32))
        self.conf_head = nn.Sequential(nn.Linear(d + N_SCALAR_CONF, conf_hidden), nn.GELU(),
                                       nn.Linear(conf_hidden, conf_hidden), nn.GELU(), nn.Linear(conf_hidden, 1))
        # low-capacity alternative: logistic regression on the scene-agnostic scalar features only
        self.conf_linear = nn.Linear(N_SCALAR_CONF, 1)
        self.num_classes = num_classes

  
    def encode_full(self, x: torch.Tensor):
        """(tokens Z (B,L_s,d), density (B,h,w))."""
        f2, f3, f4 = self.backbone(x)
        p2, p3 = self.pool2(f2), self.pool3(f3)
        h, w = f4.shape[-2:]
        p2 = F.adaptive_avg_pool2d(p2, (h, w)) if p2.shape[-2:] != (h, w) else p2
        p3 = F.adaptive_avg_pool2d(p3, (h, w)) if p3.shape[-2:] != (h, w) else p3
        m = self.fuse(torch.cat([p2, p3, f4], dim=1))                    # fused map M_u
        u = self.dens_f3(f3) + F.interpolate(self.dens_f4(f4), size=f3.shape[-2:], mode="nearest")
        dens = F.softplus(self.dens_out(u).float()).squeeze(1)           # (B, h16, w16) targets per cell
        f = F.adaptive_avg_pool2d(m, (self.grid, self.grid))             # (B, c_m, g, g)
        # each token also carries the (log) number of targets in its region, so the transmitted semantics
        # contain the counting evidence the edge model and the neighbours need
        dh, dw = dens.shape[-2:]
        dtok = F.adaptive_avg_pool2d(dens[:, None], (self.grid, self.grid)) * (dh * dw) / (self.grid * self.grid)
        f = torch.cat([f, torch.log1p(dtok).to(f.dtype)], dim=1)
        X = f.flatten(2).transpose(1, 2)                                 # (B, L_s, c_m + 1)
        Z = self.compress(X) + self.pos
        a = self.attn_norm(Z)
        Z = Z + self.attn(a, a, a, need_weights=False)[0]
        Z = Z + self.ffn(Z)
        return self.gss(Z), dens                                         # semantic tokens Z_u

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_full(x)[0]

    def head(self, Z: torch.Tensor, dens: torch.Tensor):
        """(z_bar, logits = log b, mu = E[log(1 + scene count)], log sigma)."""
        q = self.query.expand(Z.shape[0], -1, -1)
        z_bar = self.pool_attn(q, Z, Z, need_weights=False)[0].squeeze(1)
        S = dens.float().sum(dim=(-2, -1))
        mu = torch.log1p(S) + self.count_offset + 0.5 * torch.tanh(self.count_corr(z_bar).squeeze(-1).float())
        log_sig = self.count_logsig(z_bar).squeeze(-1).float().clamp(-4.0, 1.5)
        return z_bar, self.bin_logits(mu, log_sig), mu, log_sig

    def bin_logits(self, mu: torch.Tensor, log_sig: torch.Tensor) -> torch.Tensor:
        """log P(count in level k) under log(1 + n) ~ N(mu, sigma^2)."""
        z = (self.cls_bounds.view(1, -1) - mu.unsqueeze(-1)) / torch.exp(log_sig).unsqueeze(-1)
        cdf = torch.special.ndtr(z)                                       # (B, C-1)
        ones, zeros = torch.ones_like(mu).unsqueeze(-1), torch.zeros_like(mu).unsqueeze(-1)
        b = torch.cat([cdf, ones], -1) - torch.cat([zeros, cdf], -1)
        return torch.log(b.clamp_min(1e-6))

    def conf_features(self, z_bar, logits, log_count, log_sigma) -> torch.Tensor:
        p = torch.softmax(logits.detach().float(), dim=-1)
        top2 = p.topk(2, dim=-1).values
        ent = -(p * torch.log(p + 1e-12)).sum(-1, keepdim=True) / math.log(self.num_classes)
        lc = log_count.detach().float().unsqueeze(-1)
        dist = (lc - self.cls_bounds.view(1, -1)).abs().min(-1, keepdim=True).values
        ls = log_sigma.detach().float().unsqueeze(-1)
        return torch.cat([self.zbar_gate * z_bar.detach().float(), top2[:, :1], ent, top2[:, :1] - top2[:, 1:2],
                          lc / 5.0, dist, ls], dim=-1)

    def confidence(self, z_bar, logits, log_count, log_sigma) -> torch.Tensor:
        f = self.conf_features(z_bar, logits, log_count, log_sigma)
        if float(self.logistic_head) > 0.5:
            return self.conf_linear(f[:, -N_SCALAR_CONF:]).squeeze(-1)
        return self.conf_head(f).squeeze(-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        Z, dens = self.encode_full(x)
        z_bar, logits, lc, ls = self.head(Z, dens)
        c = self.confidence(z_bar, logits, lc, ls)
        return {"tokens": Z, "z_bar": z_bar, "logits": logits, "density": dens, "log_count": lc, "log_sigma": ls,
                "count": torch.expm1(lc).clamp_min(0.0),
                "belief": torch.softmax(logits.float() / self.T_b, dim=-1),
                "conf_logit": c, "delta": torch.sigmoid(c / self.T_delta), "delta_raw": torch.sigmoid(c)}

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """{'stage1': everything except the confidence head, 'stage2': confidence-head params}."""
        conf = list(self.conf_head.parameters()) + list(self.conf_linear.parameters())
        ids = {id(p) for p in conf}
        return {"stage1": [p for p in self.parameters() if id(p) not in ids], "stage2": conf}


def build(cfg_perception, num_classes: int, pretrained: bool = True, count_edges=(45, 76, 108)) -> SPCRNet:
    p = cfg_perception
    return SPCRNet(p.backbone, num_classes, p.d, p.d_hidden, p.attn_heads, p.ssm_state, pretrained=pretrained,
                   grid=getattr(p, "token_grid", 7), count_edges=count_edges)
