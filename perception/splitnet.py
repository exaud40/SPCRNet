
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resnet50(pretrained: bool):
    import torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)


def level_logits(mu: torch.Tensor, log_sig: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """log P(level k) for log(1 + n) ~ N(mu, sigma^2) and level boundaries ``bounds`` (log1p scale)."""
    z = (bounds.view(1, -1) - mu.unsqueeze(-1)) / torch.exp(log_sig).unsqueeze(-1)
    cdf = torch.special.ndtr(z)
    one, zero = torch.ones_like(mu).unsqueeze(-1), torch.zeros_like(mu).unsqueeze(-1)
    return torch.log((torch.cat([cdf, one], -1) - torch.cat([zero, cdf], -1)).clamp_min(1e-6))


def awgn(z: torch.Tensor, snr_db: torch.Tensor | float | None) -> torch.Tensor:
    """Unit-power payload + complex AWGN at the given SNR (per sample); None = clean link."""
    if snr_db is None:
        return z
    snr = torch.as_tensor(snr_db, dtype=z.dtype, device=z.device).reshape(-1, *([1] * (z.dim() - 1)))
    return z + torch.randn_like(z) * torch.sqrt(10.0 ** (-snr / 10.0))


class UAVHead(nn.Module):

    def __init__(self, c_b: int = 8, pretrained: bool = True, count_edges=(45, 76, 108), local_head: str = "payload"):
        super().__init__()
        r = _resnet50(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.encoder = nn.Sequential(nn.Conv2d(512, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, c_b, 1))
        self.local_head = local_head
        if local_head == "rich":
            w = 128
            self.back = nn.Sequential(nn.Conv2d(512, w, 1), nn.GroupNorm(16, w), nn.ReLU(inplace=True),
                                      *[m for _ in range(3) for m in (nn.Conv2d(w, w, 3, 1, 2, dilation=2),
                                                                       nn.GroupNorm(16, w), nn.ReLU(inplace=True))])
            self.dens = nn.Conv2d(w, 1, 1)
            nn.init.constant_(self.dens.bias, -4.0)
            self.glob = nn.Sequential(nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten(), nn.Dropout(0.3))
            g = w
        elif local_head == "payload":
            self.back = nn.Identity()
            self.dens = nn.Sequential(nn.Conv2d(c_b, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
            nn.init.constant_(self.dens[-1].bias, -3.0)
            self.glob = nn.Sequential(nn.Conv2d(c_b, 64, 3, 2, 1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten(), nn.Dropout(0.3))
            g = 64
        else:
            raise ValueError(f"local_head must be 'payload' or 'rich', got {local_head!r}")
        self.corr = nn.Linear(g, 1)
        self.logsig = nn.Linear(g, 1)
        nn.init.zeros_(self.corr.weight)
        nn.init.zeros_(self.corr.bias)
        nn.init.constant_(self.logsig.bias, math.log(0.3))
        self.offset = nn.Parameter(torch.tensor(0.31))
        self.register_buffer("bounds", torch.tensor([math.log1p(e - 0.5) for e in count_edges]))
        self.register_buffer("T_b", torch.tensor(1.0))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(self.stem(x)))

    def encode_from(self, f2: torch.Tensor) -> torch.Tensor:
        z = self.encoder(f2)
        return z / z.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8).sqrt()   # unit average power

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_from(self.features(x))

    def local(self, z: torch.Tensor, f2: torch.Tensor | None = None) -> dict:
        h = self.back(f2) if self.local_head == "rich" else z
        dens = F.softplus(self.dens(h).float()).squeeze(1)
        g = self.glob(h).float()
        mu = torch.log1p(dens.sum((-2, -1))) + self.offset + 0.5 * torch.tanh(self.corr(g).squeeze(-1))
        ls = self.logsig(g).squeeze(-1).clamp(-4.0, 1.5)
        logits = level_logits(mu, ls, self.bounds)
        return {"density": dens, "mu": mu, "log_sigma": ls, "logits": logits,
                "belief": torch.softmax(logits / self.T_b, -1)}


def module_cost(module: nn.Module, run) -> tuple[float, float]:
  
    macs = [0.0]

    def hook(m, inp, out):
        n = out.shape[0]
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
            macs[0] += out.numel() / n * k
        elif isinstance(m, nn.Linear):
            macs[0] += out.numel() / n * m.in_features

    hs = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hs:
            h.remove()
    return macs[0] / 1e9, sum(p.numel() for p in module.parameters()) / 1e6


class ServerBackbone(nn.Module):


    def __init__(self, c_b: int = 8, pretrained: bool = True):
        super().__init__()
        from torchvision.ops import FeaturePyramidNetwork
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

        r = _resnet50(pretrained)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(c_b, 256, 2, 2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.fpn = FeaturePyramidNetwork([512, 1024, 2048], 256, extra_blocks=LastLevelMaxPool())
        self.out_channels = 256
        self.z_in: torch.Tensor | None = None             # payload set by SplitDetector before each call
        self.last_feats: OrderedDict | None = None         # FPN output of the last call (for the edge count head)

    def forward(self, images_tensor: torch.Tensor) -> OrderedDict:
        z = self.z_in
        c2 = self.decoder(z)
        if c2.shape[-2:] != (images_tensor.shape[-2] // 8, images_tensor.shape[-1] // 8):
            c2 = F.interpolate(c2, size=(images_tensor.shape[-2] // 8, images_tensor.shape[-1] // 8), mode="nearest")
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        self.last_feats = self.fpn(OrderedDict([("0", c2), ("1", c3), ("2", c4)]))
        return self.last_feats


class EdgeCountHead(nn.Module):


    def __init__(self, count_edges=(45, 76, 108)):
        super().__init__()
        self.dens = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 1, 1))
        nn.init.constant_(self.dens[-1].bias, -4.0)
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3))
        self.corr, self.logsig = nn.Linear(128, 1), nn.Linear(128, 1)
        nn.init.zeros_(self.corr.weight)
        nn.init.zeros_(self.corr.bias)
        nn.init.constant_(self.logsig.bias, math.log(0.3))
        self.offset = nn.Parameter(torch.tensor(0.31))
        self.register_buffer("bounds", torch.tensor([math.log1p(e - 0.5) for e in count_edges]))
        self.register_buffer("T_e", torch.tensor(1.0))

    def forward(self, feats: OrderedDict) -> dict:
        dens = F.softplus(self.dens(feats["0"]).float()).squeeze(1)
        g = self.glob(feats["2"]).float()
        mu = torch.log1p(dens.sum((-2, -1))) + self.offset + 0.5 * torch.tanh(self.corr(g).squeeze(-1))
        ls = self.logsig(g).squeeze(-1).clamp(-4.0, 1.5)
        logits = level_logits(mu, ls, self.bounds)
        return {"density": dens, "mu": mu, "log_sigma": ls, "logits": logits,
                "belief": torch.softmax(logits / self.T_e, -1)}


class SplitDetector(nn.Module):
    
    def __init__(self, c_b: int = 8, pretrained: bool = True, count_edges=(45, 76, 108), min_size: int = 576,
                 max_size: int = 1024, local_head: str = "payload"):
        super().__init__()
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.ops import MultiScaleRoIAlign

        self.uav = UAVHead(c_b, pretrained, count_edges, local_head)
        self.server_backbone = ServerBackbone(c_b, pretrained)
        self.edge_count = EdgeCountHead(count_edges)
        anchors = AnchorGenerator(sizes=((16,), (32,), (64,), (128,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4)
        self.detector = FasterRCNN(
            self.server_backbone, num_classes=4, rpn_anchor_generator=anchors,
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2"], output_size=7, sampling_ratio=2),
            min_size=min_size, max_size=max_size, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
            box_detections_per_img=400, box_score_thresh=0.05, rpn_pre_nms_top_n_train=3000,
            rpn_post_nms_top_n_train=1500, rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1000,
            box_batch_size_per_image=512)
        # the detector normalises and (if needed) resizes; the UAV sees the same normalised tensor
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.c_b = c_b

    def uav_forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        f2 = self.uav.features((images - self.mean) / self.std)
        z = self.uav.encode_from(f2)
        return z, self.uav.local(z, f2)

    def uav_cost(self, h: int, w: int) -> dict:
        """Onboard cost of one frame: GMACs and parameters of the whole UAV side (backbone, encoder, local head)."""
        dev = next(self.parameters()).device
        was = self.training
        self.eval()
        g, p = module_cost(self.uav, lambda: self.uav_forward(torch.zeros(1, 3, h, w, device=dev)))
        self.train(was)
        return {"uav_gmacs": g, "uav_params_M": p}

    def server_forward(self, images: list[torch.Tensor] | torch.Tensor, z: torch.Tensor, targets=None):
        self.server_backbone.z_in = z
        imgs = list(images) if isinstance(images, torch.Tensor) else images
        return self.detector(imgs, targets)

    def edge_belief(self) -> dict:
        """Edge count head on the features of the last server_forward call."""
        return self.edge_count(self.server_backbone.last_feats)

    def forward(self, images: torch.Tensor, targets=None, snr_db=None):
        """Training: (detector loss dict, local outputs, edge outputs). Eval: (detections, local, edge)."""
        z, loc = self.uav_forward(images)
        out = self.server_forward(images, awgn(z, snr_db), targets)
        return out, loc, self.edge_belief()

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        pre = {id(p) for m in (self.uav.stem, self.uav.layer1, self.uav.layer2, self.server_backbone.layer3,
                               self.server_backbone.layer4) for p in m.parameters()}
        return {"pretrained": [p for p in self.parameters() if id(p) in pre and p.requires_grad],
                "new": [p for p in self.parameters() if id(p) not in pre and p.requires_grad]}

    def payload_symbols(self, h: int, w: int) -> int:
        """k_s: complex channel uses for one frame of size h x w (two reals per symbol)."""
        return self.c_b * math.ceil(h / 16) * math.ceil(w / 16) // 2


def load_mmdet_backbone(model: SplitDetector, path: str) -> int:
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.") and "num_batches" not in k}
    targets = {"conv1.": model.uav.stem[0], "bn1.": model.uav.stem[1], "layer1.": model.uav.layer1,
               "layer2.": model.uav.layer2, "layer3.": model.server_backbone.layer3, "layer4.": model.server_backbone.layer4}
    n = 0
    for prefix, mod in targets.items():
        part = {k[len(prefix):]: v for k, v in bb.items() if k.startswith(prefix)}
        if part:
            missing, _ = mod.load_state_dict(part, strict=False)
            n += len(part) - len(missing)
    return n
