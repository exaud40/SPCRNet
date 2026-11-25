# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:51:35 2026

@author: Exaud
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class NetworkCfg:
    N: int = 1
    M: int = 6
    J: int = 3
    area_radius: float = 200.0
    H: float = 100.0
    l_max: float = 30.0
    d_min: float = 10.0
    d_max: float = 150.0
    edge_radius: float = 1000.0
    views_per_scene: int = 2


@dataclass
class ChannelCfg:
    f_c: float = 2.6e9
    B_tot: float = 10e6
    P_max_dbm: float = 20.0
    P_l_dbm: float = 30.0
    N0_dbm_hz: float = -174.0
    NF_db: float = 10.0
    E0: float = 1.0e-4
    a: float = 12.08
    b: float = 0.11
    eta_los_db: float = 1.6
    eta_nlos_db: float = 23.0
    nakagami_m: float = 3.0


@dataclass
class SemanticCfg:
    k_b: int = 16
    k_s: int = 256
    snr_grid_db: list[float] = field(default_factory=lambda: [float(x) for x in range(-10, 21, 5)])


@dataclass
class ComputeCfg:
    chi_L: float = 1e9
    chi_F: float = 1e4
    chi_G: float = 2e10
    f_U: float = 1e12
    f_G: float = 3e13


@dataclass
class QoECfg:
    tau: float = 1.0
    D_th: float = 0.2
    zeta_D: float = 50.0


@dataclass
class ConstraintCfg:
    E_bar: float = 2e-3
    delta_min: float = 0.5
    delta_max: float = 0.95
    delta_margin: float = 0.05


@dataclass
class EnvCfg:
    episode_len: int = 200
    progress_on_correct: bool = True
    confidence_source: str = "calibrated"
    allowed_modes: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class RLCfg:
    gamma_disc: float = 0.99
    gae_lambda: float = 0.95
    kappa: float = 2.0
    kappa_anneal_iters: int = 200    # kappa decays linearly to 0 (GAE) over this many iterations; 0 = constant
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_dual: float = 0.05
    n_envs: int = 8
    ppo_epochs: int = 4
    minibatch: int = 1024
    rollout_len: int = 2048
    iterations: int = 400
    hidden: list[int] = field(default_factory=lambda: [400, 300])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class PerceptionCfg:
    backbone: str = "fastvit_t8"
    img_size: int = 224
    d: int = 64
    d_hidden: int = 256
    num_classes: int = 8
    attn_heads: int = 4
    ssm_state: int = 16
    token_grid: int = 7          # L_s = token_grid^2 semantic tokens (49 -> k_s = 49 * d / 2 complex symbols)
    img_w: int = 512             # MDMT frames pre-resized to img_w x img_h (16:9; small targets)
    img_h: int = 288
    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.60, "heldout": 0.15, "cal": 0.10, "test": 0.15}
    )


@dataclass
class Config:
    seed: int = 0
    network: NetworkCfg = field(default_factory=NetworkCfg)
    channel: ChannelCfg = field(default_factory=ChannelCfg)
    semantic: SemanticCfg = field(default_factory=SemanticCfg)
    compute: ComputeCfg = field(default_factory=ComputeCfg)
    qoe: QoECfg = field(default_factory=QoECfg)
    constraint: ConstraintCfg = field(default_factory=ConstraintCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    rl: RLCfg = field(default_factory=RLCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)


def _fill(dc_type: type, values: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, recursing into nested dataclasses.
    Unknown keys raise, so a typo in the YAML fails loudly instead of being ignored.
    """
    obj = dc_type()
    known = {f.name: f for f in fields(dc_type)}
    for key, val in (values or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {dc_type.__name__}")
        current = getattr(obj, key)
        if is_dataclass(current):
            setattr(obj, key, _fill(type(current), val))
        else:
            setattr(obj, key, _coerce(current, val, f"{dc_type.__name__}.{key}"))
    return obj


def _coerce(default: Any, val: Any, name: str) -> Any:
    #Cast a YAML value to the type of the dataclass default.

    if isinstance(default, bool):
        if not isinstance(val, bool):
            raise TypeError(f"{name}: expected bool, got {val!r}")
        return val
    if isinstance(default, (int, float)):
        try:
            num = float(val)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}: expected a number, got {val!r}") from exc
        if isinstance(default, int):
            if not num.is_integer():
                raise TypeError(f"{name}: expected an integer, got {val!r}")
            return int(num)
        return num
    if isinstance(default, list) and default and isinstance(default[0], (int, float)):
        return [type(default[0])(float(v)) for v in val]
    return val


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _fill(Config, raw)


# Unit helpers used everywhere (keep conversions in one place).
def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def db_to_lin(db):
    return 10.0 ** (db / 10.0)


def lin_to_db(x):
    import numpy as np

    return 10.0 * np.log10(x)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:

    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like key=value, got '{item}'")
        path, raw = item.split("=", 1)
        section, key = path.split(".", 1)
        sub = getattr(cfg, section, None)
        if sub is None or not is_dataclass(sub) or key not in {f.name for f in fields(sub)}:
            raise KeyError(f"unknown config key '{path}'")
        current = getattr(sub, key)
        if isinstance(current, list):
            val: Any = [v for v in raw.split(",") if v != ""]
        elif isinstance(current, bool):
            if raw.lower() not in ("true", "false", "1", "0"):
                raise TypeError(f"{path}: expected true/false, got '{raw}'")
            val = raw.lower() in ("true", "1")
        else:
            val = raw
        setattr(sub, key, _coerce(current, val, path) if not isinstance(current, str) else raw)
    return cfg
