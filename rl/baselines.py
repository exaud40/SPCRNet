
from __future__ import annotations

import numpy as np

from ..system.latency_energy import A2A, A2G, LOCAL

NO_GATE = 1.01


class Policy:
    name = "policy"
    delta_th = NO_GATE
    mu = 0.0

    def act(self, out, env) -> dict:
        raise NotImplementedError

    @staticmethod
    def _full(mode: np.ndarray) -> dict:
        M = len(mode)
        return {"mode": np.asarray(mode, dtype=np.int64), "p": np.ones(M), "beta": np.ones(M)}


class AlwaysLocal(Policy):
    name = "Always-Local"

    def act(self, out, env):
        return self._full(np.zeros(out.mask.shape[0], dtype=np.int64))


class AlwaysA2G(Policy):
    name = "Always-A2G"

    def act(self, out, env):
        return self._full(np.full(out.mask.shape[0], A2G))


class AlwaysA2A(Policy):
    name = "Always-A2A"

    def act(self, out, env):
        return self._full(np.where(out.mask[:, A2A], A2A, LOCAL))


class RandomPolicy(Policy):
    """Uniform over feasible modes, uniform power and band share ."""
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, out, env):
        M = out.mask.shape[0]
        mode = np.array([self.rng.choice(np.flatnonzero(out.mask[u])) for u in range(M)])
        return {"mode": mode, "p": self.rng.uniform(size=M), "beta": self.rng.uniform(0.05, 1.0, size=M)}


class FixedThreshold(Policy):

    name = "Fixed-Threshold (MSP)"

    def __init__(self, tau: float):
        self.tau = float(tau)

    def act(self, out, env):
        msp = env.b.max(axis=1)
        return self._full(np.where(msp < self.tau, A2G, LOCAL))


class GreedyGate(Policy):
 
    name = "Greedy-Gate"

    def __init__(self, tau: float):
        self.tau = float(tau)

    def act(self, out, env):
        M = out.mask.shape[0]
        mode = np.zeros(M, dtype=np.int64)
        for u in range(M):
            if out.delta[u] >= self.tau:
                continue
            nb = env.nbrs[u]
            if nb.size and env.delta[nb].max() > out.delta[u]:
                mode[u] = A2A
            else:
                mode[u] = A2G
        return self._full(mode)


BASELINES = {"local": AlwaysLocal, "a2g": AlwaysA2G, "a2a": AlwaysA2A, "random": RandomPolicy}
TUNABLE = {"fixed": FixedThreshold, "greedy": GreedyGate}
