
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, T: int, n_envs: int, M: int, obs_dim: int, state_dim: int):
        self.T, self.N, self.M = int(T), int(n_envs), int(M)
        shp = (self.T, self.N)
        self.state = np.zeros(shp + (state_dim,))
        self.obs = np.zeros(shp + (M, obs_dim))
        self.mask = np.zeros(shp + (M, 3), dtype=bool)
        self.delta = np.zeros(shp + (M,))
        self.mode = np.zeros(shp + (M,), dtype=np.int64)
        self.z = np.zeros(shp + (M, 2))
        self.logp = np.zeros(shp + (M,))
        self.reward = np.zeros(shp)
        self.done = np.zeros(shp)
        self.values_ext: np.ndarray | None = None
        self.delta_ext: np.ndarray | None = None
        self.last_state: np.ndarray | None = None
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr == self.T

    def add(self, *, state, obs, mask, delta, mode, z, logp, reward, done) -> None:
        """All arguments carry a leading environment axis of length N."""
        if self.full:
            raise RuntimeError("buffer is full; call finish() and clear()")
        t = self.ptr
        self.state[t], self.obs[t], self.mask[t], self.delta[t] = state, obs, mask, delta
        self.mode[t], self.z[t], self.logp[t] = mode, z, logp
        self.reward[t], self.done[t] = reward, np.asarray(done, dtype=np.float64)
        self.ptr += 1

    def finish(self, values: np.ndarray, last_values: np.ndarray, last_delta: np.ndarray) -> None:
        """values (T,N) from the batched critic pass; last_values (N,); last_delta (N,M)."""
        if not self.full:
            raise RuntimeError("finish() called before the buffer is full")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.T, self.N):
            raise ValueError(f"values must have shape {(self.T, self.N)}")
        self.values_ext = np.vstack([values, np.asarray(last_values, dtype=np.float64)[None, :]])
        self.delta_ext = np.concatenate([self.delta, np.asarray(last_delta, dtype=np.float64)[None]], axis=0)

    def actor_minibatches(self, batch_size: int, rng: np.random.Generator):
        """Yield (t_idx, e_idx, u_idx) covering all T*N*M triples once, in random order."""
        n = self.T * self.N * self.M
        perm = rng.permutation(n)
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            t, rest = np.divmod(idx, self.N * self.M)
            e, u = np.divmod(rest, self.M)
            yield t, e, u

    def critic_minibatches(self, n_batches: int, rng: np.random.Generator):
        """Yield (t_idx, e_idx) split into n_batches groups, covering all T*N slots once."""
        perm = rng.permutation(self.T * self.N)
        for chunk in np.array_split(perm, max(1, n_batches)):
            if chunk.size:
                yield np.divmod(chunk, self.N)

    def clear(self) -> None:
        self.ptr = 0
        self.values_ext = None
        self.delta_ext = None
