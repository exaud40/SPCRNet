
from __future__ import annotations
import numpy as np


def init_positions(
    rng: np.random.Generator, M: int, area_radius: float, d_min: float, d_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (leader_xy (2,), followers_xy (M,2)) with d_min <= ||w_u - w_l|| <= d_max."""
    if not 0 <= d_min < d_max:
        raise ValueError("need 0 <= d_min < d_max")
    if d_max > area_radius:
        raise ValueError("d_max must not exceed the area radius")
    leader = np.zeros(2)
    # uniform over the annulus area (not uniform in radius)
    radius = np.sqrt(rng.uniform(d_min**2, d_max**2, size=M))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=M)
    followers = leader + np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return leader, followers


def project_to_annulus(xy: np.ndarray, center: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Radially project each point into [d_min, d_max] around center."""
    rel = np.asarray(xy, dtype=np.float64) - center
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    unit = np.where(dist > 0, rel / np.where(dist > 0, dist, 1.0), np.array([[1.0, 0.0]]))
    return center + unit * np.clip(dist, d_min, d_max)


def step_followers(
    followers_xy: np.ndarray,
    leader_xy: np.ndarray,
    waypoints: np.ndarray,
    advance: np.ndarray,
    l_max: float,
    d_min: float,
    d_max: float,
) -> np.ndarray:
    """Move followers with advance[u] == True by at most l_max toward waypoints[u];
    then project each follower back into the annulus [d_min, d_max] around the leader."""
    xy = np.asarray(followers_xy, dtype=np.float64).copy()
    delta = np.asarray(waypoints, dtype=np.float64) - xy
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    step = np.where(dist > 0, delta / np.where(dist > 0, dist, 1.0) * np.minimum(dist, l_max), 0.0)
    xy = xy + np.asarray(advance, dtype=bool)[:, None] * step
    return project_to_annulus(xy, np.asarray(leader_xy, dtype=np.float64), d_min, d_max)


def edge_positions(rng: np.random.Generator, J: int, radius: float) -> np.ndarray:
    """Edge servers uniform over a disk of the given radius around the cluster center."""
    r = radius * np.sqrt(rng.uniform(size=J))
    a = rng.uniform(0.0, 2.0 * np.pi, size=J)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def random_waypoints(rng: np.random.Generator, n: int, leader_xy: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Search waypoints uniform over the annulus around the leader."""
    radius = np.sqrt(rng.uniform(d_min**2, d_max**2, size=n))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.asarray(leader_xy) + np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def neighbor_sets(
    followers_xy: np.ndarray, scene_group: np.ndarray, d_max: float
) -> list[np.ndarray]:
    """N_u(t): co-observing followers (same scene_group) within d_max, excluding u itself."""
    xy = np.asarray(followers_xy, dtype=np.float64)
    grp = np.asarray(scene_group)
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    M = xy.shape[0]
    ok = (grp[:, None] == grp[None, :]) & (dist <= d_max) & ~np.eye(M, dtype=bool)
    return [np.flatnonzero(ok[u]) for u in range(M)]
