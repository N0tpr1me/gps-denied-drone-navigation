"""Random routes through a tunnel or corridor, NumPy only (one draw per episode).

A route is ``segments + 1`` waypoints joined by straight segments. The first segment heads
along world +x from ``(0, 0, start_height_m)``; each joint turns the heading by a draw from
``turn_deg`` and each segment has its own slope from ``climb_deg``, except that a short
route turns at fewer joints (``RouteGen.min_segment_m``). The cross-section is a corridor
(rectangle, ``half_width`` x ``half_height``) or a tunnel (circle, ``radius``), drawn per
route; ``envs/route_follow/geometry.py`` builds the walls from it.

**Worlds** (``RouteGen``, all off by default): a corridor can have an arched roof (``arched``:
a mine drift or arched culvert), any segment can be narrowed (``seg_scale``: a doorway, a
collapsed stretch, a smaller pipe section), and dead-end side passages can leave the route
at right angles (``branch_*``: crosscuts, side corridors). The extra draws come after all of
the plain route's draws, so a config with them off draws exactly the routes it always did.

**Speed plan.** The target starts and ends at rest and passes each joint at a corner speed
that drops with the turn, ``speed * (1 - (1 - corner_speed_min_frac) * min(angle, 90 deg) /
90 deg)``: a 90 degree corner at a fifth of the cruise speed, a straight joint at full speed,
as PX4 Mission mode slows for a corner instead of turning at speed. Between joints it
accelerates and brakes at ``accel`` (per segment a trapezoid, or a triangle where the
segment is too short to reach cruise); ``plan_speeds`` lowers corner speeds that cannot be
reached or braked from within a segment. ``segment_times`` gives each segment's duration
(the torch version of the same formula is ``RouteGeometry.target``).

**Length.** The route is scaled so that the whole plan takes ``episode_length_s -
end_hold_s``: every episode ends with the target stopped on the last waypoint for about
``end_hold_s`` (capped at ``max_length_m``, after which it holds longer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from envs.route_follow.config import RouteGen


@dataclass
class Routes:
    """One route per env. Arrays are NumPy here; the env keeps torch copies."""

    waypoints: np.ndarray  # (N, S + 1, 3) m, world ENU
    tunnel: np.ndarray  # (N,) bool: round tunnel, else rectangular corridor
    half_width: np.ndarray  # (N,) m (corridor)
    half_height: np.ndarray  # (N,) m (corridor)
    radius: np.ndarray  # (N,) m (tunnel)
    speed: np.ndarray  # (N,) m/s, the target's cruise speed
    accel: np.ndarray  # (N,) m/s^2
    corner_speed: np.ndarray  # (N, S + 1) m/s at each waypoint (0 at both ends)
    arched: np.ndarray | None = None  # (N,) bool: a corridor with a half-circle roof
    seg_scale: np.ndarray | None = None  # (N, S) cross-section scale of each segment
    branch_start: np.ndarray | None = None  # (N, B, 3) m, on the centreline
    branch_dir: np.ndarray | None = None  # (N, B, 3) unit, horizontal
    branch_len: np.ndarray | None = None  # (N, B) m, 0 where there is no branch
    branch_scale: np.ndarray | None = None  # (N, B) cross-section scale (its segment's)

    def __post_init__(self) -> None:
        n, s = self.waypoints.shape[0], self.waypoints.shape[1] - 1
        if self.arched is None:
            self.arched = np.zeros(n, dtype=bool)
        if self.seg_scale is None:
            self.seg_scale = np.ones((n, s))
        if self.branch_len is None:
            self.branch_start, self.branch_dir = np.zeros((n, 0, 3)), np.zeros((n, 0, 3))
            self.branch_len, self.branch_scale = np.zeros((n, 0)), np.ones((n, 0))

    @property
    def seg_len(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.waypoints, axis=1), axis=-1)

    @property
    def length(self) -> np.ndarray:
        return self.seg_len.sum(-1)

    @property
    def duration(self) -> np.ndarray:
        """Time the target takes from the first waypoint to rest on the last, (N,)."""
        return segment_times(self.seg_len, self.corner_speed, self.speed, self.accel).sum(-1)


def corner_speeds(waypoints: np.ndarray, speed: np.ndarray, min_frac: float) -> np.ndarray:
    """Speed wanted at each waypoint, (N, S + 1): zero at the ends, lowered with the turn
    at each joint (module docstring)."""
    d = np.diff(waypoints, axis=1)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    cos = np.clip((d[:, 1:] * d[:, :-1]).sum(-1), -1.0, 1.0)
    turn = np.minimum(np.arccos(cos), 0.5 * math.pi) / (0.5 * math.pi)
    inner = speed[:, None] * (1.0 - (1.0 - min_frac) * turn)
    zero = np.zeros((waypoints.shape[0], 1))
    return np.concatenate([zero, inner, zero], axis=1)


def plan_speeds(seg_len: np.ndarray, wanted: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """Lower the wanted corner speeds until every segment can reach its exit speed from its
    entry speed (forward pass) and brake to it (backward pass) at ``accel``."""
    u = wanted.copy()
    reach = lambda u0, length: np.sqrt(u0 * u0 + 2.0 * accel * length)  # noqa: E731
    for k in range(1, u.shape[1]):
        u[:, k] = np.minimum(u[:, k], reach(u[:, k - 1], seg_len[:, k - 1]))
    for k in range(u.shape[1] - 2, -1, -1):
        u[:, k] = np.minimum(u[:, k], reach(u[:, k + 1], seg_len[:, k]))
    return u


def segment_times(seg_len: np.ndarray, corner: np.ndarray, speed: np.ndarray,
                  accel: np.ndarray) -> np.ndarray:
    """Duration of each segment, (N, S): from its entry corner speed up toward ``speed``,
    cruise, and down to its exit corner speed, at ``accel`` (a triangle when too short)."""
    u0, u1 = corner[:, :-1], corner[:, 1:]
    v, a = np.asarray(speed)[:, None], np.asarray(accel)[:, None]
    vp = np.minimum(v, np.sqrt(np.maximum((2 * a * seg_len + u0 * u0 + u1 * u1) / 2, 0.0)))
    vp = np.maximum(vp, np.maximum(u0, u1))
    d1, d2 = (vp * vp - u0 * u0) / (2 * a), (vp * vp - u1 * u1) / (2 * a)
    cruise = np.maximum(seg_len - d1 - d2, 0.0) / np.maximum(vp, 1e-9)
    return (vp - u0) / a + cruise + (vp - u1) / a


def _finish(waypoints: np.ndarray, tunnel, half_width, half_height, radius, speed,
            accel: np.ndarray, min_frac: float) -> Routes:
    seg_len = np.linalg.norm(np.diff(waypoints, axis=1), axis=-1)
    corner = plan_speeds(seg_len, corner_speeds(waypoints, speed, min_frac), accel)
    return Routes(waypoints=waypoints, tunnel=np.asarray(tunnel), half_width=half_width,
                  half_height=half_height, radius=radius, speed=speed, accel=accel,
                  corner_speed=corner)


def sample_routes(cfg: RouteGen, n: int, episode_length_s: float,
                  rng: np.random.Generator) -> Routes:
    s = cfg.segments
    speed = rng.uniform(*cfg.speed_mps, size=n)
    accel = np.full(n, cfg.accel_mps2)
    fly_s = episode_length_s - cfg.end_hold_s
    # A first length guess: a straight trapezoid over the flying time.
    v = np.minimum(speed, 0.5 * cfg.accel_mps2 * fly_s)
    length = np.minimum(v * (fly_s - v / cfg.accel_mps2), cfg.max_length_m)
    weights = rng.uniform(*cfg.segment_weight, size=(n, s))
    frac = weights / weights.sum(-1, keepdims=True)
    turns = np.radians(rng.uniform(*cfg.turn_deg, size=(n, s)))
    climb = np.radians(rng.uniform(*cfg.climb_deg, size=(n, s)))
    # Straight pieces: segment k belongs to piece floor(k * pieces / s); turn and new slope
    # only where a new piece starts.
    pieces = np.clip(np.floor(length / cfg.min_segment_m), 1, s)
    piece = np.floor(np.arange(s)[None, :] * pieces[:, None] / s)
    new_piece = np.concatenate([np.zeros((n, 1), bool), piece[:, 1:] != piece[:, :-1]], 1)
    turns = np.where(new_piece, turns, 0.0)  # the first segment heads along +x
    for k in range(1, s):
        climb[:, k] = np.where(new_piece[:, k], climb[:, k], climb[:, k - 1])
    heading = np.cumsum(turns, axis=1)
    unit = np.stack([np.cos(climb) * np.cos(heading), np.cos(climb) * np.sin(heading),
                     np.sin(climb)], -1)
    start = np.zeros((n, 1, 3))
    start[..., 2] = cfg.start_height_m
    tunnel = rng.random(n) < cfg.tunnel_prob
    half_width = 0.5 * rng.uniform(*cfg.corridor_width_m, size=n)
    half_height = 0.5 * rng.uniform(*cfg.corridor_height_m, size=n)
    radius = rng.uniform(*cfg.tunnel_radius_m, size=n)
    # The worlds' draws, only when switched on (module docstring).
    arched = np.zeros(n, dtype=bool)
    if cfg.arch_prob > 0:
        arched = ~tunnel & (rng.random(n) < cfg.arch_prob)
        half_height = np.where(arched, np.maximum(half_height, 0.6 * half_width), half_height)
    seg_scale = np.ones((n, s))
    if cfg.pinch_prob > 0:
        pinched = rng.random((n, s)) < cfg.pinch_prob
        seg_scale = np.where(pinched, rng.uniform(*cfg.pinch_frac, size=(n, s)), 1.0)
    b = cfg.branches
    if b > 0:
        b_on = rng.random((n, b)) < cfg.branch_prob
        b_seg = rng.integers(0, s, size=(n, b))
        b_at = rng.uniform(0.3, 0.7, size=(n, b))
        b_side = np.where(rng.random((n, b)) < 0.5, -1.0, 1.0)
        b_len = np.where(b_on, rng.uniform(*cfg.branch_len_m, size=(n, b)), 0.0)

    def build(total: np.ndarray) -> Routes:
        step = unit * (total[:, None] * frac)[..., None]
        wp = np.concatenate([start, start + np.cumsum(step, axis=1)], axis=1)
        r = _finish(wp, tunnel, half_width, half_height, radius, speed, accel,
                    cfg.corner_speed_min_frac)
        r.arched, r.seg_scale = arched, seg_scale
        if b > 0:
            rows = np.arange(n)[:, None]
            d = (wp[:, 1:] - wp[:, :-1])[rows, b_seg]  # (n, b, 3) the branch's segment
            left = np.stack([-d[..., 1], d[..., 0], np.zeros_like(d[..., 0])], -1)
            left /= np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-9)
            r.branch_start = wp[rows, b_seg] + b_at[..., None] * d
            r.branch_dir = b_side[..., None] * left
            r.branch_len = b_len
            r.branch_scale = seg_scale[rows, b_seg]
        return r

    # The corners slow the target down: shrink the route until the plan fits the flying time
    # (the time is close to linear in the length, so a few rescales converge).
    routes = build(length)
    for _ in range(6):
        scale = np.minimum(fly_s / np.maximum(routes.duration, 1e-9), 1.0)
        if np.all(scale > 0.999):
            break
        length = length * scale
        routes = build(length)
    return routes


def _one(x: float) -> np.ndarray:
    return np.array([x], dtype=float)


def straight_route(length: float, *, tunnel: bool, half_width: float = 1.5,
                   half_height: float = 1.5, radius: float = 1.5, speed: float = 1.0,
                   accel: float = 0.5, height: float = 2.0) -> Routes:
    """One straight route along +x, for tests and hand checks."""
    w = np.array([[[0.0, 0.0, height], [length, 0.0, height]]])
    return _finish(w, [tunnel], _one(half_width), _one(half_height), _one(radius),
                   _one(speed), _one(accel), 0.2)


def turn_route(first: float, second: float, turn_deg: float, *, tunnel: bool,
               half_width: float = 1.5, half_height: float = 1.5, radius: float = 1.5,
               speed: float = 1.0, accel: float = 0.5, height: float = 2.0,
               corner_speed_min_frac: float = 0.2) -> Routes:
    """Two segments with one turn (left positive), for tests and hand checks."""
    a = math.radians(turn_deg)
    w = np.array([[[0.0, 0.0, height], [first, 0.0, height],
                   [first + second * math.cos(a), second * math.sin(a), height]]])
    return _finish(w, [tunnel], _one(half_width), _one(half_height), _one(radius),
                   _one(speed), _one(accel), corner_speed_min_frac)
