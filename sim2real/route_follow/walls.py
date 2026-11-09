"""One route's walls and moving target in NumPy, for the Gazebo side of Route Follow.

The mock (``envs/route_follow/geometry.py``) holds every env's route on torch tensors. The
Gazebo runner scores a flight from its ulog on the host and the SITL image has no torch, so
this is the same geometry for a single route in NumPy: the same union of pieces (one box,
arched box or capsule per segment and per side branch), the same ``clearance`` (a lower
bound on the distance to the nearest wall, positive in free air), the same centreline
distance and the same target speed plan. ``tests/test_walls.py`` pins it to the torch
version on random routes of every world, so a Gazebo flight is judged against exactly the
walls a mock episode is.
"""

from __future__ import annotations

import math

import numpy as np

from envs.route_follow.routes import Routes, segment_times


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def left_of(x: np.ndarray) -> np.ndarray:
    """Horizontal unit vector to the left of direction ``x`` (z cross x), (..., 3)."""
    return _unit(np.stack([-x[..., 1], x[..., 0], np.zeros_like(x[..., 0])], -1))


class RouteWalls:
    """Route ``index`` of a :class:`Routes`, as NumPy arrays (module docstring)."""

    def __init__(self, routes: Routes, index: int = 0):
        i = int(index)
        wp = np.asarray(routes.waypoints[i], dtype=float)
        d = np.diff(wp, axis=0)
        self.waypoints = wp
        self.seg_len = np.linalg.norm(d, axis=-1)
        self.seg_x = _unit(d)
        self.seg_y = left_of(self.seg_x)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.tunnel = bool(routes.tunnel[i])
        self.arched = bool(routes.arched[i])
        self.half_width = float(routes.half_width[i])
        self.half_height = float(routes.half_height[i])
        self.radius = float(routes.radius[i])
        self.speed = float(routes.speed[i])
        self.accel = float(routes.accel[i])
        self.corner_speed = np.asarray(routes.corner_speed[i], dtype=float)
        on = routes.branch_len[i] > 0
        bdir = np.asarray(routes.branch_dir[i][on], dtype=float).reshape(-1, 3)
        bx = _unit(bdir) if len(bdir) else bdir
        by = left_of(bx) if len(bx) else bx
        self.piece_start = np.concatenate([wp[:-1], routes.branch_start[i][on]]).reshape(-1, 3)
        self.piece_x = np.concatenate([self.seg_x, bx]).reshape(-1, 3)
        self.piece_y = np.concatenate([self.seg_y, by]).reshape(-1, 3)
        self.piece_z = np.cross(self.piece_x, self.piece_y)
        self.piece_len = np.concatenate([self.seg_len, routes.branch_len[i][on]])
        self.piece_scale = np.concatenate([routes.seg_scale[i], routes.branch_scale[i][on]])
        self.num_branches = int(on.sum())

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    @property
    def duration(self) -> float:
        """Time the target takes from the first waypoint to rest on the last, s."""
        return float(self.segment_durations().sum())

    def segment_durations(self) -> np.ndarray:
        return segment_times(self.seg_len[None], self.corner_speed[None],
                             np.array([self.speed]), np.array([self.accel]))[0]

    # ---------------------------------------------------------------- walls

    def clearance(self, pos: np.ndarray) -> np.ndarray:
        """Distance to the nearest wall, floor or ceiling for ``pos`` (..., 3), positive in
        free air; a lower bound on the true distance, as ``RouteGeometry.clearance``."""
        p = np.asarray(pos, dtype=float)
        rel = p[..., None, :] - self.piece_start  # (..., P, 3)
        a = (rel * self.piece_x).sum(-1)
        b = (rel * self.piece_y).sum(-1)
        c = (rel * self.piece_z).sum(-1)
        length, scale = self.piece_len, self.piece_scale
        if self.tunnel:
            along = np.minimum(np.maximum(a, 0.0), length)
            off = rel - along[..., None] * self.piece_x
            per_piece = self.radius * scale - np.linalg.norm(off, axis=-1)
        else:
            hw, hh = self.half_width * scale, self.half_height * scale
            ends = np.minimum(a + hw, length + hw - a)
            if self.arched:
                # Vertical walls up to the springline c0, a half circle of radius hw above.
                c0 = hh - hw
                walls = np.minimum(c + hh, np.maximum(
                    hw - np.sqrt(b * b + (c - c0) ** 2),
                    np.minimum(hw - np.abs(b), c0 - c)))
            else:
                walls = np.minimum(hw - np.abs(b), hh - np.abs(c))
            per_piece = np.minimum(ends, walls)
        return per_piece.max(axis=-1)

    def trace(self, origin: np.ndarray, dirs: np.ndarray, max_m: float, iters: int = 40,
              tol_m: float = 1e-3) -> np.ndarray:
        """Distance along each unit beam ``dirs`` (R, 3) from ``origin`` (3,) to the first
        wall by sphere tracing, ``inf`` past ``max_m``, 0 for an origin outside the walls
        (``RouteGeometry.trace``)."""
        o = np.asarray(origin, dtype=float)
        d = np.asarray(dirs, dtype=float)
        dist = np.zeros(len(d))
        for _ in range(int(iters)):
            c = self.clearance(o + dist[:, None] * d)
            dist = dist + np.where(dist <= max_m, np.maximum(c, 0.0), 0.0)
        c = self.clearance(o + dist[:, None] * d)
        hit = np.abs(c) <= tol_m + 1e-6 * dist
        out = np.where(hit | ~(dist > max_m), dist, math.inf)
        return out if float(self.clearance(o)) > 0 else np.zeros_like(out)

    # ---------------------------------------------------------------- along the route

    def distance_to_route(self, pos: np.ndarray) -> np.ndarray:
        """Distance from ``pos`` (..., 3) to the centreline polyline: the cross-track
        error."""
        rel = np.asarray(pos, dtype=float)[..., None, :] - self.waypoints[:-1]
        along = np.minimum(np.maximum((rel * self.seg_x).sum(-1), 0.0), self.seg_len)
        return np.linalg.norm(rel - along[..., None] * self.seg_x, axis=-1).min(axis=-1)

    def progress(self, pos: np.ndarray) -> np.ndarray:
        """Arc length of the centreline point nearest ``pos`` (..., 3)."""
        rel = np.asarray(pos, dtype=float)[..., None, :] - self.waypoints[:-1]
        along = np.minimum(np.maximum((rel * self.seg_x).sum(-1), 0.0), self.seg_len)
        dist = np.linalg.norm(rel - along[..., None] * self.seg_x, axis=-1)
        k = dist.argmin(axis=-1)
        return self.cum[:-1][k] + np.take_along_axis(along, k[..., None], -1)[..., 0]

    def point_at(self, s: np.ndarray) -> np.ndarray:
        """Centreline point at arc length ``s`` (...,), clamped to the route."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        k = np.clip(np.searchsorted(self.cum[1:-1], s, side="right"), 0, len(self.seg_len) - 1)
        return self.waypoints[k] + (s - self.cum[k])[..., None] * self.seg_x[k]

    def target_s(self, t: np.ndarray) -> np.ndarray:
        """The target's arc length at time ``t`` (...,) since the route started: from rest,
        through each corner at its corner speed, to rest on the last waypoint
        (``RouteGeometry.target``, same formula)."""
        t = np.asarray(t, dtype=float)
        u0, u1 = self.corner_speed[:-1], self.corner_speed[1:]
        v, a, length = self.speed, self.accel, self.seg_len
        vp = np.minimum(v, np.sqrt(np.maximum((2 * a * length + u0 * u0 + u1 * u1) / 2, 0.0)))
        vp = np.maximum(np.maximum(vp, np.maximum(u0, u1)), 1e-9)
        t1 = (vp - u0) / a
        d1 = (vp * vp - u0 * u0) / (2 * a)
        tc = np.maximum(length - d1 - (vp * vp - u1 * u1) / (2 * a), 0.0) / vp
        dur = t1 + tc + (vp - u1) / a
        tcum = np.concatenate([[0.0], np.cumsum(dur)])
        k = np.clip(np.searchsorted(tcum[1:-1], t, side="right"), 0, len(dur) - 1)
        tau = np.maximum(t - tcum[k], 0.0)
        back = np.maximum(dur[k] - tau, 0.0)
        s_local = np.where(
            tau < t1[k], u0[k] * tau + 0.5 * a * tau * tau,
            np.where(tau < t1[k] + tc[k], d1[k] + vp[k] * (tau - t1[k]),
                     length[k] - (u1[k] * back + 0.5 * a * back * back)))
        s = np.minimum(self.cum[:-1][k] + np.maximum(s_local, 0.0), self.length)
        return np.where(t >= tcum[-1], self.length, s)
