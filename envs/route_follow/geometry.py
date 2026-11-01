"""Route geometry on tensors: the walls, the moving target, the route frame, rangefinder
beams. One ``RouteGeometry`` holds every env's route; ``set`` replaces some envs' routes.

**Walls.** A corridor is one box per segment, ``half_width`` either side and
``half_height`` above and below the centreline, each box extended ``half_width`` past both
ends of its segment so the outside of a turn is filled. A tunnel is one capsule per segment
(radius ``radius`` around it, round ends). The free air is the union of the pieces, and
``clearance`` is the largest of the per-piece distances to that piece's boundary. That is
never more than the true distance to the union's boundary (it can be less near the inside
of a turn), so contact is never missed and beams can be sphere-traced safely: stepping by
the clearance never steps through a wall.

**Target.** The point at arc length ``s`` along the centreline, ``s`` following the route's
speed plan in time (``routes.py``: from rest, through each corner at its corner speed, to
rest on the last waypoint).

**Route frame.** x points from the target to the target ``lookahead`` metres further on
(the chord, so the frame turns smoothly through a corner instead of jumping at the joint;
on the last ``lookahead`` metres it keeps the chord of that stretch), y is horizontal and
to the left, z completes it (up, tilted with a climb).
"""

from __future__ import annotations

import torch

from envs.route_follow.routes import Routes

_Z = (0.0, 0.0, 1.0)


def _unit(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def left_of(x: torch.Tensor) -> torch.Tensor:
    """Horizontal unit vector to the left of direction ``x`` (z cross x), (..., 3)."""
    z = torch.tensor(_Z, dtype=x.dtype, device=x.device).expand_as(x)
    return _unit(torch.cross(z, x, dim=-1))


def frame_from_x(x: torch.Tensor) -> torch.Tensor:
    """Rotation (..., 3, 3) whose columns are the frame's x, y (left, horizontal), z axes."""
    x = _unit(x)
    y = left_of(x)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


class RouteGeometry:
    def __init__(self, num_envs: int, segments: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        n, s = int(num_envs), int(segments)
        self.num_envs, self.segments = n, s
        z = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        self.dtype, self.device = dtype, torch.device(device)
        self.waypoints = z(n, s + 1, 3)
        self.seg_x = z(n, s, 3)  # unit tangent
        self.seg_y = z(n, s, 3)  # horizontal left
        self.seg_z = z(n, s, 3)  # x cross y
        self.seg_len = z(n, s)
        self.cum = z(n, s + 1)  # arc length at each waypoint
        self.tunnel = torch.zeros(n, dtype=torch.bool, device=device)
        self.half_width, self.half_height, self.radius = z(n), z(n), z(n)
        self.speed, self.accel = z(n), z(n)
        self.corner_speed = z(n, s + 1)
        self.seg_x[..., 0] = 1.0
        self.seg_y[..., 1] = 1.0
        self.seg_z[..., 2] = 1.0
        self.half_width[:] = self.half_height[:] = self.radius[:] = 1.0
        self.speed[:] = self.accel[:] = 1.0

    @classmethod
    def from_routes(cls, routes: Routes, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> RouteGeometry:
        n, s = routes.waypoints.shape[0], routes.waypoints.shape[1] - 1
        geo = cls(n, s, device, dtype)
        geo.set(torch.arange(n, device=geo.device), routes)
        return geo

    def set(self, env_ids: torch.Tensor, routes: Routes) -> None:
        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        wp = t(routes.waypoints)
        d = wp[:, 1:] - wp[:, :-1]
        length = d.norm(dim=-1)
        x = _unit(d)
        y = left_of(x)
        self.waypoints[env_ids] = wp
        self.seg_x[env_ids] = x
        self.seg_y[env_ids] = y
        self.seg_z[env_ids] = torch.cross(x, y, dim=-1)
        self.seg_len[env_ids] = length
        self.cum[env_ids] = torch.cat([torch.zeros_like(length[:, :1]), length.cumsum(-1)], -1)
        self.tunnel[env_ids] = torch.as_tensor(routes.tunnel, device=self.device)
        self.half_width[env_ids] = t(routes.half_width)
        self.half_height[env_ids] = t(routes.half_height)
        self.radius[env_ids] = t(routes.radius)
        self.speed[env_ids] = t(routes.speed)
        self.accel[env_ids] = t(routes.accel)
        self.corner_speed[env_ids] = t(routes.corner_speed)

    @property
    def length(self) -> torch.Tensor:
        return self.cum[:, -1]

    # ---------------------------------------------------------------- along the route

    def _segment_of(self, s: torch.Tensor) -> torch.Tensor:
        """Segment index for arc length ``s`` of shape (N,) or (N, K)."""
        squeeze = s.dim() == 1
        ss = s[:, None] if squeeze else s
        k = torch.searchsorted(self.cum[:, 1:-1].contiguous(), ss.contiguous(), right=True)
        return k[:, 0] if squeeze else k

    def point_at(self, s: torch.Tensor) -> torch.Tensor:
        """Centreline point at arc length ``s`` (N,) or (N, K), clamped to the route."""
        s = torch.minimum(s.clamp(min=0.0), self.length.view(-1, *([1] * (s.dim() - 1))))
        k = self._segment_of(s)
        if s.dim() == 1:
            idx = k[:, None, None].expand(-1, 1, 3)
            w0 = self.waypoints.gather(1, idx)[:, 0]
            x = self.seg_x.gather(1, idx)[:, 0]
            c = self.cum.gather(1, k[:, None])[:, 0]
            return w0 + (s - c)[:, None] * x
        idx = k[..., None].expand(-1, -1, 3)
        w0 = self.waypoints.gather(1, idx)
        x = self.seg_x.gather(1, idx)
        c = self.cum.gather(1, k)
        return w0 + (s - c)[..., None] * x

    def segment_x_at(self, s: torch.Tensor) -> torch.Tensor:
        """Tangent of the segment that arc length ``s`` (N,) lies on, (N, 3)."""
        k = self._segment_of(s.clamp(min=0.0))
        return self.seg_x.gather(1, k[:, None, None].expand(-1, 1, 3))[:, 0]

    def target(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The target's arc length and speed at time ``t`` (N,) since the route started: per
        segment, from its entry corner speed up toward cruise and down to its exit corner
        speed (``routes.segment_times``, same formula), then at rest on the last waypoint."""
        length, u = self.seg_len, self.corner_speed
        u0, u1 = u[:, :-1], u[:, 1:]
        v, a = self.speed[:, None], self.accel[:, None]
        vp = torch.minimum(v, torch.sqrt(((2 * a * length + u0 * u0 + u1 * u1) / 2).clamp(
            min=0.0)))
        vp = torch.maximum(vp, torch.maximum(u0, u1)).clamp(min=1e-9)
        t1 = (vp - u0) / a
        d1 = (vp * vp - u0 * u0) / (2 * a)
        d2 = (vp * vp - u1 * u1) / (2 * a)
        tc = (length - d1 - d2).clamp(min=0.0) / vp
        dur = t1 + tc + (vp - u1) / a
        tcum = torch.cat([torch.zeros_like(dur[:, :1]), dur.cumsum(-1)], -1)
        k = torch.searchsorted(tcum[:, 1:-1].contiguous(), t[:, None].contiguous(),
                               right=True)
        g = lambda x: x.gather(1, k)[:, 0]  # noqa: E731
        tau = (t - g(tcum)).clamp(min=0.0)
        a1, u0k, u1k, vpk = a[:, 0], g(u0), g(u1), g(vp)
        t1k, tck, durk, lk, d1k = g(t1), g(tc), g(dur), g(length), g(d1)
        back = (durk - tau).clamp(min=0.0)
        s_local = torch.where(
            tau < t1k, u0k * tau + 0.5 * a1 * tau * tau,
            torch.where(tau < t1k + tck, d1k + vpk * (tau - t1k),
                        lk - (u1k * back + 0.5 * a1 * back * back)))
        speed = torch.where(tau < t1k, u0k + a1 * tau,
                            torch.where(tau < t1k + tck, vpk, u1k + a1 * back))
        s = (g(self.cum[:, :-1]) + s_local.clamp(min=0.0)).minimum(self.length)
        done = t >= tcum[:, -1]
        return torch.where(done, self.length, s), torch.where(done, torch.zeros_like(speed),
                                                              speed)

    def route_frame(self, s: torch.Tensor, lookahead: float) -> torch.Tensor:
        """Route frame (N, 3, 3) at arc length ``s`` (module docstring)."""
        length = self.length
        s0 = torch.minimum(s, (length - lookahead).clamp(min=0.0))
        chord = self.point_at(torch.minimum(s0 + lookahead, length)) - self.point_at(s0)
        short = chord.norm(dim=-1, keepdim=True) < 1e-4
        chord = torch.where(short, self.segment_x_at(s), chord)
        return frame_from_x(chord)

    # ---------------------------------------------------------------- walls

    def clearance(self, pos: torch.Tensor) -> torch.Tensor:
        """Distance to the nearest wall, floor or ceiling, positive in free air, for
        ``pos`` (N, 3) or (N, K, 3). A lower bound on the true distance (module docstring)."""
        squeeze = pos.dim() == 2
        p = pos[:, None, :] if squeeze else pos  # (N, K, 3)
        rel = p[:, :, None, :] - self.waypoints[:, None, :-1, :]  # (N, K, S, 3)
        a = (rel * self.seg_x[:, None]).sum(-1)
        b = (rel * self.seg_y[:, None]).sum(-1)
        c = (rel * self.seg_z[:, None]).sum(-1)
        length = self.seg_len[:, None]
        hw = self.half_width[:, None, None]
        hh = self.half_height[:, None, None]
        box = torch.minimum(torch.minimum(a + hw, length + hw - a),
                            torch.minimum(hw - b.abs(), hh - c.abs()))
        along = torch.minimum(a.clamp(min=0.0), length)
        off = rel - along[..., None] * self.seg_x[:, None]
        capsule = self.radius[:, None, None] - off.norm(dim=-1)
        per_seg = torch.where(self.tunnel[:, None, None], capsule, box)
        out = per_seg.max(dim=-1).values
        return out[:, 0] if squeeze else out

    def distance_to_route(self, pos: torch.Tensor) -> torch.Tensor:
        """Distance from ``pos`` (N, 3) to the centreline polyline, (N,): the cross-track
        error."""
        rel = pos[:, None, :] - self.waypoints[:, :-1, :]
        along = torch.minimum((rel * self.seg_x).sum(-1).clamp(min=0.0), self.seg_len)
        return (rel - along[..., None] * self.seg_x).norm(dim=-1).min(dim=-1).values

    def progress(self, pos: torch.Tensor) -> torch.Tensor:
        """Arc length of the centreline point nearest ``pos`` (N, 3), (N,)."""
        rel = pos[:, None, :] - self.waypoints[:, :-1, :]
        along = torch.minimum((rel * self.seg_x).sum(-1).clamp(min=0.0), self.seg_len)
        dist = (rel - along[..., None] * self.seg_x).norm(dim=-1)
        k = dist.argmin(dim=-1, keepdim=True)
        return (self.cum[:, :-1].gather(1, k) + along.gather(1, k))[:, 0]

    def trace(self, origin: torch.Tensor, dirs: torch.Tensor, max_m: float,
              iters: int = 40, tol_m: float = 1e-3) -> torch.Tensor:
        """Distance along each beam from ``origin`` (N, 3) in directions ``dirs`` (N, R, 3,
        unit) to the first wall, by sphere tracing on ``clearance``; ``inf`` past ``max_m``
        and 0 for an origin already outside. A beam that has not converged after ``iters``
        steps reads short (where it got to), never long."""
        dist = torch.zeros(dirs.shape[:2], dtype=dirs.dtype, device=dirs.device)
        for _ in range(int(iters)):
            c = self.clearance(origin[:, None, :] + dist[..., None] * dirs)
            dist = dist + torch.where(dist <= max_m, c.clamp(min=0.0), torch.zeros_like(c))
        c = self.clearance(origin[:, None, :] + dist[..., None] * dirs)
        inside = self.clearance(origin) > 0
        hit = c.abs() <= tol_m + 1e-6 * dist
        out = torch.where(dist > max_m, torch.full_like(dist, float("inf")), dist)
        out = torch.where(hit | (dist > max_m), out, dist)  # not converged: reads short
        return torch.where(inside[:, None], out, torch.zeros_like(out))
