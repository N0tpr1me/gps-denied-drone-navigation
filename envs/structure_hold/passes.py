"""Surface passes: the path the hold point follows along the wall (``config.SurfacePass``,
``docs/surface-passes.md``).

A pass is a short list of straight segments in a plane parallel to the wall: the pass
itself, then for a raster a row step across it and the next pass the other way, and so on.
Each segment is flown on a trapezoidal speed profile (accelerate, cruise, brake to a stop),
one after the other with a pause at each corner, so the path is

    offset(t) = sum_k dir_k * s_k(t - t0_k)

where ``s_k`` is the distance flown along segment ``k`` (0 before it starts, its length once
it is done). Velocity is the same sum with the profile's speed. Per-episode draws are NumPy
from a generator of the passes' own; the per-step evaluation is torch on every env at once.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from envs.structure_hold.config import SurfacePass


class PassPaths:
    """Per-env pass paths, (N, K) segments each; envs that are not passes have K empty ones."""

    def __init__(self, cfg: SurfacePass, num_envs: int, seed: int,
                 device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        n, k = int(num_envs), cfg.max_segments
        self.device, self.dtype = device, dtype
        self.rng = np.random.default_rng([int(seed), 0x9A5])
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.vertical = torch.zeros(n, dtype=torch.bool, device=device)
        self.speed = z(n)  # cruise speed, m/s
        self.dir = z(n, k, 3)  # unit direction of each segment, world frame
        self.length = z(n, k)  # m; 0 for a segment the episode does not fly
        self.t0 = z(n, k)  # s after the engage at which each segment starts
        self.vp = z(n, k)  # peak speed reached on the segment (cruise, or less if short)
        self.ta = z(n, k)  # time to reach it
        self.tc = z(n, k)  # time at it
        self.end_s = z(n)  # s after the engage at which the last segment stops

    def draw(self, env_ids: torch.Tensor) -> None:
        """New paths for ``env_ids``: which are passes, which way, how fast, how many legs."""
        c, rng, n, k = self.cfg, self.rng, int(env_ids.numel()), self.cfg.max_segments
        is_pass = rng.random(n) < c.prob
        length = rng.uniform(*c.length_m, size=n)
        speed = rng.uniform(*c.speed_mps, size=n)
        start = rng.uniform(*c.start_s, size=n)
        vertical = rng.random(n) < c.vertical_prob
        legs = rng.integers(c.legs[0], c.legs[1] + 1, size=n)
        step = rng.uniform(*c.row_step_m, size=n)
        side = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        # Along the wall either way, or up it; the row step goes up a horizontal raster and
        # sideways (the same random side) on a vertical one.
        main = np.zeros((n, 3))
        main[~vertical, 1] = side[~vertical]
        main[vertical, 2] = 1.0
        across = np.zeros((n, 3))
        across[~vertical, 2] = 1.0
        across[vertical, 1] = side[vertical]

        dirs = np.zeros((n, k, 3))
        lens = np.zeros((n, k))
        for j in range(k):
            leg = j // 2
            if j % 2 == 0:  # a pass, alternating direction
                dirs[:, j] = main * (-1.0) ** leg
                lens[:, j] = np.where(leg < legs, length, 0.0)
            else:  # the row step before the next pass
                dirs[:, j] = across
                lens[:, j] = np.where(leg + 1 < legs, step, 0.0)
        lens[~is_pass] = 0.0

        a = c.accel_mps2
        vp = np.minimum(speed[:, None], np.sqrt(lens * a))  # a short segment never reaches it
        ta = vp / a
        tc = np.where(vp > 0, (lens - vp * ta) / np.maximum(vp, 1e-9), 0.0).clip(min=0.0)
        dur = 2.0 * ta + tc
        t0 = np.zeros((n, k))
        t = start.copy()
        for j in range(k):
            t0[:, j] = t
            t = t + np.where(lens[:, j] > 0, dur[:, j] + c.pause_s, 0.0)
        end = np.where(is_pass, t - c.pause_s, 0.0)

        tt = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)  # noqa: E731
        self.active[env_ids] = torch.as_tensor(is_pass, device=self.device)
        self.vertical[env_ids] = torch.as_tensor(vertical & is_pass, device=self.device)
        self.speed[env_ids] = tt(np.where(is_pass, speed, 0.0))
        self.dir[env_ids] = tt(dirs)
        self.length[env_ids] = tt(lens)
        self.t0[env_ids] = tt(t0)
        self.vp[env_ids] = tt(vp)
        self.ta[env_ids] = tt(ta)
        self.tc[env_ids] = tt(tc)
        self.end_s[env_ids] = tt(end)

    def at(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Offset of the hold point from where the pass started, and its velocity, at ``t``
        seconds after the engage, each (N, 3)."""
        a = self.cfg.accel_mps2
        dur = 2.0 * self.ta + self.tc
        tau = torch.minimum((t[:, None] - self.t0).clamp(min=0.0), dur)
        up, cruise_end = tau < self.ta, tau < self.ta + self.tc
        s = torch.where(up, 0.5 * a * tau**2,
                        torch.where(cruise_end, 0.5 * a * self.ta**2 + self.vp * (tau - self.ta),
                                    self.length - 0.5 * a * (dur - tau) ** 2))
        v = torch.where(up, a * tau, torch.where(cruise_end, self.vp, a * (dur - tau)))
        return (self.dir * s[..., None]).sum(1), (self.dir * v[..., None]).sum(1)


def pass_duration_s(cfg: SurfacePass, speed_mps: float, legs: int = 1) -> float:
    """How long a pass takes from the engage to its stop, at the longest start and length:
    for sizing an eval cell's episode so every pass finishes with time left to hold."""
    a, length, step = cfg.accel_mps2, cfg.length_m[1], cfg.row_step_m[1]

    def seg(d: float) -> float:
        if d <= 0:
            return 0.0
        vp = min(speed_mps, math.sqrt(d * a))
        return 2.0 * vp / a + (d - vp * vp / a) / vp

    return (cfg.start_s[1] + legs * seg(length) + (legs - 1) * seg(step)
            + 2 * (legs - 1) * cfg.pause_s)
