"""Heading error: the yaw the vehicle's estimator reports, off from the true yaw.

The position estimate in ``core`` is all a VIO-fed EKF2 gets wrong in position. Its heading
comes from somewhere else: on the benchmark set-up (``sim2real/estimate_bridge/ekf2_ev.params``,
``EKF2_EV_CTRL`` 7, no vision yaw) and on most partner drones it is the compass, and next to
steel (a bridge deck, a tower, a rack, rebar) the compass is wrong by up to tens of degrees
(``docs/market-research.md`` gap 10, Heimdall; ``docs/heading-error.md``). EKF2 then turns its
attitude by that error while the vision positions stay in their own (true) frame, so every
controller that turns a position error into a tilt does it in a frame turned by the error:
the classic "toilet bowl" spiral, and a wall-facing heading that is not the wall's.

This module is the error itself, ``delta`` (rad, + is counter-clockwise seen from above, a
yaw in ENU), per env, as a batched process:

- **offset** at engage (``offset_deg``, magnitude drawn uniform, random sign): the compass is
  already off where the module engages. It stays for the episode.
- **drift** (``drift_rate_deg_s``, random sign, from ``drift_start_s`` after the reset, capped
  at ``drift_max_deg``): the vehicle edges toward the steel, or EKF2 slowly gives in to a
  disturbed compass.
- **steps** (Poisson at ``step_rate_hz``, ``step_deg``, random sign): the field changes
  (a girder passes under the vehicle); EKF2's compass fusion follows with a first-order lag
  ``step_tau_s`` rather than at once.

The sum is clipped at ``max_deg``. ``prob`` is the share of episodes that have any of it, so
training can mix clean and disturbed headings. What the env does with ``delta``
(``envs/structure_hold/env.py``): the module latches its structure frame from EKF2's heading at
engage, so positions and velocities are read in a frame turned by ``delta`` at engage, and the
heading input by how far ``delta`` has moved since.

Pure batched transform, NumPy only; written against the operations NumPy and PyTorch share,
so ``torch_adapter.TorchHeadingError`` reuses it. Its own generator: turning it on changes no
other draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


def _pair(v: Any, name: str) -> tuple[float, float]:
    lo, hi = (float(v), float(v)) if isinstance(v, (int, float)) else (float(v[0]), float(v[1]))
    if not 0 <= lo <= hi:
        raise ValueError(f"heading error {name}: need 0 <= min <= max, got {(lo, hi)}")
    return lo, hi


@dataclass(frozen=True)
class HeadingErrorConfig:
    prob: float = 0.0  # share of episodes with a heading error; 0 is off
    offset_deg: tuple[float, float] = (0.0, 0.0)  # magnitude at engage, random sign
    drift_rate_deg_s: tuple[float, float] = (0.0, 0.0)  # magnitude, random sign
    drift_start_s: tuple[float, float] = (0.0, 0.0)  # after the reset (the engage)
    drift_max_deg: float = 0.0  # the drift stops growing here
    step_rate_hz: float = 0.0
    step_deg: tuple[float, float] = (0.0, 0.0)  # magnitude, random sign
    step_tau_s: float = 3.0  # EKF2 follows a step in the compass with this lag
    max_deg: float = 90.0  # the whole error is clipped here

    def __post_init__(self) -> None:
        for name in ("offset_deg", "drift_rate_deg_s", "drift_start_s", "step_deg"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("heading error prob must lie in [0, 1]")
        for name in ("drift_max_deg", "step_rate_hz", "step_tau_s", "max_deg"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"heading error {name} must be >= 0")
        if self.max_deg > 180:
            raise ValueError("heading error max_deg must be <= 180")

    @property
    def active(self) -> bool:
        return self.prob > 0 and (self.offset_deg[1] > 0
                                  or (self.drift_rate_deg_s[1] > 0 and self.drift_max_deg > 0)
                                  or (self.step_rate_hz > 0 and self.step_deg[1] > 0))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> HeadingErrorConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown heading error keys: {sorted(unknown)}")
        return cls(**d)


class HeadingError:
    """Batched heading error on NumPy arrays (float64); ``delta`` is (num_envs,) radians."""

    xp = np

    def __init__(self, cfg: HeadingErrorConfig, num_envs: int, dt: float, seed: int):
        if not dt > 0:
            raise ValueError("dt must be positive")
        self.cfg, self.num_envs, self.dt = cfg, int(num_envs), float(dt)
        self._init_rng(seed)
        self.p_step = 1.0 - math.exp(-cfg.step_rate_hz * self.dt)
        self.alpha = 1.0 if cfg.step_tau_s <= 0 else 1.0 - math.exp(-self.dt / cfg.step_tau_s)
        self.max_rad = math.radians(cfg.max_deg)
        n = self.num_envs
        self.on = self._zeros((n,)) > 0
        self.offset = self._zeros((n,))
        self.drift_rate = self._zeros((n,))  # rad/s, signed
        self.drift_start = self._zeros((n,))
        self.target = self._zeros((n,))  # the steps' sum, where the compass points
        self.lagged = self._zeros((n,))  # where EKF2's heading has got to
        self.t = self._zeros((n,))
        self.delta = self._zeros((n,))

    # Backend hooks: the only methods the PyTorch adapter overrides.

    def _init_rng(self, seed: int) -> None:
        self._rng = np.random.default_rng([int(seed), 0x4EAD])

    def _zeros(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)

    def _rand(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.random(shape)

    def _as_mask(self, env_ids: Any) -> np.ndarray:
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[np.asarray(env_ids)] = True
        return mask

    # Shared logic.

    def reset(self, env_ids: Any = None) -> None:
        """New episodes for ``env_ids`` (default all): fresh draws, error at its engage value."""
        xp, cfg, n = self.xp, self.cfg, self.num_envs
        mask = self._as_mask(range(n) if env_ids is None else env_ids)
        u = self._rand((n, 6))  # every env draws, so a reset's draws do not depend on the ids
        sign = lambda x: xp.where(x < 0.5, -1.0, 1.0)  # noqa: E731
        span = lambda lo_hi, x: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * x  # noqa: E731
        on = u[:, 0] < cfg.prob
        rad = math.pi / 180.0
        offset = on * sign(u[:, 2]) * span(cfg.offset_deg, u[:, 1]) * rad
        drift = on * sign(u[:, 4]) * span(cfg.drift_rate_deg_s, u[:, 3]) * rad
        self.on = xp.where(mask, on, self.on)
        self.offset = xp.where(mask, offset, self.offset)
        self.drift_rate = xp.where(mask, drift, self.drift_rate)
        self.drift_start = xp.where(mask, span(cfg.drift_start_s, u[:, 5]), self.drift_start)
        for name in ("target", "lagged", "t"):
            setattr(self, name, xp.where(mask, 0.0, getattr(self, name)))
        self.delta = xp.where(mask, xp.clip(self.offset, -self.max_rad, self.max_rad),
                              self.delta)

    def step(self) -> Any:
        """Advance one ``dt``; returns ``delta`` (num_envs,), rad."""
        xp, cfg = self.xp, self.cfg
        u = self._rand((self.num_envs, 3))
        self.t = self.t + self.dt
        event = (u[:, 0] < self.p_step) & self.on
        lo, hi = cfg.step_deg
        size = (lo + (hi - lo) * u[:, 1]) * xp.where(u[:, 2] < 0.5, -1.0, 1.0) * math.pi / 180.0
        self.target = xp.clip(self.target + xp.where(event, size, 0.0),
                              -self.max_rad, self.max_rad)
        self.lagged = self.lagged + self.alpha * (self.target - self.lagged)
        cap = math.radians(cfg.drift_max_deg)
        drift = xp.clip(self.drift_rate * xp.clip(self.t - self.drift_start, 0.0, None),
                        -cap, cap)
        self.delta = xp.clip(self.offset + self.lagged + drift, -self.max_rad, self.max_rad)
        return self.delta
