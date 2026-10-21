"""Body-rate PID in the shape of PX4's ``RateControl`` (``MC_*RATE_P/I/D``, ``MC_*R_INT_LIM``).

The policy commands body rates, exactly as the onboard Structure Hold module will, so the
training env closes the same inner loop PX4 closes. PX4 works in normalised torque
(``-1..1`` of the airframe's authority); ``RatePid`` does the same and scales by
``max_torque`` on the way out. As in PX4:

- D acts on the measured rate (low-pass filtered, ``IMU_DGYRO_CUTOFF``), not on the error,
  so a step in the setpoint gives no derivative kick;
- the integrator freezes while the output is saturated in the same direction;
- the integral term is clipped to ``±int_lim``.

Gains default to PX4 v1.14+ multicopter defaults. Batched: every state is ``(N, 3)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass(frozen=True)
class RatePidGains:
    # Roll, pitch, yaw: MC_ROLLRATE_P, MC_PITCHRATE_P, MC_YAWRATE_P and so on.
    p: tuple[float, float, float] = (0.15, 0.15, 0.20)
    i: tuple[float, float, float] = (0.20, 0.20, 0.10)
    d: tuple[float, float, float] = (0.003, 0.003, 0.0)
    ff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    int_lim: tuple[float, float, float] = (0.30, 0.30, 0.30)
    d_cutoff_hz: float = 30.0  # IMU_DGYRO_CUTOFF; 0 disables the filter

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RatePidGains:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown rate PID gains: {sorted(unknown)}")
        for key in ("p", "i", "d", "ff", "int_lim"):
            if key in d:
                d[key] = tuple(float(x) for x in d[key])
        return cls(**d)


class RatePid:
    def __init__(
        self,
        gains: RatePidGains,
        max_torque: Any,
        num_envs: int,
        dt: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.gains = gains
        self.dt = float(dt)
        self.num_envs = num_envs
        dev = torch.device(device)
        t = lambda v: torch.tensor(v, dtype=dtype, device=dev)  # noqa: E731
        self.kp, self.ki, self.kd, self.kff = t(gains.p), t(gains.i), t(gains.d), t(gains.ff)
        self.int_lim = t(gains.int_lim)
        self.max_torque = torch.as_tensor(max_torque, dtype=dtype, device=dev)
        if gains.d_cutoff_hz > 0:
            self.alpha = 1.0 - math.exp(-2.0 * math.pi * gains.d_cutoff_hz * self.dt)
        else:
            self.alpha = 1.0
        self.integral = torch.zeros(num_envs, 3, dtype=dtype, device=dev)
        self.rate_filt = torch.zeros(num_envs, 3, dtype=dtype, device=dev)
        self.rate_prev = torch.zeros(num_envs, 3, dtype=dtype, device=dev)
        self.out_prev = torch.zeros(num_envs, 3, dtype=dtype, device=dev)

    def reset(self, env_ids: Any = None, rate: Any = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.integral[ids] = 0.0
        self.out_prev[ids] = 0.0
        if rate is None:
            self.rate_filt[ids] = 0.0
            self.rate_prev[ids] = 0.0
        else:
            r = torch.as_tensor(rate, dtype=self.rate_filt.dtype, device=self.rate_filt.device)
            self.rate_filt[ids] = r
            self.rate_prev[ids] = r

    def __call__(self, rate_sp: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """Body torque (N, 3) in N m for a rate setpoint and measured rate (rad/s, body)."""
        err = rate_sp - rate
        self.rate_filt = self.rate_filt + self.alpha * (rate - self.rate_filt)
        d_rate = (self.rate_filt - self.rate_prev) / self.dt
        self.rate_prev = self.rate_filt
        # Anti-windup as PX4: no integration pushing further into saturation.
        saturated = (self.out_prev >= 1.0) & (err > 0) | (self.out_prev <= -1.0) & (err < 0)
        step = torch.where(saturated, torch.zeros_like(err), self.ki * err * self.dt)
        self.integral = torch.maximum(torch.minimum(self.integral + step, self.int_lim),
                                      -self.int_lim)
        out = self.kp * err + self.integral - self.kd * d_rate + self.kff * rate_sp
        self.out_prev = out.clamp(-1.0, 1.0)
        return self.out_prev * self.max_torque
