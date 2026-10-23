"""An EKF2-like fusion of the corrupted position fix with the IMU, as PX4 runs it onboard.

On the vehicle the module never sees the vision fix itself: PX4's EKF2 fuses it with the
accelerometer and the module reads ``vehicle_local_position``. This is that path, cut down
to what shapes the module's input, per axis and batched on tensors:

- a ``[position, velocity]`` Kalman filter running at the *delayed horizon*: it is
  propagated with the acceleration of ``ev_delay_s`` ago (EKF2 buffers the IMU) and fuses a
  fix the moment it arrives as if it were that old (``EKF2_EV_DELAY``: a constant, so a fix
  whose real latency differs is fused at the wrong time, as in Gazebo);
- fixes at ``vision_rate_hz``, position and velocity fused in turn with noise
  ``pos_noise_m`` / ``vel_noise_mps`` (``EKF2_EVP_NOISE`` / ``EKF2_EVV_NOISE``) behind an
  innovation gate of ``gate_sd`` standard deviations (``EKF2_EV_GATE``); none while the fix
  is stale (dropouts), so the filter dead-reckons on the IMU;
- process noise ``accel_noise_mps2`` (``EKF2_ACC_NOISE``) and accelerometer noise
  ``imu_noise_mps2`` on the true acceleration;
- EKF2's output predictor: the current state is the IMU propagated forward from the delayed
  horizon, pulled toward it with time constant ``output_tau_s`` (``EKF2_TAU_POS`` /
  ``EKF2_TAU_VEL``).

Not modelled: attitude and bias states, height and position resets (the gate and the
covariance growth during a dropout let a returning fix back in instead). Off by default;
the ``gz_x500_sitl`` airframe turns it on (docs/decisions.md 2026-10-23).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = False
    vision_rate_hz: float = 50.0  # the bridge sends 50 Hz
    ev_delay_s: float = 0.1  # EKF2_EV_DELAY
    pos_noise_m: float = 0.1  # EKF2_EVP_NOISE
    vel_noise_mps: float = 0.1  # EKF2_EVV_NOISE
    gate_sd: float = 5.0  # EKF2_EV_GATE
    accel_noise_mps2: float = 0.35  # EKF2_ACC_NOISE
    imu_noise_mps2: float = 0.05  # white noise on the true acceleration
    output_tau_s: float = 0.25  # EKF2_TAU_POS, EKF2_TAU_VEL

    def __post_init__(self) -> None:
        for name in ("vision_rate_hz", "pos_noise_m", "vel_noise_mps", "gate_sd",
                     "output_tau_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"fusion {name} must be positive")
        for name in ("ev_delay_s", "accel_noise_mps2", "imu_noise_mps2"):
            if getattr(self, name) < 0:
                raise ValueError(f"fusion {name} must be >= 0")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FusionConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown fusion keys: {sorted(unknown)}")
        return cls(**d)


class TorchFusion:
    """Batched fusion for ``num_envs`` vehicles stepped every ``dt`` seconds."""

    def __init__(self, cfg: FusionConfig, num_envs: int, dt: float, seed: int,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.cfg, self.n, self.dt = cfg, int(num_envs), float(dt)
        self.device, self.dtype = torch.device(device), dtype
        self.delay_steps = max(1, round(cfg.ev_delay_s / self.dt))
        self.period = 1.0 / cfg.vision_rate_hz
        self.clock = 0.0  # time since the last fix was due (one schedule for all envs)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed) * 104729 + 0xEF2)
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=self.device)  # noqa: E731
        n, k = self.n, self.delay_steps + 1
        self.p_d, self.v_d = z(n, 3), z(n, 3)  # delayed-horizon state
        self.P11, self.P12, self.P22 = z(n, 3), z(n, 3), z(n, 3)
        self.p_o, self.v_o = z(n, 3), z(n, 3)  # output predictor (now)
        self.hist_p, self.hist_v = z(k, n, 3), z(k, n, 3)  # output states, oldest first
        self.acc = z(k, n, 3)  # IMU samples, oldest first
        self.reset(torch.arange(n, device=self.device), z(n, 3), z(n, 3))

    def reset(self, env_ids: Any, pos: torch.Tensor, vel: torch.Tensor) -> None:
        """Start ``env_ids`` at ``pos``, ``vel`` (rows for those envs) with a settled filter."""
        ids = env_ids
        self.p_d[ids], self.v_d[ids] = pos, vel
        self.p_o[ids], self.v_o[ids] = pos, vel
        self.P11[ids], self.P12[ids], self.P22[ids] = 0.01, 0.0, 0.01
        self.hist_p[:, ids] = pos
        self.hist_v[:, ids] = vel
        self.acc[:, ids] = 0.0

    def _fuse(self, y: torch.Tensor, var: float, h_pos: bool, ok: torch.Tensor) -> None:
        """One scalar-per-axis measurement of position (``h_pos``) or velocity."""
        S = (self.P11 if h_pos else self.P22) + var
        ok = ok & (y * y < self.cfg.gate_sd ** 2 * S)
        if h_pos:
            K1, K2 = self.P11 / S, self.P12 / S
            P11, P12, P22 = (1 - K1) * self.P11, (1 - K1) * self.P12, self.P22 - K2 * self.P12
        else:
            K1, K2 = self.P12 / S, self.P22 / S
            P11, P12, P22 = self.P11 - K1 * self.P12, (1 - K2) * self.P12, (1 - K2) * self.P22
        self.p_d = torch.where(ok, self.p_d + K1 * y, self.p_d)
        self.v_d = torch.where(ok, self.v_d + K2 * y, self.v_d)
        self.P11 = torch.where(ok, P11, self.P11)
        self.P12 = torch.where(ok, P12, self.P12)
        self.P22 = torch.where(ok, P22, self.P22)

    def step(self, accel: torch.Tensor, fix_pos: torch.Tensor, fix_vel: torch.Tensor,
             fresh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one step with the true acceleration ``accel`` (N, 3) and the latest fix;
        ``fresh`` (N,) is False while the fix is stale. Returns (position, velocity) now."""
        cfg, dt = self.cfg, self.dt
        accel = torch.nan_to_num(accel) + cfg.imu_noise_mps2 * torch.randn(
            self.n, 3, generator=self.gen, device=self.device, dtype=self.dtype)
        self.acc = torch.cat([self.acc[1:], accel[None]], 0)
        a_d = self.acc[0]  # the IMU sample at the delayed horizon
        self.p_d = self.p_d + self.v_d * dt + 0.5 * a_d * dt * dt
        self.v_d = self.v_d + a_d * dt
        q = cfg.accel_noise_mps2 ** 2
        P11 = self.P11 + 2 * dt * self.P12 + dt * dt * self.P22 + q * dt ** 4 / 4
        P12 = self.P12 + dt * self.P22 + q * dt ** 3 / 2
        self.P11, self.P12, self.P22 = P11, P12, self.P22 + q * dt * dt
        self.clock += dt
        if self.clock >= self.period - 1e-9:
            self.clock -= self.period
            ok = fresh[:, None].expand(self.n, 3)
            self._fuse(fix_pos - self.p_d, cfg.pos_noise_m ** 2, True, ok)
            self._fuse(fix_vel - self.v_d, cfg.vel_noise_mps ** 2, False, ok)
        # Output predictor: propagate with the current IMU sample, then pull the whole
        # buffer toward the delayed-horizon state.
        self.v_o = self.v_o + accel * dt
        self.p_o = self.p_o + self.v_o * dt
        self.hist_p = torch.cat([self.hist_p[1:], self.p_o[None]], 0)
        self.hist_v = torch.cat([self.hist_v[1:], self.v_o[None]], 0)
        g = dt / cfg.output_tau_s
        e_p, e_v = (self.p_d - self.hist_p[0]) * g, (self.v_d - self.hist_v[0]) * g
        self.p_o, self.v_o = self.p_o + e_p, self.v_o + e_v
        self.hist_p, self.hist_v = self.hist_p + e_p, self.hist_v + e_v
        return self.p_o, self.v_o
