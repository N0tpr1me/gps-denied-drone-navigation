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
- optionally a constant accelerometer error per vehicle, ``imu_bias_mps2`` (the standard
  deviation per axis, drawn at each reset), added while the fix is stale. It stands in for
  what EKF2's bias and attitude states leave unestimated (a 0.5 deg tilt error is 0.09 m/s^2
  sideways): while the fix is fused EKF2 absorbs it, so it only shows in a dropout, as a
  drift of half the error times the dropout's length squared. Off by default: without it the
  filter dead-reckons on the true acceleration and a dropout costs almost nothing;
- EKF2's output predictor: the current state is the IMU propagated forward from the delayed
  horizon, pulled toward it with time constant ``output_tau_s`` (``EKF2_TAU_POS`` /
  ``EKF2_TAU_VEL``).

With ``resets`` on, EKF2 v1.17's vision resets (``ev_pos_control.cpp``, ``ev_vel_control.cpp``,
``ev_height_control.cpp``, ``ev_control.cpp``; vision is the only horizontal aiding and the
height reference), which put an estimate jump in front of the module as a step:

- fixes stop for more than ``stop_after_s`` (2 x ``EV_MAX_INTERVAL``): fusion stops, and the
  first fix after it resets velocity, position and height to the fix;
- horizontal position fixes rejected for ``hpos_reset_after_s`` (``no_aid_timeout_max``,
  1 s): horizontal position reset to the fix; the same for velocity;
- height fixes rejected for ``hgt_reset_after_s`` (``hgt_fusion_timeout_max``, 5 s): height
  fusion stops and restarts on the next fix with a height reset.

A reset moves the delayed state to the fix and shifts the output predictor by the same
amount, so the module sees the whole step at once. Horizontal fixes are fused or rejected as
a pair, as in EKF2.

Height reference (``height_source``): ``"vision"`` is the above, the set-up every policy up
to v22 trained on (``EKF2_HGT_REF`` 3). ``"baro"`` is the benchmark since 2026-11-01
(``EKF2_HGT_REF`` 0, ``docs/decisions.md``): the barometer is the height reference and the
vision height only aids it through EKF2's bias estimator, so the fused height follows the
barometer's own error (a random walk of ``baro_walk_m_per_sqrt_s``, capped at
``baro_walk_max_m``, plus white noise ``baro_noise_m`` per sample at ``baro_rate_hz``, fused
behind EKF2's own loose barometer gate), keeps being aided while the vision fix is stale
(no frozen height, no height resets, no IMU-bias drift in z), and restarts after a dropout
with the horizontal position and the velocity reset to the fix but the height left alone.
The vision height is then not fused at all: the bias estimator absorbs it. ``step`` needs
the true position for the barometer in this mode.

Pressure that is not height (barometer reference only; ``docs/environment-factors.md``): a
barometer reads 12 Pa as one metre, and indoors the air pressure moves by more than that
when a door or a shutter opens, the ventilation starts, or a train pushes air down a tunnel.
``baro_step_rate_hz`` draws such events: each moves the reading by ``baro_step_m`` (either
way), reached with the time constant ``baro_step_tau_s``, and with ``baro_step_decay_s`` set
it leaks back to nothing (a door that shuts again, a train that has passed). ``step`` also
takes ``baro_offset``, metres added to the reading from outside: the env passes the wind on
the static port (``envs/ambient``). The fused height follows all of it, as it follows the
barometer's own walk, because the barometer is the reference. Off by default.

Not modelled: attitude and bias states, EKF2's limit of five horizontal resets per fusion
start (after it, EKF2 stops and restarts the fusion, which resets anyway). Off by default;
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
    imu_bias_mps2: float = 0.0  # constant per-vehicle accelerometer error, std per axis
    output_tau_s: float = 0.25  # EKF2_TAU_POS, EKF2_TAU_VEL
    resets: bool = False  # EKF2's resets to the fix (module docstring)
    stop_after_s: float = 0.4  # 2 x EV_MAX_INTERVAL
    hpos_reset_after_s: float = 1.0  # no_aid_timeout_max
    hgt_reset_after_s: float = 5.0  # hgt_fusion_timeout_max
    # Height reference: "vision" (EKF2_HGT_REF 3, every policy up to v22) or "baro" (0, the
    # benchmark since 2026-11-01; module docstring).
    height_source: str = "vision"
    # The barometer as EKF2's height comes out (its raw noise is smoothed away by the vision
    # height aiding it): measured on Gazebo x500 hovers on px4_tuned_v2.params, 2026-11-03,
    # EKF2 height minus truth over a 60 s hold: std 0.006 m calm, 0.019 m in 4 m/s wind,
    # a walk of 0.002 to 0.007 m/sqrt(s), 0.001 m left after a 1 s mean, no reaction to a
    # 0.1 m vision height jump or to a 17 s vision dropout.
    baro_rate_hz: float = 50.0
    baro_noise_m: float = 0.02  # white noise per fused sample
    baro_walk_m_per_sqrt_s: float = 0.005  # the barometer's slow error (pressure, wind)
    baro_walk_max_m: float = 0.3
    # Pressure events the barometer reads as height (module docstring). A 10 Pa step, a door
    # onto a windy street, is 0.8 m.
    baro_step_rate_hz: float = 0.0
    baro_step_m: tuple[float, float] = (0.0, 0.0)  # magnitude, random sign
    baro_step_tau_s: float = 1.0
    baro_step_decay_s: float | None = None  # None: the step stays

    def __post_init__(self) -> None:
        lo, hi = ((float(self.baro_step_m),) * 2 if isinstance(self.baro_step_m, (int, float))
                  else (float(self.baro_step_m[0]), float(self.baro_step_m[1])))
        object.__setattr__(self, "baro_step_m", (lo, hi))
        if not 0 <= lo <= hi or self.baro_step_rate_hz < 0 or self.baro_step_tau_s <= 0:
            raise ValueError("fusion baro_step needs 0 <= min <= max metres, rate >= 0, tau > 0")
        if self.baro_step_decay_s is not None and self.baro_step_decay_s <= 0:
            raise ValueError("fusion baro_step_decay_s must be positive or None")
        for name in ("vision_rate_hz", "pos_noise_m", "vel_noise_mps", "gate_sd",
                     "output_tau_s", "stop_after_s", "hpos_reset_after_s", "hgt_reset_after_s",
                     "baro_rate_hz", "baro_noise_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"fusion {name} must be positive")
        for name in ("ev_delay_s", "accel_noise_mps2", "imu_noise_mps2", "imu_bias_mps2",
                     "baro_walk_m_per_sqrt_s", "baro_walk_max_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"fusion {name} must be >= 0")
        if self.height_source not in ("vision", "baro"):
            raise ValueError('fusion height_source must be "vision" or "baro"')

    @property
    def baro(self) -> bool:
        return self.height_source == "baro"

    @property
    def baro_steps(self) -> bool:
        return self.baro and self.baro_step_rate_hz > 0 and self.baro_step_m[1] > 0

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
        # Resets: time without a fix, since the last fused horizontal position, height and
        # horizontal velocity, and whether fusion has stopped for want of fixes.
        self.no_fix_s = z(n)
        self.since_fuse = z(n, 3)  # [horizontal position, height, horizontal velocity]
        self.stopped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.resets = 0  # count, for tests and diagnostics
        # Per env, as PX4's vehicle_local_position carries them: how many horizontal position
        # resets so far (xy_reset_counter) and the step of the last one (delta_xy).
        self.xy_reset_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.xy_reset_step = z(n, 3)
        self.imu_bias = z(n, 3)
        # Barometer height reference: the barometer's slow error, its own tick, and the true
        # height at the delayed horizon it is fused at (buffered like the IMU).
        self.baro_err = z(n)
        self.baro_clock = 0.0
        # Pressure events (cfg.baro_step_*): where the reading is heading, and where it is.
        # Their own generator, so turning them on leaves every other draw as it was.
        self.step_gen = torch.Generator(device=self.device).manual_seed(
            int(seed) * 104729 + 0xBA5)
        self.baro_step_target, self.baro_step = z(n), z(n)
        self.true_z = z(k, n)
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
        self.no_fix_s[ids] = 0.0
        self.since_fuse[ids] = 0.0
        self.stopped[ids] = False
        self.baro_err[ids] = 0.0  # EKF2's height origin is the barometer's level at the start
        self.baro_step_target[ids] = 0.0
        self.baro_step[ids] = 0.0
        self.true_z[:, ids] = pos[:, 2]
        if self.cfg.imu_bias_mps2 > 0:  # no draw when off, so other draws stay as they were
            self.imu_bias[ids] = self.cfg.imu_bias_mps2 * torch.randn(
                pos.shape[0], 3, generator=self.gen, device=self.device, dtype=self.dtype)

    def _pressure_events(self) -> None:
        """One step of the pressure events: new ones arrive, old ones leak away, and the
        reading follows with its lag."""
        cfg, dt = self.cfg, self.dt
        u = torch.rand(self.n, 3, generator=self.step_gen, device=self.device, dtype=self.dtype)
        lo, hi = cfg.baro_step_m
        size = (lo + (hi - lo) * u[:, 1]) * torch.where(u[:, 2] < 0.5, -1.0, 1.0)
        if cfg.baro_step_decay_s is not None:
            self.baro_step_target = self.baro_step_target * (1.0 - dt / max(
                cfg.baro_step_decay_s, dt))
        self.baro_step_target = self.baro_step_target + torch.where(
            u[:, 0] < cfg.baro_step_rate_hz * dt, size, torch.zeros_like(size))
        self.baro_step = self.baro_step + min(1.0, dt / cfg.baro_step_tau_s) * (
            self.baro_step_target - self.baro_step)

    def _fuse(self, y: torch.Tensor, var: float, h_pos: bool, ok: torch.Tensor,
              gate_var: float | None = None) -> torch.Tensor:
        """One scalar-per-axis measurement of position (``h_pos``) or velocity; returns which
        axes were fused. With ``resets`` the horizontal pair passes or fails the gate together.
        ``gate_var`` is the measurement variance the gate assumes when it is not ``var``."""
        P = self.P11 if h_pos else self.P22
        S = P + var
        ok = ok & (y * y < self.cfg.gate_sd ** 2 * (P + (var if gate_var is None else gate_var)))
        if self.cfg.resets:
            both = ok[:, 0] & ok[:, 1]
            ok = torch.stack([both, both, ok[:, 2]], dim=1)
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
        return ok

    def _reset_to(self, mask: torch.Tensor, fix_pos: torch.Tensor | None,
                  fix_vel: torch.Tensor | None) -> None:
        """Reset the masked (N, 3) components of the delayed state to the fix, and shift the
        output predictor by the same step."""
        cfg = self.cfg
        if fix_pos is not None:
            step = torch.where(mask, fix_pos - self.p_d, torch.zeros_like(self.p_d))
            self.p_d = self.p_d + step
            self.p_o, self.hist_p = self.p_o + step, self.hist_p + step
            xy = mask[:, 0] | mask[:, 1]
            self.xy_reset_count += xy.long()
            self.xy_reset_step = torch.where(xy[:, None], step, self.xy_reset_step)
            self.P11 = torch.where(mask, torch.full_like(self.P11, cfg.pos_noise_m ** 2), self.P11)
            self.P12 = torch.where(mask, torch.zeros_like(self.P12), self.P12)
        if fix_vel is not None:
            step = torch.where(mask, fix_vel - self.v_d, torch.zeros_like(self.v_d))
            self.v_d = self.v_d + step
            self.v_o, self.hist_v = self.v_o + step, self.hist_v + step
            var = torch.full_like(self.P22, cfg.vel_noise_mps ** 2)
            self.P22 = torch.where(mask, var, self.P22)
            self.P12 = torch.where(mask, torch.zeros_like(self.P12), self.P12)
        self.resets += int(mask.any(dim=1).sum())

    def step(self, accel: torch.Tensor, fix_pos: torch.Tensor, fix_vel: torch.Tensor,
             fresh: torch.Tensor, true_pos: torch.Tensor | None = None,
             baro_offset: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one step with the true acceleration ``accel`` (N, 3) and the latest fix;
        ``fresh`` (N,) is False while the fix is stale. ``true_pos`` (N, 3) is what the
        barometer measures the height of, needed with ``height_source`` "baro";
        ``baro_offset`` (N,) is metres added to its reading from outside (the wind on the
        static port). Returns (position, velocity) now."""
        cfg, dt = self.cfg, self.dt
        if cfg.baro and true_pos is None:
            raise ValueError('fusion height_source "baro" needs true_pos')
        accel = torch.nan_to_num(accel) + cfg.imu_noise_mps2 * torch.randn(
            self.n, 3, generator=self.gen, device=self.device, dtype=self.dtype)
        bias = self.imu_bias * (~fresh)[:, None]  # EKF2 absorbs it while aided
        if cfg.baro:
            bias = bias * torch.tensor([1.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
        accel = accel + bias
        self.acc = torch.cat([self.acc[1:], accel[None]], 0)
        a_d = self.acc[0]  # the IMU sample at the delayed horizon
        self.p_d = self.p_d + self.v_d * dt + 0.5 * a_d * dt * dt
        self.v_d = self.v_d + a_d * dt
        q = cfg.accel_noise_mps2 ** 2
        P11 = self.P11 + 2 * dt * self.P12 + dt * dt * self.P22 + q * dt ** 4 / 4
        P12 = self.P12 + dt * self.P22 + q * dt ** 3 / 2
        self.P11, self.P12, self.P22 = P11, P12, self.P22 + q * dt * dt
        self.clock += dt
        if cfg.resets:
            self.no_fix_s = torch.where(fresh, torch.zeros_like(self.no_fix_s), self.no_fix_s + dt)
            self.stopped = self.stopped | (self.no_fix_s > cfg.stop_after_s)
            self.since_fuse = self.since_fuse + dt
        if cfg.baro:
            self.true_z = torch.cat([self.true_z[1:], torch.nan_to_num(true_pos)[None, :, 2]], 0)
            self.baro_err = (self.baro_err + cfg.baro_walk_m_per_sqrt_s * dt ** 0.5 * torch.randn(
                self.n, generator=self.gen, device=self.device, dtype=self.dtype)
            ).clamp(-cfg.baro_walk_max_m, cfg.baro_walk_max_m)
            if cfg.baro_steps:
                self._pressure_events()
            self.baro_clock += dt
            if self.baro_clock >= 1.0 / cfg.baro_rate_hz - 1e-9:
                self.baro_clock -= 1.0 / cfg.baro_rate_hz
                z_meas = self.true_z[0] + self.baro_err + cfg.baro_noise_m * torch.randn(
                    self.n, generator=self.gen, device=self.device, dtype=self.dtype)
                if cfg.baro_steps:
                    z_meas = z_meas + self.baro_step
                if baro_offset is not None:
                    z_meas = z_meas + torch.nan_to_num(baro_offset)
                y = torch.zeros_like(self.p_d)
                y[:, 2] = z_meas - self.p_d[:, 2]
                only_z = torch.tensor([False, False, True], device=self.device).expand(self.n, 3)
                # Gated as EKF2 gates the barometer (EKF2_BARO_NOISE 3.5 m): its slow error
                # is never rejected, whatever the filter's own height variance has shrunk to.
                self._fuse(y, cfg.baro_noise_m ** 2, True, only_z, gate_var=3.5 ** 2)
        if self.clock >= self.period - 1e-9:
            self.clock -= self.period
            ok = fresh[:, None].expand(self.n, 3)
            if cfg.baro:  # the vision height only aids the barometer: nothing to fuse
                ok = ok & torch.tensor([True, True, False], device=self.device)
            if cfg.resets:
                # The first fix after fusion stopped restarts it with a full reset (the
                # height stays on the barometer when that is the reference).
                start = (self.stopped & fresh)[:, None].expand(self.n, 3)
                self._reset_to(start & ok, fix_pos, None)
                self._reset_to(start, None, fix_vel)
                self.since_fuse = torch.where(start[:, :1].expand(self.n, 3),
                                              torch.zeros_like(self.since_fuse), self.since_fuse)
                self.stopped = self.stopped & ~fresh
                ok = ok & ~start
            fused_p = self._fuse(fix_pos - self.p_d, cfg.pos_noise_m ** 2, True, ok)
            fused_v = self._fuse(fix_vel - self.v_d, cfg.vel_noise_mps ** 2, False, ok)
            if cfg.resets:
                fused = torch.stack([fused_p[:, 0], fused_p[:, 2], fused_v[:, 0]], dim=1)
                self.since_fuse = torch.where(fused, torch.zeros_like(self.since_fuse),
                                              self.since_fuse)
                # Rejected for too long while fixes keep coming: reset to the fix.
                live = fresh & ~self.stopped
                hpos = live & (self.since_fuse[:, 0] > cfg.hpos_reset_after_s)
                hgt = live & (self.since_fuse[:, 1] > cfg.hgt_reset_after_s)
                if cfg.baro:  # the height is the barometer's: no vision height resets
                    hgt = torch.zeros_like(hgt)
                hvel = live & (self.since_fuse[:, 2] > cfg.hpos_reset_after_s)
                xy = torch.tensor([True, True, False], device=self.device)
                self._reset_to(hpos[:, None] & xy | hgt[:, None] & ~xy, fix_pos, None)
                self._reset_to(hvel[:, None].expand(self.n, 3), None, fix_vel)
                for col, m in enumerate((hpos, hgt, hvel)):
                    self.since_fuse[:, col] = torch.where(m, 0.0, self.since_fuse[:, col])
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
