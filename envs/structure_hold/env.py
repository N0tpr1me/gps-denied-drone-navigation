"""Structure Hold: hold a quadrotor 1 to 3 m from a wall, in wind, on a corrupted estimate.

Composes, in order every step: the policy's action (body rates and thrust) -> the
PX4-matched rate PID (``envs/sims/rate_pid.py``) -> the simulator behind the seam
(``envs/sims``) with the wind force from ``envs/disturbance`` -> the estimate corruption
from ``envs/estimate`` -> observation and reward. Scoring with ``eval/metrics.py`` is done
by the caller from the true positions the env exposes (see ``run.py``); the env itself
only rewards and terminates.

VecEnv shape (what rsl_rl's runner expects): ``num_envs``, ``num_obs``, ``num_actions``,
``max_episode_length``, ``device``; ``reset() -> (obs, extras)``;
``get_observations() -> (obs, extras)``; ``step(actions) -> (obs, reward, dones, extras)``
with ``extras["time_outs"]`` marking episodes ended by the clock, and envs that finished
reset inside ``step`` so the returned observation is already the new episode's first.

Frames: world ENU, the wall is the plane x = 0 and the drone flies at x < 0, so the
structure frame (x toward the surface, z up) coincides with the world frame. Observations are
written in that frame so a later wall placement changes one rotation, not the policy.

Observation (``num_obs`` = 23), everything the onboard module can know:

| slice | what | units |
| --- | --- | --- |
| 0:3 | estimated position minus hold point, structure frame, clipped | m |
| 3:6 | estimated velocity, structure frame | m/s |
| 6:9 | gravity direction in the body frame (attitude) | unit |
| 9:11 | body x axis (heading) in the structure frame, x and y components | unit |
| 11:14 | body rates / max_rate | - |
| 14:18 | previous action | - |
| 18 | estimate stale flag (1 while the estimator has dropped out) | - |
| 19 | estimate age, capped at 1 s | s |
| 20 | time since engage (episode start or re-engage) / ``engage_obs_cap_s``, capped at 1 | - |
| 21 | wall range error: the hold point's wall distance minus the measured one, clipped | m |
| 22 | wall range usable (1 while a fresh forward range reading counts, else 0) | - |

Policies up to v21 read the first 20 (``NUM_OBS_V1``); ``policy.json`` says how many a
policy reads, and ``training/policy_io.py`` feeds it that many. The engage time is there
because an offset at engage is real while one later is as likely an estimate jump, and the
policy cannot tell them apart otherwise (``docs/hold-offset.md``).

Action (``num_actions`` = 4), each in [-1, 1]: body rate setpoints (roll, pitch, yaw) as a
fraction of ``max_rate_rad_s``, and thrust mapped linearly onto ``[0, max_thrust]`` newtons
of ``cfg.quad``. Those newtons reach the airframe as the PX4 module sends them: through the
hover point (``thrust_n / (mass * g) * hover_thrust``), then, when ``cfg.actuation`` says so,
the module's clamp and the vehicle's thrust curve (``envs/structure_hold/airframes.py``).

Randomness: one NumPy generator (per-episode draws: hold point, start pose, wind params),
one torch generator (estimate corruption) and a second NumPy generator for the airframe
(``cfg.airframe_rand`` and ``cfg.wall_yaw_torque``, both off by default), all from the
constructor's ``seed``. The airframe draws having their own generator means a stressed
airframe flies exactly the episodes the nominal one does.

``term_reason`` holds, per env, why its last episode ended (``TERM_REASONS``), so a scorer can
tell crashes apart after the env has already reset.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from envs.disturbance import sample_params
from envs.disturbance import torch_adapter as wind
from envs.estimate.ekf import TorchFusion
from envs.estimate.torch_adapter import TorchEstimateCorruptor
from envs.sims import QuadState, RatePid, make_sim
from envs.sims.torch_quad import GRAVITY, quat_from_yaw, quat_rotate, quat_rotate_inverse
from envs.structure_hold.config import StructureHoldConfig

OBS = {
    "pos_err": slice(0, 3),
    "vel": slice(3, 6),
    "gravity_body": slice(6, 9),
    "heading": slice(9, 11),
    "ang_vel": slice(11, 14),
    "last_action": slice(14, 18),
    "stale": slice(18, 19),
    "age": slice(19, 20),
    "engage_time": slice(20, 21),
    "wall_range": slice(21, 22),
    "range_ok": slice(22, 23),
}
NUM_OBS = 23
NUM_OBS_V1 = 20  # what v0 to v21 read
NUM_OBS_V22 = 21  # v22: plus the engage time
NUM_ACTIONS = 4

TERM_REASONS = ("none", "contact", "ground", "tilt", "box", "nonfinite", "timeout")

# The wall: a point on it and the normal pointing from the drone toward it (structure x).
WALL_POINT = (0.0, 0.0, 0.0)
WALL_NORMAL_TOWARD = (1.0, 0.0, 0.0)


class StructureHoldEnv:
    def __init__(
        self,
        cfg: StructureHoldConfig,
        num_envs: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
        sim: str = "torch",
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.num_obs = NUM_OBS
        self.num_privileged_obs = None
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = cfg.max_episode_length
        self.dt = cfg.dt
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.airframe_rng = np.random.default_rng([int(seed), 0xA1F])
        self.noise_gen = torch.Generator(device=device).manual_seed(int(seed) * 7919 + 0x5E5)

        n, dev, dt = self.num_envs, self.device, self.dtype
        self.sim = make_sim(sim, cfg.quad, n, cfg.dt, device=device)
        # PX4's unit torque on this airframe, and where the airframe saturates before it.
        act = cfg.actuation
        per_unit = act.torque_per_unit or cfg.quad.max_torque
        sat = (None if act.torque_per_unit is None else
               tuple(min(1.0, m / u) for m, u in zip(cfg.quad.max_torque, per_unit,
                                                             strict=True)))
        self.rate_pid = RatePid(cfg.rate_pid, per_unit, n, cfg.dt, dev, dt, sat_limit=sat)
        self.estimator = TorchEstimateCorruptor(cfg.estimate, n, cfg.dt, seed, dev, dt)
        # EKF2-like fusion of the corrupted fix with the IMU (cfg.fusion, off by default).
        self.fusion = (TorchFusion(cfg.fusion, n, cfg.dt, seed, dev, dt) if cfg.fusion.enabled
                       else None)
        self.vision_offset = torch.tensor([0.0, 0.0, -cfg.vision_offset_down_m], dtype=dt,
                                          device=dev)
        self.wall_normal = torch.tensor(WALL_NORMAL_TOWARD, dtype=dt, device=dev)
        self.wall_point = torch.tensor(WALL_POINT, dtype=dt, device=dev)
        self.cos_max_tilt = math.cos(math.radians(cfg.max_tilt_deg))
        self.cos_tilt_free = math.cos(math.radians(cfg.reward.tilt_free_deg))

        self.hold_point = torch.zeros(n, 3, dtype=dt, device=dev)
        self.episode_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.episode_time = torch.zeros(n, dtype=dt, device=dev)
        self.engage_time = torch.zeros(n, dtype=dt, device=dev)
        self.last_action = torch.zeros(n, NUM_ACTIONS, dtype=dt, device=dev)
        # Rate commands through first-order filters at reward.action_band_hi_hz and lo_hz:
        # their difference is the band the action_band term charges.
        self.action_fast = torch.zeros(n, 3, dtype=dt, device=dev)
        self.action_slow = torch.zeros(n, 3, dtype=dt, device=dev)
        self.band_alpha = [cfg.dt / (cfg.dt + 1.0 / (2.0 * math.pi * f))
                           for f in (cfg.reward.action_band_hi_hz, cfg.reward.action_band_lo_hz)]
        self.rate_sp =torch.zeros(n, 3, dtype=dt, device=dev)  # lagged rate setpoint
        self.rate_lag_alpha = torch.ones(n, dtype=dt, device=dev)  # dt / tau, 1 = no lag
        self.obs_buf = torch.zeros(n, NUM_OBS, dtype=dt, device=dev)
        self.reward_buf = torch.zeros(n, dtype=dt, device=dev)
        self.done_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.episode_reward = torch.zeros(n, dtype=dt, device=dev)
        self.estimate_pos = torch.zeros(n, 3, dtype=dt, device=dev)
        self.estimate_vel = torch.zeros(n, 3, dtype=dt, device=dev)
        self.estimate_stale = torch.zeros(n, dtype=torch.bool, device=dev)
        self.estimate_age = torch.zeros(n, dtype=dt, device=dev)
        self.wind_force = torch.zeros(n, 3, dtype=dt, device=dev)
        self.wind = None  # DisturbanceParams on tensors, set by the first reset
        # Per-episode airframe (cfg.airframe_rand): thrust per unit action relative to the
        # nominal airframe (mass scale x hover-thrust error), and the drawn scales for logs.
        self.thrust_gain = torch.ones(n, dtype=dt, device=dev)
        # With cfg.actuation: the normalised command per policy newton (the module's hover
        # mapping on this env's airframe) and the airframe's max thrust in newtons.
        self.thrust_via_module = act.thrust_via_module
        self.thrust_cmd_per_n = torch.full((n,), self._cmd_per_newton(1.0, 1.0, 0.0),
                                           dtype=dt, device=dev)
        self.max_thrust_act = torch.full((n,), cfg.quad.max_thrust, dtype=dt, device=dev)
        self.mass_scale = torch.ones(n, dtype=dt, device=dev)
        self.inertia_scale = torch.ones(n, dtype=dt, device=dev)
        self.yaw_torque_peak = torch.zeros(n, dtype=dt, device=dev)  # signed, N m
        self.yaw_torque = torch.zeros(n, dtype=dt, device=dev)  # applied this step, N m
        # Per-episode sensor-noise stds (cfg.sensor_noise; zero while it is off) and the
        # velocity wander state.
        self.noise_std = torch.zeros(n, 3, dtype=dt, device=dev)  # rates, gravity, vel wander
        self.vel_wander = torch.zeros(n, 3, dtype=dt, device=dev)
        self.term_reason = torch.zeros(n, dtype=torch.long, device=dev)
        # The true position after this step's physics, kept for scorers: step() resets the
        # finished episodes before it returns, so state.pos is then the new episode's.
        self.step_pos = torch.zeros(n, 3, dtype=dt, device=dev)
        self.counts = {"contact": 0, "ground": 0, "tilt": 0, "box": 0, "nonfinite": 0,
                       "timeout": 0}
        # Mid-episode re-engages (cfg.reengage_rate_hz): the episode's own hold point, which
        # they scatter about; which envs re-engaged on the last step; and their own generator,
        # so turning them on leaves every episode's other draws as they were.
        self.base_hold = torch.zeros(n, 3, dtype=dt, device=dev)
        self.reengaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self.reengage_count = 0
        self.reengage_rng = np.random.default_rng([int(seed), 0xE6A])
        # Forward rangefinder (cfg.range): per-episode presence and scale error, the newest
        # sample and its age, time to the next sample, dropout time left, and the hold point's
        # distance from the wall the input measures against. Its own generators, so turning
        # it on leaves every other draw as it was.
        self.range_rng = np.random.default_rng([int(seed), 0x7A6])
        self.range_gen = torch.Generator(device=device).manual_seed(int(seed) * 7919 + 0x7A6)
        self.range_present = torch.zeros(n, dtype=torch.bool, device=dev)
        self.range_scale = torch.zeros(n, dtype=dt, device=dev)
        self.range_meas = torch.full((n,), math.inf, dtype=dt, device=dev)
        self.range_age = torch.full((n,), math.inf, dtype=dt, device=dev)
        self.range_timer = torch.zeros(n, dtype=dt, device=dev)
        self.range_drop_left = torch.zeros(n, dtype=dt, device=dev)
        self.range_ref = torch.zeros(n, dtype=dt, device=dev)
        # Spike rejection (cfg.range.spike_gate_m): the last accepted perpendicular distance,
        # the last rejected one (NaN: none pending) and whether the newest reading was rejected.
        self.range_accepted = torch.full((n,), math.nan, dtype=dt, device=dev)
        self.range_pending = torch.full((n,), math.nan, dtype=dt, device=dev)
        self.range_rejected = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cos_range_off = math.cos(math.radians(cfg.range.max_off_axis_deg))
        self.reset()

    # ------------------------------------------------------------ state helpers

    @property
    def state(self) -> QuadState:
        """The true state (for scoring and tests), not what the policy sees."""
        return self.sim.state

    def clearance(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Distance from the wall along its normal, positive on the free-air side, (N,)."""
        p = self.sim.state.pos if pos is None else pos
        return (self.wall_point - p) @ self.wall_normal

    def _beam_cos(self) -> torch.Tensor:
        """Cosine between the body x axis (the rangefinder's beam) and the wall normal, (N,)."""
        fwd = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        return quat_rotate(self.sim.state.quat, fwd.expand(self.num_envs, 3)) @ self.wall_normal

    # ------------------------------------------------------------ curriculum hooks

    @property
    def dropout_max_s(self) -> float:
        """The longest estimate dropout the corruptor currently draws, in seconds."""
        return self.estimator.dropout_steps_max * self.dt

    def set_dropout_max_s(self, seconds: float) -> float:
        """Cap estimate dropouts at ``seconds`` from the next step on (a training curriculum
        hook; dropouts already running keep their length). The cap is rounded down to whole
        steps and never below ``cfg.estimate.dropout_min_s``. Returns the cap applied."""
        lo = self.estimator.dropout_steps_min
        hi = max(lo, math.floor(float(seconds) / self.dt + 1e-9))
        self.estimator.dropout_steps_max = hi
        return hi * self.dt

    def gust_windows(self, env_id: int) -> list[tuple[float, float]]:
        """(start, end) in episode seconds of each gust of that env's current episode."""
        assert self.wind is not None
        s = self.wind.gust_start[env_id].tolist()
        d = self.wind.gust_duration[env_id].tolist()
        return [(a, a + b) for a, b in zip(s, d, strict=True)]

    # ------------------------------------------------------------ resets

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_envs(torch.arange(self.num_envs, device=self.device))
        self._observe()
        return self.obs_buf, {"observations": {}}

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Start new episodes for ``env_ids``: new hold point, start pose, wind and
        estimator state; everything else untouched."""
        n = int(env_ids.numel())
        if n == 0:
            return
        cfg, rng = self.cfg, self.rng
        self._draw_airframe(env_ids)
        if cfg.sensor_noise.active:
            sn = cfg.sensor_noise
            top = torch.tensor([sn.ang_vel_std_rad_s, sn.gravity_std, sn.vel_wander_std_mps],
                               dtype=self.dtype, device=self.device)
            u = torch.rand(n, 3, generator=self.noise_gen, device=self.device, dtype=self.dtype)
            self.noise_std[env_ids] = u * top
            self.vel_wander[env_ids] = 0.0

        dist = rng.uniform(*cfg.hold_distance, size=n)
        height = rng.uniform(*cfg.hold_height, size=n)
        hold = np.stack([-dist, np.zeros(n), height], 1)
        start = hold + cfg.start_offset_std_m * rng.standard_normal((n, 3))
        # Never start inside the wall or the ground.
        start[:, 0] = np.minimum(start[:, 0], -(cfg.contact_radius_m + 0.2))
        start[:, 2] = np.maximum(start[:, 2], cfg.min_height_m + 0.2)
        yaw = cfg.start_yaw_std_rad * rng.standard_normal(n)

        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        state = QuadState(
            pos=t(start), quat=quat_from_yaw(t(yaw)),
            vel=torch.zeros(n, 3, dtype=self.dtype, device=self.device),
            ang_vel=torch.zeros(n, 3, dtype=self.dtype, device=self.device),
        )
        self.sim.reset(env_ids, state)
        self.hold_point[env_ids] = t(hold)
        self.base_hold[env_ids] = t(hold)
        self.rate_pid.reset(env_ids)
        self.estimator.reset(self.sim.state.pos, self.sim.state.vel, env_ids)
        if self.fusion is not None:
            self.fusion.reset(env_ids, self.sim.state.pos[env_ids], self.sim.state.vel[env_ids])

        # Wind: wall_normal here is the disturbance model's convention (into free air).
        new = wind.to_torch(
            sample_params(cfg.disturbance, n, rng, wall_point=WALL_POINT,
                          wall_normal=tuple(-x for x in WALL_NORMAL_TOWARD)),
            self.device, self.dtype,
        )
        if self.wind is None or n == self.num_envs:
            self.wind = new
        else:
            for name in new.__dataclass_fields__:
                getattr(self.wind, name)[env_ids] = getattr(new, name)

        self.episode_step[env_ids] = 0
        self.episode_time[env_ids] = 0.0
        self.engage_time[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.action_fast[env_ids] = 0.0
        self.action_slow[env_ids] = 0.0
        self.rate_sp[env_ids] = 0.0
        tau = rng.uniform(*cfg.rate_lag_s, size=n)
        self.rate_lag_alpha[env_ids] = t(np.where(tau > 0, cfg.dt / np.maximum(tau, cfg.dt), 1.0))
        self.episode_reward[env_ids] = 0.0
        self.estimate_pos[env_ids] = self.sim.state.pos[env_ids]
        self.estimate_vel[env_ids] = 0.0
        self.estimate_stale[env_ids] = False
        self.estimate_age[env_ids] = 0.0
        self.wind_force[env_ids] = 0.0
        self.range_ref[env_ids] = self.clearance(self.hold_point[env_ids])
        if cfg.range.active:
            self._reset_range(env_ids)

    def _reset_range(self, env_ids: torch.Tensor) -> None:
        """Per-episode rangefinder draws, and a first sample at the start pose: the sensor
        was running before the engage, so its newest sample is up to one period old."""
        rc, rng, n = self.cfg.range, self.range_rng, int(env_ids.numel())
        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        self.range_present[env_ids] = torch.as_tensor(rng.random(n) < rc.present_prob,
                                                      device=self.device)
        self.range_scale[env_ids] = t(rng.uniform(*rc.scale_error, size=n))
        period = 1.0 / rc.rate_hz
        age = rng.uniform(0.0, period, size=n)
        self.range_age[env_ids] = t(age)
        self.range_timer[env_ids] = t(period - age)
        self.range_drop_left[env_ids] = 0.0
        self.range_accepted[env_ids] = math.nan  # a new flight: the first reading is accepted
        self.range_pending[env_ids] = math.nan
        take = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        take[env_ids] = True
        self._take_range(take, self._range_reading())
        self.range_age[env_ids] = t(age)  # _take_range zeroed it: the sample is that old

    def _range_reading(self) -> torch.Tensor:
        """What the rangefinder would read now for every env, (N,): the beam's distance to
        the wall plane with this episode's scale error, noise and outliers."""
        rc = self.cfg.range
        cos = self._beam_cos()
        true = torch.where(cos > 1e-3, self.clearance() / cos.clamp(min=1e-3),
                           torch.full_like(cos, math.inf))
        true = torch.where(true > 0, true, torch.full_like(true, math.inf))
        n = self.num_envs
        noise = torch.randn(n, generator=self.range_gen, device=self.device, dtype=self.dtype)
        r = true * (1.0 + self.range_scale) + noise * (rc.noise_std_m + rc.noise_frac
                                                         * true.clamp(max=rc.max_m))
        if rc.outlier_prob > 0:
            u = torch.rand(n, 2, generator=self.range_gen, device=self.device, dtype=self.dtype)
            r = torch.where(u[:, 0] < rc.outlier_prob, rc.min_m + u[:, 1] * (rc.max_m - rc.min_m),
                            r)
        return r

    def _update_range(self) -> None:
        """Advance the rangefinder one step: age, dropouts, and a new sample when due."""
        rc, dt = self.cfg.range, self.cfg.dt
        self.range_age += dt
        self.range_timer -= dt
        self.range_drop_left = (self.range_drop_left - dt).clamp(min=0.0)
        if rc.dropout_rate_hz > 0:
            start = torch.as_tensor(self.range_rng.random(self.num_envs) < rc.dropout_rate_hz * dt,
                                    device=self.device) & (self.range_drop_left <= 0)
            if bool(start.any()):
                k = int(start.sum())
                self.range_drop_left[start] = torch.as_tensor(
                    self.range_rng.uniform(*rc.dropout_s, size=k), dtype=self.dtype,
                    device=self.device)
        due = self.range_timer <= 0
        self.range_timer = torch.where(due, self.range_timer + 1.0 / rc.rate_hz, self.range_timer)
        take = due & (self.range_drop_left <= 0)
        self._take_range(take, self._range_reading())

    def _take_range(self, take: torch.Tensor, reading: torch.Tensor) -> None:
        """New readings for the envs in ``take``, through the spike gate."""
        self.range_meas = torch.where(take, reading, self.range_meas)
        self.range_age = torch.where(take, torch.zeros_like(self.range_age), self.range_age)
        gate = self.cfg.range.spike_gate_m
        if gate <= 0:
            return
        wall = reading * self._beam_cos()
        accept = (torch.isnan(self.range_accepted)  # nothing accepted yet this flight
                  | ((wall - self.range_accepted).abs() <= gate)
                  | ((wall - self.range_pending).abs() <= gate))
        # A beam that missed the wall (inf) is not a wall: the gate keeps its last finite
        # reference, as the module's does (an unusable sample never reaches its gate).
        self.range_accepted = torch.where(take & accept & torch.isfinite(wall), wall,
                                          self.range_accepted)
        self.range_pending = torch.where(take, torch.where(accept, math.nan, wall),
                                         self.range_pending)
        self.range_rejected = torch.where(take, ~accept, self.range_rejected)

    def range_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The wall-range input as the module computes it: (error, usable), each (N,).

        The measured perpendicular distance is the newest reading times the beam's cosine
        now (``WallGuard::apply``'s ``wall_m``); the error is the hold point's wall distance
        (``range_ref``) minus it, so it has the sign of ``pos_err``'s x (positive toward the
        wall) but is measured on the wall, not on the estimate. Zero while unusable."""
        rc = self.cfg.range
        cos = self._beam_cos()
        ok = (self.range_present & ~self.range_rejected & (self.range_age <= rc.max_age_s + 1e-6)
              & (self.range_meas >= rc.min_m) & (self.range_meas <= rc.max_m)
              & (cos >= self.cos_range_off) & torch.isfinite(self.range_meas))
        err = (self.range_ref - self.range_meas * cos).clamp(-rc.obs_clip_m, rc.obs_clip_m)
        err = torch.where(ok, torch.nan_to_num(err), torch.zeros_like(err))
        return err, ok

    def _cmd_per_newton(self, mass_scale: Any, tw_scale: Any, hover_err: Any) -> Any:
        """The module's normalised thrust per policy newton: ``hover_thrust / (MASS_KG * g)``
        with ``MASS_KG`` the nominal mass and the hover thrust this airframe's (through its
        thrust curve), off by ``hover_err``."""
        q, act = self.cfg.quad, self.cfg.actuation
        hover_frac = np.minimum(1.0, 1.0 / (np.asarray(tw_scale, dtype=np.float64)
                                            * q.max_thrust / (q.mass * GRAVITY)))
        u_hover = act.command_for_fraction(hover_frac)
        return u_hover * (1.0 + np.asarray(hover_err, dtype=np.float64)) / (q.mass * GRAVITY)

    def _draw_airframe(self, env_ids: torch.Tensor) -> None:
        """Per-episode airframe and near-wall yaw moment, from their own generator. Nothing
        is drawn (and the sim keeps ``cfg.quad``) while both are off."""
        cfg, rng, n = self.cfg, self.airframe_rng, int(env_ids.numel())
        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        ar = cfg.airframe_rand
        if ar.active:
            q = cfg.quad
            mass = rng.uniform(*ar.mass_scale, size=n)
            inertia = rng.uniform(*ar.inertia_scale, size=n)
            tw = rng.uniform(*ar.thrust_to_weight_scale, size=n)
            tau = (np.full(n, q.motor_tau) if ar.motor_tau_s is None
                   else rng.uniform(*ar.motor_tau_s, size=n))
            hover_err = rng.uniform(*ar.hover_thrust_error, size=n)
            self.sim.set_airframe(
                env_ids, mass=q.mass * mass,
                inertia=np.asarray(q.inertia)[None, :] * inertia[:, None],
                max_thrust=q.max_thrust * mass * tw, motor_tau=tau,
            )
            self.thrust_gain[env_ids] = t(mass * (1.0 + hover_err))
            self.thrust_cmd_per_n[env_ids] = t(self._cmd_per_newton(mass, tw, hover_err))
            self.max_thrust_act[env_ids] = t(q.max_thrust * mass * tw)
            self.mass_scale[env_ids] = t(mass)
            self.inertia_scale[env_ids] = t(inertia)
        wy = cfg.wall_yaw_torque
        if wy.active:
            peak = rng.uniform(*wy.peak_nm, size=n)
            sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
            self.yaw_torque_peak[env_ids] = t(peak * sign)

    def _add_sensor_noise(self, o: torch.Tensor) -> None:
        """Gyro and attitude noise, and slow velocity wander (cfg.sensor_noise)."""
        def randn(k: int) -> torch.Tensor:
            return torch.randn(self.num_envs, k, generator=self.noise_gen, device=self.device,
                               dtype=self.dtype)

        a = math.exp(-self.dt / self.cfg.sensor_noise.vel_wander_tau_s)
        self.vel_wander = a * self.vel_wander + math.sqrt(1.0 - a * a) * (
            self.noise_std[:, 2:3] * randn(3))
        o[:, OBS["vel"]] += self.vel_wander
        o[:, OBS["ang_vel"]] += self.noise_std[:, 0:1] * randn(3) / self.cfg.max_rate_rad_s
        o[:, OBS["gravity_body"]] += self.noise_std[:, 1:2] * randn(3)

    def _wall_yaw_torque(self, pos: torch.Tensor) -> torch.Tensor | None:
        wy = self.cfg.wall_yaw_torque
        if not wy.active:
            return None
        assert self.wind is not None
        q = (self.wind.mean_wind.norm(dim=-1) / wy.ref_wind_mps) ** 2
        self.yaw_torque = (self.yaw_torque_peak
                           * torch.exp(-self.clearance(pos).clamp(min=0.0) / wy.decay_m) * q)
        tau = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        tau[:, 2] = self.yaw_torque
        return tau

    # ------------------------------------------------------------ stepping

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.obs_buf, {"observations": {}}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   dict]:
        cfg = self.cfg
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        if actions.shape != (self.num_envs, NUM_ACTIONS):
            raise ValueError(
                f"actions must be ({self.num_envs}, {NUM_ACTIONS}), got {tuple(actions.shape)}"
            )
        actions = torch.nan_to_num(actions).clamp(-1.0, 1.0)

        # Action -> lagged rate setpoint -> rate loop -> simulator, with this step's wind.
        rate_cmd = actions[:, :3] * cfg.max_rate_rad_s
        self.rate_sp += self.rate_lag_alpha[:, None] * (rate_cmd - self.rate_sp)
        # Thrust as the module sends it: through the hover point, so x thrust_gain on an
        # airframe other than the nominal one (1 unless cfg.airframe_rand is on).
        thrust = 0.5 * (actions[:, 3] + 1.0) * cfg.quad.max_thrust
        if self.thrust_via_module:
            u = (thrust * self.thrust_cmd_per_n).clamp(*cfg.actuation.thrust_cmd_range)
            thrust = self.max_thrust_act * cfg.actuation.thrust_fraction(u)
        else:
            thrust = thrust * self.thrust_gain
        thrust_n, weight_n = thrust, cfg.quad.mass * self.mass_scale * GRAVITY
        torque = self.rate_pid(self.rate_sp, self.sim.state.ang_vel)
        self.wind_force = wind.disturbance_force(
            self.wind, self.episode_time, self.sim.state.pos, self.sim.state.vel
        )
        tau_ext = self._wall_yaw_torque(self.sim.state.pos)
        vel_before = self.sim.state.vel.clone()
        state = (self.sim.step(thrust, torque, self.wind_force) if tau_ext is None
                 else self.sim.step(thrust, torque, self.wind_force, tau_ext))
        self.episode_step += 1
        self.episode_time += cfg.dt
        self.engage_time += cfg.dt

        # What the estimator reports.
        finite = state.finite()
        safe_pos = torch.where(finite[:, None], state.pos, self.hold_point)
        safe_vel = torch.where(finite[:, None], state.vel, torch.zeros_like(state.vel))
        if self.cfg.vision_offset_down_m > 0:
            # The fix tracks a point below the centre of mass (body -z), less its level offset.
            r = self.vision_offset.expand(self.num_envs, 3)
            safe_pos = safe_pos + quat_rotate(state.quat, r) - r
            safe_vel = safe_vel + quat_rotate(state.quat, torch.cross(state.ang_vel, r, dim=-1))
        est = self.estimator.step(safe_pos, safe_vel)
        self.estimate_pos, self.estimate_vel = est.pos, est.vel
        if self.fusion is not None:
            accel = torch.where(finite[:, None], (state.vel - vel_before) / cfg.dt,
                                torch.zeros_like(vel_before))
            self.estimate_pos, self.estimate_vel = self.fusion.step(accel, est.pos, est.vel,
                                                                    ~est.stale)
        self.estimate_stale, self.estimate_age = est.stale, est.age

        # Termination on the TRUE state.
        err = state.pos - self.hold_point
        clearance = self.clearance(state.pos)
        gravity_body = quat_rotate_inverse(state.quat, torch.tensor(
            [0.0, 0.0, -1.0], dtype=self.dtype, device=self.device).expand(self.num_envs, 3))
        contact = clearance <= cfg.contact_radius_m
        ground = state.pos[:, 2] <= cfg.min_height_m
        tilt = -gravity_body[:, 2] < self.cos_max_tilt
        box = err.norm(dim=-1) > cfg.max_pos_error_m
        nonfinite = ~finite
        crash = (contact | ground | tilt | box | nonfinite)
        timeout = (self.episode_step >= self.max_episode_length) & ~crash
        self.done_buf = crash | timeout
        self.time_out_buf = timeout
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for name, mask in (("contact", contact), ("ground", ground), ("tilt", tilt),
                           ("box", box), ("nonfinite", nonfinite), ("timeout", timeout)):
            self.counts[name] += int(mask.sum())
            reason = torch.where(mask & (reason == 0), TERM_REASONS.index(name), reason)
        self.term_reason = reason

        # Reward on the TRUE state (the policy only ever sees the estimate).
        w = cfg.reward
        toward = (err @ self.wall_normal).clamp(0.0, 1.0)
        approach = ((state.vel @ self.wall_normal) - w.toward_wall_vel_deadband_mps).clamp(
            0.0, 1.0)
        d_action = actions - self.last_action
        rates = actions[:, :3]
        self.action_fast = self.action_fast + self.band_alpha[0] * (rates - self.action_fast)
        self.action_slow = self.action_slow + self.band_alpha[1] * (rates - self.action_slow)
        band_power = ((self.action_fast - self.action_slow) ** 2).sum(-1)
        # Tilting past tilt_free_deg costs more the further it goes, so the policy learns to
        # stay level when its estimate is stale or jumps instead of flipping to chase it.
        over_tilt = (self.cos_tilt_free + gravity_body[:, 2]).clamp(min=0.0)
        # Shaping terms are bounded (see RewardWeights) so living always beats crashing.
        speed_w = w.vel + w.stale_vel * est.stale.to(self.dtype)  # slow down on a dropout
        # Heading: 1 - cos(yaw error) to the wall-facing heading, from the body x axis.
        fwd_world = quat_rotate(state.quat, torch.tensor(
            [1.0, 0.0, 0.0], dtype=self.dtype, device=self.device).expand(self.num_envs, 3))
        cos_yaw = fwd_world[:, 0] / fwd_world[:, :2].norm(dim=-1).clamp(min=1e-6)
        reward = (
            w.alive
            - w.pos * (1.0 - torch.exp(-err.norm(dim=-1) / w.pos_scale_m))
            - w.toward_wall * toward
            - w.toward_wall_vel * approach
            - w.height * (1.0 - torch.exp(-err[:, 2].abs() / w.height_scale_m))
            - speed_w * state.vel.norm(dim=-1).clamp(max=5.0)
            - w.ang_vel * state.ang_vel.norm(dim=-1).clamp(max=10.0)
            - w.thrust * (thrust_n / weight_n - 1.0).abs().clamp(max=1.0)
            - w.action * (actions[:, :3] ** 2).sum(-1)
            - w.action_rate * (d_action**2).sum(-1)
            - w.action_band * band_power.clamp(max=w.action_band_cap)
            - w.tilt * over_tilt
            - w.heading * (1.0 - cos_yaw)
            - w.yaw_rate * state.ang_vel[:, 2].abs().clamp(max=10.0)
            - w.crash * crash.to(self.dtype)
        )
        self.reward_buf = torch.nan_to_num(reward, nan=-w.crash)
        self.episode_reward += self.reward_buf
        self.last_action = actions
        self.step_pos = state.pos.clone()

        extras: dict[str, Any] = {"time_outs": self.time_out_buf, "observations": {}}
        done_ids = self.done_buf.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            ep = {
                "reward": self.episode_reward[done_ids].mean(),
                "length_s": self.episode_time[done_ids].mean(),
                "crash_rate": crash[done_ids].to(self.dtype).mean(),
                "contact_rate": contact[done_ids].to(self.dtype).mean(),
            }
            extras["episode"] = ep
            self.reset_envs(done_ids)
        self.reengaged = self._reengage(~self.done_buf)
        if cfg.range.active:
            self._update_range()
        self._observe()
        return self.obs_buf, self.reward_buf, self.done_buf, extras

    def _reengage(self, eligible: torch.Tensor) -> torch.Tensor:
        """Mid-episode re-engages (``cfg.reengage_rate_hz``), for envs in ``eligible`` at
        least ``reengage_min_s`` into their episode. The hold point moves to the episode's own
        plus a fresh ``start_offset_std_m`` scatter, never nearer the wall than the nearest
        hold distance nor within 0.5 m of the ground, and the previous action is zeroed, as
        the module's engage does. Returns the envs that re-engaged; the caller refills their
        policy history (``training/ppo_env.py``), so the policy meets the flat history and
        fresh offset it meets at a start. Gate 1 opens every episode with that transient
        (0.2 m of scatter), and a 60 s training episode has only one."""
        cfg = self.cfg
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if cfg.reengage_rate_hz <= 0:
            return mask
        draw = self.reengage_rng.random(self.num_envs) < cfg.reengage_rate_hz * cfg.dt
        mask = (torch.as_tensor(draw, device=self.device) & eligible
                & (self.episode_time >= cfg.reengage_min_s))
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return mask
        offset = cfg.start_offset_std_m * self.reengage_rng.standard_normal((ids.numel(), 3))
        hold = self.base_hold[ids] + torch.as_tensor(offset, dtype=self.dtype,
                                                     device=self.device)
        hold[:, 0] = hold[:, 0].clamp(max=-cfg.hold_distance[0])
        hold[:, 2] = hold[:, 2].clamp(min=cfg.min_height_m + 0.5)
        self.hold_point[ids] = hold
        self.range_ref[ids] = self.clearance(hold)
        self.last_action[ids] = 0.0
        self.engage_time[ids] = 0.0
        self.reengage_count += int(ids.numel())
        return mask

    def _observe(self) -> None:
        cfg, state = self.cfg, self.sim.state
        n = self.num_envs
        down = torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device).expand(n, 3)
        fwd = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device).expand(n, 3)
        o = self.obs_buf
        o[:, OBS["pos_err"]] = (self.estimate_pos - self.hold_point).clamp(
            -cfg.obs_pos_clip_m, cfg.obs_pos_clip_m)
        o[:, OBS["vel"]] = self.estimate_vel
        o[:, OBS["gravity_body"]] = quat_rotate_inverse(state.quat, down)
        o[:, OBS["heading"]] = quat_rotate(state.quat, fwd)[:, :2]
        rates = state.ang_vel
        pid = self.rate_pid
        if pid.gyro_alpha < 1.0:
            # PX4's filtered gyro as the rate loop last read it, one tick old: the delay
            # stands in for SITL's actuator and transport latency, which the mock otherwise
            # lacks. With it the mock's lift and wobble match Gazebo's (v8 +0.46 m against
            # +0.43 m); with the current sample they come out 40 % low (docs/gazebo_check.md).
            rates = pid.gyro
        o[:, OBS["ang_vel"]] = rates / cfg.max_rate_rad_s
        o[:, OBS["last_action"]] = self.last_action
        o[:, OBS["stale"]] = self.estimate_stale.to(self.dtype)[:, None]
        o[:, OBS["age"]] = self.estimate_age.clamp(max=1.0)[:, None]
        o[:, OBS["engage_time"]] = (self.engage_time / cfg.engage_obs_cap_s).clamp(
            max=1.0)[:, None]
        if cfg.range.active:
            err, ok = self.range_input()
            o[:, OBS["wall_range"]] = err[:, None]
            o[:, OBS["range_ok"]] = ok.to(self.dtype)[:, None]
        else:
            o[:, OBS["wall_range"]] = 0.0
            o[:, OBS["range_ok"]] = 0.0
        if cfg.sensor_noise.active:
            self._add_sensor_noise(o)
        torch.nan_to_num_(o)
