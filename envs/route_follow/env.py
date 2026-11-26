"""Route Follow: fly a given route through a tunnel or corridor on a corrupted estimate.

Each step, as Structure Hold (``envs/structure_hold/env.py``): the policy's action (body
rates and thrust) -> the PX4-matched rate PID -> the simulator with the wind force -> the
estimate corruption (and EKF2-like fusion when ``cfg.fusion`` is on) -> observation and
reward. What differs: the hold point is a target moving along the route
(``geometry.RouteGeometry.target``), the walls are the route's tunnel or corridor, and the
policy can read up to five rangefinders. The design is in ``docs/route-follow.md``.

Frames: world ENU. Observations are written in the **route frame** at the target
(``RouteGeometry.route_frame``: x along the route, y left, z up), so the policy is the same
on any heading.

Observation (``NUM_OBS`` = 38):

| slice | what | units |
| --- | --- | --- |
| 0:3 | estimated position minus the target, route frame, clipped | m |
| 3:6 | estimated velocity, route frame | m/s |
| 6 | the target's speed now | m/s |
| 7:13 | the target ``preview_m`` metres on (two points), minus the target, route frame | m |
| 13:16 | gravity direction in the body frame | unit |
| 16:18 | body x axis in the route frame, x and y | unit |
| 18:21 | body rates / max_rate | - |
| 21:25 | previous action | - |
| 25 | estimate stale flag | - |
| 26 | estimate age, capped at 1 s | s |
| 27 | distance left to the last waypoint / ``remaining_cap_m``, capped at 1 | - |
| 28:33 | rangefinders forward, left, right, up, down, clipped at ``ranges.obs_max_m`` | m |
| 33:38 | rangefinder usable flags (0: absent, out of range or not yet sampled; reading 0) | - |

Action (4, each in [-1, 1]): body rates as a fraction of ``max_rate_rad_s`` and thrust onto
``[0, quad.max_thrust]`` N, through the module's hover mapping when ``cfg.actuation`` says
so, exactly as Structure Hold.

Termination on the TRUE state: ``contact`` (any wall, floor or ceiling within
``contact_radius_m``), ``tilt``, ``lost`` (more than ``max_track_error_m`` from the target),
``nonfinite``, else ``timeout`` at the episode's end. ``step`` leaves the last step's true
cross-track error, error to the target, clearance and distance to the last waypoint in
``step_*`` attributes before resetting finished envs, for the scorer
(``eval/route_follow.py``).

The surroundings (``cfg.ambient``, ``envs/ambient``, off by default; drawn per episode from a
generator of their own, so a config without them flies the same episodes): air density,
battery level and rain scale the thrust and torque a command gives and the draft's push, the
battery sags under load, and near a surface the rotors feel it. A tunnel's nearest surface
can be a wall, the floor or the roof, so the cushion (floor), the pull upward (roof) and the
pull sideways (wall) act on whichever is nearest, split by which way it faces; the downwash
coming back grows as that surface comes closer. A barometer fused for height
(``fusion.height_source: baro``) also reads the airspeed past its port.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from envs.ambient import nearness, sag_gain, sample_ambient, surface_gain
from envs.disturbance import sample_params
from envs.disturbance import torch_adapter as wind
from envs.estimate.ekf import TorchFusion
from envs.estimate.torch_adapter import TorchEstimateCorruptor
from envs.route_follow.config import RANGE_DIRS, RouteFollowConfig
from envs.route_follow.geometry import RouteGeometry
from envs.route_follow.routes import sample_routes
from envs.sims import QuadState, RatePid, make_sim
from envs.sims.torch_quad import (
    GRAVITY,
    body_z_world,
    quat_from_yaw,
    quat_rotate,
    quat_rotate_inverse,
)

OBS = {
    "pos_err": slice(0, 3),
    "vel": slice(3, 6),
    "target_speed": slice(6, 7),
    "preview": slice(7, 13),
    "gravity_body": slice(13, 16),
    "heading": slice(16, 18),
    "ang_vel": slice(18, 21),
    "last_action": slice(21, 25),
    "stale": slice(25, 26),
    "age": slice(26, 27),
    "remaining": slice(27, 28),
    "ranges": slice(28, 33),
    "range_ok": slice(33, 38),
}
NUM_OBS = 38
NUM_ACTIONS = 4
NUM_RANGES = len(RANGE_DIRS)
TERM_REASONS = ("none", "contact", "tilt", "lost", "nonfinite", "timeout")


class RouteFollowEnv:
    def __init__(self, cfg: RouteFollowConfig, num_envs: int, seed: int = 0,
                 device: str | torch.device = "cpu", sim: str = "torch"):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.cfg = cfg
        self.num_envs = n = int(num_envs)
        self.device = dev = torch.device(device)
        self.dtype = dt = torch.float32
        self.num_obs = NUM_OBS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = cfg.max_episode_length
        self.dt = cfg.dt
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.range_rng = np.random.default_rng([int(seed), 0x7A6])
        self.range_gen = torch.Generator(device=dev).manual_seed(int(seed) * 7919 + 0x7A6)

        self.sim = make_sim(sim, cfg.quad, n, cfg.dt, device=dev)
        act = cfg.actuation
        per_unit = act.torque_per_unit or cfg.quad.max_torque
        sat = (None if act.torque_per_unit is None else
               tuple(min(1.0, m / u) for m, u in zip(cfg.quad.max_torque, per_unit,
                                                     strict=True)))
        self.rate_pid = RatePid(cfg.rate_pid, per_unit, n, cfg.dt, dev, dt, sat_limit=sat)
        self.estimator = TorchEstimateCorruptor(cfg.estimate, n, cfg.dt, seed, dev, dt)
        self.fusion = (TorchFusion(cfg.fusion, n, cfg.dt, seed, dev, dt) if cfg.fusion.enabled
                       else None)
        self.vision_offset = torch.tensor([0.0, 0.0, -cfg.vision_offset_down_m], dtype=dt,
                                          device=dev)
        # The module's thrust mapping on the nominal airframe (StructureHoldEnv's
        # _cmd_per_newton with no airframe randomisation).
        q = cfg.quad
        hover_frac = min(1.0, q.mass * GRAVITY / q.max_thrust)
        self.thrust_cmd_per_n = float(act.command_for_fraction(hover_frac)) / (q.mass * GRAVITY)
        self.cos_max_tilt = math.cos(math.radians(cfg.max_tilt_deg))
        self.cos_tilt_free = math.cos(math.radians(cfg.reward.tilt_free_deg))
        self.range_dirs = torch.tensor(RANGE_DIRS, dtype=dt, device=dev)
        self.range_period_steps = max(1, round(1.0 / (cfg.ranges.rate_hz * cfg.dt)))

        self.geo = RouteGeometry(n, cfg.route.segments, dev, dt, branches=cfg.route.branches)
        z = lambda *shape: torch.zeros(*shape, dtype=dt, device=dev)  # noqa: E731
        self.episode_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.episode_time = z(n)
        self.target_s, self.target_speed = z(n), z(n)
        self.target_pos, self.target_vel = z(n, 3), z(n, 3)
        self.frame = torch.eye(3, dtype=dt, device=dev).repeat(n, 1, 1)  # route frame (cols)
        self.last_action = z(n, NUM_ACTIONS)
        self.rate_sp = z(n, 3)
        self.rate_lag_alpha = torch.ones(n, dtype=dt, device=dev)
        self.obs_buf = z(n, NUM_OBS)
        self.reward_buf = z(n)
        self.done_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.episode_reward = z(n)
        self.estimate_pos, self.estimate_vel = z(n, 3), z(n, 3)
        self.estimate_stale = torch.zeros(n, dtype=torch.bool, device=dev)
        self.estimate_age = z(n)
        self.wind_force = z(n, 3)
        self.wind = None
        self.draft_sign = z(n)
        self.range_present = torch.zeros(n, NUM_RANGES, dtype=torch.bool, device=dev)
        self.range_meas = torch.full((n, NUM_RANGES), math.inf, dtype=dt, device=dev)
        self.range_sampled = torch.zeros(n, dtype=torch.bool, device=dev)
        self.range_dusty = torch.zeros(n, dtype=torch.bool, device=dev)
        self.global_step = 0
        self.term_reason = torch.zeros(n, dtype=torch.long, device=dev)
        self.counts = {k: 0 for k in TERM_REASONS[1:]}
        # Last step's true diagnostics, before finished envs reset (module docstring).
        self.step_cross_track, self.step_track_err = z(n), z(n)
        self.step_clearance, self.step_to_end = z(n), z(n)
        # The surroundings (cfg.ambient): their own generators, so nothing else moves.
        self.ambient_rng = np.random.default_rng([int(seed), 0xA3B])
        self.ambient_gen = torch.Generator(device=dev).manual_seed(int(seed) * 7907 + 0xA3B)
        ones = lambda: torch.ones(n, dtype=dt, device=dev)  # noqa: E731
        self.amb_power, self.amb_drag, self.amb_mass = ones(), ones(), ones()
        self.amb_sag, self.amb_down, self.amb_radius = z(n), z(n), z(n)
        self.amb_ground_coef, self.amb_ceiling_coef = z(n), z(n)
        self.amb_wall_pull, self.amb_unsteady, self.amb_baro_wind = z(n), z(n), z(n)
        self.amb_cmd_per_n = torch.full((n,), self.thrust_cmd_per_n, dtype=dt, device=dev)
        self.amb_max_thrust = torch.full((n,), cfg.quad.max_thrust, dtype=dt, device=dev)
        self.ambient_force, self.unsteady_noise = z(n, 3), z(n, 3)
        self.baro_wind_ref, self.baro_wind_err = z(n), z(n)
        self.reset()

    # ------------------------------------------------------------ helpers

    @property
    def state(self) -> QuadState:
        """The true state (for scoring and tests), not what the policy sees."""
        return self.sim.state

    def to_route(self, v: torch.Tensor) -> torch.Tensor:
        """World vectors (N, 3) into the route frame at the target."""
        return torch.einsum("nij,ni->nj", self.frame, v)

    def from_route(self, v: torch.Tensor) -> torch.Tensor:
        """Route-frame vectors (N, 3) into the world."""
        return torch.einsum("nij,nj->ni", self.frame, v)

    def _update_target(self) -> None:
        geo, cfg = self.geo, self.cfg
        self.target_s, self.target_speed = geo.target(self.episode_time)
        self.target_pos = geo.point_at(self.target_s)
        self.target_vel = self.target_speed[:, None] * geo.segment_x_at(self.target_s)
        self.frame = geo.route_frame(self.target_s, cfg.heading_lookahead_m)

    # ------------------------------------------------------------ resets

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_envs(torch.arange(self.num_envs, device=self.device))
        self._observe()
        return self.obs_buf, {"observations": {}}

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        k = int(env_ids.numel())
        if k == 0:
            return
        cfg, rng = self.cfg, self.rng
        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        routes = sample_routes(cfg.route, k, cfg.episode_length_s, rng)
        self.geo.set(env_ids, routes)
        start = routes.waypoints[:, 0] + cfg.start_offset_std_m * rng.standard_normal((k, 3))
        yaw = cfg.start_yaw_std_rad * rng.standard_normal(k)  # the first segment is along +x
        zeros = torch.zeros(k, 3, dtype=self.dtype, device=self.device)
        self.sim.reset(env_ids, QuadState(pos=t(start), quat=quat_from_yaw(t(yaw)), vel=zeros,
                                          ang_vel=zeros.clone()))
        self.rate_pid.reset(env_ids)
        self.estimator.reset(self.sim.state.pos, self.sim.state.vel, env_ids)
        if self.fusion is not None:
            self.fusion.reset(env_ids, self.sim.state.pos[env_ids], self.sim.state.vel[env_ids])
        new = wind.to_torch(sample_params(cfg.disturbance, k, rng), self.device, self.dtype)
        if self.wind is None or k == self.num_envs:
            self.wind = new
        else:
            for name in new.__dataclass_fields__:
                getattr(self.wind, name)[env_ids] = getattr(new, name)
        self.draft_sign[env_ids] = t(np.where(rng.random(k) < 0.5, -1.0, 1.0))
        self.episode_step[env_ids] = 0
        self.episode_time[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.rate_sp[env_ids] = 0.0
        tau = rng.uniform(*cfg.rate_lag_s, size=k)
        self.rate_lag_alpha[env_ids] = t(np.where(tau > 0, cfg.dt / np.maximum(tau, cfg.dt), 1.0))
        self.episode_reward[env_ids] = 0.0
        self.estimate_pos[env_ids] = self.sim.state.pos[env_ids]
        self.estimate_vel[env_ids] = 0.0
        self.estimate_stale[env_ids] = False
        self.estimate_age[env_ids] = 0.0
        self.wind_force[env_ids] = 0.0
        if cfg.ranges.active:
            self.range_present[env_ids] = torch.as_tensor(
                self.range_rng.random((k, NUM_RANGES)) < cfg.ranges.present_prob,
                device=self.device)
            if cfg.ranges.dust_prob > 0:  # drawn only when on, so other configs draw as before
                self.range_dusty[env_ids] = torch.as_tensor(
                    self.range_rng.random(k) < cfg.ranges.dust_prob, device=self.device)
        self.range_meas[env_ids] = math.inf
        self.range_sampled[env_ids] = False
        if cfg.ambient.active:
            self._draw_ambient(env_ids)
        self._update_target()

    def place_on_target(self) -> None:
        """Put every drone on its target as it stands for ``episode_time``: level, headed
        along the route, moving at the target's speed, estimate and rangefinders restarted
        there. Used when the episode phase is randomised at the start of training."""
        self._update_target()
        ids = torch.arange(self.num_envs, device=self.device)
        along = self.geo.segment_x_at(self.target_s)
        yaw = torch.atan2(along[:, 1], along[:, 0])
        zeros = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        self.sim.reset(ids, QuadState(pos=self.target_pos.clone(), quat=quat_from_yaw(yaw),
                                      vel=self.target_vel.clone(), ang_vel=zeros))
        self.rate_pid.reset(ids)
        self.estimator.reset(self.sim.state.pos, self.sim.state.vel, ids)
        if self.fusion is not None:
            self.fusion.reset(ids, self.sim.state.pos, self.sim.state.vel)
        self.estimate_pos[:] = self.sim.state.pos
        self.estimate_vel[:] = self.sim.state.vel
        self.estimate_stale[:] = False
        self.estimate_age[:] = 0.0
        self.last_action[:] = 0.0
        self.rate_sp[:] = 0.0
        self.range_meas[:] = math.inf
        self.range_sampled[:] = False
        self._observe()

    # ------------------------------------------------------------ sensors and air

    def _sample_ranges(self) -> None:
        """New readings for every env with a rangefinder: the beams traced through the walls,
        with noise and outliers, and in a dusty episode lost and short returns."""
        rc = self.cfg.ranges
        dirs = quat_rotate(self.sim.state.quat[:, None, :].expand(-1, NUM_RANGES, 4).reshape(
            -1, 4), self.range_dirs.repeat(self.num_envs, 1)).reshape(self.num_envs,
                                                                      NUM_RANGES, 3)
        true = self.geo.trace(self.sim.state.pos, dirs, rc.max_m, rc.trace_iters)
        shape = (self.num_envs, NUM_RANGES)
        noise = torch.randn(shape, generator=self.range_gen, device=self.device,
                            dtype=self.dtype)
        r = true + noise * (rc.noise_std_m + rc.noise_frac * true.clamp(max=rc.max_m))
        if rc.outlier_prob > 0:
            u = torch.rand(*shape, 2, generator=self.range_gen, device=self.device,
                           dtype=self.dtype)
            r = torch.where(u[..., 0] < rc.outlier_prob,
                            rc.min_m + u[..., 1] * (rc.max_m - rc.min_m), r)
        if rc.dust_prob > 0:
            # Dusty episodes: no return (inf reads as not usable) or a short one off the dust.
            u = torch.rand(*shape, 3, generator=self.range_gen, device=self.device,
                           dtype=self.dtype)
            lo, hi = rc.dust_short_frac
            dusty = self.range_dusty[:, None]
            r = torch.where(dusty & (u[..., 1] < rc.dust_short_prob),
                            true.clamp(max=rc.max_m) * (lo + (hi - lo) * u[..., 2]), r)
            r = torch.where(dusty & (u[..., 0] < rc.dust_loss_prob),
                            torch.full_like(r, math.inf), r)
        self.range_meas = r
        self.range_sampled[:] = True

    def range_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(readings clipped at ``obs_max_m``, usable) each (N, 5); readings 0 where not
        usable."""
        rc = self.cfg.ranges
        ok = (self.range_present & self.range_sampled[:, None] & torch.isfinite(self.range_meas)
              & (self.range_meas >= rc.min_m) & (self.range_meas <= rc.max_m))
        val = torch.where(ok, self.range_meas.clamp(max=rc.obs_max_m),
                          torch.zeros_like(self.range_meas))
        return val, ok

    def _draft_force(self, state: QuadState) -> torch.Tensor:
        """The wind force with the mean wind, gusts and turbulence turned so that the
        disturbance model's +x lies along the target's segment (either way, per episode): a
        draft that follows the tunnel."""
        x = self.geo.segment_x_at(self.target_s)
        x = x * self.draft_sign[:, None]
        heading = torch.atan2(x[:, 1], x[:, 0])
        c, s = torch.cos(heading), torch.sin(heading)

        def rot(v: torch.Tensor, sign: float) -> torch.Tensor:
            return torch.stack([c * v[:, 0] - sign * s * v[:, 1],
                                sign * s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)

        local = wind.disturbance_force(self.wind, self.episode_time, state.pos,
                                       rot(state.vel, -1.0))
        return rot(local, 1.0)

    def _draw_ambient(self, env_ids: torch.Tensor) -> None:
        """One episode's surroundings for ``env_ids``, as Structure Hold draws them: PX4's
        hover-thrust estimate keeps hover where it was, so what changes is the headroom above
        it (maximum thrust and torque times ``power``), the draft's push (times the density)
        and the weight of the rain on the airframe."""
        cfg, q = self.cfg, self.cfg.quad
        p = sample_ambient(cfg.ambient, int(env_ids.numel()), self.ambient_rng)
        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        self.sim.set_airframe(env_ids, mass=q.mass * p.mass_scale,
                              max_thrust=q.max_thrust * p.power,
                              max_torque=np.asarray(q.max_torque)[None, :] * p.power[:, None])
        hover_frac = np.minimum(1.0, q.mass * p.mass_scale * GRAVITY / (q.max_thrust * p.power))
        u_hover = cfg.actuation.command_for_fraction(hover_frac)
        self.amb_cmd_per_n[env_ids] = t(u_hover / (q.mass * GRAVITY))
        self.amb_max_thrust[env_ids] = t(q.max_thrust * p.power)
        self.amb_mass[env_ids] = t(p.mass_scale)
        self.amb_power[env_ids] = t(p.power)
        self.amb_drag[env_ids] = t(p.density_ratio)
        self.amb_sag[env_ids] = t(p.battery_sag)
        self.amb_down[env_ids] = t(p.down_force_n)
        self.amb_radius[env_ids] = t(p.rotor_radius_m)
        self.amb_ground_coef[env_ids] = t(p.ground_coef)
        self.amb_ceiling_coef[env_ids] = t(p.ceiling_coef)
        self.amb_wall_pull[env_ids] = t(p.wall_pull)
        self.amb_unsteady[env_ids] = t(p.unsteady)
        self.amb_baro_wind[env_ids] = t(p.baro_wind_m_per_mps2)
        self.ambient_force[env_ids] = 0.0
        self.unsteady_noise[env_ids] = 0.0
        self.baro_wind_ref[env_ids] = 0.0
        self.baro_wind_err[env_ids] = 0.0

    def _ambient_force(self, thrust: torch.Tensor) -> torch.Tensor:
        """What the surroundings add to the draft's force this step, (N, 3) N, world: the
        rain's momentum and, near the nearest surface, the rotors' cushion off a floor, the
        pull up into a roof or sideways into a wall, and the downwash coming back. The
        surface's direction is the clearance's gradient (it points away from the surface)."""
        amb, st = self.cfg.ambient, self.sim.state
        f = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        f[:, 2] = -self.amb_down
        if not amb.surfaces:
            return f
        pos = torch.nan_to_num(st.pos)
        d = self.geo.clearance(pos)
        h = 0.02
        eye = torch.eye(3, dtype=self.dtype, device=self.device) * h
        grad = torch.stack([self.geo.clearance(pos + eye[i]) - d for i in range(3)], -1) / h
        away = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # from the surface
        floor, roof = away[:, 2].clamp(min=0.0), (-away[:, 2]).clamp(min=0.0)
        side = torch.sqrt((1.0 - away[:, 2] ** 2).clamp(min=0.0))
        r = self.amb_radius
        x = nearness(r, d)
        gain = (1.0 + floor * (surface_gain(r, d, self.amb_ground_coef) - 1.0)
                + roof * (surface_gain(r, d, self.amb_ceiling_coef) - 1.0))
        f = f + ((gain - 1.0) * thrust)[:, None] * body_z_world(st.quat)
        toward = -away.clone()
        toward[:, 2] = 0.0
        toward = toward / toward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        f = f + (self.amb_wall_pull * thrust * x * x * side)[:, None] * toward
        if amb.unsteady[1] > 0:
            a = math.exp(-self.dt / amb.unsteady_tau_s)
            self.unsteady_noise = a * self.unsteady_noise + math.sqrt(1.0 - a * a) * torch.randn(
                self.num_envs, 3, generator=self.ambient_gen, device=self.device,
                dtype=self.dtype)
            near = (x * x).clamp(max=1.0)
            f = f + (self.amb_unsteady * thrust * near)[:, None] * self.unsteady_noise
        return torch.nan_to_num(f)

    def _down_range(self, state: QuadState) -> torch.Tensor:
        """The downward rangefinder's height above the floor beneath (``fusion.height_source``
        range), (N,): the beam along body -z traced to the tunnel's or corridor's floor, times
        the tilt's cosine, as EKF2 uses it; NaN past 45 deg of tilt or without a return."""
        down = -body_z_world(state.quat)
        cos = -down[:, 2]
        pos = torch.nan_to_num(state.pos)
        slant = self.geo.trace(pos, down[:, None, :], max_m=self.cfg.fusion.range_max_m / 0.7)
        h = slant[:, 0] * cos
        ok = (cos > 0.7071) & torch.isfinite(h) & (slant[:, 0] > 0)
        return torch.where(ok, h, torch.full_like(h, float("nan")))

    def _baro_wind(self, state: QuadState) -> torch.Tensor:
        """How far the barometer's reading has moved since the episode began because of the
        air moving past its port, (N,) m (``cfg.ambient.baro_wind_m_per_mps2``). The draft is
        turned along the route, which does not change its speed against the drone."""
        x = self.geo.segment_x_at(self.target_s) * self.draft_sign[:, None]
        heading = torch.atan2(x[:, 1], x[:, 0])
        c, s = torch.cos(heading), torch.sin(heading)
        v = state.vel
        local_vel = torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1],
                                 v[:, 2]], -1)
        air = wind.air_velocity(self.wind, self.episode_time, state.pos) - local_vel
        err = torch.nan_to_num(self.amb_baro_wind * (air * air).sum(-1))
        self.baro_wind_ref = torch.where(self.episode_step == 1, err, self.baro_wind_ref)
        self.baro_wind_err = err - self.baro_wind_ref
        return self.baro_wind_err

    # ------------------------------------------------------------ stepping

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.obs_buf, {"observations": {}}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   dict]:
        cfg = self.cfg
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        if actions.shape != (self.num_envs, NUM_ACTIONS):
            raise ValueError(
                f"actions must be ({self.num_envs}, {NUM_ACTIONS}), got {tuple(actions.shape)}")
        actions = torch.nan_to_num(actions).clamp(-1.0, 1.0)

        rate_cmd = actions[:, :3] * cfg.max_rate_rad_s
        self.rate_sp += self.rate_lag_alpha[:, None] * (rate_cmd - self.rate_sp)
        thrust = 0.5 * (actions[:, 3] + 1.0) * cfg.quad.max_thrust
        ambient = cfg.ambient.active
        if ambient:
            # The module's mapping through this episode's hover point, then the battery's sag
            # under load (the top of the throttle gives less than its share).
            cap = self.amb_max_thrust
            if cfg.actuation.thrust_via_module:
                u = (thrust * self.amb_cmd_per_n).clamp(*cfg.actuation.thrust_cmd_range)
                thrust = cap * cfg.actuation.thrust_fraction(u)
            else:
                thrust = thrust * self.amb_mass
            weight_n = cfg.quad.mass * self.amb_mass * GRAVITY
            thrust = thrust * sag_gain(self.amb_sag, (thrust / cap).clamp(0.0, 1.0),
                                       (weight_n / cap).clamp(max=1.0))
        elif cfg.actuation.thrust_via_module:
            u = (thrust * self.thrust_cmd_per_n).clamp(*cfg.actuation.thrust_cmd_range)
            thrust = cfg.quad.max_thrust * cfg.actuation.thrust_fraction(u)
        torque = self.rate_pid(self.rate_sp, self.sim.state.ang_vel)
        self.wind_force = self._draft_force(self.sim.state)
        force = self.wind_force
        if ambient:
            torque = torque * self.amb_power[:, None]
            self.wind_force = self.wind_force * self.amb_drag[:, None]
            self.ambient_force = self._ambient_force(self.sim.thrust)
            force = self.wind_force + self.ambient_force
        vel_before = self.sim.state.vel.clone()
        state = self.sim.step(thrust, torque, force)
        self.episode_step += 1
        self.episode_time += cfg.dt
        self._update_target()

        finite = state.finite()
        safe_pos = torch.where(finite[:, None], state.pos, self.target_pos)
        safe_vel = torch.where(finite[:, None], state.vel, torch.zeros_like(state.vel))
        if cfg.vision_offset_down_m > 0:
            r = self.vision_offset.expand(self.num_envs, 3)
            safe_pos = safe_pos + quat_rotate(state.quat, r) - r
            safe_vel = safe_vel + quat_rotate(state.quat, torch.cross(state.ang_vel, r, dim=-1))
        est = self.estimator.step(safe_pos, safe_vel)
        self.estimate_pos, self.estimate_vel = est.pos, est.vel
        if self.fusion is not None:
            accel = torch.where(finite[:, None], (state.vel - vel_before) / cfg.dt,
                                torch.zeros_like(vel_before))
            extra = {}
            if cfg.ambient.baro_wind and cfg.fusion.baro:
                extra["baro_offset"] = self._baro_wind(state)
            if cfg.fusion.range:
                extra["range_height"] = self._down_range(state)
            self.estimate_pos, self.estimate_vel = self.fusion.step(accel, est.pos, est.vel,
                                                                    ~est.stale, true_pos=state.pos,
                                                                    **extra)
        self.estimate_stale, self.estimate_age = est.stale, est.age

        # Termination and reward on the TRUE state.
        pos_now = torch.where(finite[:, None], state.pos, self.target_pos)
        err = pos_now - self.target_pos
        clearance = self.geo.clearance(pos_now)
        down = torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device)
        gravity_body = quat_rotate_inverse(state.quat, down.expand(self.num_envs, 3))
        contact = clearance <= cfg.contact_radius_m
        tilt = -gravity_body[:, 2] < self.cos_max_tilt
        lost = err.norm(dim=-1) > cfg.max_track_error_m
        nonfinite = ~finite
        crash = contact | tilt | lost | nonfinite
        timeout = (self.episode_step >= self.max_episode_length) & ~crash
        self.done_buf = crash | timeout
        self.time_out_buf = timeout
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for name, mask in (("contact", contact), ("tilt", tilt), ("lost", lost),
                           ("nonfinite", nonfinite), ("timeout", timeout)):
            self.counts[name] += int(mask.sum())
            reason = torch.where(mask & (reason == 0), TERM_REASONS.index(name), reason)
        self.term_reason = reason
        self.step_cross_track = self.geo.distance_to_route(pos_now)
        self.step_track_err = err.norm(dim=-1)
        self.step_clearance = clearance
        self.step_to_end = (pos_now - self.geo.waypoints[:, -1]).norm(dim=-1)

        w = cfg.reward
        vel = torch.where(finite[:, None], state.vel, torch.zeros_like(state.vel))
        margin_used = ((w.clear_margin_m - (clearance - cfg.contact_radius_m))
                       / w.clear_margin_m).clamp(0.0, 1.0)
        over_tilt = (self.cos_tilt_free + gravity_body[:, 2]).clamp(min=0.0)
        fwd = quat_rotate(state.quat, torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype,
                                                   device=self.device).expand(self.num_envs, 3))
        route_x = self.frame[:, :, 0]
        cos_yaw = ((fwd[:, :2] * route_x[:, :2]).sum(-1)
                   / (fwd[:, :2].norm(dim=-1) * route_x[:, :2].norm(dim=-1)).clamp(min=1e-6))
        d_action = actions - self.last_action
        reward = (
            w.alive
            - w.track * (1.0 - torch.exp(-err.norm(dim=-1) / w.track_scale_m))
            - w.vel * (vel - self.target_vel).norm(dim=-1).clamp(max=3.0)
            - w.clearance * margin_used
            - w.heading * (1.0 - cos_yaw)
            - w.ang_vel * state.ang_vel.norm(dim=-1).clamp(max=10.0)
            - w.action * (actions[:, :3] ** 2).sum(-1)
            - w.action_rate * (d_action ** 2).sum(-1)
            - w.tilt * over_tilt
            - w.crash * crash.to(self.dtype)
        )
        self.reward_buf = torch.nan_to_num(reward, nan=-w.crash)
        self.episode_reward += self.reward_buf
        self.last_action = actions

        extras: dict[str, Any] = {"time_outs": self.time_out_buf, "observations": {}}
        done_ids = self.done_buf.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            extras["episode"] = {
                "reward": self.episode_reward[done_ids].mean(),
                "length_s": self.episode_time[done_ids].mean(),
                "crash_rate": crash[done_ids].to(self.dtype).mean(),
                "contact_rate": contact[done_ids].to(self.dtype).mean(),
            }
            self.reset_envs(done_ids)
        self.global_step += 1
        if cfg.ranges.active and self.global_step % self.range_period_steps == 0:
            self._sample_ranges()
        self._observe()
        return self.obs_buf, self.reward_buf, self.done_buf, extras

    def _observe(self) -> None:
        cfg, state, n = self.cfg, self.sim.state, self.num_envs
        o = self.obs_buf
        o[:, OBS["pos_err"]] = self.to_route(self.estimate_pos - self.target_pos).clamp(
            -cfg.obs_pos_clip_m, cfg.obs_pos_clip_m)
        o[:, OBS["vel"]] = self.to_route(self.estimate_vel)
        o[:, OBS["target_speed"]] = self.target_speed[:, None]
        ahead = self.target_s[:, None] + torch.tensor(cfg.preview_m, dtype=self.dtype,
                                                      device=self.device)
        rel = self.geo.point_at(ahead) - self.target_pos[:, None, :]  # (N, 2, 3)
        o[:, OBS["preview"]] = torch.einsum("nij,nki->nkj", self.frame, rel).reshape(n, 6)
        down = torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device)
        fwd = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        o[:, OBS["gravity_body"]] = quat_rotate_inverse(state.quat, down.expand(n, 3))
        o[:, OBS["heading"]] = self.to_route(quat_rotate(state.quat, fwd.expand(n, 3)))[:, :2]
        rates = self.rate_pid.gyro if self.rate_pid.gyro_alpha < 1.0 else state.ang_vel
        o[:, OBS["ang_vel"]] = rates / cfg.max_rate_rad_s
        o[:, OBS["last_action"]] = self.last_action
        o[:, OBS["stale"]] = self.estimate_stale.to(self.dtype)[:, None]
        o[:, OBS["age"]] = self.estimate_age.clamp(max=1.0)[:, None]
        o[:, OBS["remaining"]] = ((self.geo.length - self.target_s) / cfg.remaining_cap_m).clamp(
            0.0, 1.0)[:, None]
        val, ok = self.range_input()
        o[:, OBS["ranges"]] = val
        o[:, OBS["range_ok"]] = ok.to(self.dtype)
        torch.nan_to_num_(o)
