"""Configuration for the Structure Hold task: numbers only, read from one YAML mapping."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from envs.disturbance import DisturbanceConfig
from envs.estimate import EstimateCorruptionConfig
from envs.estimate.ekf import FusionConfig
from envs.sims import QuadParams, RatePidGains
from envs.structure_hold.airframes import airframe_config
from eval.metrics import CONTACT_RADIUS_M


def _check_keys(cls: type, d: dict[str, Any], what: str) -> None:
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {what} keys: {sorted(unknown)}")


@dataclass(frozen=True)
class RewardWeights:
    """Per-step reward = alive - bounded penalties, minus ``crash`` once on a crash.

    Every shaping term is bounded so a live step is always worth more than a crash: with
    an unbounded error penalty a drone that has drifted past about 1 m earns less per step
    than nothing and learns to end the episode by tilting over, which is exactly what it
    did during estimate dropouts.
    """

    alive: float = 2.0  # per step; above the worst-case sum of the bounded penalties
    pos: float = 1.5  # times 1 - exp(-error / pos_scale_m), so at most ``pos``
    pos_scale_m: float = 1.0  # error at which the position penalty reaches 63 % of ``pos``
    toward_wall: float = 0.5  # extra, times the error toward the wall clipped at 1 m
    # Extra on the height error alone, times 1 - exp(-|error| / height_scale_m). Off by
    # default. Inside the 3D norm a few centimetres of height error cost almost nothing next
    # to a larger horizontal error, so the policy's thrust at zero error drifted between runs
    # and v13 settled 0.05 m below the hold point on a clean estimate (docs/gazebo_check.md).
    height: float = 0.0
    height_scale_m: float = 0.1
    vel: float = 0.05  # times true speed clipped at 5 m/s
    # Extra speed penalty while the estimate is stale (a dropout): the policy cannot know
    # where it is, so moving fast on a frozen estimate is how it leaves the 3 m box. Off by
    # default: two runs with 0.3 did not reduce crashes (docs/decisions.md, 2026-10-22).
    stale_vel: float = 0.0  # times true speed clipped at 5 m/s, only while stale
    ang_vel: float = 0.01  # times body rate norm clipped at 10 rad/s
    # Collective thrust away from the weight: |thrust / (m g) - 1| clipped at 1. Off by
    # default. A 2 to 3 Hz thrust cycle around hover moves the drone only centimetres, so the
    # position term barely sees it and the step-to-step action_rate term weights 2 Hz by
    # (2 pi f dt)^2 ~ 0.016; this term charges the swing itself (docs/decisions.md 2026-10-22).
    thrust: float = 0.0
    action: float = 0.02  # times squared normalised rate command
    # Smoothness: squared change in normalised action between steps. 0.15 (was 0.05) after
    # v2 chattered between the rate limits on PX4 (docs/decisions.md, 2026-10-22).
    action_rate: float = 0.15
    # Ring: the action_band_lo_hz to action_band_hi_hz part of the normalised rate commands
    # (two first-order filters), squared and summed over the three axes, clipped at
    # action_band_cap. Off by default. v16 to v18 commanded a 3.3 Hz roll and pitch ring that
    # Gazebo flew too (PR #58); action_rate weights 3.3 Hz by (2 pi f dt)^2 ~ 0.06 and the
    # body-rate term at 0.2 did not remove it. v18's band power is about 0.045, v10's 0.0035.
    action_band: float = 0.0
    action_band_lo_hz: float = 1.0
    action_band_hi_hz: float = 6.0
    action_band_cap: float = 0.2
    tilt: float = 2.0  # times (cos(tilt_free) - cos(tilt)) when tilted beyond tilt_free_deg
    tilt_free_deg: float = 45.0  # no tilt penalty up to here (fighting 12 m/s wind needs ~30)
    # Heading hold: the camera must stay on the structure, and a policy with no heading
    # term yawed freely (v2 commanded a saturated yaw rate on PX4, docs/decisions.md
    # 2026-10-22). Target is the wall-facing heading (+x of the structure frame).
    # 2.0: at 0.5 the first v3 run settled on a 30 deg offset (30 deg cost 0.07 per step, less
    # than the position shaping it traded for); 2.0 makes 30 deg cost 0.27.
    heading: float = 2.0  # times 1 - cos(yaw error): 0 facing the wall, at most 2 x heading
    yaw_rate: float = 0.05  # times |yaw rate| clipped at 10 rad/s
    crash: float = 20.0  # once, on contact, ground, tilt or leaving the box

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RewardWeights:
        d = dict(d or {})
        _check_keys(cls, d, "reward")
        return cls(**d)


def _range(v: Any, name: str) -> tuple[float, float]:
    lo, hi = (float(v), float(v)) if isinstance(v, (int, float)) else (float(v[0]), float(v[1]))
    if lo > hi:
        raise ValueError(f"{name}: min {lo} > max {hi}")
    return lo, hi


@dataclass(frozen=True)
class AirframeRandomization:
    """Per-episode spread of the airframe around ``quad``: domain randomisation for training
    and the stress conditions of the Gate 1 protocol. Each range is (min, max), drawn uniform
    per episode from a generator of its own, so switching it on changes the airframe and
    nothing else (same hold points, wind and estimate as without it). The defaults are the
    nominal airframe exactly: nothing is drawn.

    The PX4 module scales the policy's thrust through the vehicle's hover thrust
    (``sim2real/px4``: ``thrust_n / (MASS_KG * g) * hover_thrust``), so a heavier airframe
    gets proportionally more thrust for the same action. The env reproduces that: on a
    thrust-linear airframe, thrust in newtons = action thrust x ``mass_scale`` x (1 +
    ``hover_thrust_error``), where the error is how far PX4's hover-thrust estimate is off;
    with ``actuation`` set, the same mapping goes through the airframe's thrust curve.
    """

    mass_scale: tuple[float, float] = (1.0, 1.0)  # times quad.mass
    inertia_scale: tuple[float, float] = (1.0, 1.0)  # times quad.inertia, all axes together
    # Max thrust over weight, relative to the nominal airframe's (placeholder 16 N / 7.85 N =
    # 2.04, gz_x500 34.2 N / 20.2 N = 1.69).
    thrust_to_weight_scale: tuple[float, float] = (1.0, 1.0)
    motor_tau_s: tuple[float, float] | None = None  # None: quad.motor_tau
    hover_thrust_error: tuple[float, float] = (0.0, 0.0)  # fraction, + means too much thrust

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                object.__setattr__(self, f.name, _range(v, f.name))
        for name in ("mass_scale", "inertia_scale", "thrust_to_weight_scale"):
            if getattr(self, name)[0] <= 0:
                raise ValueError(f"{name} must be positive")
        if self.motor_tau_s is not None and self.motor_tau_s[0] < 0:
            raise ValueError("motor_tau_s must be >= 0")
        if self.hover_thrust_error[0] <= -1:
            raise ValueError("hover_thrust_error must be above -1")

    @property
    def active(self) -> bool:
        return (any(getattr(self, n) != (1.0, 1.0) for n in
                    ("mass_scale", "inertia_scale", "thrust_to_weight_scale"))
                or self.motor_tau_s is not None or self.hover_thrust_error != (0.0, 0.0))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AirframeRandomization:
        d = dict(d or {})
        _check_keys(cls, d, "airframe randomization")
        return cls(**d)


@dataclass(frozen=True)
class WallYawTorque:
    """Placeholder near-wall yaw moment (not in envs/disturbance: one published data point,
    showing the moment changing sign, is not enough to fit it; the fan rig will). Per episode
    a peak moment is drawn from ``peak_nm`` with a random sign; the moment on the body is

        tau_z = sign * peak * exp(-clearance / decay_m) * (|mean wind| / ref_wind_mps)^2

    so it grows near the wall and with the wind's dynamic pressure, and reaches ``peak`` at
    the wall in a ``ref_wind_mps`` wind. The nominal airframe's yaw authority is 0.05 N m.
    Off by default (peak 0); drawn from the airframe generator so it changes nothing else.
    """

    peak_nm: tuple[float, float] = (0.0, 0.0)
    decay_m: float = 1.0
    ref_wind_mps: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "peak_nm", _range(self.peak_nm, "peak_nm"))
        if self.peak_nm[0] < 0 or self.decay_m <= 0 or self.ref_wind_mps <= 0:
            raise ValueError("peak_nm >= 0, decay_m > 0 and ref_wind_mps > 0")

    @property
    def active(self) -> bool:
        return self.peak_nm[1] > 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> WallYawTorque:
        d = dict(d or {})
        _check_keys(cls, d, "wall yaw torque")
        return cls(**d)


@dataclass(frozen=True)
class SensorNoise:
    """Noise on the inputs the mock otherwise reads from the true state, plus slow wander on
    the velocity estimate, for policies that fly on a real estimator and IMU. Per episode
    each std is drawn uniformly between 0 and its maximum here:

    - ``ang_vel_std_rad_s``: white noise on the body rates (gyro);
    - ``gravity_std``: white noise on the gravity direction (attitude estimate);
    - ``vel_wander_std_mps`` with ``vel_wander_tau_s``: first-order (Ornstein-Uhlenbeck)
      noise on the velocity input, the slow error of an EKF's velocity, which a 15-step
      history cannot average out the way it averages white noise.

    v5 sat 0.3 to 0.45 m above its hold point in Gazebo while its rate setpoints jittered
    ten times more than in the mock; in the mock, noisy rate or velocity inputs lift its hold
    the same way (docs/decisions.md, 2026-10-22). Off by default; its own generator, so it
    changes nothing else.
    """

    ang_vel_std_rad_s: float = 0.0
    gravity_std: float = 0.0
    vel_wander_std_mps: float = 0.0
    vel_wander_tau_s: float = 0.3

    def __post_init__(self) -> None:
        if min(self.ang_vel_std_rad_s, self.gravity_std, self.vel_wander_std_mps) < 0:
            raise ValueError("sensor noise stds must be >= 0")
        if self.vel_wander_tau_s <= 0:
            raise ValueError("vel_wander_tau_s must be positive")

    @property
    def active(self) -> bool:
        return max(self.ang_vel_std_rad_s, self.gravity_std, self.vel_wander_std_mps) > 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SensorNoise:
        d = dict(d or {})
        _check_keys(cls, d, "sensor noise")
        return cls(**d)


@dataclass(frozen=True)
class Actuation:
    """How PX4's normalised commands become force and torque on the airframe, between the
    rate loop and the simulator. The defaults are the placeholder's: PX4's unit torque is
    the airframe's ``quad.max_torque`` and thrust is linear in the command, so the policy's
    newtons arrive unchanged (``envs/structure_hold/airframes.py`` derives a real vehicle's).
    """

    # N m of body torque per unit of PX4's normalised torque (roll, pitch, yaw), linearised
    # at hover. None: ``quad.max_torque``, i.e. a unit command is exactly the airframe's
    # limit. Larger than ``max_torque``: PX4 saturates the torque before a unit command.
    torque_per_unit: tuple[float, float, float] | None = None
    # Rotor speed is linear in the command from this fraction of full speed (gz:
    # SIM_GZ_EC_MIN / MAX) and thrust goes with speed squared. None: thrust linear in it.
    esc_min_frac: float | None = None
    # The module's clamp on the normalised collective thrust it sends.
    thrust_cmd_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.torque_per_unit is not None:
            object.__setattr__(self, "torque_per_unit",
                               tuple(float(x) for x in self.torque_per_unit))
            if len(self.torque_per_unit) != 3 or min(self.torque_per_unit) <= 0:
                raise ValueError("torque_per_unit must be three positive values")
        if self.esc_min_frac is not None and not 0.0 <= self.esc_min_frac < 1.0:
            raise ValueError("esc_min_frac must be in [0, 1)")
        object.__setattr__(self, "thrust_cmd_range",
                           _range(self.thrust_cmd_range, "thrust_cmd_range"))
        lo, hi = self.thrust_cmd_range
        if lo < 0 or hi > 1:
            raise ValueError("thrust_cmd_range must lie in [0, 1]")

    @property
    def thrust_via_module(self) -> bool:
        """True when thrust must go through the module's normalised command (a thrust
        curve or a clamp); False keeps the placeholder's direct newtons."""
        return self.esc_min_frac is not None or self.thrust_cmd_range != (0.0, 1.0)

    def thrust_fraction(self, u: Any) -> Any:
        """Thrust as a fraction of max thrust for normalised commands ``u`` in [0, 1]."""
        if self.esc_min_frac is None:
            return u
        w0 = self.esc_min_frac
        return (w0 + (1.0 - w0) * u) ** 2

    def command_for_fraction(self, f: Any) -> Any:
        """Inverse of ``thrust_fraction``: the command giving thrust fraction ``f``."""
        if self.esc_min_frac is None:
            return f
        w0 = self.esc_min_frac
        return (np.sqrt(f) - w0) / (1.0 - w0)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Actuation:
        d = dict(d or {})
        _check_keys(cls, d, "actuation")
        return cls(**d)


@dataclass(frozen=True)
class StructureHoldConfig:
    """The task. The wall is the plane x = 0 (world ENU); free air is x < 0 and the hold
    point sits ``hold_distance`` metres from it, so the structure frame's x axis is world +x.

    ``airframe`` names a vehicle in ``envs/structure_hold/airframes.py``; ``from_dict``
    merges its numbers (``quad``, ``rate_pid``, ``actuation``, ``rate_lag_s``, and for
    ``gz_x500_sitl`` also ``dt``, ``vision_offset_down_m`` and ``fusion``) under the ones the
    mapping states itself.

    The position fix and what the policy reads from it: ``estimate`` corrupts the fix;
    ``vision_offset_down_m`` makes the fix track a point that far below the centre of mass
    along the body z axis (zero offset while level, so the hold point is unchanged), as the
    Gazebo bridge sends the x500's model origin, 0.24 m below its IMU, with no lever-arm
    correction in EKF2; ``fusion`` (``envs/estimate/ekf.py``), when enabled, fuses the fix
    with the IMU the way EKF2 does and the policy reads the fused estimate. Both off by
    default.
    """

    airframe: str = "placeholder"
    dt: float = 0.01  # s, policy and rate-loop step (100 Hz)
    episode_length_s: float = 20.0
    hold_distance: tuple[float, float] = (1.0, 3.0)  # m from the wall, per episode
    hold_height: tuple[float, float] = (2.0, 2.0)  # m above the origin
    start_offset_std_m: float = 0.2  # start position scatter around the hold point
    start_yaw_std_rad: float = 0.2  # start heading scatter about facing the wall (+x)
    # Mid-episode re-engages, for training: at this rate per second the hold point moves to
    # the episode's own plus a fresh start_offset_std_m scatter, as if the module were
    # engaged again there (env.StructureHoldEnv._reengage). 0, the default, is off.
    reengage_rate_hz: float = 0.0
    reengage_min_s: float = 2.0  # none in an episode's first seconds (its start is one)
    # Action 1.0 = this body rate. 2 rad/s: on PX4 dynamics a policy trained with 6 rad/s
    # banged its roll and pitch commands between the limits and tipped over through the rate
    # loop's lag; the module flew it once it clamped the commands to 1 to 2 rad/s.
    max_rate_rad_s: float = 2.0
    # The autopilot's rate loop does not follow a rate setpoint at once: a first-order lag
    # with this time constant, drawn per episode (PX4 showed 100 to 200 ms). (0, 0) disables.
    rate_lag_s: tuple[float, float] = (0.1, 0.2)
    max_pos_error_m: float = 3.0  # episode ends beyond this (true error)
    min_height_m: float = 0.3  # the ground, also a crash
    max_tilt_deg: float = 80.0
    contact_radius_m: float = CONTACT_RADIUS_M
    obs_pos_clip_m: float = 5.0
    quad: QuadParams = field(default_factory=QuadParams)
    rate_pid: RatePidGains = field(default_factory=RatePidGains)
    actuation: Actuation = field(default_factory=Actuation)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    estimate: EstimateCorruptionConfig = field(default_factory=EstimateCorruptionConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    airframe_rand: AirframeRandomization = field(default_factory=AirframeRandomization)
    wall_yaw_torque: WallYawTorque = field(default_factory=WallYawTorque)
    sensor_noise: SensorNoise = field(default_factory=SensorNoise)
    vision_offset_down_m: float = 0.0
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def __post_init__(self) -> None:
        if self.vision_offset_down_m < 0:
            raise ValueError("vision_offset_down_m must be >= 0")
        if self.dt <= 0 or self.episode_length_s < self.dt:
            raise ValueError("dt must be positive and episode_length_s at least one step")
        for name in ("hold_distance", "hold_height"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name}: min {lo} > max {hi}")
        if self.hold_distance[0] <= self.contact_radius_m:
            raise ValueError("hold_distance must start outside the contact radius")
        if self.max_rate_rad_s <= 0:
            raise ValueError("max_rate_rad_s must be positive")
        if self.reengage_rate_hz < 0 or self.reengage_min_s < 0:
            raise ValueError("reengage_rate_hz and reengage_min_s must be >= 0")
        r = self.reward
        if not 0 < r.action_band_lo_hz < r.action_band_hi_hz or r.action_band_cap <= 0:
            raise ValueError("reward.action_band needs 0 < lo_hz < hi_hz and cap > 0")
        lo, hi = self.rate_lag_s
        if lo < 0 or lo > hi:
            raise ValueError(f"rate_lag_s: need 0 <= min <= max, got {self.rate_lag_s}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> StructureHoldConfig:
        d = dict(d or {})
        _check_keys(cls, d, "structure_hold config")
        # The named airframe first, then the mapping's own values on top (one level deep, so
        # ``quad: {motor_tau: 0.03}`` changes one number of the vehicle's quad).
        for key, value in airframe_config(str(d.get("airframe", "placeholder"))).items():
            if isinstance(value, dict) and isinstance(d.get(key), dict):
                d[key] = {**value, **d[key]}
            elif key not in d:
                d[key] = value
        nested = {
            "quad": QuadParams.from_dict,
            "rate_pid": RatePidGains.from_dict,
            "actuation": Actuation.from_dict,
            "disturbance": DisturbanceConfig.from_dict,
            "estimate": lambda x: EstimateCorruptionConfig.from_dict(x or {}),
            "reward": RewardWeights.from_dict,
            "airframe_rand": AirframeRandomization.from_dict,
            "wall_yaw_torque": WallYawTorque.from_dict,
            "sensor_noise": SensorNoise.from_dict,
            "fusion": FusionConfig.from_dict,
        }
        for key, build in nested.items():
            if key in d:
                d[key] = build(d[key])
        for key in ("hold_distance", "hold_height", "rate_lag_s"):
            if key in d:
                v = d[key]
                d[key] = (float(v), float(v)) if isinstance(v, (int, float)) else tuple(
                    float(x) for x in v)
        return cls(**d)

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_length_s / self.dt))
