"""Configuration for the Structure Hold task: numbers only, read from one YAML mapping."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from envs.disturbance import DisturbanceConfig
from envs.estimate import EstimateCorruptionConfig
from envs.sims import QuadParams, RatePidGains
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
    vel: float = 0.05  # times true speed clipped at 5 m/s
    # Extra speed penalty while the estimate is stale (a dropout): the policy cannot know
    # where it is, so moving fast on a frozen estimate is how it leaves the 3 m box. Off by
    # default: two runs with 0.3 did not reduce crashes (docs/decisions.md, 2026-10-22).
    stale_vel: float = 0.0  # times true speed clipped at 5 m/s, only while stale
    ang_vel: float = 0.01  # times body rate norm clipped at 10 rad/s
    action: float = 0.02  # times squared normalised rate command
    # Smoothness: squared change in normalised action between steps. 0.15 (was 0.05) after
    # v2 chattered between the rate limits on PX4 (docs/decisions.md, 2026-10-22).
    action_rate: float = 0.15
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
    gets proportionally more thrust for the same action. The env reproduces that: thrust in
    newtons = action thrust x ``mass_scale`` x (1 + ``hover_thrust_error``), where the error is
    how far PX4's hover-thrust estimate is off.
    """

    mass_scale: tuple[float, float] = (1.0, 1.0)  # times quad.mass
    inertia_scale: tuple[float, float] = (1.0, 1.0)  # times quad.inertia, all axes together
    # Max thrust over weight, relative to the nominal airframe's (16 N / 7.85 N = 2.04).
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
class StructureHoldConfig:
    """The task. The wall is the plane x = 0 (world ENU); free air is x < 0 and the hold
    point sits ``hold_distance`` metres from it, so the structure frame's x axis is world +x.
    """

    dt: float = 0.01  # s, policy and rate-loop step (100 Hz)
    episode_length_s: float = 20.0
    hold_distance: tuple[float, float] = (1.0, 3.0)  # m from the wall, per episode
    hold_height: tuple[float, float] = (2.0, 2.0)  # m above the origin
    start_offset_std_m: float = 0.2  # start position scatter around the hold point
    start_yaw_std_rad: float = 0.2  # start heading scatter about facing the wall (+x)
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
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    estimate: EstimateCorruptionConfig = field(default_factory=EstimateCorruptionConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    airframe_rand: AirframeRandomization = field(default_factory=AirframeRandomization)
    wall_yaw_torque: WallYawTorque = field(default_factory=WallYawTorque)

    def __post_init__(self) -> None:
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
        lo, hi = self.rate_lag_s
        if lo < 0 or lo > hi:
            raise ValueError(f"rate_lag_s: need 0 <= min <= max, got {self.rate_lag_s}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> StructureHoldConfig:
        d = dict(d or {})
        _check_keys(cls, d, "structure_hold config")
        nested = {
            "quad": QuadParams.from_dict,
            "rate_pid": RatePidGains.from_dict,
            "disturbance": DisturbanceConfig.from_dict,
            "estimate": lambda x: EstimateCorruptionConfig.from_dict(x or {}),
            "reward": RewardWeights.from_dict,
            "airframe_rand": AirframeRandomization.from_dict,
            "wall_yaw_torque": WallYawTorque.from_dict,
        }
        for key, build in nested.items():
            if key in d:
                d[key] = build(d[key])
        for key in ("hold_distance", "hold_height"):
            if key in d:
                v = d[key]
                d[key] = (float(v), float(v)) if isinstance(v, (int, float)) else tuple(
                    float(x) for x in v)
        return cls(**d)

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_length_s / self.dt))
