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
    ang_vel: float = 0.01  # times body rate norm clipped at 10 rad/s
    action: float = 0.02  # times squared normalised rate command
    action_rate: float = 0.05  # times squared change in normalised action between steps
    tilt: float = 2.0  # times (cos(tilt_free) - cos(tilt)) when tilted beyond tilt_free_deg
    tilt_free_deg: float = 45.0  # no tilt penalty up to here (fighting 12 m/s wind needs ~30)
    crash: float = 20.0  # once, on contact, ground, tilt or leaving the box

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RewardWeights:
        d = dict(d or {})
        _check_keys(cls, d, "reward")
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
    max_rate_rad_s: float = 6.0  # action 1.0 = this body rate
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
