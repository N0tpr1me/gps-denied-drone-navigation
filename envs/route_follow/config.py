"""Configuration for the Route Follow task: numbers only, read from one YAML mapping.

The airframe, rate loop, actuation, estimate corruption and fusion are the same building
blocks Structure Hold uses (``envs/structure_hold``), so a named airframe (``gz_x500``,
``gz_x500_sitl``) means exactly the same vehicle in both tasks. What is new: the route and
its tunnel or corridor (``RouteGen``), the rangefinders (``RouteRanges``) and the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from envs.disturbance import DisturbanceConfig
from envs.estimate import EstimateCorruptionConfig
from envs.estimate.ekf import FusionConfig
from envs.sims import QuadParams, RatePidGains
from envs.structure_hold.airframes import airframe_config
from envs.structure_hold.config import Actuation
from eval.metrics import CONTACT_RADIUS_M


def _check_keys(cls: type, d: dict[str, Any], what: str) -> None:
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {what} keys: {sorted(unknown)}")


def _pair(v: Any, name: str) -> tuple[float, float]:
    lo, hi = (float(v), float(v)) if isinstance(v, (int, float)) else (float(v[0]), float(v[1]))
    if lo > hi:
        raise ValueError(f"{name}: min {lo} > max {hi}")
    return lo, hi


def _pairs(obj: Any) -> None:
    """Turn every tuple-typed field of a frozen dataclass into a checked (min, max) pair."""
    for f in fields(obj):
        if str(f.type).startswith("tuple"):
            object.__setattr__(obj, f.name, _pair(getattr(obj, f.name), f.name))


@dataclass(frozen=True)
class RouteGen:
    """One random route per episode: ``segments`` straight pieces joined at random turns,
    through a corridor (rectangular cross-section) or a tunnel (round), drawn per episode.

    The route's length is set by its speed so that the target reaches the last waypoint
    ``end_hold_s`` before the episode ends: every flight has a stop and a short hold at the
    end, and "completed" means the same thing in every episode.
    """

    segments: int = 4
    # A short route turns less: it gets at most length / min_segment_m straight pieces (each
    # one or more of the segments, turning only between pieces), so a slow, short route is
    # not four 1 m zig-zags.
    min_segment_m: float = 3.0
    # Relative segment lengths, drawn uniform and scaled to the route length.
    segment_weight: tuple[float, float] = (1.0, 2.0)
    turn_deg: tuple[float, float] = (-90.0, 90.0)  # heading change at each joint
    climb_deg: tuple[float, float] = (-10.0, 10.0)  # each segment's slope
    tunnel_prob: float = 0.5  # else a corridor
    corridor_width_m: tuple[float, float] = (2.0, 4.0)
    corridor_height_m: tuple[float, float] = (2.4, 4.0)
    tunnel_radius_m: tuple[float, float] = (1.1, 2.0)
    # The worlds beyond plain corridors and round tunnels (all off by default, so the v0
    # sets are unchanged; docs/route-follow.md, "Worlds"):
    # a corridor whose roof is a half circle as wide as the corridor, on vertical walls (a
    # mine drift or an arched culvert), with this probability; its height is kept at least
    # 0.6 of its width so the walls stand above the floor;
    arch_prob: float = 0.0
    # each segment narrowed (a doorway, a collapsed stretch, a smaller pipe section) with
    # this probability, its cross-section scaled by a draw from pinch_frac;
    pinch_prob: float = 0.0
    pinch_frac: tuple[float, float] = (0.6, 0.85)
    # up to ``branches`` dead-end side passages, each present with branch_prob, leaving a
    # random segment at right angles, left or right, branch_len_m long (crosscuts in a mine,
    # side corridors, junctions): open air the side rangefinders see past.
    branches: int = 0
    branch_prob: float = 1.0
    branch_len_m: tuple[float, float] = (2.0, 6.0)
    speed_mps: tuple[float, float] = (0.5, 1.5)  # the target's cruise speed
    accel_mps2: float = 0.5  # the target's acceleration and braking
    # The target's speed at a 90 degree (or sharper) corner, as a fraction of cruise; it
    # scales linearly to full speed at a straight joint (envs/route_follow/routes.py).
    corner_speed_min_frac: float = 0.2
    end_hold_s: float = 5.0
    max_length_m: float = 60.0
    start_height_m: float = 2.0  # world z of the first waypoint (nothing depends on it)

    def __post_init__(self) -> None:
        _pairs(self)
        if self.segments < 1:
            raise ValueError("segments must be >= 1")
        if self.min_segment_m <= 0:
            raise ValueError("min_segment_m must be positive")
        if self.segment_weight[0] <= 0:
            raise ValueError("segment_weight must be positive")
        for name in ("tunnel_prob", "arch_prob", "pinch_prob", "branch_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.branches < 0 or self.branch_len_m[0] <= 0:
            raise ValueError("branches must be >= 0 and branch_len_m positive")
        if not (0.0 < self.pinch_frac[0] and self.pinch_frac[1] <= 1.0):
            raise ValueError("pinch_frac must lie in (0, 1]")
        if self.speed_mps[0] <= 0 or self.accel_mps2 <= 0:
            raise ValueError("speed_mps and accel_mps2 must be positive")
        if not 0.0 <= self.corner_speed_min_frac <= 1.0:
            raise ValueError("corner_speed_min_frac must be in [0, 1]")
        if abs(self.climb_deg[0]) >= 60 or abs(self.climb_deg[1]) >= 60:
            raise ValueError("climb_deg must stay within +-60 deg (the route frame needs a "
                             "horizontal left axis)")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RouteGen:
        d = dict(d or {})
        _check_keys(cls, d, "route")
        return cls(**d)


# The rangefinders, in this order in the observation; body-frame beam directions (FLU).
RANGE_NAMES = ("forward", "left", "right", "up", "down")
RANGE_DIRS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0),
              (0.0, 0.0, -1.0))


@dataclass(frozen=True)
class RouteRanges:
    """Up to five 1D rangefinders, each present per episode with ``present_prob`` (so a
    policy learns to fly with any subset, or none). A reading is the beam's distance to the
    tunnel or corridor wall with noise and outliers, sampled every ``1 / rate_hz`` s; it is
    usable while it lies inside ``[min_m, max_m]``. The input is the reading clipped at
    ``obs_max_m`` (past that a wall is far enough not to matter) and a usable flag."""

    present_prob: float = 0.0  # off by default: the inputs stay zero
    rate_hz: float = 20.0
    noise_std_m: float = 0.01
    noise_frac: float = 0.005
    outlier_prob: float = 0.0
    min_m: float = 0.1
    max_m: float = 8.0
    obs_max_m: float = 4.0
    trace_iters: int = 40  # sphere-tracing steps per beam (envs/route_follow/geometry.py)

    def __post_init__(self) -> None:
        if not 0.0 <= self.present_prob <= 1.0 or not 0.0 <= self.outlier_prob <= 1.0:
            raise ValueError("present_prob and outlier_prob must be in [0, 1]")
        if self.rate_hz <= 0 or not 0 < self.min_m < self.max_m or self.obs_max_m <= 0:
            raise ValueError("need rate_hz > 0, 0 < min_m < max_m and obs_max_m > 0")
        if self.trace_iters < 1:
            raise ValueError("trace_iters must be >= 1")

    @property
    def active(self) -> bool:
        return self.present_prob > 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RouteRanges:
        d = dict(d or {})
        _check_keys(cls, d, "ranges")
        return cls(**d)


@dataclass(frozen=True)
class RouteRewardWeights:
    """Per-step reward = alive - bounded penalties, minus ``crash`` once on a crash; every
    shaping term is bounded so living always beats crashing (as Structure Hold)."""

    alive: float = 2.0
    track: float = 1.5  # times 1 - exp(-|error to the target| / track_scale_m)
    track_scale_m: float = 0.5
    vel: float = 0.1  # times |velocity - target velocity| clipped at 3 m/s
    # Inside ``clear_margin_m`` of the contact distance: times the fraction of the margin used.
    clearance: float = 2.0
    clear_margin_m: float = 0.4
    heading: float = 0.5  # times 1 - cos(yaw error to the route direction)
    ang_vel: float = 0.01
    action: float = 0.02
    action_rate: float = 0.15
    tilt: float = 2.0
    tilt_free_deg: float = 45.0
    crash: float = 20.0

    def __post_init__(self) -> None:
        if self.track_scale_m <= 0 or self.clear_margin_m <= 0:
            raise ValueError("track_scale_m and clear_margin_m must be positive")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RouteRewardWeights:
        d = dict(d or {})
        _check_keys(cls, d, "route reward")
        return cls(**d)


def tunnel_estimate() -> EstimateCorruptionConfig:
    """The default estimate for tunnels and corridors: Structure Hold's VIO-like corruption
    with the drift allowed to grow further (walk 0.02 m/sqrt(s), cap 0.5 m against 0.01 and
    0.3), since dark, repetitive walls are where visual odometry drifts most. A guess until a
    tunnel VIO fit exists (docs/route-follow.md)."""
    return EstimateCorruptionConfig(bias_walk_std_m_per_sqrt_s=0.02, bias_max_m=0.5)


def tunnel_air() -> DisturbanceConfig:
    """The default air in a tunnel: a draft along it (the env turns the mean wind onto the
    local route direction, either way), smaller gusts and turbulence than outdoors."""
    return DisturbanceConfig(mean_speed=(0.0, 3.0), mean_heading_deg=(0.0, 0.0),
                             gust_start=(2.0, 40.0), gust_speed=(0.0, 2.0),
                             turb_sigma=(0.0, 0.5))


@dataclass(frozen=True)
class RouteFollowConfig:
    """The task. ``airframe`` names a vehicle in ``envs/structure_hold/airframes.py``, merged
    under the mapping's own values as ``StructureHoldConfig`` does."""

    airframe: str = "placeholder"
    dt: float = 0.01
    episode_length_s: float = 30.0
    start_offset_std_m: float = 0.1
    start_yaw_std_rad: float = 0.1
    max_rate_rad_s: float = 2.0
    rate_lag_s: tuple[float, float] = (0.1, 0.2)
    max_track_error_m: float = 3.0  # lost the target: a crash
    max_tilt_deg: float = 80.0
    contact_radius_m: float = CONTACT_RADIUS_M
    obs_pos_clip_m: float = 3.0
    heading_lookahead_m: float = 1.0  # the route frame's x points at the target this far on
    preview_m: tuple[float, float] = (1.0, 2.5)  # the two preview points
    remaining_cap_m: float = 10.0
    vision_offset_down_m: float = 0.0
    route: RouteGen = field(default_factory=RouteGen)
    ranges: RouteRanges = field(default_factory=RouteRanges)
    reward: RouteRewardWeights = field(default_factory=RouteRewardWeights)
    quad: QuadParams = field(default_factory=QuadParams)
    rate_pid: RatePidGains = field(default_factory=RatePidGains)
    actuation: Actuation = field(default_factory=Actuation)
    disturbance: DisturbanceConfig = field(default_factory=tunnel_air)
    estimate: EstimateCorruptionConfig = field(default_factory=tunnel_estimate)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate_lag_s", _pair(self.rate_lag_s, "rate_lag_s"))
        object.__setattr__(self, "preview_m", _pair(self.preview_m, "preview_m"))
        if self.dt <= 0 or self.episode_length_s < self.dt:
            raise ValueError("dt must be positive and episode_length_s at least one step")
        if self.episode_length_s <= self.route.end_hold_s + 1.0:
            raise ValueError("episode_length_s must leave time to fly before the end hold")
        if self.rate_lag_s[0] < 0:
            raise ValueError("rate_lag_s must be >= 0")
        if self.max_rate_rad_s <= 0 or self.heading_lookahead_m <= 0 or self.preview_m[0] <= 0:
            raise ValueError("max_rate_rad_s, heading_lookahead_m and preview_m must be > 0")
        r = self.route
        narrow = min(0.5 * r.corridor_width_m[0], 0.5 * r.corridor_height_m[0],
                     r.tunnel_radius_m[0])
        if r.pinch_prob > 0:
            narrow *= r.pinch_frac[0]
        if narrow <= self.contact_radius_m + 0.1:
            raise ValueError("the narrowest corridor or tunnel leaves no room for the drone")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RouteFollowConfig:
        d = dict(d or {})
        _check_keys(cls, d, "route_follow config")
        for key, value in airframe_config(str(d.get("airframe", "placeholder"))).items():
            if isinstance(value, dict) and isinstance(d.get(key), dict):
                d[key] = {**value, **d[key]}
            elif key not in d:
                d[key] = value
        defaults = {"disturbance": tunnel_air(), "estimate": tunnel_estimate()}
        nested = {
            "route": RouteGen.from_dict,
            "ranges": RouteRanges.from_dict,
            "reward": RouteRewardWeights.from_dict,
            "quad": QuadParams.from_dict,
            "rate_pid": RatePidGains.from_dict,
            "actuation": Actuation.from_dict,
            "fusion": FusionConfig.from_dict,
        }
        for key, build in nested.items():
            if key in d:
                d[key] = build(d[key])
        # The tunnel defaults, with the mapping's own values on top.
        if "disturbance" in d:
            base = {f.name: getattr(defaults["disturbance"], f.name)
                    for f in fields(DisturbanceConfig)}
            d["disturbance"] = DisturbanceConfig.from_dict({**base, **(d["disturbance"] or {})})
        if "estimate" in d:
            base = {f.name: getattr(defaults["estimate"], f.name)
                    for f in fields(EstimateCorruptionConfig)}
            d["estimate"] = EstimateCorruptionConfig.from_dict({**base, **(d["estimate"] or {})})
        return cls(**d)

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_length_s / self.dt))
