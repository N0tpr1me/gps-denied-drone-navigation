"""Scenario files for the Gazebo worlds (``worlds/*.sdf``).

A scenario is one YAML file (``sim2real/gazebo/scenarios/*.yaml``) with the world, the spawn
pose, the wall, an optional second surface (the deck over the drone in ``bridge_deck``;
docs/bridge-holds.md), the ``envs.disturbance`` config, the seed, the hover to fly, and
optionally a surface pass flown in it (``pass``; docs/surface-passes.md "In Gazebo"). The
world defaults to ``wall_slab``, so every scenario written before the others flies where it
did. ``sample_scenario``
draws the episode's disturbance exactly as ``envs/structure_hold`` does (same
``sample_params``, same seed, one env), so a Gazebo run and a torch_quad run with the same
scenario see identical wind.

Frames follow docs/conventions.md: ENU, the wall normal ``normal_toward`` points from the
drone into the structure (the disturbance model takes its negative). Angles in the file are
degrees; ``wind_heading_deg`` is the direction the mean wind blows toward, 0 = +x, 90 = +y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.disturbance import DisturbanceConfig, DisturbanceParams, sample_params

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
WORLD_DIR = Path(__file__).resolve().parent / "worlds"
DEFAULT_WORLD = "wall_slab"


@dataclass(frozen=True)
class Spawn:
    x: float = -2.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0

    def px4_pose(self) -> str:
        """``PX4_GZ_MODEL_POSE`` value: x,y,z,roll,pitch,yaw with yaw in radians."""
        return f"{self.x},{self.y},{self.z},0,0,{math.radians(self.yaw_deg)}"


@dataclass(frozen=True)
class Hover:
    altitude: float = 2.5  # m above the spawn point
    hover_s: float = 30.0


@dataclass(frozen=True)
class SurfacePass:
    """A surface pass flown in the scenario's hover (docs/surface-passes.md): after
    ``start_s`` of plain hold the hold point moves ``length_m`` along the wall at the same
    standoff (to the right of the nose for a positive length, or up with ``up``), at
    ``speed_mps``, as the module's ``SH_PASS_*`` parameters say it. ``engage_at_s`` is the
    simulation time at which every controller starts its hold, so they meet the same wind and
    the same estimate faults at the same point of the pass (0: as soon as it is ready)."""

    length_m: float = 5.0
    speed_mps: float = 0.5
    start_s: float = 3.0
    up: bool = False
    engage_at_s: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < abs(self.length_m) <= 10.0:
            raise ValueError("pass.length_m must be within 10 m and not 0 (SH_PASS_LEN)")
        if not 0.1 <= self.speed_mps <= 1.0:
            raise ValueError("pass.speed_mps must lie in 0.1 to 1 m/s (SH_PASS_SPD)")
        if not 0.0 <= self.start_s <= 60.0 or self.engage_at_s < 0:
            raise ValueError("pass.start_s must lie in 0 to 60 s and engage_at_s be >= 0")

    def module_params(self) -> list[str]:
        """The module parameters that fly this pass, as ``NAME=VALUE``."""
        return [f"SH_PASS_LEN={float(self.length_m)}", f"SH_PASS_DIR={int(self.up)}",
                f"SH_PASS_SPD={float(self.speed_mps)}", f"SH_PASS_T0={float(self.start_s)}"]

    def flight_args(self) -> list[str]:
        """The same pass as ``sitl_test.py`` and ``pass_flight.py`` options."""
        args = ["--pass-len", str(self.length_m), "--pass-speed", str(self.speed_mps),
                "--pass-t0", str(self.start_s)]
        if self.up:
            args.append("--pass-up")
        if self.engage_at_s > 0:
            args += ["--engage-at-boot-s", str(self.engage_at_s)]
        return args


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int = 0
    spawn: Spawn = field(default_factory=Spawn)
    wall_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wall_normal_toward: tuple[float, float, float] = (1.0, 0.0, 0.0)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    wind_heading_deg: float | None = None
    hover: Hover = field(default_factory=Hover)
    world: str = DEFAULT_WORLD
    # The second surface: a point on it, the normal from the drone toward it, and the
    # horizontal direction across it that drives its edge shedding (None: the whole wind).
    surface2_point: tuple[float, float, float] | None = None
    surface2_normal_toward: tuple[float, float, float] | None = None
    surface2_across: tuple[float, float, float] | None = None
    # A surface pass flown in the hover (the YAML key is ``pass``); None: a plain hold.
    surface_pass: SurfacePass | None = None

    @property
    def wall_clearance(self) -> float:
        """Spawn distance from the wall along its normal, m."""
        p = np.asarray(self.wall_point) - np.array([self.spawn.x, self.spawn.y, self.spawn.z])
        return float(p @ np.asarray(self.wall_normal_toward))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        d = dict(d)
        known = {
            "name", "seed", "spawn", "wall", "disturbance", "wind_heading_deg", "hover",
            "world", "surface2", "pass",
        }
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        if "name" not in d:
            raise ValueError("scenario needs a name")
        wall = dict(d.get("wall") or {})
        unknown_wall = set(wall) - {"point", "normal_toward"}
        if unknown_wall:
            raise ValueError(f"unknown wall keys: {sorted(unknown_wall)}")
        normal = tuple(float(v) for v in wall.get("normal_toward", (1.0, 0.0, 0.0)))
        if len(normal) != 3 or np.linalg.norm(normal) < 1e-9:
            raise ValueError("wall.normal_toward must be a non-zero 3-vector")
        heading = d.get("wind_heading_deg")
        world = str(d.get("world", DEFAULT_WORLD))
        if not (WORLD_DIR / f"{world}.sdf").exists():
            raise ValueError(f"unknown world {world!r}: no {world}.sdf in {WORLD_DIR}")
        s2 = d.get("surface2")
        s2_point = s2_normal = s2_across = None
        if s2 is not None:
            s2 = dict(s2)
            unknown_s2 = set(s2) - {"point", "normal_toward", "across"}
            if unknown_s2 or not {"point", "normal_toward"} <= set(s2):
                raise ValueError("surface2 needs point and normal_toward (and optionally "
                                 f"across); got {sorted(s2)}")
            s2_point = _vec3(s2["point"], "surface2.point")
            s2_normal = _vec3(s2["normal_toward"], "surface2.normal_toward", nonzero=True)
            if s2.get("across") is not None:
                s2_across = _vec3(s2["across"], "surface2.across", nonzero=True)
        return cls(
            name=str(d["name"]),
            seed=int(d.get("seed", 0)),
            spawn=Spawn(**(d.get("spawn") or {})),
            wall_point=tuple(float(v) for v in wall.get("point", (0.0, 0.0, 0.0))),
            wall_normal_toward=normal,
            disturbance=DisturbanceConfig.from_dict(d.get("disturbance")),
            wind_heading_deg=None if heading is None else float(heading),
            hover=Hover(**(d.get("hover") or {})),
            world=world,
            surface2_point=s2_point,
            surface2_normal_toward=s2_normal,
            surface2_across=s2_across,
            surface_pass=SurfacePass(**d["pass"]) if d.get("pass") else None,
        )


def _vec3(value: Any, name: str, nonzero: bool = False) -> tuple[float, float, float]:
    v = tuple(float(x) for x in value)
    if len(v) != 3 or (nonzero and np.linalg.norm(v) < 1e-9):
        raise ValueError(f"{name} must be a{' non-zero' if nonzero else ''} 3-vector")
    return v  # type: ignore[return-value]


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario YAML; a bare name is looked up in ``scenarios/``."""
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = SCENARIO_DIR / f"{path}.yaml"
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: scenario must be a mapping")
    return Scenario.from_dict(data)


def sample_scenario(sc: Scenario) -> DisturbanceParams:
    """The scenario's episode disturbance for one env, as the training env would draw it."""
    normal = tuple(-v for v in sc.wall_normal_toward)
    cfg = sc.disturbance
    if sc.wind_heading_deg is not None:
        # The heading fixed in the config rather than written over the drawn wind: the same
        # draws and the same mean wind as before, and the near-wall flow, which depends on
        # which side of the wall the wind blows from, sees the scenario's heading.
        h = float(sc.wind_heading_deg)
        cfg = replace(cfg, mean_heading_deg=(h, h))
    second = {}
    if sc.surface2_normal_toward is not None:
        second = dict(surface2_point=sc.surface2_point,
                      surface2_normal=tuple(-v for v in sc.surface2_normal_toward),
                      surface2_across=sc.surface2_across)
    return sample_params(cfg, 1, sc.seed, wall_point=sc.wall_point, wall_normal=normal,
                         **second)
