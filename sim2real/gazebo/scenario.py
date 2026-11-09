"""Scenario files for the wall-and-slab world.

A scenario is one YAML file (``sim2real/gazebo/scenarios/*.yaml``) with the spawn pose, the
wall, the ``envs.disturbance`` config, the seed, the hover to fly and, optionally, a payload
fixed under the x500 (``payload: {mass_kg, offset_down_m}``; the SITL image adds it to the
model before PX4 spawns it, ``sim2real/docker/add_payload.py``). ``sample_scenario``
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
class Payload:
    """A rigid payload fixed under the vehicle's base_link: a 0.10 x 0.10 x 0.05 m box of
    ``mass_kg`` whose centre sits ``offset_down_m`` below it. Mass 0 means none."""

    mass_kg: float = 0.0
    offset_down_m: float = 0.05

    def __post_init__(self) -> None:
        if self.mass_kg < 0 or self.offset_down_m < 0:
            raise ValueError("payload mass_kg and offset_down_m must be >= 0")


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
    payload: Payload = field(default_factory=Payload)

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
            "payload",
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
        return cls(
            name=str(d["name"]),
            seed=int(d.get("seed", 0)),
            spawn=Spawn(**(d.get("spawn") or {})),
            wall_point=tuple(float(v) for v in wall.get("point", (0.0, 0.0, 0.0))),
            wall_normal_toward=normal,
            disturbance=DisturbanceConfig.from_dict(d.get("disturbance")),
            wind_heading_deg=None if heading is None else float(heading),
            hover=Hover(**(d.get("hover") or {})),
            payload=Payload(**(d.get("payload") or {})),
        )


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
    return sample_params(cfg, 1, sc.seed, wall_point=sc.wall_point, wall_normal=normal)


def payload_environment(sc: Scenario) -> dict[str, str]:
    """The SITL image's environment for the scenario's payload (``sim2real/docker/
    entrypoint.sh`` hands it to ``add_payload.py``); empty without one."""
    if sc.payload.mass_kg <= 0:
        return {}
    return {"STRUCTURE_HOLD_PAYLOAD_KG": repr(float(sc.payload.mass_kg)),
            "STRUCTURE_HOLD_PAYLOAD_OFFSET_M": repr(float(sc.payload.offset_down_m))}
