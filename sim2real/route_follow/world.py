"""Gazebo route worlds: one fixed route through a tunnel or corridor, as one YAML file.

A route file (``sim2real/route_follow/routes/*.yaml``) names the cross-section, the
waypoints, the target's speed plan and the air; :func:`load_route_world` turns it into the
mock's own :class:`envs.route_follow.routes.Routes` (one route), so the speed plan, the
walls and the scoring are the mock's formulas (``walls.py``). :func:`world_sdf` writes the
Gazebo world for it from ``sim2real/gazebo/worlds/wall_slab.sdf`` (same physics, stock
systems and wind plugin) with the wall-and-slab structure replaced by the route's walls.

    name: tunnel_bend
    seed: 1                  # the wind draw; run_mission's --scenario-seed overrides it
    kind: tunnel             # tunnel (round), corridor (rectangle) or arched (a mine drift)
    radius_m: 1.5            # tunnel
    width_m: 3.0             # corridor and arched
    height_m: 2.6
    waypoints:               # ENU m: x, y from the take-off point, z the centreline height
      - [0.0, 0.0, 1.5]
      - [10.0, 0.0, 1.5]
    segment_scale: [1, 0.7]  # optional, per segment: a narrowed stretch
    speed_mps: 1.0           # the target's cruise speed (PX4's mission speed too)
    accel_mps2: 0.5
    corner_speed_min_frac: 0.2
    end_hold_s: 5.0          # hold on the last waypoint before landing
    draft: along             # the mean wind blows along the first segment (or "against")
    disturbance: {...}       # envs.disturbance keys over the mock's tunnel air

**The walls are drawn, not solid.** Gazebo shows them (watch image) but they have no
collision: a wall contact is judged on the true position against the route's exact walls,
as in the mock (``walls.RouteWalls.clearance`` under ``CONTACT_RADIUS_M``), and a drone that
touches a wall is scored a crash whether or not Gazebo would have bounced it. That keeps the
Gazebo score and the mock score the same measurement; solid walls built from panels would
only approximate the mock's union of pieces at the corners.

**The floor is the ground.** The drone takes off from the floor of the first segment (the
spawn point), so every centreline must sit at least one half cross-section above z = 0;
``load_route_world`` refuses a route that dips into the ground. Side branches are not in the
Gazebo set yet (the drawn main wall would cover the opening).

**The draft.** The mock turns its mean wind onto the local route direction; Gazebo's wind
plugin blows in one fixed direction, so a route world's draft blows along its first segment.
Gusts and turbulence are the same model as the mock's.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.disturbance import DisturbanceConfig, DisturbanceParams, sample_params
from envs.route_follow.config import tunnel_air
from envs.route_follow.routes import Routes, corner_speeds, plan_speeds
from sim2real.route_follow.walls import RouteWalls

HERE = Path(__file__).resolve().parent
ROUTE_DIR = HERE / "routes"
TEMPLATE = HERE.parent / "gazebo" / "worlds" / "wall_slab.sdf"
KINDS = ("tunnel", "corridor", "arched")
FLOOR_TOL_M = 0.05
PANEL_M = 0.05  # drawn wall thickness
TUNNEL_SIDES = 16


@dataclass(frozen=True)
class RouteWorld:
    name: str
    seed: int
    kind: str
    routes: Routes  # one route, the mock's own type
    end_hold_s: float
    disturbance: DisturbanceConfig
    draft_heading_deg: float  # the direction the mean wind blows toward, 0 = +x

    @property
    def walls(self) -> RouteWalls:
        return RouteWalls(self.routes)

    @property
    def world_name(self) -> str:
        return f"route_{self.name}"

    @property
    def waypoints(self) -> np.ndarray:
        return self.routes.waypoints[0]

    @property
    def speed(self) -> float:
        return float(self.routes.speed[0])

    @property
    def takeoff_alt_m(self) -> float:
        return float(self.waypoints[0, 2])

    @property
    def spawn_pose(self) -> str:
        """``PX4_GZ_MODEL_POSE``: on the floor under the first waypoint, nose along the
        first segment (yaw in radians, ENU)."""
        d = self.waypoints[1] - self.waypoints[0]
        x, y = self.waypoints[0, :2]
        return f"{x},{y},0,0,0,{math.atan2(d[1], d[0])}"

    def wind(self) -> DisturbanceParams:
        """The flight's disturbance (free air: the walls do not shelter it), drawn as the
        mock draws one episode's."""
        h = self.draft_heading_deg
        return sample_params(replace(self.disturbance, mean_heading_deg=(h, h)), 1, self.seed)

    def with_seed(self, seed: int) -> RouteWorld:
        return replace(self, seed=int(seed))


def _floats(v: Any, n: int | None = None, what: str = "") -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if n is not None and a.shape != (n,):
        raise ValueError(f"{what} needs {n} numbers")
    return a


def route_world_from_dict(d: dict[str, Any]) -> RouteWorld:
    d = dict(d)
    known = {"name", "seed", "kind", "radius_m", "width_m", "height_m", "waypoints",
             "segment_scale", "speed_mps", "accel_mps2", "corner_speed_min_frac",
             "end_hold_s", "draft", "disturbance"}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown route world keys: {sorted(unknown)}")
    kind = str(d.get("kind", "tunnel"))
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    wp = np.asarray(d["waypoints"], dtype=float)
    if wp.ndim != 2 or wp.shape[1] != 3 or len(wp) < 2:
        raise ValueError("waypoints need at least two [x, y, z] points")
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=-1)
    if np.any(seg < 0.5):
        raise ValueError("every segment must be at least 0.5 m long")
    s = len(wp) - 1
    scale = _floats(d.get("segment_scale", [1.0] * s), s, "segment_scale")
    if np.any(scale <= 0) or np.any(scale > 1):
        raise ValueError("segment_scale must lie in (0, 1]")
    speed = float(d.get("speed_mps", 1.0))
    accel = float(d.get("accel_mps2", 0.5))
    if speed <= 0 or accel <= 0:
        raise ValueError("speed_mps and accel_mps2 must be positive")
    radius = float(d.get("radius_m", 1.5))
    half_w = 0.5 * float(d.get("width_m", 3.0))
    half_h = 0.5 * float(d.get("height_m", 2.6))
    if kind == "arched":
        half_h = max(half_h, 0.6 * half_w)  # as RouteGen: walls stand above the floor
    one = lambda x: np.array([x], dtype=float)  # noqa: E731
    w = wp[None]
    corner = plan_speeds(seg[None], corner_speeds(w, one(speed),
                                                  float(d.get("corner_speed_min_frac", 0.2))),
                         one(accel))
    routes = Routes(waypoints=w, tunnel=np.array([kind == "tunnel"]), half_width=one(half_w),
                    half_height=one(half_h), radius=one(radius), speed=one(speed),
                    accel=one(accel), corner_speed=corner,
                    arched=np.array([kind == "arched"]), seg_scale=scale[None])
    # The floor is the ground: the lowest point of each segment's cross-section at or above 0.
    half = radius if kind == "tunnel" else half_h
    floor = np.minimum(wp[:-1, 2], wp[1:, 2]) - half * scale
    if np.any(floor < -FLOOR_TOL_M):
        raise ValueError(f"the route dips {-floor.min():.2f} m into the ground: raise its "
                         "centreline to at least half the cross-section above z = 0")
    first = wp[1] - wp[0]
    heading = math.degrees(math.atan2(first[1], first[0]))
    draft = str(d.get("draft", "along"))
    if draft not in ("along", "against"):
        raise ValueError("draft must be 'along' or 'against'")
    if draft == "against":
        heading += 180.0
    base = {k: getattr(tunnel_air(), k) for k in DisturbanceConfig.__dataclass_fields__}
    dist = DisturbanceConfig.from_dict({**base, **(d.get("disturbance") or {})})
    if "name" not in d:
        raise ValueError("route world needs a name")
    return RouteWorld(name=str(d["name"]), seed=int(d.get("seed", 0)), kind=kind,
                      routes=routes, end_hold_s=float(d.get("end_hold_s", 5.0)),
                      disturbance=dist, draft_heading_deg=heading)


def load_route_world(path: str | Path) -> RouteWorld:
    """Load a route world; a bare name is looked up in ``routes/``."""
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = ROUTE_DIR / f"{path}.yaml"
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: route world must be a mapping")
    return route_world_from_dict(data)


def route_world_names() -> list[str]:
    return sorted(p.stem for p in ROUTE_DIR.glob("*.yaml"))


# ---------------------------------------------------------------- the drawn walls


def _rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """SDF roll, pitch, yaw of a rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = -math.asin(max(-1.0, min(1.0, float(rot[2, 0]))))
    return (math.atan2(float(rot[2, 1]), float(rot[2, 2])), pitch,
            math.atan2(float(rot[1, 0]), float(rot[0, 0])))


def _panel(i: int, center: np.ndarray, rot: np.ndarray, size: tuple[float, float, float],
           rgb: str) -> str:
    r, p, y = _rpy(rot)
    c = center
    return (f'      <visual name="panel_{i}">\n'
            f"        <pose>{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {r:.5f} {p:.5f} {y:.5f}</pose>\n"
            f"        <geometry><box><size>{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"
            f"</size></box></geometry>\n"
            f"        <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse>"
            f"</material>\n      </visual>\n")


def _turns(wp: np.ndarray) -> np.ndarray:
    """Signed heading change at each waypoint (0 at both ends), left positive, rad."""
    d = np.diff(wp, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    t = np.zeros(len(wp))
    t[1:-1] = (np.diff(h) + math.pi) % (2 * math.pi) - math.pi
    return t


def wall_panels(rw: RouteWorld) -> list[str]:
    """The route's walls as thin visual boxes: a 16-sided tube per tunnel segment; walls,
    floor and ceiling (or an arched roof) per corridor segment, mitred at the corners, with
    end walls where the mock's boxes end."""
    w = rw.walls
    wp, turn = w.waypoints, _turns(w.waypoints)
    out: list[str] = []
    for k in range(len(w.seg_len)):
        x, y = w.seg_x[k], w.seg_y[k]
        z = np.cross(x, y)
        frame = np.stack([x, y, z], axis=1)
        length, scale = float(w.seg_len[k]), float(w.piece_scale[k])
        mid = 0.5 * (wp[k] + wp[k + 1])
        if rw.kind == "tunnel":
            r = w.radius * scale
            ext = 0.5 * r  # overlap at the joints so the tube has no gaps
            chord = 2 * (r + PANEL_M) * math.tan(math.pi / TUNNEL_SIDES)
            for j in range(TUNNEL_SIDES):
                a = 2 * math.pi * j / TUNNEL_SIDES
                radial = math.cos(a) * y + math.sin(a) * z
                tang = np.cross(radial, x)
                rot = np.stack([x, tang, radial], axis=1)
                out.append(_panel(len(out), mid + (r + 0.5 * PANEL_M) * radial, rot,
                                  (length + 2 * ext, chord, PANEL_M), "0.45 0.42 0.38"))
            continue
        hw, hh = w.half_width * scale, w.half_height * scale
        # Mitre: at a left turn the left (inner) wall stops short and the right runs on.
        t0, t1 = turn[k], turn[k + 1]
        m0, m1 = hw * math.tan(0.5 * t0), hw * math.tan(0.5 * t1)
        ends = {1: (m0, m1), -1: (-m0, -m1)}
        if k == 0:
            ends = {s: (-hw, e[1]) for s, e in ends.items()}
        if k == len(w.seg_len) - 1:
            ends = {s: (e[0], -hw) for s, e in ends.items()}
        # Arched: the walls stand from the floor (-hh) to the springline (hh - hw).
        wall_h = 2 * hh if rw.kind == "corridor" else 2 * hh - hw
        wall_zc = 0.0 if rw.kind == "corridor" else -0.5 * hw
        for side in (1, -1):
            a0, a1 = ends[side]  # extension before the start (-a0) and past the end (-a1)
            start, stop = a0, length - a1
            c = wp[k] + 0.5 * (start + stop) * x + side * (hw + 0.5 * PANEL_M) * y + wall_zc * z
            out.append(_panel(len(out), c, frame, (stop - start, PANEL_M, wall_h),
                              "0.55 0.55 0.5"))
        e0 = hw if k == 0 else hw * abs(math.tan(0.5 * t0))
        e1 = hw if k == len(w.seg_len) - 1 else hw * abs(math.tan(0.5 * t1))
        span = length + e0 + e1
        c = wp[k] + (0.5 * (length + e1 - e0)) * x
        out.append(_panel(len(out), c - (hh + 0.5 * PANEL_M) * z, frame,
                          (span, 2 * hw, PANEL_M), "0.4 0.4 0.4"))  # floor
        if rw.kind == "corridor":
            out.append(_panel(len(out), c + (hh + 0.5 * PANEL_M) * z, frame,
                              (span, 2 * hw, PANEL_M), "0.5 0.5 0.5"))
        else:
            c0 = hh - hw  # springline; half circle of radius hw above it
            n = TUNNEL_SIDES // 2
            chord = 2 * (hw + PANEL_M) * math.tan(0.5 * math.pi / n)
            for j in range(n):
                a = math.pi * (j + 0.5) / n
                radial = math.cos(a) * y + math.sin(a) * z
                rot = np.stack([x, np.cross(radial, x), radial], axis=1)
                out.append(_panel(len(out), c + c0 * z + (hw + 0.5 * PANEL_M) * radial, rot,
                                  (span, chord, PANEL_M), "0.5 0.48 0.44"))
        for end, at in ((k == 0, -hw), (k == len(w.seg_len) - 1, length + hw)):
            if end:
                out.append(_panel(len(out), wp[k] + at * x, frame,
                                  (PANEL_M, 2 * hw, 2 * hh), "0.55 0.55 0.5"))
    return out


def world_sdf(rw: RouteWorld, template: Path = TEMPLATE) -> str:
    """The route's Gazebo world: the wall-and-slab world's physics, systems and wind plugin
    with its structure replaced by the route's drawn walls (module docstring)."""
    text = template.read_text()
    text = text.replace('<world name="wall_slab">', f'<world name="{rw.world_name}">')
    model = ('    <model name="route">\n      <static>true</static>\n'
             '      <link name="walls">\n' + "".join(wall_panels(rw)) +
             "        <enable_wind>false</enable_wind>\n      </link>\n    </model>\n")
    new, n = re.subn(r'    <!-- The structure:.*?</model>\n', model, text, flags=re.S)
    if n != 1:
        raise ValueError(f"{template}: the structure model to replace was not found")
    header = (f"<!--\n  Route world {rw.name} ({rw.kind}), generated by "
              "sim2real/route_follow/world.py from\n  its route file. Walls are drawn only; "
              "contact is judged on the route's exact walls\n  (sim2real/route_follow/walls.py)."
              "\n-->\n")
    return new.replace("<sdf ", header + "<sdf ", 1)
