"""Route worlds: the shipped files load, the world SDF is well formed, the mission is the
route, and a flight is scored the way the mock scores an episode."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sim2real.estimate_bridge.config import load_profile
from sim2real.route_follow.run_mission import (
    ORIGIN,
    mission_json,
    mission_window,
    score_route,
)
from sim2real.route_follow.world import (
    TEMPLATE,
    load_route_world,
    route_world_from_dict,
    route_world_names,
    world_sdf,
)

NAMES = route_world_names()


def test_the_benchmark_set() -> None:
    assert set(NAMES) >= {"tunnel_bend", "corridor_ramp", "pipe_narrow", "drift_arched"}


@pytest.mark.parametrize("name", NAMES)
def test_route_world_loads_and_draws(name: str) -> None:
    rw = load_route_world(name)
    w = rw.walls
    # The take-off point is on the floor of the first segment, the first waypoint inside.
    assert w.clearance(rw.waypoints[0]) > 0.4
    assert rw.walls.duration > 10.0
    root = ET.fromstring(world_sdf(rw))
    world = root.find("world")
    assert world.get("name") == f"route_{name}"
    names = [p.get("name") for p in world.findall("plugin")]
    assert "structure_hold::WindAdapter" in names  # the wind plugin stays
    assert world.find("model[@name='structure']") is None
    route = world.find("model[@name='route']")
    visuals = route.findall("link/visual")
    assert len(visuals) >= 4 * len(w.seg_len)
    assert route.find("link/collision") is None  # drawn only (world.py docstring)
    for v in visuals:
        pose = [float(x) for x in v.find("pose").text.split()]
        assert all(math.isfinite(x) for x in pose)


def test_tunnel_panels_sit_on_the_wall() -> None:
    """Every tube panel's centre is the radius (plus half a panel) from its segment's axis,
    i.e. on the wall the scorer uses."""
    rw = load_route_world("tunnel_bend")
    w = rw.walls
    root = ET.fromstring(world_sdf(rw))
    centres = np.array([[float(x) for x in v.find("pose").text.split()[:3]]
                        for v in root.iter("visual") if v.get("name", "").startswith("panel")])
    clear = w.clearance(centres)
    assert np.all(clear <= 0.0 + 1e-6)  # never inside the free air
    assert np.median(clear) == pytest.approx(-0.025, abs=0.01)


def test_template_origin_is_the_mission_origin() -> None:
    text = TEMPLATE.read_text()
    lat = float(re.search(r"<latitude_deg>\s*([0-9.]+)", text).group(1))
    lon = float(re.search(r"<longitude_deg>\s*([0-9.]+)", text).group(1))
    assert (lat, lon) == ORIGIN[:2]


def test_mission_is_the_route() -> None:
    rw = load_route_world("corridor_ramp")
    m = mission_json(rw)
    assert len(m["waypoints"]) == len(rw.waypoints)
    for wp, p in zip(m["waypoints"], rw.waypoints, strict=True):
        assert wp["rel_alt_m"] == pytest.approx(p[2])
        # metres per degree near 47.4 deg N
        assert (wp["lat"] - ORIGIN[0]) * 111_200 == pytest.approx(p[1], abs=0.05)
        assert (wp["lon"] - ORIGIN[1]) * 75_300 == pytest.approx(p[0], abs=0.1)
    assert m["takeoff_alt_m"] == pytest.approx(1.3)
    assert m["planned_s"] == pytest.approx(rw.walls.duration)


def test_rejects_a_route_in_the_ground() -> None:
    with pytest.raises(ValueError, match="ground"):
        route_world_from_dict({"name": "x", "kind": "tunnel", "radius_m": 1.5,
                               "waypoints": [[0, 0, 1.0], [10, 0, 1.0]]})
    with pytest.raises(ValueError, match="unknown"):
        route_world_from_dict({"name": "x", "branches": 2,
                               "waypoints": [[0, 0, 2.0], [10, 0, 2.0]]})


def test_mission_window() -> None:
    seq = [{"t": 1.0, "mode": "Hold"}, {"t": 10.0, "mode": "Takeoff"},
           {"t": 20.0, "mode": "Mission"}, {"t": 56.0, "mode": "Hold"},
           {"t": 61.5, "mode": "Land"}]
    assert mission_window({"sequence": seq}) == (20.0, 61.5)  # the end hold is scored
    assert mission_window({"sequence": seq[:2]}) is None
    failed = seq[:3] + [{"t": 30.0, "mode": "Return"}]
    assert mission_window({"sequence": failed}) == (20.0, 30.0)


def test_score_route_on_the_target_is_perfect() -> None:
    rw = load_route_world("tunnel_bend")
    w = rw.walls
    t = np.arange(0.0, w.duration + rw.end_hold_s, 0.02) + 30.0
    pos = w.point_at(w.target_s(t - 30.0))
    m = score_route(t, pos, rw, (30.0, float(t[-1])), mission_finished=True)
    assert m["cross_track_max_m"] < 1e-9 and m["track_p95_m"] < 1e-9
    assert m["min_clearance_m"] == pytest.approx(1.5)
    assert m["completed"] and not m["crash"]
    assert m["time_s"] == pytest.approx(w.duration, abs=2.0)


def test_score_route_judges_contact_on_truth() -> None:
    rw = load_route_world("pipe_narrow")
    w = rw.walls
    t = np.arange(0.0, w.duration + 5.0, 0.02)
    pos = w.point_at(w.target_s(t))
    pos[len(t) // 2:, 1] += 0.6  # 0.6 m sideways in a 0.9 m pipe: 0.3 m from the wall
    m = score_route(t, pos, rw, (0.0, float(t[-1])), mission_finished=True)
    assert m["contact"] and m["crash"] and not m["completed"]
    assert m["min_clearance_m"] < 0.4


@pytest.mark.parametrize("name", ["route_clean", "route_vio", "route_drift", "route_degrade"])
def test_route_profiles(name: str) -> None:
    prof = load_profile(name).with_anchor_altitude(1.5)
    assert prof.gps is None and prof.rangefinder is None
    if name != "route_clean":
        assert prof.fault_anchor == "altitude"
        assert prof.fault_anchor_alt_m == pytest.approx(1.2)
        assert prof.estimate.bias_max_m >= 0.5  # at least the mock's tunnel drift


def test_drifted_end_is_not_a_crash() -> None:
    """Finishing the mission 0.6 m off the last waypoint (the estimate drifted) is not
    completed, but not a crash either; a mission PX4 never finished is a crash."""
    rw = load_route_world("tunnel_bend")
    w = rw.walls
    t = np.arange(0.0, w.duration + 5.0, 0.02)
    pos = w.point_at(w.target_s(t))
    pos[:, 1] += np.clip(t / t[-1], 0, 1) * 0.6
    m = score_route(t, pos, rw, (0.0, float(t[-1])), mission_finished=True)
    assert not m["completed"] and not m["crash"] and not m["contact"]
    m = score_route(t, pos, rw, (0.0, float(t[-1])), mission_finished=False)
    assert m["crash"]
