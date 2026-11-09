"""The NumPy route geometry is the mock's torch geometry, on random routes of every world."""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.route_follow.config import RouteGen
from envs.route_follow.routes import sample_routes, straight_route, turn_route
from sim2real.route_follow.walls import RouteWalls

torch = pytest.importorskip("torch")
from envs.route_follow.geometry import RouteGeometry  # noqa: E402

WORLDS = {
    "plain": RouteGen(),
    "arched": RouteGen(tunnel_prob=0.0, arch_prob=1.0),
    "pinched": RouteGen(pinch_prob=0.5),
    "branches": RouteGen(branches=2, branch_prob=0.7),
    "pipes": RouteGen(tunnel_prob=1.0, tunnel_radius_m=(0.8, 1.1), climb_deg=(-20.0, 20.0)),
}


@pytest.mark.parametrize("world", sorted(WORLDS))
def test_matches_torch_geometry(world: str) -> None:
    routes = sample_routes(WORLDS[world], 12, 30.0, np.random.default_rng(5))
    geo = RouteGeometry.from_routes(routes, dtype=torch.float64)
    rng = np.random.default_rng(1)
    for i in range(12):
        w = RouteWalls(routes, i)
        s = rng.uniform(0, w.length, 64)
        pts = w.point_at(s) + rng.normal(0, 1.2, (64, 3))
        p = torch.as_tensor(pts)[None].expand(12, -1, -1).clone()
        want = geo.clearance(p)[i].numpy()
        np.testing.assert_allclose(w.clearance(pts), want, atol=1e-9)
        sel = torch.zeros(12, 3, dtype=torch.float64)
        for k in range(0, 64, 16):
            sel[i] = torch.as_tensor(pts[k])
            np.testing.assert_allclose(w.distance_to_route(pts[k]),
                                       geo.distance_to_route(sel)[i].item(), atol=1e-9)
            np.testing.assert_allclose(w.progress(pts[k]), geo.progress(sel)[i].item(),
                                       atol=1e-9)
        t = torch.as_tensor(rng.uniform(0, 35, 12))
        s_torch, _ = geo.target(t)
        np.testing.assert_allclose(w.target_s(t[i].item()), s_torch[i].item(), atol=1e-9)
        assert w.duration == pytest.approx(float(routes.duration[i]))


def test_beams_match_torch() -> None:
    routes = sample_routes(WORLDS["branches"], 4, 30.0, np.random.default_rng(2))
    geo = RouteGeometry.from_routes(routes, dtype=torch.float64)
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    for i in range(4):
        w = RouteWalls(routes, i)
        origin = w.point_at(0.4 * w.length)
        o = torch.zeros(4, 3, dtype=torch.float64)
        o[i] = torch.as_tensor(origin)
        d = torch.as_tensor(dirs)[None].expand(4, -1, -1)
        want = geo.trace(o, d, 8.0)[i].numpy()
        np.testing.assert_allclose(w.trace(origin, dirs, 8.0), want, atol=1e-9)


def test_straight_tunnel_by_hand() -> None:
    w = RouteWalls(straight_route(10.0, tunnel=True, radius=1.5, height=2.0))
    assert w.clearance(np.array([5.0, 0.0, 2.0])) == pytest.approx(1.5)
    assert w.clearance(np.array([5.0, 1.0, 2.0])) == pytest.approx(0.5)
    assert w.distance_to_route(np.array([5.0, 0.3, 2.4])) == pytest.approx(0.5)
    left = w.trace(np.array([5.0, 0.0, 2.0]), np.array([[0.0, 1.0, 0.0]]), 8.0)
    assert left[0] == pytest.approx(1.5, abs=2e-3)
    assert w.target_s(0.0) == 0.0 and w.target_s(1e3) == pytest.approx(10.0)


def test_corner_target_slows() -> None:
    w = RouteWalls(turn_route(8.0, 8.0, 90.0, tunnel=False, speed=1.5))
    t = np.linspace(0, w.duration, 400)
    s = w.target_s(t)
    v = np.gradient(s, t)
    at_corner = np.argmin(np.abs(s - 8.0))
    assert v[at_corner] == pytest.approx(0.3, abs=0.05)  # a fifth of cruise at 90 degrees
    assert math.isclose(s[-1], 16.0, abs_tol=1e-6)
