"""Route sampling (NumPy) and the route geometry on tensors: walls, beams, target, frame."""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.route_follow.config import RouteFollowConfig, RouteGen
from envs.route_follow.routes import sample_routes, straight_route, turn_route

torch = pytest.importorskip("torch")

from envs.route_follow.geometry import RouteGeometry  # noqa: E402


def beams():
    # left, right, up, down, forward
    return torch.tensor([[[0.0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0]]])


def test_routes_fit_the_episode_and_are_seeded():
    cfg = RouteGen()
    a = sample_routes(cfg, 64, 30.0, np.random.default_rng(1))
    b = sample_routes(cfg, 64, 30.0, np.random.default_rng(1))
    np.testing.assert_array_equal(a.waypoints, b.waypoints)
    assert a.waypoints.shape == (64, cfg.segments + 1, 3)
    # The target flies the whole route and stops with end_hold_s to spare.
    np.testing.assert_allclose(a.duration, 30.0 - cfg.end_hold_s, rtol=0.01)
    first = a.waypoints[:, 1] - a.waypoints[:, 0]
    assert np.all(first[:, 0] > 0) and np.allclose(first[:, 1], 0.0)  # starts along +x
    assert a.tunnel.any() and (~a.tunnel).any()
    assert np.all(a.corner_speed[:, [0, -1]] == 0.0)
    assert np.all(a.corner_speed <= a.speed[:, None] + 1e-12)


def test_corner_speed_drops_with_the_turn():
    sharp = turn_route(10.0, 10.0, 90.0, tunnel=False, speed=1.0)
    gentle = turn_route(10.0, 10.0, 45.0, tunnel=False, speed=1.0)
    straight = turn_route(10.0, 10.0, 0.0, tunnel=False, speed=1.0)
    assert sharp.corner_speed[0, 1] == pytest.approx(0.2)
    assert gentle.corner_speed[0, 1] == pytest.approx(0.6)
    assert straight.corner_speed[0, 1] == pytest.approx(1.0)
    # A corner the target cannot reach from rest within its segment is lowered.
    short = turn_route(0.25, 10.0, 0.0, tunnel=False, speed=1.0, accel=0.5)
    assert short.corner_speed[0, 1] == pytest.approx(0.5)  # sqrt(2 * 0.5 * 0.25)


def test_profile_length_triangle_and_trapezoid():
    # Plenty of room: 1 m/s cruise with 2 s of ramps each way, 8 m in 10 s.
    r = straight_route(8.0, tunnel=False, speed=1.0, accel=0.5)
    assert r.duration[0] == pytest.approx(10.0)
    # Too short to reach 1 m/s: a triangle peaking at 0.5 m/s, 0.5 m in 2 s.
    r = straight_route(0.5, tunnel=False, speed=1.0, accel=0.5)
    assert r.duration[0] == pytest.approx(2.0)


def test_short_routes_turn_at_fewer_joints():
    cfg = RouteGen(speed_mps=(0.5, 0.5))  # 0.5 m/s for 5 s of flight: a 2 m route
    r = sample_routes(cfg, 32, 10.0, np.random.default_rng(2))
    d = np.diff(r.waypoints, axis=1)
    heading = np.arctan2(d[..., 1], d[..., 0])
    np.testing.assert_allclose(heading, 0.0, atol=1e-9)  # one straight piece


def test_corridor_clearance_and_beams_on_a_straight():
    g = RouteGeometry.from_routes(straight_route(10.0, tunnel=False, half_width=1.0,
                                                 half_height=1.5))
    p = torch.tensor([[5.0, 0.3, 2.0]])
    assert float(g.clearance(p)) == pytest.approx(0.7, abs=1e-5)
    r = g.trace(p, beams(), 20.0)[0]
    # The end wall sits half_width past the last waypoint.
    np.testing.assert_allclose(r.numpy(), [0.7, 1.3, 1.5, 1.5, 6.0], atol=1e-3)
    outside = torch.tensor([[5.0, 1.2, 2.0]])
    assert float(g.clearance(outside)) < 0
    assert float(g.trace(outside, beams(), 20.0).abs().max()) == 0.0


def test_tunnel_clearance_and_beams():
    g = RouteGeometry.from_routes(straight_route(10.0, tunnel=True, radius=1.5))
    p = torch.tensor([[5.0, 0.5, 2.0]])
    assert float(g.clearance(p)) == pytest.approx(1.0, abs=1e-5)
    r = g.trace(p, beams(), 20.0)[0]
    up = math.sqrt(1.5**2 - 0.5**2)
    np.testing.assert_allclose(r[:4].numpy(), [1.0, 2.0, up, up], atol=2e-3)
    assert float(r[4]) == pytest.approx(5.0 + up, abs=5e-3)  # the round end cap


def test_beams_past_max_range_read_inf():
    g = RouteGeometry.from_routes(straight_route(30.0, tunnel=False))
    r = g.trace(torch.tensor([[1.0, 0.0, 2.0]]), beams(), 8.0)[0]
    assert math.isinf(float(r[4])) and math.isfinite(float(r[0]))


def test_the_outside_of_a_turn_is_filled_and_the_side_branch_is_open():
    g = RouteGeometry.from_routes(turn_route(6.0, 6.0, 90.0, tunnel=False, half_width=1.0))
    # At the joint: forward hits the outer wall 1 m on, left looks down the new branch.
    r = g.trace(torch.tensor([[6.0, 0.0, 2.0]]), beams(), 20.0)[0]
    assert float(r[4]) == pytest.approx(1.0, abs=1e-3)
    assert float(r[0]) == pytest.approx(7.0, abs=1e-2)
    assert float(r[1]) == pytest.approx(1.0, abs=1e-3)
    # The outer corner (6.9, -0.9) is inside the extended box: free air.
    assert float(g.clearance(torch.tensor([[6.9, -0.9, 2.0]]))) > 0


def test_clearance_never_exceeds_the_true_distance_near_an_inner_corner():
    g = RouteGeometry.from_routes(turn_route(6.0, 6.0, 90.0, tunnel=False, half_width=1.0))
    # Inner corner at (5, 1): a point 0.1 m in from both walls is 0.14 m from the corner.
    c = float(g.clearance(torch.tensor([[4.9, 0.9, 2.0]])))
    assert 0 < c <= math.hypot(0.1, 0.1) + 1e-6


def test_target_profile_and_points_along_the_route():
    g = RouteGeometry.from_routes(straight_route(10.0, tunnel=False, speed=1.0, accel=0.5))
    sv = [g.target(torch.tensor([t])) for t in (0.0, 1.0, 2.0, 6.0, 12.0, 30.0)]
    s, v = np.array([float(x[0]) for x in sv]), np.array([float(x[1]) for x in sv])
    np.testing.assert_allclose(s, [0.0, 0.25, 1.0, 5.0, 10.0, 10.0], atol=1e-5)
    np.testing.assert_allclose(v, [0.0, 0.5, 1.0, 1.0, 0.0, 0.0], atol=1e-5)
    # Through a 90 degree corner the target slows to a fifth of cruise, and the torch plan
    # takes as long as the NumPy one.
    r = turn_route(6.0, 6.0, 90.0, tunnel=False, speed=1.0, accel=0.5)
    gt = RouteGeometry.from_routes(r)
    ts = torch.linspace(0.0, float(r.duration[0]) + 1.0, 2001)
    s_all, v_all = [], []
    for tt in ts.tolist():
        sv, vv = gt.target(torch.tensor([tt]))
        s_all.append(float(sv))
        v_all.append(float(vv))
    s_all, v_all = np.array(s_all), np.array(v_all)
    near_corner = np.abs(s_all - 6.0) < 0.02
    assert near_corner.any() and v_all[near_corner].max() < 0.25
    assert np.all(np.diff(s_all) >= -1e-6) and s_all[-1] == pytest.approx(12.0)
    reached = ts.numpy()[np.argmax(s_all >= 12.0 - 1e-5)]
    assert reached == pytest.approx(float(r.duration[0]), abs=0.01)
    dt = float(ts[1] - ts[0])
    assert np.abs(np.diff(v_all)).max() <= 0.5 * dt + 1e-4  # speed changes at accel only
    g2 = RouteGeometry.from_routes(turn_route(6.0, 4.0, 90.0, tunnel=True))
    p = g2.point_at(torch.tensor([[2.0, 8.0, 50.0]]))[0]
    np.testing.assert_allclose(p.numpy(), [[2.0, 0, 2], [6.0, 2.0, 2], [6.0, 4.0, 2]], atol=1e-5)
    q = torch.tensor([[6.5, 2.0, 2.0]])
    assert float(g2.distance_to_route(q)) == pytest.approx(0.5)
    assert float(g2.progress(q)) == pytest.approx(8.0)


def test_route_frame_turns_smoothly_through_a_corner():
    g = RouteGeometry.from_routes(turn_route(6.0, 6.0, 90.0, tunnel=False))
    yaws = []
    for s in np.linspace(4.0, 8.0, 161):
        f = g.route_frame(torch.tensor([float(s)]), 1.0)[0]
        assert torch.allclose(f.T @ f, torch.eye(3), atol=1e-5)  # orthonormal
        assert abs(float(f[2, 1])) < 1e-6  # y stays horizontal
        yaws.append(math.atan2(float(f[1, 0]), float(f[0, 0])))
    assert yaws[0] == pytest.approx(0.0, abs=1e-6)
    assert yaws[-1] == pytest.approx(math.pi / 2, abs=1e-5)
    assert np.abs(np.diff(yaws)).max() < math.radians(5)  # no jump at the joint
    # On the last metre the frame keeps the last stretch's direction.
    f_end = g.route_frame(torch.tensor([12.0]), 1.0)[0]
    assert float(f_end[1, 0]) == pytest.approx(1.0, abs=1e-5)


def test_config_rejects_a_corridor_too_narrow_for_the_drone():
    with pytest.raises(ValueError, match="no room"):
        RouteFollowConfig.from_dict({"contact_radius_m": 0.9})
    with pytest.raises(ValueError, match="unknown"):
        RouteFollowConfig.from_dict({"route": {"segmnts": 3}})
