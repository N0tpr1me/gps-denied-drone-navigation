import math

import numpy as np
import pytest

from envs.disturbance import disturbance_force, wind_velocity
from sim2real.gazebo.scenario import SCENARIO_DIR, Scenario, load_scenario, sample_scenario

NAMES = ["calm_0ms", "wind_4ms", "wind_8ms"]


@pytest.mark.parametrize("name", NAMES)
def test_shipped_scenarios_load_and_spawn_near_the_wall(name):
    sc = load_scenario(name)
    assert sc.name == name
    assert 1.0 <= sc.wall_clearance <= 3.0
    assert (SCENARIO_DIR / f"{name}.yaml").exists()


def test_calm_scenario_is_exactly_still_air():
    p = sample_scenario(load_scenario("calm_0ms"))
    for t in (0.0, 5.0, 41.0):
        assert np.array_equal(wind_velocity(p, t), np.zeros((1, 3)))


@pytest.mark.parametrize("name, speed", [("wind_4ms", 4.0), ("wind_8ms", 8.0)])
def test_wind_scenarios_blow_into_the_wall(name, speed):
    sc = load_scenario(name)
    p = sample_scenario(sc)
    np.testing.assert_allclose(p.mean_wind[0], (speed, 0.0, 0.0), atol=1e-12)
    assert p.has_wall[0]
    np.testing.assert_allclose(p.wall_normal[0], (-1.0, 0.0, 0.0))
    # With the mean wind toward the wall the drone at its spawn point is pushed toward +x.
    f = disturbance_force(p, 0.0, np.array([[sc.spawn.x, 0.0, 2.5]]), np.zeros((1, 3)))
    assert f[0, 0] > 0


def test_same_seed_same_params():
    a, b = sample_scenario(load_scenario("wind_8ms")), sample_scenario(load_scenario("wind_8ms"))
    for name in a.__dataclass_fields__:
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))
    other = sample_scenario(Scenario.from_dict({"name": "x", "seed": 2,
                                                "disturbance": {"mean_speed": 8.0}}))
    assert not np.array_equal(other.turb_phase, a.turb_phase)


def test_heading_sets_direction_not_speed():
    sc = Scenario.from_dict({"name": "h", "seed": 3, "wind_heading_deg": 90.0,
                             "disturbance": {"mean_speed": 3.0}})
    p = sample_scenario(sc)
    np.testing.assert_allclose(p.mean_wind[0], (0.0, 3.0, 0.0), atol=1e-12)


def test_px4_pose_uses_radians():
    sc = Scenario.from_dict({"name": "p", "spawn": {"x": -1.5, "yaw_deg": 90.0}})
    x, y, z, r, pch, yaw = sc.spawn.px4_pose().split(",")
    assert float(x) == -1.5 and float(yaw) == pytest.approx(math.pi / 2)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown scenario keys"):
        Scenario.from_dict({"name": "x", "wind": {}})
    with pytest.raises(ValueError, match="wall"):
        Scenario.from_dict({"name": "x", "wall": {"normal_toward": [0, 0, 0]}})
