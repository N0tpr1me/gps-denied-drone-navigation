import math
from dataclasses import replace

import numpy as np
import pytest

from envs.disturbance import disturbance_force, wind_velocity
from sim2real.gazebo.scenario import SCENARIO_DIR, Scenario, load_scenario, sample_scenario

NAMES = ["calm_0ms", "wind_4ms", "wind_8ms",
         "wall1m_calm_0ms", "wall1m_wind_4ms", "wall1m_wind_8ms",
         "wall1m_wind_8ms_flow", "wall1m_lee_8ms_flow"]
RIG = ["wall1m_calm_0ms", "wall1m_wind_4ms", "wall1m_wind_8ms",
       "wall1m_wind_8ms_flow", "wall1m_lee_8ms_flow"]
NAMES += [f"rig_{n}" for n in RIG]
BRIDGE = ["deck_corner_8ms_flow", "deck_open_8ms_flow", "pier_lee_8ms_flow"]


@pytest.mark.parametrize("name", NAMES)
def test_shipped_scenarios_load_and_spawn_near_the_wall(name):
    sc = load_scenario(name)
    assert sc.name == name
    assert 1.0 <= sc.wall_clearance <= 3.0
    assert (SCENARIO_DIR / f"{name}.yaml").exists()


@pytest.mark.parametrize("name", RIG)
def test_rig_copies_differ_only_in_name_and_height(name):
    # docs/fan-rig.md: the rig holds at 1.5 m; everything else is the 2.5 m scenario's.
    base, rig = load_scenario(name), load_scenario(f"rig_{name}")
    assert rig.hover.altitude == 1.5 and base.hover.altitude == 2.5
    assert replace(rig, name=base.name, hover=base.hover) == base
    a, b = sample_scenario(base), sample_scenario(rig)
    for f in a.__dataclass_fields__:
        np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))


def test_calm_scenario_is_exactly_still_air():
    p = sample_scenario(load_scenario("calm_0ms"))
    for t in (0.0, 5.0, 41.0):
        assert np.array_equal(wind_velocity(p, t), np.zeros((1, 3)))


@pytest.mark.parametrize("name, speed", [("wind_4ms", 4.0), ("wind_8ms", 8.0),
                                         ("wall1m_wind_4ms", 4.0), ("wall1m_wind_8ms", 8.0)])
def test_wind_scenarios_blow_into_the_wall(name, speed):
    sc = load_scenario(name)
    p = sample_scenario(sc)
    np.testing.assert_allclose(p.mean_wind[0], (speed, 0.0, 0.0), atol=1e-12)
    assert p.has_wall[0]
    np.testing.assert_allclose(p.wall_normal[0], (-1.0, 0.0, 0.0))
    # With the mean wind toward the wall the drone at its spawn point is pushed toward +x.
    f = disturbance_force(p, 0.0, np.array([[sc.spawn.x, 0.0, 2.5]]), np.zeros((1, 3)))
    assert f[0, 0] > 0


def test_flow_scenarios_shed_only_in_the_lee():
    # The near-wall flow sees the scenario's heading: shedding needs the wind blowing away.
    into = sample_scenario(load_scenario("wall1m_wind_8ms_flow"))
    lee = sample_scenario(load_scenario("wall1m_lee_8ms_flow"))
    np.testing.assert_allclose(lee.mean_wind[0], (-8.0, 0.0, 0.0), atol=1e-12)
    assert into.shed_amp[0] == 0.0 < lee.shed_amp[0]
    assert into.wall_deflect[0] > 0.0 and lee.wall_recirc[0] > 0.0


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


def test_old_scenarios_fly_wall_slab_without_a_second_surface():
    sc = load_scenario("wall1m_wind_8ms")
    assert sc.world == "wall_slab" and sc.surface2_normal_toward is None
    assert not sample_scenario(sc).has_surface2[0]


@pytest.mark.parametrize("name", BRIDGE)
def test_bridge_scenarios_load_in_their_worlds(name):
    sc = load_scenario(name)
    assert sc.name == name and sc.world in ("bridge_deck", "bridge_pier")
    # 1 m from the abutment or pier face, or 6 m out under the deck alone.
    assert sc.wall_clearance in (1.0, 6.0)
    p = sample_scenario(sc)
    assert p.has_wall[0]
    if sc.world == "bridge_deck":
        # The deck 1.5 m over the 2.5 m hold, and the crosswind sheds from its edges.
        assert p.has_surface2[0]
        assert sc.surface2_point[2] - sc.hover.altitude == pytest.approx(1.5)
        assert p.surface2_shed_amp[0] > 0
    else:
        assert not p.has_surface2[0] and p.shed_amp[0] > 0  # the pier's wake


def test_deck_shedding_follows_the_wind_across_the_deck():
    # deck_open: the wind blows across the deck, all of it drives the shedding; the same
    # wind along the bridge's axis would shed nothing.
    sc = load_scenario("deck_open_8ms_flow")
    across = sample_scenario(sc).surface2_shed_amp[0]
    along = sample_scenario(replace(sc, wind_heading_deg=0.0)).surface2_shed_amp[0]
    assert across > 0 and along == pytest.approx(0.0, abs=1e-9)


def test_scenario_world_and_surface2_are_checked():
    base = {"name": "x", "spawn": {"x": -1.0}}
    with pytest.raises(ValueError, match="unknown world"):
        Scenario.from_dict({**base, "world": "no_such_world"})
    with pytest.raises(ValueError, match="surface2 needs"):
        Scenario.from_dict({**base, "surface2": {"point": [0, 0, 4]}})
    with pytest.raises(ValueError, match="non-zero"):
        Scenario.from_dict({**base, "surface2": {"point": [0, 0, 4],
                                                 "normal_toward": [0, 0, 0]}})


PASSES = ["wall1m_pass_calm_0ms", "wall1m_pass_wind_4ms", "wall1m_pass_wind_8ms"]


@pytest.mark.parametrize("name", PASSES)
def test_pass_scenarios_fly_the_benchmark_wind_along_the_wall(name):
    # docs/surface-passes.md "In Gazebo": the wall1m benchmark with a 5 m pass in its hold.
    sc, base = load_scenario(name), load_scenario(name.replace("_pass", ""))
    p = sc.surface_pass
    assert p is not None and base.surface_pass is None
    assert (p.length_m, p.speed_mps, p.start_s, p.up) == (5.0, 0.5, 3.0, False)
    assert sc.wall_clearance == base.wall_clearance == 1.0
    assert sc.disturbance.mean_speed == base.disturbance.mean_speed
    assert sc.disturbance.gust_speed == base.disturbance.gust_speed
    # The pass runs to the right of the nose (-y here) and stays on the 20 m wall.
    assert sc.spawn.yaw_deg == 0.0 and abs(sc.spawn.y - p.length_m) < 10.0
    # The hold covers the settle, the pass and a hold at its end; the gusts land in it.
    cruise_s = (p.length_m - p.speed_mps ** 2 / 0.5) / p.speed_mps
    flown = p.start_s + 2 * p.speed_mps / 0.5 + cruise_s
    assert sc.hover.hover_s >= flown + 10.0
    for start in list(sc.disturbance.gust_start)[:sc.disturbance.num_gusts]:
        assert p.engage_at_s + 2.0 < start < p.engage_at_s + sc.hover.hover_s - 3.0
    assert sc.surface_pass.module_params() == [
        "SH_PASS_LEN=5.0", "SH_PASS_DIR=0", "SH_PASS_SPD=0.5", "SH_PASS_T0=3.0"]
    assert sc.surface_pass.flight_args()[-2:] == ["--engage-at-boot-s", str(p.engage_at_s)]


def test_pass_block_is_checked_against_the_module_limits():
    base = {"name": "x", "pass": {"length_m": 5.0}}
    assert Scenario.from_dict(base).surface_pass.engage_at_s == 0.0
    assert "--engage-at-boot-s" not in Scenario.from_dict(base).surface_pass.flight_args()
    up = Scenario.from_dict({"name": "x", "pass": {"length_m": -2.0, "up": True}}).surface_pass
    assert "--pass-up" in up.flight_args() and "SH_PASS_DIR=1" in up.module_params()
    for bad in ({"length_m": 0.0}, {"length_m": 12.0}, {"speed_mps": 2.0}, {"start_s": -1.0},
                {"engage_at_s": -5.0}):
        with pytest.raises(ValueError):
            Scenario.from_dict({"name": "x", "pass": {"length_m": 5.0, **bad}})
    with pytest.raises(TypeError):
        Scenario.from_dict({"name": "x", "pass": {"lenght_m": 5.0}})
