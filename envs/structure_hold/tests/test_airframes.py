"""Named airframes: the gz x500's derived numbers, the preset merge, and how the env flies it."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig  # noqa: E402
from envs.sims.torch_quad import GRAVITY, quat_rotate  # noqa: E402
from envs.structure_hold import NUM_ACTIONS, StructureHoldConfig, StructureHoldEnv  # noqa: E402
from envs.structure_hold.airframes import airframe_config, gz_x500, px4_allocation  # noqa: E402
from envs.structure_hold.config import Actuation, AirframeRandomization  # noqa: E402


def calm(airframe: str, **kw) -> StructureHoldConfig:
    return StructureHoldConfig.from_dict({
        "airframe": airframe,
        "disturbance": asdict(DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0,
                                                turb_sigma=(0.0, 0.0))),
        "estimate": asdict(EstimateCorruptionConfig.clean()),
        "start_offset_std_m": 0.0, "start_yaw_std_rad": 0.0, **kw,
    })


def hover_action(env: StructureHoldEnv) -> torch.Tensor:
    a = torch.zeros(env.num_envs, NUM_ACTIONS)
    a[:, 3] = 2.0 * env.cfg.quad.hover_thrust / env.cfg.quad.max_thrust - 1.0
    return a


def test_px4_allocation_of_a_symmetric_quad_is_px4s_textbook_mix():
    a = 0.2
    mix = px4_allocation(((a, a, 0.05), (-a, -a, 0.05), (a, -a, -0.05), (-a, a, -0.05)))
    np.testing.assert_allclose(np.abs(mix[:, :2]), np.sqrt(0.5), rtol=1e-9)
    np.testing.assert_allclose(np.abs(mix[:, 2]), 1.0, rtol=1e-9)
    np.testing.assert_allclose(mix[:, 5], -1.0, rtol=1e-9)


def test_gz_x500_numbers_follow_from_its_sources():
    x = gz_x500()
    q, act = x["quad"], x["actuation"]
    assert q["mass"] == pytest.approx(2.0643, abs=1e-4)
    assert q["max_thrust"] == pytest.approx(34.194, abs=1e-3)
    assert q["max_thrust"] / (q["mass"] * GRAVITY) == pytest.approx(1.69, abs=0.01)
    assert q["inertia"] == pytest.approx((0.02367, 0.02367, 0.04400), abs=1e-5)
    assert q["motor_tau"] == pytest.approx(0.01875)
    assert act["esc_min_frac"] == pytest.approx(0.15)
    assert act["thrust_cmd_range"] == (0.08, 0.9)
    # PX4 puts the pitch arm at 0.13 m, the rotors sit at 0.174 m: pitch is the stronger axis.
    assert act["torque_per_unit"] == pytest.approx((3.41, 5.50, 0.683), abs=0.01)
    assert q["max_torque"] == pytest.approx((2.11, 2.11, 0.185), abs=0.01)
    assert x["rate_pid"]["p"] == (0.1466, 0.1102, 0.20)  # eval/baselines/px4_tuned_v1.params
    assert x["rate_lag_s"] == (0.0, 0.0)
    # Hover command where PX4's hover thrust estimate converges, and the curve's slope there.
    a = Actuation(**act)
    u_h = a.command_for_fraction(q["mass"] * GRAVITY / q["max_thrust"])
    assert u_h == pytest.approx(0.729, abs=1e-3)
    slope = (a.thrust_fraction(u_h + 1e-6) - a.thrust_fraction(u_h)) / 1e-6
    assert slope / (a.thrust_fraction(u_h) / u_h) == pytest.approx(1.61, abs=0.01)


def test_preset_merges_under_the_configs_own_values():
    cfg = StructureHoldConfig.from_dict({"airframe": "gz_x500", "quad": {"motor_tau": 0.03}})
    assert cfg.quad.motor_tau == 0.03
    assert cfg.quad.mass == pytest.approx(2.0643, abs=1e-4)
    assert cfg.rate_lag_s == (0.0, 0.0)
    assert cfg.actuation.esc_min_frac == pytest.approx(0.15)
    assert StructureHoldConfig.from_dict({}) == StructureHoldConfig()
    assert StructureHoldConfig.from_dict({"airframe": "placeholder"}) == StructureHoldConfig()
    with pytest.raises(ValueError, match="unknown airframe"):
        StructureHoldConfig.from_dict({"airframe": "x501"})
    # policy.json stores asdict(env_cfg); reading it back gives the same config.
    x500 = StructureHoldConfig.from_dict({"airframe": "gz_x500"})
    assert StructureHoldConfig.from_dict(asdict(x500)) == x500
    assert airframe_config("placeholder") == {}


def test_x500_hovers_on_the_hover_action_through_the_module_mapping():
    env = StructureHoldEnv(calm("gz_x500"), num_envs=4, seed=0)
    env.reset()
    z0 = env.sim.state.pos[:, 2].clone()
    for _ in range(100):
        env.step(hover_action(env))
    assert (env.sim.state.pos[:, 2] - z0).abs().max() < 0.01
    # A hover-thrust estimate 10 % high: the same action climbs.
    cfg = calm("gz_x500", airframe_rand={"hover_thrust_error": [0.1, 0.1]})
    env = StructureHoldEnv(cfg, num_envs=4, seed=0)
    env.reset()
    for _ in range(100):
        env.step(hover_action(env))
    assert (env.sim.state.pos[:, 2] - z0).min() > 0.2


def test_x500_thrust_is_steeper_than_the_linear_mapping_and_clamped_like_the_module():
    env = StructureHoldEnv(calm("gz_x500"), num_envs=3, seed=0)
    env.reset()
    a = hover_action(env)
    a[:, 3] += torch.tensor([0.0, 0.05, 1.0])
    q = env.cfg.quad
    for _ in range(50):  # long past the motor lag
        env.step(a)
    thrust = env.sim.thrust
    lin = 0.5 * (a[:, 3] + 1.0) * q.max_thrust  # what a thrust-linear airframe would give
    assert thrust[0] == pytest.approx(q.hover_thrust, rel=1e-4)
    assert (thrust[1] - thrust[0]) / (lin[1] - lin[0]) == pytest.approx(1.61, abs=0.03)
    # Full action is clamped at the module's 0.9 command: (0.15 + 0.85 * 0.9)^2 of max.
    assert thrust[2] == pytest.approx(q.max_thrust * (0.15 + 0.85 * 0.9) ** 2, rel=1e-4)


def rise_time_63(airframe: str, axis: int) -> float:
    env = StructureHoldEnv(calm(airframe, rate_lag_s=0.15) if airframe == "placeholder"
                           else calm(airframe), num_envs=1, seed=0)
    env.reset()
    a = hover_action(env)
    a[:, axis] = 0.25
    target = 0.25 * env.cfg.max_rate_rad_s
    for k in range(1, 100):
        env.step(a)
        if env.sim.state.ang_vel[0, axis] >= 0.632 * target:
            return k * env.cfg.dt
    return float("inf")


def test_x500_rate_loop_answers_three_to_four_times_faster_than_the_placeholder():
    # 63 % of a roll step: per-rotor model of PX4 at 250 Hz on the x500 gives 64 ms.
    assert 0.05 <= rise_time_63("gz_x500", 0) <= 0.08
    assert 0.05 <= rise_time_63("gz_x500", 1) <= 0.08
    assert rise_time_63("placeholder", 0) >= 3 * rise_time_63("gz_x500", 0)


def test_airframe_randomisation_on_the_x500_moves_the_module_mapping():
    ar = AirframeRandomization(mass_scale=(1.2, 1.2), thrust_to_weight_scale=(0.9, 0.9),
                               hover_thrust_error=(0.0, 0.0))
    cfg = calm("gz_x500", airframe_rand=asdict(ar))
    env = StructureHoldEnv(cfg, num_envs=2, seed=0)
    env.reset()
    z0 = env.sim.state.pos[:, 2].clone()
    for _ in range(100):
        env.step(hover_action(env))  # the nominal hover action still hovers the heavier one
    assert (env.sim.state.pos[:, 2] - z0).abs().max() < 0.01
    assert env.max_thrust_act[0].item() == pytest.approx(cfg.quad.max_thrust * 1.2 * 0.9,
                                                         rel=1e-6)


def test_gz_x500_sitl_adds_the_px4_and_bridge_pieces_and_round_trips():
    cfg = StructureHoldConfig.from_dict({"airframe": "gz_x500_sitl"})
    x500 = StructureHoldConfig.from_dict({"airframe": "gz_x500"})
    assert cfg.quad == x500.quad and cfg.actuation == x500.actuation
    assert cfg.rate_pid.p == x500.rate_pid.p
    assert cfg.rate_pid.yaw_torque_cutoff_hz == 2.0  # MC_YAW_TQ_CUTOFF
    assert cfg.rate_pid.gyro_cutoff_hz == 40.0  # IMU_GYRO_CUTOFF
    assert cfg.dt == 0.012  # the module's 83 Hz tick
    assert cfg.vision_offset_down_m == 0.24 and cfg.fusion.enabled
    assert StructureHoldConfig.from_dict(asdict(cfg)) == cfg
    # Every other airframe keeps the new pieces off, so earlier results stay reproducible.
    for name in ("placeholder", "gz_x500"):
        c = StructureHoldConfig.from_dict({"airframe": name})
        assert c.vision_offset_down_m == 0.0 and not c.fusion.enabled
        assert c.rate_pid.yaw_torque_cutoff_hz == 0.0 and c.rate_pid.gyro_cutoff_hz == 0.0


def test_vision_offset_moves_the_fix_only_when_the_vehicle_tilts():
    env = StructureHoldEnv(calm("gz_x500", vision_offset_down_m=0.24), num_envs=2, seed=0)
    env.reset()
    env.step(hover_action(env))
    assert torch.allclose(env.estimate_pos, env.sim.state.pos, atol=1e-6)  # level: no offset
    a = hover_action(env)
    a[0, 0] = 0.5  # roll env 0
    for _ in range(15):
        env.step(a)
    s = env.sim.state
    r = torch.tensor([0.0, 0.0, -0.24]).expand(2, 3)
    expect = s.pos + quat_rotate(s.quat, r) - r
    assert torch.allclose(env.estimate_pos, expect, atol=1e-5)
    # Rolling right swings the point below the CG left: the fix sees a sideways move.
    assert (env.estimate_pos - s.pos)[0].norm() > 0.02
    assert (env.estimate_pos - s.pos)[1].norm() < 1e-4


def test_gz_x500_sitl_hovers_and_its_estimate_follows_the_truth():
    env = StructureHoldEnv(calm("gz_x500_sitl"), num_envs=3, seed=0)
    env.reset()
    z0 = env.sim.state.pos[:, 2].clone()
    for _ in range(int(2.0 / env.cfg.dt)):
        env.step(hover_action(env))
    assert (env.sim.state.pos[:, 2] - z0).abs().max() < 0.02
    assert (env.estimate_pos - env.sim.state.pos).norm(dim=1).max() < 0.02
