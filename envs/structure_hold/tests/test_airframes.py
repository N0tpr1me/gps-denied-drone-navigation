"""Named airframes: the gz x500's derived numbers, the preset merge, and how the env flies it."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import NOMINAL_K_QUAD, DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig  # noqa: E402
from envs.sims.torch_quad import GRAVITY, quat_rotate  # noqa: E402
from envs.structure_hold import NUM_ACTIONS, StructureHoldConfig, StructureHoldEnv  # noqa: E402
from envs.structure_hold.airframes import (  # noqa: E402
    airframe_config,
    gz_x500,
    gz_x500_sitl,
    heavy_16kg,
    mid_3kg,
    px4_allocation,
    small_300g,
)
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


def test_small_300g_has_its_published_numbers_and_the_x500s_tuned_loop():
    x, s3 = gz_x500(), small_300g()
    q = s3["quad"]
    # Starling 2: 285 g, 4 x 185 g of thrust (thrust/weight 2.6).
    assert q["mass"] == pytest.approx(0.285)
    assert q["max_thrust"] == pytest.approx(4 * 0.185 * GRAVITY)
    assert q["max_thrust"] / (q["mass"] * GRAVITY) == pytest.approx(2.6, abs=0.01)
    assert 3e-4 < q["inertia"][0] < 1e-3 and q["inertia"][2] > q["inertia"][0]
    assert q["motor_tau"] == pytest.approx(0.012)

    # Autotune's equal rise time: rate P x torque per unit / inertia is the x500's, per axis.
    def loop(c):
        return (np.asarray(c["rate_pid"]["p"]) * np.asarray(c["actuation"]["torque_per_unit"])
                / np.asarray(c["quad"]["inertia"]))

    np.testing.assert_allclose(loop(s3), loop(x), rtol=1e-9)
    ratio = np.asarray(s3["rate_pid"]["p"]) / np.asarray(x["rate_pid"]["p"])
    np.testing.assert_allclose(np.asarray(s3["rate_pid"]["i"]),
                               np.asarray(x["rate_pid"]["i"]) * ratio, rtol=1e-9)
    # Only the vehicle differs from the benchmark's PX4 pieces.
    sitl = gz_x500_sitl()
    assert s3["dt"] == sitl["dt"] and s3["fusion"] == sitl["fusion"]
    assert s3["rate_pid"]["gyro_cutoff_hz"] == sitl["rate_pid"]["gyro_cutoff_hz"]
    assert s3["vision_offset_down_m"] == 0.02
    # Drag scaled by frontal area: 230 mm against the x500's 492 mm diagonal.
    lo, hi = s3["disturbance"]["k_quad"]
    assert (lo, hi) == pytest.approx((0.8 * NOMINAL_K_QUAD * 0.2185, 1.2 * NOMINAL_K_QUAD * 0.2185),
                                     rel=1e-3)


def test_small_300g_hovers_through_the_module_mapping_and_round_trips():
    # The preset's drag stands unless a config spells k_quad itself (calm() spells it all).
    assert StructureHoldConfig.from_dict({"airframe": "small_300g"}).disturbance.k_quad[1] < (
        0.3 * NOMINAL_K_QUAD)
    cfg = calm("small_300g")
    assert StructureHoldConfig.from_dict(asdict(cfg)) == cfg
    env = StructureHoldEnv(cfg, num_envs=3, seed=0)
    env.reset()
    z0 = env.sim.state.pos[:, 2].clone()
    for _ in range(int(2.0 / env.cfg.dt)):
        env.step(hover_action(env))
    assert (env.sim.state.pos[:, 2] - z0).abs().max() < 0.02
    assert (env.estimate_pos - env.sim.state.pos).norm(dim=1).max() < 0.02


def test_small_300g_rate_loop_answers_like_the_x500s():
    """Same loop gain, faster rotors: a roll-rate step reaches 63 % about as fast."""
    def rise(name):
        env = StructureHoldEnv(calm(name), num_envs=1, seed=0)
        env.reset()
        a = hover_action(env)
        a[:, 0] = 0.25
        for k in range(1, 100):
            env.step(a)
            if env.sim.state.ang_vel[0, 0] >= 0.63 * 0.25 * env.cfg.max_rate_rad_s:
                return k * env.cfg.dt
        return float("inf")

    t_x500, t_small = rise("gz_x500_sitl"), rise("small_300g")
    assert t_small <= t_x500 + 0.013 and t_small > 0.5 * t_x500


def test_mid_and_heavy_have_their_published_numbers():
    m, h = mid_3kg()["quad"], heavy_16kg()["quad"]
    # Holybro X650: 3.0 kg with a battery, 6.3 kg take-off limit at 70 % of full thrust.
    assert m["mass"] == pytest.approx(3.0)
    assert m["max_thrust"] / GRAVITY == pytest.approx(9.0)
    # Freefly Alta X: thrust/weight 3.5 at 19.63 kg, here at 16 kg.
    assert h["mass"] == pytest.approx(16.0)
    assert h["max_thrust"] / (19.63 * GRAVITY) == pytest.approx(3.5)
    assert m["motor_tau"] == pytest.approx(0.0375) and h["motor_tau"] == pytest.approx(0.12)
    assert 1.0 < h["inertia"][0] < 3.0 and 0.03 < m["inertia"][0] < 0.15
    # Wind push per kilogram within a fifth of the x500's on both (area over mass).
    for name, mass in (("mid_3kg", 3.0), ("heavy_16kg", 16.0)):
        k = StructureHoldConfig.from_dict({"airframe": name}).disturbance.k_quad
        per_kg = 0.5 * (k[0] + k[1]) / mass / (NOMINAL_K_QUAD / 2.0643)
        assert 1.0 < per_kg < 1.25


def test_slow_rotors_slow_the_rate_loop_in_proportion():
    """Loop gain: the x500's times its rotor lag over the airframe's, never above it."""
    def loop(c):
        return (np.asarray(c["rate_pid"]["p"]) * np.asarray(c["actuation"]["torque_per_unit"])
                / np.asarray(c["quad"]["inertia"]))

    x = gz_x500()
    for c in (mid_3kg(), heavy_16kg()):
        np.testing.assert_allclose(
            loop(c), loop(x) * x["quad"]["motor_tau"] / c["quad"]["motor_tau"], rtol=1e-9)

    def rise(name):
        env = StructureHoldEnv(calm(name), num_envs=1, seed=0)
        env.reset()
        a = hover_action(env)
        a[:, 0] = 0.25
        peak = 0.0
        t63 = float("inf")
        for k in range(1, 84):
            env.step(a)
            w = float(env.sim.state.ang_vel[0, 0]) / (0.25 * env.cfg.max_rate_rad_s)
            peak = max(peak, w)
            if w >= 0.63:
                t63 = min(t63, k * env.cfg.dt)
        return t63, peak

    (t_mid, peak_mid), (t_heavy, peak_heavy) = rise("mid_3kg"), rise("heavy_16kg")
    assert 0.08 < t_mid < 0.14 and 0.25 < t_heavy < 0.40
    assert peak_mid < 1.2 and peak_heavy < 1.35  # slower, not ringing


@pytest.mark.parametrize("name", ["mid_3kg", "heavy_16kg"])
def test_mid_and_heavy_hover_through_the_module_mapping(name):
    cfg = calm(name, hold_distance=(2.0, 3.0), contact_radius_m=1.13)
    assert StructureHoldConfig.from_dict(asdict(cfg)) == cfg
    env = StructureHoldEnv(cfg, num_envs=3, seed=0)
    env.reset()
    z0 = env.sim.state.pos[:, 2].clone()
    for _ in range(int(2.0 / env.cfg.dt)):
        env.step(hover_action(env))
    assert (env.sim.state.pos[:, 2] - z0).abs().max() < 0.02


def test_rate_loop_follows_a_drawn_rotor_lag():
    """airframe_rand.rate_loop_follows_lag: the gains of an episode scale with the config's
    rotor lag over the drawn one, neither counted below the x500's."""
    def gains(name, tau, follows=True):
        cfg = calm(name, hold_distance=(2.0, 3.0), contact_radius_m=1.13, airframe_rand={
            "motor_tau_s": tau, "rate_loop_follows_lag": follows})
        env = StructureHoldEnv(cfg, num_envs=2, seed=0)
        env.reset()
        return env.rate_pid.kp.reshape(-1, 3)[0].numpy(), np.asarray(cfg.rate_pid.p)

    kp, p = gains("gz_x500_sitl", 0.075)
    np.testing.assert_allclose(kp, p * 0.01875 / 0.075, rtol=1e-6)
    kp, p = gains("gz_x500_sitl", 0.008)  # faster rotors: the tune stays
    np.testing.assert_allclose(kp, p, rtol=1e-6)
    kp, p = gains("heavy_16kg", 0.04)  # the 16 kg airframe with rotors three times as fast
    np.testing.assert_allclose(kp, p * 3.0, rtol=1e-6)
    kp, p = gains("gz_x500_sitl", 0.075, follows=False)
    np.testing.assert_allclose(kp, p, rtol=1e-6)
