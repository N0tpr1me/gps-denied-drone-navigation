"""The surroundings in the Structure Hold env (cfg.ambient, the slow IMU errors, a rangefinder
on glass; docs/environment-factors.md): off by default with the same episodes as before,
and each factor doing what its physics says."""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from envs.ambient import AmbientConfig  # noqa: E402
from envs.disturbance import DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig  # noqa: E402
from envs.structure_hold import NUM_ACTIONS, StructureHoldConfig, StructureHoldEnv  # noqa: E402
from envs.structure_hold.baseline import CascadedPd  # noqa: E402
from envs.structure_hold.config import RangeSensor, SensorNoise  # noqa: E402
from envs.structure_hold.env import OBS  # noqa: E402

CALM = DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0, turb_sigma=(0.0, 0.0))
STILL = dict(disturbance=CALM, estimate=EstimateCorruptionConfig.clean(),
             start_offset_std_m=0.0, start_yaw_std_rad=0.0, rate_lag_s=(0.0, 0.0))


def on(airframe: str, **kw) -> StructureHoldConfig:
    """A named airframe's config (``from_dict`` merges its numbers) with ``kw`` on top."""
    return dataclasses.replace(StructureHoldConfig.from_dict({"airframe": airframe}), **kw)


def place(env: StructureHoldEnv, pos: torch.Tensor) -> None:
    """Put every env level and at rest at ``pos`` (N, 3)."""
    s = env.sim.state
    env.sim.reset(torch.arange(env.num_envs), type(s)(
        pos.clone(), s.quat.clone(), torch.zeros_like(s.vel), torch.zeros_like(s.ang_vel)))


def hover(env: StructureHoldEnv) -> torch.Tensor:
    a = torch.zeros(env.num_envs, NUM_ACTIONS)
    a[:, 3] = 2.0 * env.cfg.quad.hover_thrust / env.cfg.quad.max_thrust - 1.0
    return a


def fly_pd(env: StructureHoldEnv, seconds: float) -> None:
    pd = CascadedPd(env)
    obs, _ = env.reset()
    for _ in range(int(seconds / env.dt)):
        obs, _, _, _ = env.step(pd(obs))


def test_off_by_default_and_a_neutral_block_flies_the_same_episodes():
    neutral = AmbientConfig(temperature_c=15.0, battery_voltage=1.0, rain_mm_h=0.0)
    assert not StructureHoldConfig().ambient.active and not neutral.active
    a = StructureHoldEnv(StructureHoldConfig(), num_envs=16, seed=3)
    b = StructureHoldEnv(StructureHoldConfig(ambient=neutral), num_envs=16, seed=3)
    # A block that is on but neutral in effect (a dry standard day stated as a range that
    # only ever draws it) must also leave the flight alone to float precision.
    c = StructureHoldEnv(StructureHoldConfig(ambient=AmbientConfig(battery_sag=(0.0, 1e-9))),
                         num_envs=16, seed=3)
    act = hover(a)
    for _ in range(50):
        oa, ra, _, _ = a.step(act)
        ob, rb, _, _ = b.step(act)
        oc, _, _, _ = c.step(act)
    assert torch.equal(oa, ob) and torch.equal(ra, rb)
    torch.testing.assert_close(oa, oc, atol=1e-5, rtol=0)


def test_thin_air_and_a_low_battery_cut_thrust_and_torque_but_not_the_hover():
    amb = AmbientConfig(temperature_c=35.0, altitude_m=500.0, battery_voltage=0.95)
    cfg = on("gz_x500", ambient=amb, episode_length_s=8.0, **STILL)
    env = StructureHoldEnv(cfg, num_envs=4, seed=0)
    power = float(env.amb_power[0])
    assert 0.76 < power < 0.82  # density 0.88 x voltage^2 0.90
    torch.testing.assert_close(env.sim.max_thrust, torch.full((4,), cfg.quad.max_thrust * power))
    torch.testing.assert_close(env.sim.max_torque[0],
                               torch.tensor(cfg.quad.max_torque) * power)
    assert float(env.amb_drag[0]) == pytest.approx(0.88, abs=0.02)
    # PX4's hover-thrust estimate has followed: the PD still holds its point.
    fly_pd(env, 6.0)
    assert env.counts["contact"] + env.counts["ground"] + env.counts["box"] == 0
    assert float((env.state.pos - env.hold_point).norm(dim=-1).max()) < 0.1


def test_the_x500_runs_out_of_thrust_in_air_too_thin_for_it():
    """The Gazebo x500 has 1.69 times its weight in thrust and the module sends at most 0.9
    of the command, 84 % of that thrust: below 0.71 of standard power it cannot hover."""
    amb = AmbientConfig(temperature_c=40.0, altitude_m=2500.0, humidity=0.9)
    env = StructureHoldEnv(on("gz_x500", ambient=amb, episode_length_s=8.0, **STILL),
                           num_envs=4, seed=0)
    assert float(env.amb_power[0]) < 0.7
    fly_pd(env, 6.0)
    assert env.counts["ground"] >= 4


def test_thin_air_pushes_less():
    wind = DisturbanceConfig(mean_speed=8.0, mean_heading_deg=0.0, num_gusts=0, turb_sigma=0.0)
    base = dict(STILL, disturbance=wind)
    a = StructureHoldEnv(StructureHoldConfig(**base), num_envs=4, seed=1)
    b = StructureHoldEnv(StructureHoldConfig(**base, ambient=AmbientConfig(altitude_m=3000.0)),
                         num_envs=4, seed=1)
    a.step(hover(a)), b.step(hover(b))
    ratio = b.wind_force[:, 0] / a.wind_force[:, 0]
    torch.testing.assert_close(ratio, b.amb_drag, rtol=1e-4, atol=0)
    assert 0.67 < float(ratio[0]) < 0.72  # 15 degC air at 3000 m


def test_rain_adds_weight_and_the_hover_still_holds():
    cfg = on("gz_x500", ambient=AmbientConfig(rain_mm_h=50.0), episode_length_s=8.0, **STILL)
    env = StructureHoldEnv(cfg, num_envs=2, seed=0)
    assert float(env.mass_scale[0]) == pytest.approx(1.02)
    assert float(env.amb_power[0]) == pytest.approx(0.95, abs=1e-3)
    fly_pd(env, 6.0)
    assert float(env.ambient_force[0, 2]) < 0  # the drops push down
    assert float((env.state.pos - env.hold_point).norm(dim=-1).max()) < 0.1


def test_battery_sag_takes_from_the_top_of_the_throttle_only():
    def climb(sag: float) -> float:
        amb = AmbientConfig(battery_sag=sag) if sag else AmbientConfig()
        env = StructureHoldEnv(StructureHoldConfig(ambient=amb, **STILL), num_envs=1, seed=0)
        a = hover(env)
        for _ in range(20):
            env.step(a)
        level = float(env.state.vel[0, 2])
        a[:, 3] = 1.0  # full throttle
        for _ in range(30):
            env.step(a)
        return level, float(env.state.vel[0, 2])

    (level0, up0), (level1, up1) = climb(0.0), climb(0.2)
    assert abs(level0) < 1e-3 and abs(level1) < 1e-3  # hover is hover with or without sag
    assert 0.5 * up0 < up1 < 0.9 * up0  # full throttle climbs slower on a sagging pack


def test_rotors_near_the_ground_a_deck_and_the_wall():
    amb = AmbientConfig(rotor_radius_m=0.25, wall_pull=0.05)
    cfg = StructureHoldConfig(ambient=amb, hold_distance=(1.0, 1.0), hold_height=(2.0, 2.0),
                              deck_clearance=(0.7, 0.7), **STILL)
    env = StructureHoldEnv(cfg, num_envs=3, seed=0)
    place(env, torch.tensor([[-3.0, 0.0, 0.5],  # low over the ground, far from wall and deck
                             [-3.0, 0.0, 2.4],  # 0.3 m under the deck
                             [-0.5, 0.0, 1.4]]))  # 0.5 m from the wall
    f = env._ambient_force()
    thrust = float(env.sim.thrust[0])
    # Ground cushion: Cheeseman and Bennett at z = 2 R.
    cushion = 1 / (1 - (0.25 / 0.5) ** 2 / 16)
    deck_far = 1 / (1 - float(env.amb_ceiling_coef[0]) * (0.25 / 2.2) ** 2)
    assert float(f[0, 2]) == pytest.approx(thrust * (cushion * deck_far - 1), rel=1e-3)
    assert float(f[1, 2]) > 2.0 * float(f[0, 2])  # the deck pulls up, harder than the ground
    assert float(f[2, 0]) == pytest.approx(0.05 * thrust * 0.25, rel=0.05)  # toward the wall
    assert abs(float(f[0, 0])) < 0.002 * thrust  # far from the wall: no pull to speak of


def test_downwash_coming_back_shakes_only_near_a_surface():
    amb = AmbientConfig(rotor_radius_m=0.25, unsteady=0.1)
    cfg = StructureHoldConfig(ambient=amb, hold_distance=(3.0, 3.0), episode_length_s=30.0,
                              **STILL)
    env = StructureHoldEnv(cfg, num_envs=2, seed=0)
    place(env, torch.tensor([[-0.5, 0.0, 2.0], [-3.0, 0.0, 2.0]]))
    near, far = [], []
    for _ in range(400):
        f = env._ambient_force()
        near.append(f[0].clone()), far.append(f[1].clone())
    near, far = torch.stack(near).std(0), torch.stack(far).std(0)
    assert float(near.mean()) > 10.0 * float(far.mean()) > 0.0


def test_gyro_bias_and_a_tipped_attitude_reach_the_loop_and_the_inputs():
    cfg = StructureHoldConfig(sensor_noise=SensorNoise(gyro_bias_rad_s=0.05, tilt_bias_deg=2.0),
                              **STILL)
    env = StructureHoldEnv(cfg, num_envs=64, seed=2)
    obs, _ = env.reset()
    assert 0.005 < float(env.gyro_bias.abs().mean()) < 0.05
    torch.testing.assert_close(obs[:, OBS["ang_vel"]] * cfg.max_rate_rad_s, env.gyro_bias,
                               atol=1e-6, rtol=0)
    # Level in truth, tipped in the estimate: by the drawn angle, about a horizontal axis.
    tilt = torch.acos((-obs[:, OBS["gravity_body"]][:, 2]).clamp(max=1.0))
    angle = 2.0 * torch.acos(env.tilt_quat[:, 0].clamp(max=1.0))
    torch.testing.assert_close(tilt, angle, atol=2e-3, rtol=0)
    assert 0.2 < math.degrees(float(tilt.mean())) < 2.0
    assert float(env.tilt_quat[:, 3].abs().max()) < 1e-6  # no yaw in it
    # The PD steers with the same wrong attitude: it holds, offset by what the tilt costs.
    fly_pd(env, 8.0)
    err = (env.state.pos - env.hold_point).norm(dim=-1)
    assert env.counts["contact"] == 0 and 0.005 < float(err.mean()) < 0.5
    # Without the block nothing is drawn.
    plain = StructureHoldEnv(StructureHoldConfig(**STILL), num_envs=4, seed=2)
    assert not plain.gyro_bias.any() and bool((plain.tilt_quat[:, 0] == 1).all())


def test_a_rangefinder_on_glass_reads_what_is_behind_it():
    rng = RangeSensor(present_prob=1.0, noise_std_m=0.0, noise_frac=0.0, glass_prob=0.5,
                      glass_extra_m=(1.0, 2.0), glass_frac=(0.5, 0.5), glass_dwell_s=0.5)
    cfg = StructureHoldConfig(range=rng, hold_distance=(1.5, 1.5), episode_length_s=30.0,
                              **STILL)
    env = StructureHoldEnv(cfg, num_envs=200, seed=5)
    env.reset()
    assert 0.35 < float(env.glass.float().mean()) < 0.65
    a = hover(env)
    far, seen_ok = torch.zeros(200), torch.zeros(200)
    for _ in range(300):
        obs, _, _, _ = env.step(a)
        ok = obs[:, OBS["range_ok"].start] > 0.5
        # The input's error is the hold point's distance minus the measured one: a reading
        # through the glass says the wall is further, so the error goes negative.
        far += (ok & (obs[:, OBS["wall_range"].start] < -0.5)).float()
        seen_ok += ok.float()
    assert bool((far[~env.glass] == 0).all())  # a plain wall never reads far
    share = far[env.glass] / seen_ok[env.glass].clamp(min=1.0)
    assert 0.2 < float(share.mean()) < 0.7  # glass does, about half the time
    # Off by default: no draw, no state.
    plain = StructureHoldEnv(StructureHoldConfig(range=RangeSensor(present_prob=1.0), **STILL),
                             num_envs=8, seed=5)
    assert not plain.glass.any()


def test_wind_on_the_static_port_moves_the_fused_height():
    wind = DisturbanceConfig(mean_speed=4.0, mean_heading_deg=90.0, num_gusts=1,
                             gust_start=2.0, gust_duration=20.0, gust_speed=6.0,
                             turb_sigma=0.0)
    base = dict(STILL, disturbance=wind, episode_length_s=20.0, hold_distance=(3.0, 3.0))
    amb = AmbientConfig(baro_wind_m_per_mps2=0.005)
    env = StructureHoldEnv(on("gz_x500_sitl", **base, ambient=amb), num_envs=8, seed=0)
    ref = StructureHoldEnv(on("gz_x500_sitl", **base), num_envs=8, seed=0)
    assert env.cfg.fusion.baro
    for e in (env, ref):
        fly_pd(e, 10.0)
    # The gust changes the airspeed past the vehicle, the barometer reads it as height, and
    # the controller follows the estimate: the true height moves by about what the port saw.
    port = env.baro_wind_err
    assert float(port.abs().mean()) > 0.05
    moved = (env.state.pos[:, 2] - env.hold_point[:, 2]) - (
        ref.state.pos[:, 2] - ref.hold_point[:, 2])
    torch.testing.assert_close(moved, -port, atol=0.1, rtol=0.5)
    assert not ref.baro_wind_err.any()
