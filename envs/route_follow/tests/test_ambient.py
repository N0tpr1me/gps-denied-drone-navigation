"""Route Follow with the surroundings on (``cfg.ambient``, docs/environment-factors.md): off by
default it changes nothing; on, thin air takes headroom, the nearest surface pushes and pulls,
and the airspeed shows on a fused barometer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from envs.ambient import AmbientConfig  # noqa: E402
from envs.route_follow import NUM_ACTIONS, RouteFollowConfig, RouteFollowEnv  # noqa: E402
from envs.route_follow.baseline import RoutePd  # noqa: E402
from envs.sims import QuadState  # noqa: E402
from envs.sims.torch_quad import quat_from_yaw  # noqa: E402

CLEAN_ESTIMATE = {"delay_min_s": 0.0, "delay_max_s": 0.0, "pos_noise_std_m": 0.0,
                  "vel_noise_std_mps": 0.0, "bias_walk_std_m_per_sqrt_s": 0.0,
                  "bias_max_m": 0.0, "jump_rate_hz": 0.0, "dropout_rate_hz": 0.0}
STILL_AIR = {"mean_speed": 0.0, "num_gusts": 0, "turb_sigma": 0.0}
CORRIDOR = {"segments": 1, "tunnel_prob": 0.0, "corridor_width_m": 3.0}


def cfg(**over):
    return RouteFollowConfig.from_dict({"airframe": "gz_x500_sitl", "episode_length_s": 10.0,
                                        "contact_radius_m": 0.4, **over})


def rollout(c, steps=150, seed=3):
    env = RouteFollowEnv(c, 8, seed=seed)
    act = torch.zeros(8, NUM_ACTIONS)
    act[:, 3] = 0.1
    out = []
    for _ in range(steps):
        obs, rew, *_ = env.step(act)
        out.append(torch.cat([obs, rew[:, None]], 1))
    return torch.stack(out)


def test_a_neutral_block_flies_the_same_episodes_as_none():
    neutral = {"temperature_c": [15, 15], "battery_voltage": [1.0, 1.0]}
    assert torch.equal(rollout(cfg()), rollout(cfg(ambient=neutral)))
    assert not RouteFollowConfig.from_dict({"ambient": neutral}).ambient.active


def test_thin_air_and_a_low_battery_take_thrust_and_torque_headroom():
    c = cfg(ambient={"temperature_c": [35, 35], "altitude_m": [1500, 1500],
                     "battery_voltage": [0.95, 0.95]})
    env = RouteFollowEnv(c, 4, seed=0)
    power = float(env.amb_power[0])
    assert 0.65 < power < 0.8
    q = c.quad
    torch.testing.assert_close(env.amb_max_thrust, torch.full((4,), q.max_thrust * power),
                               rtol=1e-5, atol=1e-5)
    # Full throttle gives that much less; hover is where it was (the module's mapping).
    full = torch.zeros(4, NUM_ACTIONS)
    full[:, 3] = 1.0
    for _ in range(40):
        env.step(full)
    assert float(env.sim.thrust.max()) < q.max_thrust * power * 1.001


def test_the_pd_still_flies_a_route_in_hot_humid_air_on_a_part_used_pack():
    c = cfg(episode_length_s=15.0, estimate=CLEAN_ESTIMATE, disturbance=STILL_AIR,
            ambient={"temperature_c": [30, 40], "humidity": [0.5, 1.0],
                     "battery_voltage": [0.97, 1.0], "battery_sag": [0.0, 0.1]})
    env = RouteFollowEnv(c, 8, seed=6)
    pd = RoutePd(env)
    obs, _ = env.get_observations()
    worst = 0.0
    for _ in range(env.max_episode_length - 1):
        obs, _, done, _ = env.step(pd(obs))
        assert not done.any()
        worst = max(worst, float(env.step_cross_track.max()))
    assert worst < 0.6


def place(env, pos):
    n = env.num_envs
    p = torch.tensor(pos, dtype=torch.float32).expand(n, 3).clone()
    zeros = torch.zeros(n, 3)
    env.sim.reset(torch.arange(n), QuadState(pos=p, quat=quat_from_yaw(torch.zeros(n)),
                                             vel=zeros.clone(), ang_vel=zeros.clone()))


def surface_force(pos, **amb):
    c = cfg(disturbance=STILL_AIR, route=CORRIDOR,
            ambient={"rotor_radius_m": [0.2, 0.2], **amb})
    env = RouteFollowEnv(c, 2, seed=0)
    place(env, pos)
    return env._ambient_force(torch.full((2,), 20.0))[0], env


def test_the_nearest_surface_cushions_from_below_pulls_from_above_and_from_the_side():
    _, env = surface_force([0.0, 0.0, 0.0])
    start = env.geo.waypoints[0, 0].tolist()
    half_h = float(env.geo.half_height[0])
    x0, y0, z0 = start[0] + 2.0, start[1], start[2]
    floor, _ = surface_force([x0, y0, z0 - half_h + 0.5], ground_coef=[1 / 16, 1 / 16])
    roof, _ = surface_force([x0, y0, z0 + half_h - 0.5], ceiling_coef=[0.15, 0.15])
    wall, _ = surface_force([x0, y0 + 1.5 - 0.5, z0], wall_pull=[0.05, 0.05])
    assert floor[2] > 0.05  # the cushion lifts
    assert roof[2] > 0.1  # the roof pulls up into it
    assert wall[1] > 0.1 and abs(float(wall[2])) < 1e-3  # pulled toward the left wall
    mid, _ = surface_force([x0, y0, z0], ground_coef=[1 / 16, 1 / 16], wall_pull=[0.05, 0.05])
    assert float(mid.norm()) < float(floor.norm())  # weaker in the middle of the corridor


def test_the_downwash_coming_back_shakes_the_drone_near_a_wall_only():
    c = cfg(disturbance=STILL_AIR, route=CORRIDOR,
            ambient={"rotor_radius_m": [0.2, 0.2], "unsteady": [0.1, 0.1]})
    env = RouteFollowEnv(c, 2, seed=0)
    start = env.geo.waypoints[0, 0].tolist()
    near = [start[0] + 2.0, start[1] + 1.0, start[2]]
    place(env, near)
    pushes = [env._ambient_force(torch.full((2,), 20.0))[0].norm() for _ in range(50)]
    assert max(float(p) for p in pushes) > 0.05


def test_a_fused_barometer_reads_the_draft_on_its_port():
    c = cfg(disturbance={"mean_speed": [3.0, 3.0], "num_gusts": 0, "turb_sigma": [1.0, 1.0]},
            fusion={"height_source": "baro"},
            ambient={"baro_wind_m_per_mps2": [0.01, 0.01]})
    assert c.fusion.baro and c.ambient.baro_wind
    env = RouteFollowEnv(c, 4, seed=1)
    act = torch.zeros(4, NUM_ACTIONS)
    seen = 0.0
    for _ in range(200):
        env.step(act)
        seen = max(seen, float(env.baro_wind_err.abs().max()))
    assert seen > 0.01
    assert AmbientConfig().baro_wind is False
