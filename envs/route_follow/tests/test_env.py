"""The Route Follow env: shapes, frames, sensors, termination, the classical yardsticks."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from envs.route_follow import (  # noqa: E402
    NUM_ACTIONS,
    NUM_OBS,
    OBS,
    RouteFollowConfig,
    RouteFollowEnv,  # noqa: E402
)
from envs.route_follow.baseline import RoutePd, RoutePdGains  # noqa: E402
from envs.route_follow.routes import straight_route  # noqa: E402
from envs.sims import QuadState  # noqa: E402
from envs.sims.torch_quad import quat_from_yaw  # noqa: E402

CLEAN_ESTIMATE = {"delay_min_s": 0.0, "delay_max_s": 0.0, "pos_noise_std_m": 0.0,
                  "vel_noise_std_mps": 0.0, "bias_walk_std_m_per_sqrt_s": 0.0,
                  "bias_max_m": 0.0, "jump_rate_hz": 0.0, "dropout_rate_hz": 0.0}
STILL_AIR = {"mean_speed": 0.0, "num_gusts": 0, "turb_sigma": 0.0}


def cfg(**over):
    return RouteFollowConfig.from_dict({"episode_length_s": 10.0, "contact_radius_m": 0.4,
                                        **over})


def test_shapes_and_obs_layout_cover_every_input():
    env = RouteFollowEnv(cfg(), 8, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (8, NUM_OBS)
    covered = sorted(i for s in OBS.values() for i in range(s.start, s.stop))
    assert covered == list(range(NUM_OBS))
    obs, rew, done, extras = env.step(torch.zeros(8, NUM_ACTIONS))
    assert rew.shape == (8,) and done.dtype == torch.bool and "time_outs" in extras
    with pytest.raises(ValueError):
        env.step(torch.zeros(8, 3))


def test_same_seed_same_episodes():
    a, b = RouteFollowEnv(cfg(), 4, seed=5), RouteFollowEnv(cfg(), 4, seed=5)
    act = torch.full((4, NUM_ACTIONS), 0.1)
    for _ in range(20):
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
    assert torch.equal(oa, ob)


def test_start_on_the_route_facing_along_it():
    env = RouteFollowEnv(cfg(episode_length_s=30.0), 16, seed=1)
    obs, _ = env.get_observations()
    assert obs[:, OBS["pos_err"]].norm(dim=-1).max() < 0.6
    assert obs[:, OBS["heading"]][:, 0].min() > 0.9  # body x along the route frame x
    assert obs[:, OBS["remaining"]].min() > 0.0
    assert (env.geo.clearance(env.state.pos) > env.cfg.contact_radius_m).all()


def test_route_frame_observations_do_not_depend_on_the_route_heading():
    """The same straight route along +x and along +y gives the same observations."""
    c = cfg(estimate=CLEAN_ESTIMATE, disturbance=STILL_AIR, rate_lag_s=0.0,
            route={"segments": 1})
    obs = []
    for yaw in (0.0, math.pi / 2):
        env = RouteFollowEnv(c, 1, seed=2)
        r = straight_route(20.0, tunnel=False)
        if yaw:
            r.waypoints = r.waypoints[..., [1, 0, 2]]  # along +y instead of +x
        ids = torch.tensor([0])
        env.geo.set(ids, r)
        start = torch.as_tensor(r.waypoints[:, 0], dtype=torch.float32)
        z = torch.zeros(1, 3)
        env.sim.reset(ids, QuadState(pos=start, quat=quat_from_yaw(torch.tensor([yaw])),
                                     vel=z, ang_vel=z.clone()))
        env.estimator.reset(env.state.pos, env.state.vel, ids)
        env.estimate_pos[:] = env.state.pos
        env._update_target()
        for _ in range(50):
            o, *_ = env.step(torch.tensor([[0.0, 0.1, 0.0, 0.2]]))
        obs.append(o)
    assert torch.allclose(obs[0], obs[1], atol=1e-4)


def test_rangefinders_read_the_walls_and_absent_ones_read_zero():
    c = cfg(ranges={"present_prob": 1.0, "noise_std_m": 0.0, "noise_frac": 0.0},
            route={"segments": 1, "tunnel_prob": 0.0})
    env = RouteFollowEnv(c, 4, seed=3)
    for _ in range(env.range_period_steps):
        obs, *_ = env.step(torch.zeros(4, NUM_ACTIONS))
    ok, r = obs[:, OBS["range_ok"]], obs[:, OBS["ranges"]]
    assert (ok[:, 1:] == 1).all()  # left, right, up, down always see a wall
    # Left plus right is about the corridor width (the drone starts nearly level and aligned).
    assert torch.allclose(r[:, 1] + r[:, 2], 2 * env.geo.half_width, atol=0.1)
    off = RouteFollowEnv(cfg(), 4, seed=3)
    o, *_ = off.step(torch.zeros(4, NUM_ACTIONS))
    assert (o[:, OBS["ranges"]] == 0).all() and (o[:, OBS["range_ok"]] == 0).all()


def test_flying_into_the_wall_ends_in_a_crash_not_a_timeout():
    env = RouteFollowEnv(cfg(), 4, seed=4)
    act = torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(4, 1)  # full roll and full thrust
    seen: set[int] = set()
    for _ in range(300):
        _, _, done, _ = env.step(act)
        seen |= {int(x) for x in env.term_reason[done]}
    assert seen and seen <= {1, 2, 3}  # contact, tilt or lost
    assert env.counts["timeout"] == 0


def test_the_pd_flies_a_clean_route_to_the_end():
    c = cfg(episode_length_s=20.0, estimate=CLEAN_ESTIMATE, disturbance=STILL_AIR)
    env = RouteFollowEnv(c, 8, seed=6)
    pd = RoutePd(env)
    obs, _ = env.get_observations()
    worst = 0.0
    for _ in range(env.max_episode_length - 1):
        obs, _, done, _ = env.step(pd(obs))
        assert not done.any()
        worst = max(worst, float(env.step_cross_track.max()))
    assert worst < 0.5
    assert float(env.step_to_end.max()) < 0.3


def test_centring_uses_the_walls_when_the_estimate_is_biased():
    """A constant 0.4 m estimate offset to the left: the plain PD flies 0.4 m right of the
    centreline, the centring PD stays near it."""
    c = cfg(episode_length_s=12.0, estimate=CLEAN_ESTIMATE, disturbance=STILL_AIR,
            ranges={"present_prob": 1.0},
            route={"segments": 1, "tunnel_prob": 0.0, "corridor_width_m": 3.0})
    offsets = {}
    for centring in (0.0, 1.0):
        env = RouteFollowEnv(c, 4, seed=7)
        pd = RoutePd(env, RoutePdGains(centering=centring))
        obs, _ = env.get_observations()
        for _ in range(env.max_episode_length - 2):
            obs = obs.clone()
            obs[:, OBS["pos_err"].start + 1] += 0.4  # the estimate says 0.4 m further left
            obs, *_ = env.step(pd(obs))
        offsets[centring] = float(env.step_cross_track.mean())
    assert offsets[0.0] > 0.3
    assert offsets[1.0] < 0.15


def test_the_pd_flies_the_other_worlds_on_a_clean_estimate():
    """Arched drifts, narrowed stretches and side branches: the geometry only, so the PD on a
    perfect estimate still flies them without touching a wall."""
    c = cfg(episode_length_s=20.0, estimate=CLEAN_ESTIMATE, disturbance=STILL_AIR,
            route={"arch_prob": 1.0, "tunnel_prob": 0.3, "pinch_prob": 0.5,
                   "pinch_frac": [0.8, 0.9], "branches": 2, "corridor_width_m": [2.4, 4.0],
                   "tunnel_radius_m": [1.2, 2.0]},
            ranges={"present_prob": 1.0})
    env = RouteFollowEnv(c, 8, seed=8)
    assert env.geo.arched.any() and (env.geo.piece_scale < 1).any()
    assert env.geo.piece_on[:, -2:].all()
    pd = RoutePd(env)
    obs, _ = env.get_observations()
    for _ in range(env.max_episode_length - 1):
        obs, _, done, _ = env.step(pd(obs))
        assert not done.any()
    assert float(env.step_to_end.max()) < 0.3


def test_config_checks_the_narrowed_worlds_leave_room():
    with pytest.raises(ValueError, match="no room"):
        RouteFollowConfig.from_dict({"contact_radius_m": 0.6, "route": {"pinch_prob": 0.5}})
    with pytest.raises(ValueError, match="pinch_frac"):
        RouteFollowConfig.from_dict({"route": {"pinch_frac": [0.0, 0.5]}})
