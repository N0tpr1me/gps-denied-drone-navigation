from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig  # noqa: E402
from envs.structure_hold import (  # noqa: E402
    NUM_ACTIONS,
    NUM_OBS,
    OBS,
    StructureHoldConfig,
    StructureHoldEnv,  # noqa: E402
)
from envs.structure_hold.baseline import CascadedPd  # noqa: E402
from envs.structure_hold.config import RewardWeights  # noqa: E402


def still_air(**kw) -> StructureHoldConfig:
    return StructureHoldConfig(
        disturbance=DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0, turb_sigma=(0.0, 0.0)),
        estimate=EstimateCorruptionConfig.clean(),
        start_offset_std_m=0.0,
        start_yaw_std_rad=0.0,
        **kw,
    )


def hover_action(env: StructureHoldEnv) -> torch.Tensor:
    a = torch.zeros(env.num_envs, NUM_ACTIONS)
    a[:, 3] = 2.0 * env.cfg.quad.hover_thrust / env.cfg.quad.max_thrust - 1.0
    return a


def test_shapes_and_vecenv_interface():
    env = StructureHoldEnv(StructureHoldConfig(episode_length_s=1.0), num_envs=5, seed=1)
    assert (env.num_obs, env.num_actions, env.max_episode_length) == (NUM_OBS, NUM_ACTIONS, 100)
    obs, extras = env.get_observations()
    assert obs.shape == (5, NUM_OBS) and "observations" in extras
    obs, reward, dones, extras = env.step(torch.zeros(5, NUM_ACTIONS))
    assert obs.shape == (5, NUM_OBS) and reward.shape == (5,) and dones.shape == (5,)
    assert dones.dtype == torch.bool and extras["time_outs"].shape == (5,)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    with pytest.raises(ValueError, match="actions"):
        env.step(torch.zeros(5, 3))


def test_hold_point_is_between_one_and_three_metres_from_the_wall():
    env = StructureHoldEnv(still_air(), num_envs=64, seed=3)
    clearance = env.clearance()
    assert (clearance >= 1.0).all() and (clearance <= 3.0).all()
    assert torch.allclose(env.hold_point[:, 0], -clearance)
    obs, _ = env.get_observations()
    assert torch.allclose(obs[:, OBS["pos_err"]], torch.zeros(64, 3), atol=1e-6)
    # Level and facing the wall: gravity straight down in the body, heading +x.
    assert torch.allclose(obs[:, OBS["gravity_body"]], torch.tensor([0.0, 0, -1]).expand(64, 3))
    assert torch.allclose(obs[:, OBS["heading"]], torch.tensor([1.0, 0.0]).expand(64, 2))


def test_hover_in_still_air_holds_and_times_out():
    env = StructureHoldEnv(still_air(episode_length_s=2.0), num_envs=4, seed=0)
    start = env.state.pos.clone()
    for i in range(env.max_episode_length):
        obs, reward, dones, extras = env.step(hover_action(env))
        if i < env.max_episode_length - 1:
            assert not dones.any()
            assert torch.allclose(env.state.pos, start, atol=1e-3)
    assert dones.all() and extras["time_outs"].all()
    assert extras["episode"]["crash_rate"] == 0.0
    assert env.counts["timeout"] == 4 and env.counts["contact"] == 0
    assert (obs[:, OBS["last_action"]] == 0).all()  # a fresh episode's first observation


def test_flying_into_the_wall_is_a_contact_and_resets_that_env_only():
    env = StructureHoldEnv(still_air(episode_length_s=5.0), num_envs=3, seed=0)
    env.sim.reset(torch.tensor([1]), env.state._replace(
        pos=torch.tensor([[-0.2, 0.0, 2.0]]), quat=env.state.quat[1:2],
        vel=torch.zeros(1, 3), ang_vel=torch.zeros(1, 3)))
    obs, reward, dones, extras = env.step(hover_action(env))
    assert dones.tolist() == [False, True, False]
    assert not extras["time_outs"][1]
    assert reward[1] < -env.cfg.reward.crash + 2.0 and reward[0] > reward[1]
    assert env.counts["contact"] == 1
    assert env.clearance()[1] >= 1.0  # env 1 restarted at a fresh hold point
    assert env.episode_step.tolist() == [1, 0, 1]


def test_nonfinite_state_is_counted_reset_and_never_shown_to_the_policy():
    env = StructureHoldEnv(still_air(), num_envs=2, seed=0)
    env.sim._state.pos[0, 0] = float("nan")
    obs, reward, dones, _ = env.step(hover_action(env))
    assert dones.tolist() == [True, False]
    assert env.counts["nonfinite"] == 1
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert env.state.finite().all()


def test_estimate_corruption_reaches_the_observation():
    cfg = still_air()
    cfg = StructureHoldConfig(
        disturbance=cfg.disturbance,
        estimate=EstimateCorruptionConfig(delay_min_s=0.0, delay_max_s=0.0, pos_noise_std_m=0.5,
                                          jump_rate_hz=0.0, dropout_rate_hz=0.0),
        start_offset_std_m=0.0, start_yaw_std_rad=0.0,
    )
    env = StructureHoldEnv(cfg, num_envs=32, seed=0)
    obs, *_ = env.step(hover_action(env))
    assert obs[:, OBS["pos_err"]].std() > 0.2  # noisy estimate, though the drone sits still
    assert torch.allclose(env.state.pos, env.hold_point, atol=1e-3)  # truth is unaffected


def test_wind_pushes_the_hovering_drone():
    cfg = StructureHoldConfig(
        disturbance=DisturbanceConfig(mean_speed=(8.0, 8.0), num_gusts=0, turb_sigma=(0.0, 0.0)),
        estimate=EstimateCorruptionConfig.clean(), episode_length_s=1.0,
    )
    env = StructureHoldEnv(cfg, num_envs=8, seed=0)
    start = env.state.pos.clone()
    for _ in range(50):
        env.step(hover_action(env))
    assert (env.state.pos - start).norm(dim=-1).min() > 0.05
    assert env.wind_force.norm(dim=-1).min() > 0.1


def test_same_seed_same_rollout_different_seed_differs():
    def run(seed: int) -> torch.Tensor:
        env = StructureHoldEnv(StructureHoldConfig(), num_envs=4, seed=seed)
        for _ in range(30):
            obs, *_ = env.step(hover_action(env))
        return obs.clone()

    assert torch.equal(run(7), run(7))
    assert not torch.equal(run(7), run(8))


def test_baseline_holds_in_still_air_and_mild_wind():
    cfg = StructureHoldConfig(
        disturbance=DisturbanceConfig(mean_speed=(0.0, 4.0), num_gusts=1, gust_speed=(0.0, 2.0),
                                      gust_start=(1.0, 2.0), turb_sigma=(0.0, 0.5)),
        estimate=EstimateCorruptionConfig(jump_rate_hz=0.0, dropout_rate_hz=0.0),
        episode_length_s=6.0,
    )
    env = StructureHoldEnv(cfg, num_envs=16, seed=0)
    controller = CascadedPd(env)
    obs, _ = env.get_observations()
    worst = 0.0
    for i in range(env.max_episode_length):
        obs, reward, dones, extras = env.step(controller(obs))
        assert dones.any() == (i == env.max_episode_length - 1), f"crash at step {i}"
        if i > 200 and not dones.any():  # settled, and not yet reset for the next episode
            worst = max(worst, float((env.state.pos - env.hold_point).norm(dim=-1).max()))
    assert env.counts["timeout"] == 16 and env.counts["contact"] == 0
    assert worst < 0.5, f"baseline drifted {worst:.2f} m"


def test_tilt_beyond_the_free_angle_costs_reward():
    import math

    from envs.sims import QuadState

    cfg = StructureHoldConfig(disturbance=DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0,
                                                            turb_sigma=(0.0, 0.0)),
                              estimate=EstimateCorruptionConfig.clean())
    env = StructureHoldEnv(cfg, num_envs=3, seed=0)
    st = env.sim.state
    quat = torch.zeros(3, 4)
    for i, deg in enumerate((0.0, 30.0, 60.0)):  # level, inside the free cone, beyond it
        half = math.radians(deg) / 2
        quat[i] = torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0])
    env.sim.reset(torch.arange(3), QuadState(pos=env.hold_point.clone(), quat=quat,
                                             vel=st.vel.clone(), ang_vel=st.ang_vel.clone()))
    _, reward, dones, _ = env.step(hover_action(env))
    assert not dones.any()
    level, inside, beyond = reward.tolist()
    assert abs(level - inside) < 0.2  # no tilt term inside the free cone (only drift)
    expected = cfg.reward.tilt * (math.cos(math.radians(45)) - math.cos(math.radians(60)))
    assert inside - beyond > 0.8 * expected


def test_dropout_cap_limits_how_long_the_estimate_stays_stale():
    cfg = StructureHoldConfig(
        disturbance=still_air().disturbance,
        estimate=EstimateCorruptionConfig(dropout_rate_hz=5.0, dropout_min_s=0.2,
                                          dropout_max_s=3.0, jump_rate_hz=0.0),
        start_offset_std_m=0.0, start_yaw_std_rad=0.0, episode_length_s=6.0,
    )

    def longest_dropout_steps(cap: float | None) -> int:
        env = StructureHoldEnv(cfg, num_envs=64, seed=3)
        if cap is not None:
            applied = env.set_dropout_max_s(cap)
            assert applied == pytest.approx(max(cap, cfg.estimate.dropout_min_s))
            assert env.dropout_max_s == pytest.approx(applied)
        longest = 0
        for _ in range(env.max_episode_length - 1):
            env.step(hover_action(env))
            # steps still to go right after a dropout of L steps starts: L - 1
            longest = max(longest, int(env.estimator.dropout_left.max()) + 1)
        return longest

    assert longest_dropout_steps(None) > 100  # up to 3 s with the config as is
    assert 20 <= longest_dropout_steps(0.5) <= 50  # capped at 0.5 s
    assert longest_dropout_steps(0.05) == 20  # never below dropout_min_s (0.2 s = 20 steps)


def test_moving_fast_on_a_stale_estimate_costs_extra_reward():
    cfg = StructureHoldConfig(
        disturbance=still_air().disturbance,
        estimate=EstimateCorruptionConfig(dropout_rate_hz=0.0, jump_rate_hz=0.0,
                                          delay_min_s=0.0, delay_max_s=0.0),
        start_offset_std_m=0.0, start_yaw_std_rad=0.0, reward=RewardWeights(stale_vel=0.3),
    )
    from envs.sims import QuadState

    env = StructureHoldEnv(cfg, num_envs=2, seed=0)
    # Same true state in both envs, 2 m/s sideways; env 1 is in a dropout, env 0 is not.
    st = env.sim.state
    vel = torch.tensor([[0.0, 2.0, 0.0]]).expand(2, 3).clone()
    env.sim.reset(torch.arange(2), QuadState(pos=st.pos.clone(), quat=st.quat.clone(),
                                             vel=vel, ang_vel=st.ang_vel.clone()))
    env.estimator.dropout_left[1] = 50
    _, reward, *_ = env.step(hover_action(env))
    assert bool(env.estimate_stale[1]) and not bool(env.estimate_stale[0])
    expected = cfg.reward.stale_vel * 2.0
    assert reward[0] - reward[1] == pytest.approx(expected, abs=0.05)


def test_facing_away_from_the_wall_and_yawing_cost_reward():
    cfg = StructureHoldConfig(
        disturbance=still_air().disturbance, estimate=EstimateCorruptionConfig.clean(),
        start_offset_std_m=0.0, start_yaw_std_rad=0.0,
    )
    from envs.sims import QuadState
    from envs.sims.torch_quad import quat_from_yaw

    env = StructureHoldEnv(cfg, num_envs=3, seed=0)
    st = env.sim.state
    quat = quat_from_yaw(torch.tensor([0.0, math.pi / 2, 0.0]))  # env 1 faces along the wall
    ang_vel = torch.zeros(3, 3)
    ang_vel[2, 2] = 2.0  # env 2 faces the wall but yaws at 2 rad/s
    env.sim.reset(torch.arange(3), QuadState(pos=st.pos.clone(), quat=quat, vel=st.vel.clone(),
                                             ang_vel=ang_vel))
    _, reward, *_ = env.step(hover_action(env))
    w = cfg.reward
    assert reward[0] - reward[1] == pytest.approx(w.heading * 1.0, abs=0.05)  # 1 - cos(90 deg)
    assert reward[0] - reward[2] == pytest.approx(w.yaw_rate * 2.0, abs=0.05)


def test_rate_setpoint_lags_the_command_with_the_episode_time_constant():
    cfg = still_air(rate_lag_s=(0.2, 0.2))
    env = StructureHoldEnv(cfg, num_envs=2, seed=0)
    a = hover_action(env)
    a[:, 2] = 1.0  # full yaw rate command
    env.step(a)
    assert env.rate_sp[0, 2] == pytest.approx(cfg.max_rate_rad_s * cfg.dt / 0.2)
    for _ in range(19):
        env.step(a)
    # after one time constant the setpoint has covered 1 - (1 - dt/tau)^(tau/dt) of the way
    want = cfg.max_rate_rad_s * (1 - (1 - cfg.dt / 0.2) ** 20)
    assert env.rate_sp[0, 2] == pytest.approx(want, rel=1e-4)
    assert env.state.ang_vel[0, 2] < cfg.max_rate_rad_s * 0.9  # the body lags further still

    env = StructureHoldEnv(still_air(rate_lag_s=(0.0, 0.0)), num_envs=1, seed=0)
    env.step(a[:1])
    assert env.rate_sp[0, 2] == pytest.approx(env.cfg.max_rate_rad_s)  # no lag
    with pytest.raises(ValueError):
        StructureHoldConfig(rate_lag_s=(0.3, 0.1))


# ---------------------------------------------------------------- airframe stress knobs


def test_airframe_randomisation_changes_only_the_airframe():
    """Same seed with and without airframe draws: same hold points, wind and estimate."""
    base = StructureHoldConfig()
    rand = StructureHoldConfig.from_dict({
        "airframe_rand": {"mass_scale": [0.7, 1.5], "inertia_scale": [0.5, 2.0],
                          "hover_thrust_error": [-0.1, 0.1], "motor_tau_s": [0.02, 0.08]},
        "wall_yaw_torque": {"peak_nm": [0.0, 0.02]},
    })
    a = StructureHoldEnv(base, num_envs=16, seed=4)
    b = StructureHoldEnv(rand, num_envs=16, seed=4)
    assert torch.equal(a.hold_point, b.hold_point)
    assert torch.equal(a.wind.mean_wind, b.wind.mean_wind)
    assert torch.equal(a.estimator.delay_steps, b.estimator.delay_steps)
    assert torch.equal(a.thrust_gain, torch.ones(16))
    assert not torch.equal(b.thrust_gain, torch.ones(16))
    assert float(b.mass_scale.min()) >= 0.7 and float(b.mass_scale.max()) <= 1.5
    assert torch.allclose(b.sim.inv_mass * b.mass_scale * b.cfg.quad.mass, torch.ones(16))
    assert (b.yaw_torque_peak.abs() <= 0.02).all() and (b.yaw_torque_peak != 0).any()


def test_heavier_airframe_hovers_on_the_nominal_hover_action():
    """The module scales thrust through the hover point, so with no hover-thrust error the
    nominal hover action holds a heavier airframe level."""
    env = StructureHoldEnv(StructureHoldConfig.from_dict({
        "disturbance": {"mean_speed": 0.0, "num_gusts": 0, "turb_sigma": 0.0},
        "estimate": {"jump_rate_hz": 0.0, "dropout_rate_hz": 0.0},
        "start_offset_std_m": 0.0, "start_yaw_std_rad": 0.0,
        "airframe_rand": {"mass_scale": 1.5}}), num_envs=3, seed=0)
    assert env.cfg.airframe_rand.active
    z0 = env.state.pos[:, 2].clone()
    for _ in range(100):
        env.step(hover_action(env))
    assert (env.state.pos[:, 2] - z0).abs().max() < 0.01


def test_wall_yaw_torque_spins_the_drone_near_the_wall():
    cfg = StructureHoldConfig.from_dict({
        "disturbance": {"mean_speed": 10.0, "num_gusts": 0, "turb_sigma": 0.0},
        "estimate": {"jump_rate_hz": 0.0, "dropout_rate_hz": 0.0},
        "wall_yaw_torque": {"peak_nm": 0.05}, "hold_distance": 1.0})
    env = StructureHoldEnv(cfg, num_envs=4, seed=2)
    env.step(hover_action(env))
    expected = env.yaw_torque_peak * torch.exp(-env.clearance())
    assert torch.allclose(env.yaw_torque, expected, rtol=0.05)
    assert float(env.state.ang_vel[:, 2].abs().min()) > 0  # nothing else yaws it


def test_term_reason_names_the_crash():
    from envs.structure_hold import TERM_REASONS

    env = StructureHoldEnv(still_air(), num_envs=2, seed=0)
    a = torch.zeros(2, NUM_ACTIONS)
    a[:, 3] = -1.0  # no thrust: falls to the ground
    for _ in range(200):
        _, _, done, _ = env.step(a)
        if bool(done.all()):
            break
    assert [TERM_REASONS[int(r)] for r in env.term_reason] == ["ground", "ground"]
