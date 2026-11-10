"""Surface passes: the hold point moves along the wall at a fixed standoff."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig  # noqa: E402
from envs.structure_hold import OBS, StructureHoldConfig, StructureHoldEnv  # noqa: E402
from envs.structure_hold.baseline import CascadedPd  # noqa: E402
from envs.structure_hold.config import SurfacePass  # noqa: E402
from envs.structure_hold.passes import pass_duration_s  # noqa: E402


def still_air(**kw) -> StructureHoldConfig:
    return StructureHoldConfig(
        disturbance=DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0, turb_sigma=(0.0, 0.0)),
        estimate=EstimateCorruptionConfig.clean(),
        start_offset_std_m=0.0,
        start_yaw_std_rad=0.0,
        **kw,
    )


def fly_path(env: StructureHoldEnv, seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The hold point and its velocity at every step, driving only the pass clock."""
    pts, vels = [], []
    for _ in range(int(round(seconds / env.dt))):
        env.pass_time += env.dt
        env._move_hold_point()
        pts.append(env.hold_point.clone())
        vels.append(env.hold_vel.clone())
    return torch.stack(pts), torch.stack(vels)


def test_a_pass_moves_the_hold_point_five_metres_along_the_wall_and_stops():
    sp = SurfacePass(prob=1.0, speed_mps=0.5, start_s=2.0, accel_mps2=0.5)
    env = StructureHoldEnv(still_air(surface_pass=sp), num_envs=16, seed=3)
    start = env.hold_point.clone()
    pts, vels = fly_path(env, 20.0)
    moved = pts[-1] - start
    # 5 m along y, either way; nothing toward the wall or up.
    torch.testing.assert_close(moved[:, 1].abs(), torch.full((16,), 5.0), atol=1e-4, rtol=0)
    assert moved[:, [0, 2]].abs().max() < 1e-5 and (moved[:, 1] > 0).any() and (
        moved[:, 1] < 0).any()
    assert (pts[:, :, 0] - start[:, 0]).abs().max() < 1e-5  # the standoff never changes
    # Still for the settle, cruise at 0.5 m/s, stopped at the end.
    k0 = int(1.9 / env.dt)
    assert vels[:k0].abs().max() == 0.0
    assert vels[:, :, 1].abs().max() == pytest.approx(0.5, abs=1e-5)
    assert vels[-1].abs().max() == 0.0
    # Velocity is the path's derivative.
    fd = (pts[2:] - pts[:-2]) / (2 * env.dt)
    assert (fd - vels[1:-1]).abs().max() < 0.01
    # And the end time matches the sizing helper: 2 s + 1 s up + 9 s cruise + 1 s down.
    torch.testing.assert_close(env.passes.end_s, torch.full((16,), 13.0), atol=1e-4, rtol=0)
    assert pass_duration_s(sp, 0.5) == pytest.approx(13.0)


def test_vertical_passes_go_up_and_a_raster_comes_back_one_row_up():
    up = SurfacePass(prob=1.0, speed_mps=1.0, start_s=0.0, vertical_prob=1.0, length_m=2.0)
    env = StructureHoldEnv(still_air(surface_pass=up), num_envs=4, seed=1)
    start = env.hold_point.clone()
    pts, _ = fly_path(env, 6.0)
    torch.testing.assert_close(pts[-1] - start, torch.tensor([0.0, 0.0, 2.0]).expand(4, 3),
                               atol=1e-4, rtol=0)

    raster = SurfacePass(prob=1.0, speed_mps=1.0, start_s=0.0, length_m=2.0, legs=2,
                         row_step_m=0.5, pause_s=0.5)
    env = StructureHoldEnv(still_air(surface_pass=raster), num_envs=4, seed=1)
    start = env.hold_point.clone()
    pts, _ = fly_path(env, 15.0)
    # Out 2 m, up 0.5 m, back 2 m: over the start, one row up.
    torch.testing.assert_close(pts[-1] - start, torch.tensor([0.0, 0.0, 0.5]).expand(4, 3),
                               atol=1e-4, rtol=0)
    assert (pts[:, :, 1] - start[:, 1]).abs().max() == pytest.approx(2.0, abs=1e-4)


def test_passes_off_change_nothing_and_on_leave_the_other_draws_alone():
    cfg = StructureHoldConfig(episode_length_s=1.0)
    a, b = (StructureHoldEnv(c, num_envs=8, seed=5) for c in (
        cfg, StructureHoldConfig(episode_length_s=1.0, surface_pass=SurfacePass(prob=1.0))))
    assert not a.passes.active.any() and b.passes.active.all()
    torch.testing.assert_close(a.hold_point, b.hold_point)  # same engage points ...
    torch.testing.assert_close(a.state.pos, b.state.pos)
    torch.testing.assert_close(a.wind.mean_wind, b.wind.mean_wind)  # ... and wind
    act = torch.zeros(8, 4)
    for _ in range(20):
        oa, ra, *_ = a.step(act)
        assert (a.hold_vel == 0).all()
    # Passes start after their settle, so the first 0.2 s of both are the same flight.
    for _ in range(20):
        ob, rb, *_ = b.step(act)
    torch.testing.assert_close(oa, ob)
    torch.testing.assert_close(ra, rb)


def test_the_velocity_input_is_relative_to_the_moving_hold_point():
    sp = SurfacePass(prob=1.0, speed_mps=0.8, start_s=0.0)
    env = StructureHoldEnv(still_air(surface_pass=sp), num_envs=4, seed=2)
    pd = CascadedPd(env)
    obs, _ = env.get_observations()
    for _ in range(300):
        obs, *_ = env.step(pd(obs))
    assert env.hold_vel[:, 1].abs().min() > 0.5  # mid-pass
    torch.testing.assert_close(obs[:, OBS["vel"]], env.estimate_vel - env.hold_vel)
    # Keeping up costs nothing: the drone moves at the pass speed, its input reads ~0.
    assert obs[:, OBS["vel"]].norm(dim=-1).max() < 0.15


def test_the_pd_flies_a_pass_in_still_air():
    sp = SurfacePass(prob=1.0, speed_mps=0.5, start_s=1.0)
    env = StructureHoldEnv(still_air(surface_pass=sp, episode_length_s=14.0), num_envs=8, seed=4)
    pd = CascadedPd(env)
    obs, _ = env.get_observations()
    worst = torch.zeros(8)
    for _ in range(env.max_episode_length - 1):
        obs, _, done, _ = env.step(pd(obs))
        assert not done.any()
        worst = torch.maximum(worst, (env.step_pos - env.step_hold).norm(dim=-1))
    assert worst.max() < 0.3
    assert (env.hold_point[:, 1] - env.base_hold[:, 1]).abs().min() == pytest.approx(5.0, abs=1e-3)


def test_a_pass_is_not_re_engaged():
    cfg = still_air(episode_length_s=5.0, reengage_rate_hz=50.0, reengage_min_s=0.1,
                    surface_pass=SurfacePass(prob=0.5))
    env = StructureHoldEnv(cfg, num_envs=32, seed=0)
    assert 0 < int(env.passes.active.sum()) < 32
    hits = torch.zeros(32, dtype=torch.bool)
    for _ in range(50):
        env.step(torch.zeros(32, 4))
        hits |= env.reengaged
    assert not (hits & env.passes.active).any() and hits[~env.passes.active].any()


def test_bad_pass_configs_are_refused():
    for bad in ({"prob": 1.5}, {"speed_mps": 0.0}, {"legs": 0}, {"legs": [1, 9]},
                {"length_m": -1.0}, {"accel_mps2": 0.0}):
        with pytest.raises(ValueError):
            SurfacePass.from_dict(bad)
    with pytest.raises(ValueError, match="surface pass"):
        SurfacePass.from_dict({"speed": 1.0})
    cfg = StructureHoldConfig.from_dict({"surface_pass": {"prob": 0.5, "legs": [1, 3]}})
    assert cfg.surface_pass.legs == (1, 3) and cfg.surface_pass.max_segments == 5
