"""The env with the estimator's heading made wrong (cfg.heading_error, docs/heading-error.md)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from envs.disturbance import DisturbanceConfig  # noqa: E402
from envs.estimate import EstimateCorruptionConfig, HeadingErrorConfig  # noqa: E402
from envs.sims.torch_quad import quat_from_yaw, quat_rotate, quat_rotate_inverse  # noqa: E402
from envs.structure_hold import OBS, StructureHoldConfig, StructureHoldEnv  # noqa: E402
from envs.structure_hold.baseline import CascadedPd  # noqa: E402


def cfg(heading: HeadingErrorConfig | None = None, **kw) -> StructureHoldConfig:
    return StructureHoldConfig(
        disturbance=DisturbanceConfig(mean_speed=(0.0, 0.0), num_gusts=0, turb_sigma=(0.0, 0.0)),
        estimate=EstimateCorruptionConfig.clean(),
        start_offset_std_m=0.3,
        start_yaw_std_rad=0.2,
        heading_error=heading or HeadingErrorConfig(),
        **kw,
    )


def test_off_by_default_leaves_the_observation_alone():
    a = StructureHoldEnv(cfg(), num_envs=4, seed=3)
    assert a.heading_err is None
    assert torch.equal(a.believed_quat, a.state.quat)
    obs, _ = a.get_observations()
    fwd = quat_rotate(a.state.quat, torch.tensor([[1.0, 0.0, 0.0]]).expand(4, 3))
    assert torch.allclose(obs[:, OBS["heading"]], fwd[:, :2])
    assert torch.allclose(obs[:, OBS["pos_err"]], a.estimate_pos - a.hold_point)


def test_offset_at_engage_turns_position_not_heading():
    d = 30.0
    env = StructureHoldEnv(cfg(HeadingErrorConfig(prob=1.0, offset_deg=d)), num_envs=6, seed=1)
    delta = env.heading_err.delta
    assert torch.allclose(delta.abs(), torch.full((6,), math.radians(d)))
    assert torch.equal(env.heading_latch, delta)
    obs, _ = env.get_observations()
    # Latched at engage: the heading input is the true one ...
    fwd = quat_rotate(env.state.quat, torch.tensor([[1.0, 0.0, 0.0]]).expand(6, 3))
    assert torch.allclose(obs[:, OBS["heading"]], fwd[:, :2], atol=1e-6)
    # ... and the position error is read in a frame turned by the offset.
    err = quat_rotate_inverse(quat_from_yaw(delta), env.estimate_pos - env.hold_point)
    assert torch.allclose(obs[:, OBS["pos_err"]], err, atol=1e-6)


def test_the_module_sees_the_error_turned_by_the_current_heading_error():
    """Whatever the latch, the body-frame error the module computes is the true one turned by
    -delta(t): the frame mismatch every position loop on that heading suffers."""
    h = HeadingErrorConfig(prob=1.0, offset_deg=(10.0, 40.0), drift_rate_deg_s=5.0,
                           drift_max_deg=30.0)
    env = StructureHoldEnv(cfg(h), num_envs=8, seed=2)
    for _ in range(100):
        env.step(torch.zeros(8, 4))
    assert not torch.allclose(env.heading_err.delta, env.heading_latch)
    v = torch.randn(8, 3)
    seen = quat_rotate_inverse(env.believed_quat, env.to_latched(v))
    true = quat_rotate_inverse(env.state.quat,
                               quat_rotate_inverse(quat_from_yaw(env.heading_err.delta), v))
    assert torch.allclose(seen, true, atol=1e-5)


def test_reengage_latches_the_heading_again():
    h = HeadingErrorConfig(prob=1.0, drift_rate_deg_s=10.0, drift_max_deg=30.0)
    env = StructureHoldEnv(cfg(h, reengage_rate_hz=1e6, reengage_min_s=0.5), num_envs=4, seed=0)
    for _ in range(int(0.6 / env.dt)):
        env.step(torch.zeros(4, 4))
    assert env.reengage_count > 0
    assert torch.allclose(env.heading_latch, env.heading_err.delta)


def test_pd_holds_through_10_deg_and_spirals_out_at_30():
    """The mechanism the docs describe: a turned frame turns the PD's corrections and eats its
    phase margin. With its attitude and rate-loop lag the PD has about 20 deg to give: 10 deg
    still settles, 30 deg grows into a circle over a metre wide (docs/heading-error.md)."""
    def fly(deg: float) -> tuple[float, float]:
        env = StructureHoldEnv(cfg(HeadingErrorConfig(prob=1.0, offset_deg=deg),
                                   episode_length_s=14.0), num_envs=16, seed=4)
        pd = CascadedPd(env)
        obs, _ = env.get_observations()
        worst = torch.zeros(16)
        for _ in range(env.max_episode_length - 1):
            obs, _, done, _ = env.step(pd(obs))
            err = (env.step_pos - env.hold_point).norm(dim=-1)
            worst = torch.where(done, torch.full_like(worst, 9.0), torch.maximum(worst, err))
        return float((env.state.pos - env.hold_point).norm(dim=-1).max()), float(worst.max())

    end10, _ = fly(10.0)
    end30, worst30 = fly(30.0)
    assert end10 < 0.1
    assert worst30 > 1.0 and end30 > 0.5
