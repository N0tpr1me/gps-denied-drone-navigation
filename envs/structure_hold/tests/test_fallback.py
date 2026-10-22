"""The brake-and-hover fallback: silent while fresh, brakes after the threshold, hands back."""

from __future__ import annotations

import pytest
import torch

from envs.estimate import EstimateCorruptionConfig
from envs.sims import QuadState
from envs.structure_hold.baseline import CascadedPd
from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import StructureHoldEnv
from envs.structure_hold.fallback import FallbackRule, StaleFallback
from envs.structure_hold.tests.test_env import hover_action, still_air


def quiet_env(num_envs: int = 2) -> StructureHoldEnv:
    cfg = StructureHoldConfig(
        disturbance=still_air().disturbance,
        estimate=EstimateCorruptionConfig(dropout_rate_hz=0.0, jump_rate_hz=0.0,
                                          delay_min_s=0.0, delay_max_s=0.0),
        start_offset_std_m=0.0, start_yaw_std_rad=0.0,
    )
    return StructureHoldEnv(cfg, num_envs=num_envs, seed=0)


def test_fallback_is_silent_while_the_estimate_is_fresh():
    env = quiet_env()
    pd = CascadedPd(env)
    wrapped = StaleFallback(env, CascadedPd(env), FallbackRule(threshold_s=0.5))
    obs, _ = env.get_observations()
    for _ in range(20):
        a, b = pd(obs), wrapped(obs)
        assert torch.equal(a, b)
        obs, *_ = env.step(a)
    assert not bool(wrapped.active.any()) and wrapped.active_fraction == 0.0


def test_fallback_takes_over_after_the_threshold_brakes_and_hands_back():
    env = quiet_env()
    rule = FallbackRule(threshold_s=0.5)
    # Inner controller: a level hover that never brakes, so any braking is the fallback's.
    wrapped = StaleFallback(env, lambda obs: hover_action(env), rule)
    st = env.sim.state
    vel = torch.tensor([[0.0, 1.5, 0.0]]).expand(2, 3).clone()  # 1.5 m/s along the wall
    env.sim.reset(torch.arange(2), QuadState(pos=st.pos.clone(), quat=st.quat.clone(),
                                             vel=vel, ang_vel=st.ang_vel.clone()))
    env.estimator.dropout_left[1] = 150  # env 1 loses its estimate for 1.5 s; env 0 never
    obs, _ = env.get_observations()
    hover = hover_action(env)
    active_steps = 0
    for _ in range(149):
        a = wrapped(obs)
        assert torch.equal(a[0], hover[0])  # the fresh env is untouched
        if bool(wrapped.active[1]):
            active_steps += 1
            assert not torch.equal(a[1], hover[1])
        else:
            assert torch.equal(a[1], hover[1])
            assert wrapped.stale_s[1] < rule.threshold_s
        obs, *_ = env.step(a)
    assert bool(wrapped.active[1]) and active_steps == pytest.approx(100, abs=2)
    speed = env.state.vel.norm(dim=-1)
    assert speed[1] < 0.3 and speed[0] > 1.0  # braked to a hover, the hover-only env coasts
    # Fresh again: hand back on the first fresh observation.
    for _ in range(5):
        obs, *_ = env.step(wrapped(obs))
        if not bool(env.estimate_stale[1]):
            break
    assert not bool(env.estimate_stale[1])
    a = wrapped(obs)
    assert not bool(wrapped.active[1]) and torch.equal(a[1], hover[1])
    assert 0.0 < wrapped.active_fraction < 0.5


def test_fallback_rule_from_dict_and_history_reset_pass_through():
    env = quiet_env()
    assert FallbackRule.from_dict({"threshold_s": 2.0}).threshold_s == 2.0
    assert FallbackRule.from_dict(None) == FallbackRule()
    calls = []
    inner = lambda obs: hover_action(env)  # noqa: E731
    inner.reset_history = lambda *a: calls.append(a)
    wrapped = StaleFallback(env, inner)
    assert wrapped.reset_history is inner.reset_history
