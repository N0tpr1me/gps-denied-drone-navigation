import math

import numpy as np
import pytest

from envs.estimate.core import EstimateCorruptionConfig, EstimateCorruptor
from envs.estimate.tests.helpers import OFF, only, ramp, run

DT = 0.01


def test_clean_config_is_identity():
    pos, vel = ramp(200, 3)
    out = run(EstimateCorruptor(OFF, 3, DT, seed=0), pos, vel)
    np.testing.assert_array_equal(out["pos"], pos)
    np.testing.assert_array_equal(out["vel"], vel)
    assert not out["stale"].any()
    assert not out["age"].any()


def test_delay_exact_to_one_sample():
    pos, vel = ramp(300, 2)
    cfg = only(delay_min_s=0.05, delay_max_s=0.05)
    out = run(EstimateCorruptor(cfg, 2, DT, seed=0), pos, vel)
    d = 5
    np.testing.assert_array_equal(out["pos"][d:], pos[:-d])
    np.testing.assert_array_equal(out["vel"][d:], vel[:-d])
    # Before d samples have arrived the start state is repeated.
    np.testing.assert_array_equal(out["pos"][:d], np.broadcast_to(pos[0], (d, 2, 3)))


def test_delay_drawn_per_env_within_range_and_overridable():
    cfg = only(delay_min_s=0.02, delay_max_s=0.12)
    model = EstimateCorruptor(cfg, 1000, DT, seed=1)
    model.reset(np.zeros((1000, 3)), np.zeros((1000, 3)))
    assert model.delay_steps.min() == 2 and model.delay_steps.max() == 12

    pos, vel = ramp(100, 3)
    model = EstimateCorruptor(cfg, 3, DT, seed=1)
    out = run(model, pos, vel, delay_steps=np.array([2, 7, 12]))
    for env, d in enumerate([2, 7, 12]):
        np.testing.assert_array_equal(out["pos"][d:, env], pos[:-d, env])


@pytest.mark.parametrize("pos_std, vel_std", [(0.02, 0.05), (0.3, 0.0)])
def test_noise_spread_within_5_percent(pos_std, vel_std):
    n = 10_000
    model = EstimateCorruptor(only(pos_noise_std_m=pos_std, vel_noise_std_mps=vel_std), n, DT, 2)
    truth = np.full((n, 3), 4.0)
    model.reset(truth, truth)
    s = model.step(truth, truth)
    err_p, err_v = s.pos - truth, s.vel - truth
    for axis in range(3):
        assert err_p[:, axis].std() == pytest.approx(pos_std, rel=0.05)
        assert abs(err_p[:, axis].mean()) < 4 * pos_std / math.sqrt(n) + 1e-12
        assert err_v[:, axis].std() == pytest.approx(vel_std, rel=0.05)


def test_bias_walk_grows_with_sqrt_time_and_stays_clipped():
    n, steps, walk = 10_000, 400, 0.05
    model = EstimateCorruptor(only(bias_walk_std_m_per_sqrt_s=walk, bias_max_m=10.0), n, DT, 3)
    z = np.zeros((n, 3))
    model.reset(z, z)
    for _ in range(steps):
        s = model.step(z, z)
    assert s.pos.std(axis=0) == pytest.approx(np.full(3, walk * math.sqrt(steps * DT)), rel=0.05)

    clipped = EstimateCorruptor(only(bias_walk_std_m_per_sqrt_s=1.0, bias_max_m=0.3), 500, DT, 3)
    z = np.zeros((500, 3))
    clipped.reset(z, z)
    worst = max(np.abs(clipped.step(z, z).pos).max() for _ in range(500))
    assert worst == pytest.approx(0.3)


def test_jump_sizes_and_rate_within_configured_range():
    n, steps, rate = 200, 1000, 0.5
    cfg = only(jump_rate_hz=rate, jump_min_m=0.1, jump_max_m=2.0, jump_decay_s=2.0)
    model = EstimateCorruptor(cfg, n, DT, seed=4)
    z = np.zeros((n, 3))
    model.reset(z, z)
    prev = z
    sizes = []
    for _ in range(steps):
        s = model.step(z, z)
        step = s.pos - model.jump_keep * prev  # exactly the new jump, zero otherwise
        norms = np.linalg.norm(step, axis=1)
        np.testing.assert_allclose(norms[~s.jumped], 0.0, atol=1e-12)
        sizes.extend(norms[s.jumped])
        prev = s.pos
    sizes = np.array(sizes)
    assert sizes.min() >= 0.1 and sizes.max() <= 2.0
    assert sizes.min() < 0.2 and sizes.max() > 1.9  # the whole range is used
    expected = n * steps * DT * rate  # 1000 events
    assert len(sizes) == pytest.approx(expected, rel=0.1)


def test_jump_decays_with_time_constant_or_persists():
    for decay, keep in [(0.5, math.exp(-DT / 0.5)), (None, 1.0)]:
        model = EstimateCorruptor(only(jump_rate_hz=1e9, jump_decay_s=decay), 1, DT, seed=5)
        z = np.zeros((1, 3))
        model.reset(z, z)
        first = model.step(z, z).pos.copy()
        model.p_jump = 0.0  # no more jumps after the first
        later = [model.step(z, z).pos for _ in range(50)]
        np.testing.assert_allclose(later[-1], first * keep**50, rtol=1e-12)


def test_dropouts_hold_output_flag_stale_and_last_within_range():
    n, steps = 100, 3000
    cfg = only(pos_noise_std_m=0.01, dropout_rate_hz=0.3, dropout_min_s=0.2, dropout_max_s=3.0,
               dropout_mode="hold")
    pos, vel = ramp(steps, n)
    out = run(EstimateCorruptor(cfg, n, DT, seed=6), pos, vel)
    stale, started = out["stale"], out["dropout_started"]
    assert started.sum() > 300
    assert np.array_equal(started & ~stale, np.zeros_like(started))

    lengths = []
    for env in range(n):
        for k in np.flatnonzero(started[:, env]):
            j = k + 1
            while j < steps and stale[j, env] and not started[j, env]:
                j += 1
            if j < steps:  # skip dropouts cut off by the end of the run
                lengths.append(j - k)
            # Output frozen at the last value, age counting up (it carries on when a dropout
            # starts right after another: the output has been stale the whole time).
            held = np.broadcast_to(out["pos"][k - 1, env], (j - k, 3))
            np.testing.assert_array_equal(out["pos"][k:j, env], held)
            if not stale[k - 1, env]:
                np.testing.assert_allclose(out["age"][k:j, env], DT * np.arange(1, j - k + 1))
    durations = np.array(lengths) * DT
    assert durations.min() >= 0.2 - 1e-9 and durations.max() <= 3.0 + 1e-9
    assert durations.min() < 0.3 and durations.max() > 2.9
    assert not out["age"][~stale].any()


def test_seeded_runs_identical_and_seeds_differ():
    cfg = EstimateCorruptionConfig(jump_rate_hz=1.0, dropout_rate_hz=1.0, delay_max_s=0.1)
    pos, vel = ramp(500, 8)
    a = run(EstimateCorruptor(cfg, 8, DT, seed=7), pos, vel)
    b = run(EstimateCorruptor(cfg, 8, DT, seed=7), pos, vel)
    c = run(EstimateCorruptor(cfg, 8, DT, seed=8), pos, vel)
    for f in a:
        np.testing.assert_array_equal(a[f], b[f])
    assert not np.array_equal(a["pos"], c["pos"])


def test_reset_subset_leaves_other_envs_alone():
    cfg = only(bias_walk_std_m_per_sqrt_s=0.5, bias_max_m=5.0, delay_min_s=0.03, delay_max_s=0.03)
    model = EstimateCorruptor(cfg, 4, DT, seed=9)
    z = np.zeros((4, 3))
    model.reset(z, z)
    for _ in range(50):
        model.step(z, z)
    bias_before = model.bias.copy()
    start = np.full((4, 3), 7.0)
    model.reset(start, start, env_ids=[1, 3])
    np.testing.assert_array_equal(model.bias[[0, 2]], bias_before[[0, 2]])
    assert not model.bias[[1, 3]].any()
    s = model.step(start, start)
    # Reset envs report the new start (plus one step of bias); the others still see zeros.
    assert np.allclose(s.pos[[1, 3]], 7.0, atol=0.3)
    assert np.allclose(s.pos[[0, 2]], 0.0, atol=2.0)


def test_step_before_reset_and_bad_configs_raise():
    with pytest.raises(RuntimeError):
        EstimateCorruptor(OFF, 2, DT, 0).step(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        EstimateCorruptionConfig.from_dict({"pos_noise": 0.1})
    with pytest.raises(ValueError):
        EstimateCorruptor(only(jump_min_m=3.0), 1, DT, 0)
    with pytest.raises(ValueError):
        EstimateCorruptor(only(pos_noise_std_m=-1.0), 1, DT, 0)
    with pytest.raises(ValueError):
        EstimateCorruptor(
            only(dropout_rate_hz=1.0, dropout_min_s=0.001, dropout_max_s=0.002), 1, DT, 0
        )
    cfg = EstimateCorruptionConfig.from_dict({"pos_noise_std_m": 0.1, "jump_decay_s": None})
    assert cfg.pos_noise_std_m == 0.1 and cfg.jump_decay_s is None


def test_dead_reckoned_dropout_tracks_true_motion_and_drifts_with_the_bias():
    n, steps = 64, 300
    bias = 0.5
    cfg = only(dropout_rate_hz=1e4, dropout_min_s=3.0, dropout_max_s=3.0,
               dropout_accel_bias_mps2=bias)  # one 3 s dropout starting at the first step
    _, vel = ramp(steps, n)
    pos = np.cumsum(vel, axis=0) * DT  # a trajectory consistent with its velocity
    model = EstimateCorruptor(cfg, n, DT, seed=4)
    out = run(model, pos, vel)
    assert out["stale"][1:].all() and out["dropout_started"][0].all()
    # Velocity error is the bias times the age; position error grows with age squared.
    age = out["age"]
    v_err = out["vel"] - vel
    p_err = out["pos"] - pos
    per_env_bias = v_err[-1] / age[-1][:, None]
    np.testing.assert_allclose(v_err, per_env_bias[None] * age[..., None], atol=1e-9)
    assert abs(per_env_bias.std() - bias) < 0.15 * bias  # drawn as N(0, bias) per axis
    expect_p = 0.5 * per_env_bias[None] * age[..., None] ** 2
    np.testing.assert_allclose(p_err[-1], expect_p[-1], rtol=1e-2, atol=0.01)  # discrete sum
    # Without the bias the dead reckoning is exact (no noise in this config).
    exact = run(EstimateCorruptor(only(dropout_rate_hz=1e4, dropout_min_s=3.0,
                                       dropout_max_s=3.0, dropout_accel_bias_mps2=0.0),
                                  n, DT, seed=4), pos, vel)
    np.testing.assert_allclose(exact["vel"], vel, atol=1e-9)
    np.testing.assert_allclose(exact["pos"], pos, atol=1e-6)


def test_bad_dropout_mode_raises():
    with pytest.raises(ValueError, match="dropout_mode"):
        EstimateCorruptor(only(dropout_mode="freeze"), 2, DT, seed=0)


def test_fault_start_holds_faults_back_and_keeps_the_later_draws():
    # Jumps and dropouts at rates that fire often; everything else as the defaults.
    busy = EstimateCorruptionConfig(jump_rate_hz=1.0, dropout_rate_hz=1.0)
    late = EstimateCorruptionConfig(jump_rate_hz=1.0, dropout_rate_hz=1.0, fault_start_s=2.0)
    n, steps, start = 256, 800, 200  # 8 s at 0.01 s; faults from step 200 (2 s) on
    pos, vel = ramp(steps, n)
    a = run(EstimateCorruptor(busy, n, DT, seed=4), pos, vel)
    b = run(EstimateCorruptor(late, n, DT, seed=4), pos, vel)
    assert a["jumped"][:start].any() and a["dropout_started"][:start].any()
    assert not b["jumped"][:start].any() and not b["dropout_started"][:start].any()
    assert not b["stale"][:start].any()
    # The same draws: every jump from the start on is the same event, step for step.
    np.testing.assert_array_equal(b["jumped"][start:], a["jumped"][start:])
    assert b["jumped"][start:].sum() > 50 and b["dropout_started"][start:].sum() > 50
    # Envs whose first fault came after the start see exactly the same estimate.
    clean_early = ~(a["jumped"][:start].any(0) | a["dropout_started"][:start].any(0))
    np.testing.assert_array_equal(b["pos"][:, clean_early], a["pos"][:, clean_early])

    # A reset restarts the delay for the envs it resets, and only for them.
    model = EstimateCorruptor(late, n, DT, seed=4)
    model.reset(pos[0], vel[0])
    for k in range(400):
        model.step(pos[k], vel[k])
    model.reset(pos[400], vel[400], env_ids=np.arange(n // 2))
    jumped = np.stack([model.step(pos[k], vel[k]).jumped for k in range(400, 600)])
    assert not jumped[:, : n // 2].any() and jumped[:, n // 2:].any()



def test_poor_seeing_is_off_by_default_and_scales_what_it_says():
    """degrade_*: a share of the episodes gets noisier, walks further and loses the fix and
    jumps more often; the others, and every config without it, fly the estimate they had."""
    from dataclasses import replace

    n = 400
    pos, vel = ramp(600, n)
    base = EstimateCorruptionConfig()
    a = run(EstimateCorruptor(base, n, DT, seed=4), pos, vel)
    # Ranges without a probability do nothing, and neither does a probability without ranges.
    for off in (replace(base, degrade_noise_scale=(2.0, 4.0)), replace(base, degrade_prob=1.0)):
        assert not off.degrades
        b = run(EstimateCorruptor(off, n, DT, seed=4), pos, vel)
        np.testing.assert_array_equal(a["pos"], b["pos"])
        np.testing.assert_array_equal(a["stale"], b["stale"])

    cfg = replace(base, degrade_prob=0.5, degrade_noise_scale=(3.0, 3.0),
                  degrade_walk_scale=(2.0, 2.0), degrade_jump_rate_scale=(10.0, 10.0),
                  degrade_dropout_rate_scale=(20.0, 20.0))
    assert cfg.degrades
    model = EstimateCorruptor(cfg, n, DT, seed=4)
    c = run(model, pos, vel)
    poor = model.quality[:, 0] > 1.0
    assert 0.4 < poor.mean() < 0.6
    np.testing.assert_array_equal(model.quality[poor], np.tile([3.0, 2.0, 10.0, 20.0],
                                                               (poor.sum(), 1)))
    # The episodes with the standard estimate are the ones they were, bit for bit.
    np.testing.assert_array_equal(c["pos"][:, ~poor], a["pos"][:, ~poor])
    np.testing.assert_array_equal(c["stale"][:, ~poor], a["stale"][:, ~poor])
    # The poor ones lose the fix and jump far more often.
    assert c["dropout_started"][:, poor].mean() > 5 * max(a["dropout_started"][:, poor].mean(),
                                                           1e-5)
    assert c["jumped"][:, poor].mean() > 4 * max(a["jumped"][:, poor].mean(), 1e-5)
    # A new episode draws its seeing again.
    before = model.quality.copy()
    model.reset(pos[0], vel[0], env_ids=np.arange(n // 2))
    assert (model.quality[: n // 2] != before[: n // 2]).any()
    np.testing.assert_array_equal(model.quality[n // 2:], before[n // 2:])


def test_seeing_factors_scale_the_noise_and_the_walk():
    n = 2000
    still = np.zeros((300, n, 3))
    noisy = only(pos_noise_std_m=0.02, vel_noise_std_mps=0.05)
    model = EstimateCorruptor(noisy, n, DT, seed=0)
    model.reset(still[0], still[0])
    model.set_quality([4.0, 1.0, 1.0, 1.0], env_ids=np.arange(n // 2))
    out = model.step(still[0], still[0])
    assert out.pos[: n // 2].std() == pytest.approx(0.08, rel=0.1)
    assert out.pos[n // 2:].std() == pytest.approx(0.02, rel=0.1)
    assert out.vel[: n // 2].std() == pytest.approx(0.2, rel=0.1)

    walk = only(bias_walk_std_m_per_sqrt_s=0.01, bias_max_m=0.05)
    model = EstimateCorruptor(walk, n, DT, seed=0)
    model.reset(still[0], still[0])
    model.set_quality([1.0, 3.0, 1.0, 1.0], env_ids=np.arange(n // 2))
    for _ in range(100):
        out = model.step(still[0], still[0])
    assert out.pos[: n // 2].std() == pytest.approx(0.03, rel=0.15)  # 3 x 0.01 x sqrt(1 s)
    assert out.pos[n // 2:].std() == pytest.approx(0.01, rel=0.15)
    for _ in range(3000):
        out = model.step(still[0], still[0])
    assert np.abs(out.pos[: n // 2]).max() == pytest.approx(0.15, abs=1e-9)  # the cap scales too
    assert np.abs(out.pos[n // 2:]).max() == pytest.approx(0.05, abs=1e-9)


def test_seeing_config_is_validated_and_read_from_yaml_lists():
    cfg = EstimateCorruptionConfig.from_dict({"degrade_prob": 0.3,
                                              "degrade_noise_scale": [1, 5],
                                              "degrade_walk_scale": 2})
    assert cfg.degrade_noise_scale == (1.0, 5.0) and cfg.degrade_walk_scale == (2.0, 2.0)
    cfg.validate(DT)
    for bad in (dict(degrade_prob=1.5), dict(degrade_noise_scale=(3.0, 1.0)),
                dict(degrade_walk_scale=(-1.0, 1.0))):
        with pytest.raises(ValueError):
            EstimateCorruptionConfig(**bad).validate(DT)
