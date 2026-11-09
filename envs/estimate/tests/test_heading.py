import math

import numpy as np
import pytest

from envs.estimate.heading import HeadingError, HeadingErrorConfig

DT = 0.01


def run(cfg: HeadingErrorConfig, steps: int, n: int = 64, seed: int = 3) -> np.ndarray:
    h = HeadingError(cfg, n, DT, seed)
    h.reset()
    out = [h.delta.copy()]
    for _ in range(steps):
        out.append(h.step().copy())
    return np.degrees(np.stack(out))  # (steps + 1, n), deg


def test_off_by_default():
    cfg = HeadingErrorConfig()
    assert not cfg.active
    assert np.all(run(cfg, 50) == 0.0)


def test_offset_is_there_at_engage_and_stays():
    d = run(HeadingErrorConfig(prob=1.0, offset_deg=(10.0, 30.0)), 200)
    assert np.all(np.abs(d[0]) >= 10.0 - 1e-9) and np.all(np.abs(d[0]) <= 30.0 + 1e-9)
    assert np.allclose(d, d[0])
    assert (d[0] > 0).any() and (d[0] < 0).any()  # both signs


def test_prob_leaves_some_episodes_clean():
    d = run(HeadingErrorConfig(prob=0.5, offset_deg=20.0), 1, n=400)
    clean = np.mean(d[0] == 0.0)
    assert 0.4 < clean < 0.6


def test_drift_starts_late_ramps_and_stops_at_its_cap():
    cfg = HeadingErrorConfig(prob=1.0, drift_rate_deg_s=2.0, drift_start_s=1.0,
                             drift_max_deg=5.0)
    d = np.abs(run(cfg, 500))
    t = np.arange(d.shape[0]) * DT
    assert np.all(d[t <= 1.0] < 1e-9)
    k = int(round(2.0 / DT))  # 1 s into the ramp: 2 deg
    assert np.allclose(d[k], 2.0, atol=1e-6)
    assert np.allclose(d[-1], 5.0)


def test_steps_are_followed_with_the_lag():
    cfg = HeadingErrorConfig(prob=1.0, step_rate_hz=0.5, step_deg=(20.0, 20.0), step_tau_s=2.0,
                             max_deg=180.0)
    h = HeadingError(cfg, 256, DT, 1)
    h.reset()
    for _ in range(int(10 / DT)):
        h.step()
    stepped = np.abs(h.target) > 0
    assert stepped.mean() > 0.5  # about 5 steps of random sign in 10 s
    # Every target is a whole number of 20 deg steps.
    k = np.degrees(h.target) / 20.0
    assert np.allclose(k, np.round(k), atol=1e-6)
    # One isolated step: after one time constant 63 % of it is in delta.
    one = HeadingError(HeadingErrorConfig(prob=1.0, step_rate_hz=1e9, step_deg=10.0,
                                          step_tau_s=2.0), 1, DT, 0)
    one.reset()
    one.step()  # the step lands (rate is huge) ...
    one.p_step = 0.0  # ... and no more come
    target = one.target[0]
    for _ in range(int(2.0 / DT) - 1):
        one.step()
    assert one.delta[0] / target == pytest.approx(1 - math.exp(-1), abs=0.01)


def test_clipped_at_max():
    cfg = HeadingErrorConfig(prob=1.0, offset_deg=40.0, drift_rate_deg_s=10.0,
                             drift_max_deg=40.0, max_deg=60.0)
    assert np.all(np.abs(run(cfg, 500)) <= 60.0 + 1e-9)


def test_reset_only_touches_the_named_envs_and_is_seeded():
    cfg = HeadingErrorConfig(prob=1.0, offset_deg=(5.0, 30.0), drift_rate_deg_s=1.0,
                             drift_max_deg=10.0)
    a, b = HeadingError(cfg, 8, DT, 5), HeadingError(cfg, 8, DT, 5)
    for h in (a, b):
        h.reset()
        for _ in range(100):
            h.step()
    assert np.array_equal(a.delta, b.delta)
    before = a.delta.copy()
    a.reset([2])
    assert np.array_equal(np.delete(a.delta, 2), np.delete(before, 2))
    assert a.delta[2] == a.offset[2] and a.t[2] == 0.0


def test_bad_config_refused():
    with pytest.raises(ValueError):
        HeadingErrorConfig(prob=1.5)
    with pytest.raises(ValueError):
        HeadingErrorConfig(offset_deg=(10.0, 5.0))
    with pytest.raises(ValueError):
        HeadingErrorConfig.from_dict({"yaw": 3})
    with pytest.raises(ValueError):
        HeadingErrorConfig(max_deg=200.0)
