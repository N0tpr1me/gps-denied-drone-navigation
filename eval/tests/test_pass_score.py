"""The surface pass scorer: a linear moving reference and the standoff / along-pass split."""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import Trace, compute_metrics
from eval.pass_score import moving_reference, moving_trace, pass_numbers


def _pass_reference(t: np.ndarray, start: float = 3.0, speed: float = 0.5,
                    length: float = 5.0) -> np.ndarray:
    """A hold point 1 m from the wall x = 0 that slides ``length`` along -y at ``speed``."""
    s = np.clip((t - start) * speed, 0.0, length)
    return np.stack([np.full_like(t, -1.0), -s, np.full_like(t, 2.5)], axis=1)


def test_moving_reference_is_linear_between_sparse_samples():
    ref_t = np.arange(0.0, 20.0, 0.1)  # PX4 logs its setpoint every 100 ms
    t = np.arange(0.0, 19.9, 0.004)
    got = moving_reference(t, ref_t, _pass_reference(ref_t))
    assert np.abs(got - _pass_reference(t)).max() < 0.03  # the corners of the ramp only
    cruise = (t > 3.2) & (t < 12.8)
    assert np.abs(got[cruise] - _pass_reference(t)[cruise]).max() < 1e-9
    # Holding each sample instead leaves the reference up to 5 cm behind at 0.5 m/s.
    idx = np.searchsorted(ref_t, t, side="right") - 1
    held = _pass_reference(ref_t)[idx]
    assert np.abs(held[cruise] - _pass_reference(t)[cruise]).max() > 0.04


def test_moving_reference_skips_bad_samples_and_holds_the_ends():
    ref_t = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    ref = np.array([[0, 0, 0], [0, 1, 0], [9, 9, 9], [np.nan, 0, 0], [0, 3, 0]], dtype=float)
    got = moving_reference(np.array([0.0, 1.5, 3.0, 9.0]), ref_t, ref)
    assert np.allclose(got[:, 1], [0.0, 0.5, 2.0, 3.0])
    with pytest.raises(ValueError):
        moving_reference(np.array([0.0]), np.array([1.0]), np.full((1, 3), np.nan))


def test_pass_numbers_split_the_error_along_the_normal_and_the_pass():
    t = np.arange(0.0, 20.0, 0.01)
    sp = _pass_reference(t)
    pos = sp.copy()
    pos[:, 0] += 0.10  # 10 cm toward the wall (normal +x) all the way
    moving = (t > 3.0) & (t < 13.0)
    pos[moving, 1] += 0.20  # 20 cm behind the point while it moves (it moves along -y)
    out = pass_numbers(Trace(t=t, pos=pos, pos_sp=sp), (1.0, 19.0), (1.0, 0.0, 0.0))
    assert out["travel_m"] == pytest.approx(5.0)
    assert out["travel_along_normal_m"] == pytest.approx(0.0)
    assert out["moving_s"] == pytest.approx(10.0, abs=0.05)
    assert out["standoff_p95_m"] == pytest.approx(0.10)
    assert out["standoff_max_toward_m"] == pytest.approx(0.10)
    assert out["standoff_max_m"] == pytest.approx(0.10)
    assert out["error_max_m"] == pytest.approx(np.hypot(0.1, 0.2))
    assert out["end_error_m"] == pytest.approx(0.10)
    assert out["along_p95_m"] == pytest.approx(0.20)
    assert out["along_mean_while_moving_m"] == pytest.approx(-0.20, abs=0.01)
    assert out["across_p95_m"] == pytest.approx(0.0, abs=1e-9)
    assert out["p95_while_moving_m"] == pytest.approx(np.hypot(0.1, 0.2), abs=1e-3)
    assert out["p95_while_still_m"] == pytest.approx(0.10, abs=1e-3)
    assert out["moving_from_s"] == pytest.approx(3.0, abs=0.05)
    assert out["moving_until_s"] == pytest.approx(13.0, abs=0.05)


def test_pass_numbers_on_a_plain_hold():
    t = np.arange(0.0, 10.0, 0.01)
    sp = np.tile([-1.0, 0.0, 2.5], (len(t), 1))
    pos = sp + [0.0, 0.05, 0.0]
    out = pass_numbers(Trace(t=t, pos=pos, pos_sp=sp), (0.0, 10.0), (1.0, 0.0, 0.0))
    assert out["travel_m"] == 0.0 and out["moving_s"] == 0.0
    assert out["standoff_p95_m"] == pytest.approx(0.0)
    assert "along_p95_m" not in out


def test_moving_trace_scores_a_perfect_pass_as_zero_whatever_the_log_rates():
    """Truth at 25 Hz and the setpoint at 10 Hz, both exactly on the pass: eval.metrics on
    the resampled trace reads a millimetre in the cruise, where holding the setpoint between
    truth samples would read centimetres."""
    t = np.arange(0.0, 20.0, 0.04)
    ref_t = np.arange(0.0, 20.0, 0.1)
    truth = Trace(t=t, pos=_pass_reference(t), pos_sp=np.zeros(3))
    trace = moving_trace(truth, ref_t, _pass_reference(ref_t))
    assert trace.meta["reference"] == "moving"
    assert np.allclose(np.diff(trace.t), 0.002)
    cruise = compute_metrics(trace, window=(4.0, 12.0))
    assert cruise["max_m"] < 0.0011  # one 2 ms step of the reference at 0.5 m/s
    whole = compute_metrics(trace, window=(1.0, 19.0))
    assert whole["max_m"] < 0.03  # the two corners of the ramp, between 100 ms samples
    held = Trace(t=t, pos=_pass_reference(t),
                 pos_sp=_pass_reference(ref_t)[np.searchsorted(ref_t, t, side="right") - 1])
    assert compute_metrics(held, window=(4.0, 12.0))["max_m"] > 0.04
