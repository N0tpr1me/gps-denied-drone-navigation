import math

import numpy as np
import pytest

from eval.height_split import ground_offset, split_error


def climb(t: np.ndarray, lift_at: float = 10.0, height: float = 2.5) -> np.ndarray:
    return np.clip((t - lift_at) * 0.5, 0.0, height)


def test_ground_offset_is_the_estimate_minus_truth_before_lift_off():
    t = np.arange(0.0, 40.0, 0.02)
    up = climb(t)
    # 0.3 m on the ground; whatever the estimate does once flying must not move the number.
    est = up + 0.3 + np.where(t > 20.0, 0.5, 0.0)
    assert ground_offset(t, up, est) == pytest.approx(0.3, abs=1e-9)


def test_ground_offset_is_nan_when_the_flight_never_lifts():
    t = np.arange(0.0, 10.0, 0.02)
    assert math.isnan(ground_offset(t, np.zeros_like(t), np.full_like(t, 0.2)))


def test_ground_offset_skips_the_estimate_before_it_is_valid():
    t = np.arange(0.0, 40.0, 0.02)
    up = climb(t)
    est = up + 0.1
    est[t < 8.0] = np.nan
    assert ground_offset(t, up, est) == pytest.approx(0.1, abs=1e-9)


def test_a_constant_frame_offset_is_all_of_the_vertical_error_and_comes_out():
    n = 1000
    err = np.zeros((n, 3))
    err[:, 0] = 0.1          # truth 0.1 m off horizontally the whole hold
    err[:, 2] = -0.4         # setpoint 0.4 m above the truth: the estimator's frame offset
    s = split_error(err, 0.4)
    assert s["h_p95_m"] == pytest.approx(0.1)
    assert s["v_p95_m"] == pytest.approx(0.4)
    assert s["p95_m"] == pytest.approx(math.hypot(0.1, 0.4))
    assert s["v_p95_no_offset_m"] == pytest.approx(0.0, abs=1e-12)
    assert s["p95_no_offset_m"] == pytest.approx(0.1)


def test_height_error_that_grows_during_the_hold_stays_in():
    n = 1000
    err = np.zeros((n, 3))
    err[:, 2] = -0.4 - np.linspace(0.0, 0.2, n)   # the offset, then the drone sinks 0.2 m
    s = split_error(err, 0.4)
    assert s["v_p95_no_offset_m"] == pytest.approx(0.19, abs=0.005)


def test_without_an_offset_the_corrected_numbers_are_nan_not_a_guess():
    s = split_error(np.full((10, 3), 0.1), math.nan)
    assert math.isnan(s["v_p95_no_offset_m"]) and math.isnan(s["p95_no_offset_m"])
    assert s["v_p95_m"] == pytest.approx(0.1)
