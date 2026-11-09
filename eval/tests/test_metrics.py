"""Synthetic traces with answers worked out by hand."""

import math

import numpy as np
import pytest

from eval.metrics import (
    Surface,
    Trace,
    compute_metrics,
    count_episodes,
    main,
    recovery_time,
)

HOLD = np.zeros(3)
WALL = Surface(point=[1.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])  # wall 1 m east of the hold


def line_x(t, x):
    """Positions moving only along ENU x."""
    return np.stack([x, np.zeros_like(t), np.zeros_like(t)], axis=1)


def test_constant_offset():
    t = np.arange(0, 10, 0.01)
    m = compute_metrics(Trace(t, line_x(t, np.full_like(t, 0.3)), HOLD))
    assert m["rms_m"] == pytest.approx(0.3)
    assert m["p95_m"] == pytest.approx(0.3)
    assert m["max_m"] == pytest.approx(0.3)
    assert m["contacts"] == 0
    assert math.isnan(m["worst_toward_surface_m"])


def test_sine_rms_and_p95():
    a, f = 0.2, 0.5  # 10 s = 5 full periods
    t = np.arange(0, 10 + 1e-9, 0.001)
    m = compute_metrics(Trace(t, line_x(t, a * np.sin(2 * np.pi * f * t)), HOLD))
    assert m["rms_m"] == pytest.approx(a / math.sqrt(2), rel=1e-3)
    # |sin| is below a*sin(q*pi/2) a fraction q of the time.
    assert m["p95_m"] == pytest.approx(a * math.sin(0.95 * math.pi / 2), rel=2e-3)
    assert m["max_m"] == pytest.approx(a, rel=1e-3)


def test_metrics_do_not_depend_on_log_rate():
    rng = np.random.default_rng(0)

    def x(t):
        return 0.15 * np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.sin(2 * np.pi * 1.1 * t)

    t_fast = np.arange(0, 20, 0.004)
    t_slow = np.sort(rng.uniform(0, 20, 600))  # ~30 Hz, irregular
    t_slow = np.concatenate([[0.0], t_slow, [20 - 0.004]])
    fast = compute_metrics(Trace(t_fast, line_x(t_fast, x(t_fast)), HOLD), WALL)
    slow = compute_metrics(Trace(t_slow, line_x(t_slow, x(t_slow)), HOLD), WALL)
    for k in ["rms_m", "p95_m", "max_m", "worst_toward_surface_m"]:
        assert slow[k] == pytest.approx(fast[k], rel=0.03), k


def test_toward_surface_and_clearance():
    t = np.linspace(0, 3, 301)
    x = np.interp(t, [0, 1, 2, 3], [0.0, 0.4, -0.7, 0.0])
    m = compute_metrics(Trace(t, line_x(t, x), HOLD), WALL)
    assert m["worst_toward_surface_m"] == pytest.approx(0.4)
    assert m["min_clearance_m"] == pytest.approx(0.6)
    assert m["max_m"] == pytest.approx(0.7)
    assert m["contacts"] == 0


def test_moving_away_only_scores_zero_toward_surface():
    t = np.linspace(0, 2, 201)
    m = compute_metrics(Trace(t, line_x(t, -0.5 * t), HOLD), WALL)
    assert m["worst_toward_surface_m"] == 0.0
    assert m["min_clearance_m"] == pytest.approx(1.0)


def test_toward_surface_uses_wall_normal_not_axes():
    # Wall to the north-west, normal (-1, 1, 0)/sqrt2; drone drifts 0.3 m straight north.
    wall = Surface(point=[-2.0, 2.0, 0.0], normal=[-1.0, 1.0, 0.0])
    t = np.linspace(0, 1, 101)
    pos = np.stack([np.zeros_like(t), 0.3 * t, np.zeros_like(t)], axis=1)
    m = compute_metrics(Trace(t, pos, HOLD), wall)
    assert m["worst_toward_surface_m"] == pytest.approx(0.3 / math.sqrt(2))
    assert m["min_clearance_m"] == pytest.approx(math.sqrt(8) - 0.3 / math.sqrt(2))


def test_contacts_count_episodes():
    t = np.linspace(0, 6, 601)
    # Two approaches to 0.9 m east (0.1 m from the wall), back to the hold between them.
    x = np.interp(t, [0, 1, 1.5, 2, 3, 3.5, 4, 6], [0, 0, 0.9, 0, 0, 0.9, 0, 0])
    m = compute_metrics(Trace(t, line_x(t, x), HOLD), WALL, contact_radius_m=0.25)
    assert m["contacts"] == 2
    assert m["min_clearance_m"] == pytest.approx(0.1)


def test_second_surface_scores_its_own_clearance_and_contacts():
    # A deck 0.8 m above the hold; the drone rises 0.6 m (0.2 m under it) once, the wall
    # 1 m east is never approached.
    deck = Surface(point=[0.0, 0.0, 0.8], normal=[0.0, 0.0, 1.0])
    t = np.linspace(0, 3, 301)
    z = np.interp(t, [0, 1, 1.5, 2, 3], [0.0, 0.0, 0.6, 0.0, 0.0])
    pos = np.stack([np.zeros_like(t), np.zeros_like(t), z], axis=1)
    m = compute_metrics(Trace(t, pos, HOLD), WALL, surface2=deck)
    assert m["worst_toward_surface2_m"] == pytest.approx(0.6)
    assert m["min_clearance_surface2_m"] == pytest.approx(0.2)
    assert m["min_clearance_m"] == pytest.approx(1.0)  # still the wall's
    assert m["contacts"] == 1
    assert "min_clearance_surface2_m" not in compute_metrics(Trace(t, pos, HOLD), WALL)


def test_contacts_from_simulator_flag():
    t = np.linspace(0, 1, 101)
    flag = np.zeros_like(t, dtype=bool)
    flag[10:20] = flag[50:51] = flag[80:] = True
    m = compute_metrics(Trace(t, line_x(t, np.zeros_like(t)), HOLD, contact=flag))
    assert m["contacts"] == 3


def test_count_episodes():
    assert count_episodes([]) == 0
    assert count_episodes([True]) == 1
    assert count_episodes([False, True, True, False, True]) == 2


def decay_trace(tau=0.5, t_end=10.0):
    t = np.arange(0, t_end, 0.001)
    err = np.where(t < 3, np.where(t >= 2, 0.5, 0.0), 0.5 * np.exp(-(t - 3) / tau))
    return t, err


def test_recovery_exponential_decay():
    t, err = decay_trace()
    m = compute_metrics(Trace(t, line_x(t, err), HOLD), gusts=[(2.0, 3.0)])
    expected = 0.5 * math.log(5)  # 0.5 exp(-s/0.5) = 0.1
    assert m["recovery_s"] == [pytest.approx(expected, abs=0.011)]
    assert m["recovery_max_s"] == pytest.approx(expected, abs=0.011)


def test_recovery_zero_when_already_inside():
    t = np.arange(0, 5, 0.01)
    m = compute_metrics(Trace(t, line_x(t, np.full_like(t, 0.05)), HOLD), gusts=[(1.0, 2.0)])
    assert m["recovery_s"] == [0.0]


def test_recovery_inf_when_next_gust_comes_first():
    t, err = decay_trace(tau=2.0)  # needs 3.2 s to reach 0.1 m
    m = compute_metrics(Trace(t, line_x(t, err), HOLD), gusts=[(2.0, 3.0), (5.0, 5.0)])
    assert m["recovery_s"][0] == math.inf
    assert m["recovery_max_s"] == math.inf


def test_recovery_needs_the_hold_and_restarts_on_a_break():
    t = np.arange(0, 10, 0.01)
    # Inside at 4 s, pops out 4.5 to 4.7 s, inside for good from 4.7 s.
    err = np.where((t < 4) | ((t >= 4.5) & (t < 4.7)), 0.3, 0.02)
    assert recovery_time(t, err, 3.0, t[-1], 0.1, 1.0) == pytest.approx(1.7)
    assert recovery_time(t, err, 3.0, 5.5, 0.1, 1.0) == math.inf  # hold cannot finish


def test_rate_roughness_ramp_and_square_wave():
    t = np.arange(0, 2, 0.01)
    pos = line_x(t, np.zeros_like(t))
    ramp = np.stack([3.0 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    m = compute_metrics(Trace(t, pos, HOLD, cmd_t=t, rates_cmd=ramp,
                              thrust_cmd=np.full_like(t, 0.6)))
    assert m["rate_roughness_rad_s2"] == pytest.approx(3.0)
    assert m["thrust_roughness_1_s"] == pytest.approx(0.0)

    square = np.zeros((len(t), 3))
    square[:, 2] = np.where(np.arange(len(t)) % 2 == 0, 0.5, -0.5)
    m = compute_metrics(Trace(t, pos, HOLD, cmd_t=t, rates_cmd=square))
    assert m["rate_roughness_rad_s2"] == pytest.approx(1.0 / 0.01)
    assert math.isnan(m["thrust_roughness_1_s"])


def test_window_limits_scoring():
    t = np.arange(0, 10, 0.01)
    x = np.where(t < 2, 1.5, 0.1)  # big take-off transient, then 10 cm off
    m = compute_metrics(Trace(t, line_x(t, x), HOLD), window=(3.0, 9.0))
    assert m["max_m"] == pytest.approx(0.1)
    assert m["duration_s"] == pytest.approx(6.0)


def test_setpoint_is_held_not_interpolated():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.zeros((3, 3))
    sp = np.array([[0, 0, 0], [0, 0, 0], [1.0, 0, 0]])  # setpoint steps at t = 2
    m = compute_metrics(Trace(t, pos, sp))
    assert m["max_m"] == pytest.approx(1.0)
    assert m["p95_m"] == pytest.approx(0.0)  # only the final grid sample sees the step


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(t=[0.0], pos=[[0, 0, 0]], pos_sp=HOLD),
        dict(t=[0.0, 0.0], pos=np.zeros((2, 3)), pos_sp=HOLD),
        dict(t=[0.0, 1.0], pos=[[0, 0, np.nan], [0, 0, 0]], pos_sp=HOLD),
        dict(t=[0.0, 1.0], pos=np.zeros((2, 3)), pos_sp=np.zeros((3, 3))),
        dict(t=[0.0, 1.0], pos=np.zeros((2, 3)), pos_sp=HOLD, rates_cmd=np.zeros((2, 3))),
    ],
)
def test_bad_traces_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Trace(**kwargs)


def test_cli_needs_both_wall_arguments(capsys):
    with pytest.raises(SystemExit):
        main(["x.ulg", "--wall-point", "1,0,0"])
