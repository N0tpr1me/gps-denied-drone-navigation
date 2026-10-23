from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from envs.sims import QuadParams, RatePid, RatePidGains, TorchQuadSim  # noqa: E402

DT = 0.01


def test_proportional_term_and_scaling():
    pid = RatePid(RatePidGains(i=(0, 0, 0), d=(0, 0, 0)), max_torque=(0.3, 0.3, 0.05),
                  num_envs=2, dt=DT)
    out = pid(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), torch.zeros(2, 3))
    assert torch.allclose(out[0], torch.tensor([0.15 * 0.3, 0.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.0, 0.20 * 0.05]))


def test_output_saturates_and_integral_is_limited():
    pid = RatePid(RatePidGains(d=(0, 0, 0)), max_torque=(1.0, 1.0, 1.0), num_envs=1, dt=DT)
    for _ in range(500):
        out = pid(torch.full((1, 3), 50.0), torch.zeros(1, 3))
    assert torch.allclose(out, torch.ones(1, 3))
    assert (pid.integral <= torch.tensor(pid.gains.int_lim) + 1e-6).all()


def test_no_derivative_kick_on_setpoint_step():
    pid = RatePid(RatePidGains(p=(0, 0, 0), i=(0, 0, 0), d=(0.01, 0.01, 0.01)),
                  max_torque=(1.0, 1.0, 1.0), num_envs=1, dt=DT)
    pid(torch.zeros(1, 3), torch.zeros(1, 3))
    out = pid(torch.ones(1, 3), torch.zeros(1, 3))  # setpoint steps, measurement does not
    assert torch.allclose(out, torch.zeros(1, 3))
    out = pid(torch.ones(1, 3), torch.ones(1, 3))  # measurement steps: D opposes it
    assert (out < 0).all()


def test_closed_loop_tracks_a_rate_step_on_the_mock():
    n = 3
    params = QuadParams()
    sim = TorchQuadSim(params, n, DT)
    pid = RatePid(RatePidGains(), params.max_torque, n, DT)
    rate_sp = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.5]])
    thrust = torch.full((n,), params.hover_thrust)
    for _ in range(100):  # 1 s
        s = sim.step(thrust, pid(rate_sp, sim.state.ang_vel), torch.zeros(n, 3))
    assert torch.allclose(s.ang_vel, rate_sp, atol=0.1)


def _tracking(gains: RatePidGains, f_hz: float, axis: int, dt: float = 0.004):
    """Gain and lag (deg) of the PID output against its pure-P value for a sine error."""
    pid = RatePid(gains, max_torque=(1.0, 1.0, 1.0), num_envs=1, dt=dt)
    t = torch.arange(0, 4.0, dt)
    err = 0.5 * torch.sin(2 * torch.pi * f_hz * t)
    out = torch.empty_like(t)
    for k in range(len(t)):
        sp = torch.zeros(1, 3)
        sp[0, axis] = err[k]
        out[k] = pid(sp, torch.zeros(1, 3))[0, axis]
    late = t > 2.0  # past the start transient
    ref = torch.exp(-2j * torch.pi * f_hz * t[late])
    a = (out[late] * ref).sum() / (gains.p[axis] * err[late] * ref).sum()
    return float(a.abs()), float(-torch.rad2deg(torch.angle(a)))


def test_px4_filters_are_off_by_default_and_change_nothing():
    pid = RatePid(RatePidGains(), max_torque=(1.0, 1.0, 1.0), num_envs=1, dt=DT)
    assert pid.gyro_alpha == 1.0 and pid.yaw_alpha == 1.0
    gains = RatePidGains(i=(0, 0, 0), d=(0, 0, 0))
    gain, lag = _tracking(gains, 3.0, 2)
    assert gain == pytest.approx(1.0, abs=1e-3) and abs(lag) < 1.0


def test_yaw_torque_filter_is_px4s_2_hz_low_pass_on_yaw_only():
    gains = RatePidGains(i=(0, 0, 0), d=(0, 0, 0), yaw_torque_cutoff_hz=2.0)
    # First-order 2 Hz: at 3 Hz the gain is 1/sqrt(1 + 1.5^2) = 0.55, the lag 56 deg.
    gain, lag = _tracking(gains, 3.0, 2)
    assert gain == pytest.approx(0.55, abs=0.03) and lag == pytest.approx(56.0, abs=4.0)
    gain, lag = _tracking(gains, 3.0, 0)  # roll is untouched
    assert gain == pytest.approx(1.0, abs=1e-3) and abs(lag) < 1.0


def test_gyro_filter_low_passes_the_measured_rate():
    pid = RatePid(RatePidGains(i=(0, 0, 0), d=(0, 0, 0), gyro_cutoff_hz=40.0),
                  max_torque=(1.0, 1.0, 1.0), num_envs=1, dt=0.004)
    pid.reset(rate=torch.zeros(1, 3))
    pid(torch.zeros(1, 3), torch.ones(1, 3))  # the rate steps to 1 rad/s
    # One 4 ms step of a 40 Hz first-order filter passes alpha = 0.50 of the step.
    assert pid.gyro[0, 0] == pytest.approx(0.501, abs=0.01)
    for _ in range(50):
        pid(torch.zeros(1, 3), torch.ones(1, 3))
    assert torch.allclose(pid.gyro, torch.ones(1, 3), atol=1e-4)
