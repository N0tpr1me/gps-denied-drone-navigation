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
