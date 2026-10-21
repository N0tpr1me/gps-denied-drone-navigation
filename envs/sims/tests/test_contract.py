"""Backend contract: every simulator behind the seam must pass these, parametrised by
backend. Only ``torch`` exists today; ``isaac`` is added to BACKENDS_UNDER_TEST when
envs/sims/isaac_quad.py lands, and this file checks it unchanged."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from envs.sims import QuadParams, QuadState, make_sim  # noqa: E402
from envs.sims.torch_quad import GRAVITY, quat_from_yaw, quat_rotate  # noqa: E402

BACKENDS_UNDER_TEST = ["torch"]
DT = 0.01
N = 4


def sim_for(backend: str, **kw):
    params = QuadParams(**kw) if kw else QuadParams()
    return make_sim(backend, params, N, DT)


def hover_inputs(sim, thrust=None):
    t = torch.full((N,), sim.params.hover_thrust if thrust is None else thrust)
    return t, torch.zeros(N, 3), torch.zeros(N, 3)


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_free_fall_matches_g(backend):
    sim = sim_for(backend, motor_tau=0.0)
    sim.reset(state=QuadState.hover(N, pos=(0.0, 0.0, 10.0)))
    steps = 100
    for _ in range(steps):
        s = sim.step(torch.zeros(N), torch.zeros(N, 3), torch.zeros(N, 3))
    t = steps * DT
    # Semi-implicit Euler lands within one sub-step's worth of 0.5 g t^2.
    assert torch.allclose(s.vel[:, 2], torch.full((N,), -GRAVITY * t), atol=1e-4)
    assert torch.allclose(s.pos[:, 2], torch.full((N,), 10.0 - 0.5 * GRAVITY * t**2),
                          atol=GRAVITY * t * sim.dt)
    assert torch.allclose(s.pos[:, :2], torch.zeros(N, 2))


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_hover_thrust_holds_still(backend):
    sim = sim_for(backend)
    sim.reset(state=QuadState.hover(N, pos=(1.0, -2.0, 3.0)))
    for _ in range(200):
        s = sim.step(*hover_inputs(sim))
    assert torch.allclose(s.pos, torch.tensor([1.0, -2.0, 3.0]).expand(N, 3), atol=1e-4)
    assert torch.allclose(s.vel, torch.zeros(N, 3), atol=1e-4)
    assert torch.allclose(s.quat, torch.tensor([1.0, 0, 0, 0]).expand(N, 4), atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_body_torque_gives_right_rotation_sign_and_rate(backend, axis):
    sim = sim_for(backend, motor_tau=0.0)
    sim.reset()
    torque = torch.zeros(N, 3)
    torque[:, axis] = 0.01
    steps = 50
    for _ in range(steps):
        s = sim.step(sim.params.hover_thrust * torch.ones(N), torque, torch.zeros(N, 3))
    expected = 0.01 / sim.params.inertia[axis] * steps * DT
    assert torch.allclose(s.ang_vel[:, axis], torch.full((N,), expected), rtol=1e-3)
    other = [i for i in range(3) if i != axis]
    assert s.ang_vel[:, other].abs().max() < 1e-3
    # Positive torque about +axis rotates body vectors by the right hand rule: the body
    # axis after the next one moves toward the one after that (x: y -> z, y: z -> x, z: x -> y).
    e = torch.zeros(N, 3)
    e[:, (axis + 1) % 3] = 1.0
    rotated = quat_rotate(s.quat, e)
    assert (rotated[:, (axis + 2) % 3] > 0).all()
    assert (rotated[:, (axis + 1) % 3] < 1.0).all()


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_external_force_gives_f_over_m(backend):
    sim = sim_for(backend, motor_tau=0.0)
    sim.reset()
    f = torch.tensor([0.4, -0.2, 0.0]).expand(N, 3)
    steps = 100
    for _ in range(steps):
        s = sim.step(sim.params.hover_thrust * torch.ones(N), torch.zeros(N, 3), f)
    expected = f[0] / sim.params.mass * steps * DT
    assert torch.allclose(s.vel, expected.expand(N, 3), atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_frames_are_enu_and_flu(backend):
    """Yawed 90 deg left (about +z, ENU up) the body x axis (FLU forward) points north."""
    sim = sim_for(backend, motor_tau=0.0)
    state = QuadState.hover(N)
    state = state._replace(quat=quat_from_yaw(torch.full((N,), math.pi / 2)))
    sim.reset(state=state)
    s = sim.step(*hover_inputs(sim))
    forward = quat_rotate(s.quat, torch.tensor([1.0, 0, 0]).expand(N, 3))
    assert torch.allclose(forward, torch.tensor([0.0, 1.0, 0.0]).expand(N, 3), atol=1e-5)
    # Thrust is along body +z, which is world up when level: more thrust means climbing.
    sim.reset()
    s = sim.step(*hover_inputs(sim, thrust=sim.params.max_thrust))
    assert (s.vel[:, 2] > 0).all()
    assert (s.pos[:, 2] > 0).all()


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_inputs_are_clipped_to_the_airframe(backend):
    sim = sim_for(backend, motor_tau=0.0)
    sim.reset()
    s = sim.step(torch.full((N,), -5.0), torch.full((N, 3), 100.0), torch.zeros(N, 3))
    assert (s.vel[:, 2] < 0).all()  # negative thrust clips to zero: it falls
    lim = torch.tensor(sim.params.max_torque)
    assert torch.allclose(sim.torque, lim.expand(N, 3))


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_partial_reset_touches_only_those_envs(backend):
    sim = sim_for(backend)
    sim.reset(state=QuadState.hover(N, pos=(5.0, 5.0, 5.0)))
    sim.reset(env_ids=torch.tensor([1, 3]), state=QuadState.hover(2, pos=(0.0, 0.0, 1.0)))
    assert torch.allclose(sim.state.pos[[0, 2]], torch.full((2, 3), 5.0))
    assert torch.allclose(sim.state.pos[[1, 3]], torch.tensor([0.0, 0.0, 1.0]).expand(2, 3))
    mask = torch.tensor([True, False, False, False])
    sim.reset(env_ids=mask, state=QuadState.hover(1, pos=(2.0, 0.0, 0.0)))
    assert torch.allclose(sim.state.pos[0], torch.tensor([2.0, 0.0, 0.0]))


@pytest.mark.parametrize("backend", BACKENDS_UNDER_TEST)
def test_bad_shapes_and_backends_fail_loudly(backend):
    sim = sim_for(backend)
    with pytest.raises(ValueError, match="thrust"):
        sim.step(torch.zeros(N, 1), torch.zeros(N, 3), torch.zeros(N, 3))
    with pytest.raises(ValueError, match="force_ext_world"):
        sim.step(torch.zeros(N), torch.zeros(N, 3), torch.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        sim.reset(state=QuadState.hover(N, pos=(float("nan"), 0.0, 0.0)))
    with pytest.raises(NotImplementedError, match="isaac"):
        make_sim("isaac", QuadParams(), N, DT)
    with pytest.raises(ValueError, match="unknown"):
        make_sim("gazebo", QuadParams(), N, DT)


def test_torch_sim_is_deterministic_and_fast_enough():
    """Same inputs, same state; and a few thousand envs step at least in real time."""
    import time

    params = QuadParams()
    a = make_sim("torch", params, 8, DT)
    b = make_sim("torch", params, 8, DT)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        thrust = params.hover_thrust * (1 + 0.1 * torch.randn(8, generator=g))
        torque = 0.01 * torch.randn(8, 3, generator=g)
        force = torch.randn(8, 3, generator=g)
        sa, sb = a.step(thrust, torque, force), b.step(thrust, torque, force)
    assert all(torch.equal(x, y) for x, y in zip(sa, sb, strict=True))

    big = make_sim("torch", params, 2048, DT)
    inputs = (torch.full((2048,), params.hover_thrust), torch.zeros(2048, 3), torch.zeros(2048, 3))
    big.step(*inputs)
    t0 = time.perf_counter()
    for _ in range(20):
        big.step(*inputs)
    per_step = (time.perf_counter() - t0) / 20
    # Real time at 100 Hz is 10 ms per step; allow a slow CI box 10x headroom.
    assert per_step < 0.1, f"{per_step * 1e3:.1f} ms per step for 2048 envs"
