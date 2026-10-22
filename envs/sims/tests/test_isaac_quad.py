"""The Isaac backend's own code, without Isaac Lab: ``IsaacQuadSim`` over the PhysX stand-in.

The physics inside Isaac Lab is checked against the mock by the parity check
(``eval/sim_parity.py``) on a machine that has Isaac Lab; these tests make sure everything
this repo adds around PhysX (frames, clipping, motor lag, resets, origins, the lazy import)
is right on every CI run.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from envs.sims import QuadParams, QuadState, make_sim  # noqa: E402
from envs.sims.isaac_quad import (  # noqa: E402
    IsaacLabUnavailable,
    IsaacQuadSim,
    isaac_available,
    parameter_map,
)
from envs.sims.tests.fake_physx import FakePhysxWorld  # noqa: E402
from envs.sims.torch_quad import quat_from_axis_angle  # noqa: E402

DT = 0.01


def glue(params: QuadParams, n: int) -> IsaacQuadSim:
    return IsaacQuadSim(params, n, DT, device="cpu",
                        world=FakePhysxWorld(params, n, DT / params.substeps))


def random_start(n: int, g: torch.Generator) -> QuadState:
    axis = torch.randn(n, 3, generator=g)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    quat = quat_from_axis_angle(axis, 0.6 * torch.rand(n, generator=g))
    return QuadState(torch.randn(n, 3, generator=g), quat, torch.randn(n, 3, generator=g),
                     2.0 * torch.randn(n, 3, generator=g))


def test_importing_the_backend_does_not_import_isaac_lab():
    code = ("import sys, envs.sims, envs.sims.isaac_quad; "
            "assert not any(m.split('.')[0] in ('isaaclab', 'isaacsim', 'omni') "
            "for m in sys.modules), 'Isaac imported eagerly'")
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(isaac_available(), reason="Isaac Lab is installed here")
def test_isaac_without_isaac_lab_says_where_to_look():
    with pytest.raises(IsaacLabUnavailable, match="docs/isaac-lab.md"):
        make_sim("isaac", QuadParams(), 4, DT)


def test_every_quad_param_is_mapped():
    mapped = {row[0] for row in parameter_map(QuadParams(), DT)}
    assert {f.name for f in dataclasses.fields(QuadParams)} <= mapped


def test_glue_matches_the_mock_step_for_step():
    """Same inputs, same random starts (tilted, moving, spinning) and per-env resets: the
    Isaac glue over a PhysX-like integrator tracks the mock to float32 rounding, so frames,
    clipping, motor lag and resets agree."""
    g = torch.Generator().manual_seed(0)
    n, params = 64, QuadParams()
    mock = make_sim("torch", params, n, DT)
    isaac = glue(params, n)
    start = random_start(n, g)
    mock.reset(state=start)
    isaac.reset(state=start)
    worst = 0.0
    for k in range(150):
        thrust = params.hover_thrust * (1 + 0.6 * torch.randn(n, generator=g))
        torque = 0.4 * torch.tensor(params.max_torque) * torch.randn(n, 3, generator=g)
        force = 0.5 * torch.randn(n, 3, generator=g)
        a = mock.step(thrust, torque, force)
        b = isaac.step(thrust, torque, force)
        if k == 70:  # reset a few envs mid-run, with a new random state
            ids = torch.tensor([3, 17, 40])
            s = random_start(3, g)
            mock.reset(ids, s)
            isaac.reset(ids, s)
        for x, y in zip(a, b, strict=True):
            worst = max(worst, float((x - y).abs().max()))
    assert worst < 1e-3, worst
    assert torch.allclose(mock.thrust, isaac.thrust) and torch.allclose(mock.torque, isaac.torque)


def test_state_is_relative_to_each_env_origin():
    sim = glue(QuadParams(), 16)
    assert sim.world.origins.abs().sum() > 0  # the stand-in places envs on a grid
    sim.reset(state=QuadState.hover(16, pos=(1.0, 2.0, 3.0)))
    s = sim.step(torch.full((16,), sim.params.hover_thrust), torch.zeros(16, 3),
                 torch.zeros(16, 3))
    assert torch.allclose(s.pos, torch.tensor([1.0, 2.0, 3.0]).expand(16, 3), atol=1e-4)


def test_a_replaced_world_refuses_to_step():
    sim = glue(QuadParams(), 2)
    sim.close()
    with pytest.raises(RuntimeError, match="one physics scene per process"):
        sim.step(torch.zeros(2), torch.zeros(2, 3), torch.zeros(2, 3))


def test_float64_is_refused():
    with pytest.raises(ValueError, match="float32"):
        IsaacQuadSim(QuadParams(), 2, DT, device="cpu", dtype=torch.float64,
                     world=FakePhysxWorld(QuadParams(), 2, DT))
