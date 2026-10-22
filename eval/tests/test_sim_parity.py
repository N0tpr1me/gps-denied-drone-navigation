"""Simulator parity: the check that Isaac Lab must pass, and proof that the check has teeth.

Without Isaac Lab (CI, the Mac) this runs the check on stand-ins: the mock against itself at
an 8x finer step (where the tolerances come from), the Isaac glue over a PhysX stand-in, and
deliberately wrong backends that must fail. With Isaac Lab installed (docs/isaac-lab.md),
``test_isaac_lab_matches_the_mock`` runs the real thing on the GPU.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from envs.sims import QuadParams, make_sim  # noqa: E402
from envs.sims.isaac_quad import IsaacQuadSim, isaac_available  # noqa: E402
from envs.sims.tests.fake_physx import FakePhysxWorld  # noqa: E402
from eval import sim_parity  # noqa: E402


def glue(params, n, dt, device):
    return IsaacQuadSim(params, n, dt, device=device,
                        world=FakePhysxWorld(params, n, dt / params.substeps, device))


def wrong(**changes):
    """A backend that gets one physical parameter wrong."""
    return lambda params, n, dt, device: make_sim("torch", replace(params, **changes), n, dt,
                                                  device=device)


class SwappedTorqueAxes:
    """A backend that applies roll torque about pitch and vice versa (a frame slip)."""

    def __init__(self, params, n, dt, device):
        self.sim = make_sim("torch", params, n, dt, device=device)

    def reset(self, *a, **kw):
        self.sim.reset(*a, **kw)

    def step(self, thrust, torque, force):
        return self.sim.step(thrust, torque[:, [1, 0, 2]], force)


TORCH = sim_parity.factory("torch")


def test_finer_mock_is_within_the_open_loop_tolerances():
    report = sim_parity.open_loop(TORCH, sim_parity.factory("torch-fine"))
    assert report["ok"], report["failures"]


def test_finer_mock_is_within_the_closed_loop_tolerances():
    report = sim_parity.closed_loop("torch", "torch-fine", num_envs=128, seconds=10.0)
    assert report["ok"], report["failures"]


def test_isaac_glue_passes_open_loop_parity_tightly():
    tight = sim_parity.OpenLoopTolerance(pos_m=1e-4, vel_m_s=1e-4, att_rad=1e-4,
                                         rate_rad_s=1e-4)
    report = sim_parity.open_loop(TORCH, glue, tol=tight)
    assert report["ok"], report["failures"]


@pytest.mark.parametrize("make, scenario", [
    (wrong(mass=0.8 * 1.05), "climb"),
    (wrong(inertia=(0.0035 * 1.1, 0.0035, 0.0060)), "roll_pulse"),
    (wrong(motor_tau=0.04), "saturated"),
    (SwappedTorqueAxes, "pitch_pulse"),
])
def test_parity_catches_a_wrong_backend(make, scenario):
    report = sim_parity.open_loop(TORCH, make)
    assert not report["ok"]
    assert any(f.startswith(scenario) for f in report["failures"]), report["failures"]


def test_parity_command_line_writes_a_report(tmp_path):
    out = tmp_path / "parity.json"
    code = sim_parity.main(["--a", "torch", "--b", "torch-fine", "--device", "cpu",
                            "--num-envs", "32", "--seconds", "3", "--out", str(out)])
    assert code == 0 and out.exists()


ISAAC_SKIP = ("Isaac Lab not installed (docs/isaac-lab.md)" if not isaac_available() else
              "no CUDA GPU: Isaac Lab parity runs on the GPU" if not torch.cuda.is_available()
              else "")


@pytest.mark.skipif(bool(ISAAC_SKIP), reason=ISAAC_SKIP)
def test_isaac_lab_matches_the_mock():
    ol = sim_parity.open_loop(TORCH, sim_parity.factory("isaac"), QuadParams(), device="cuda")
    assert ol["ok"], ol["failures"]
    cl = sim_parity.closed_loop("torch", "isaac", num_envs=256, seconds=10.0, device="cuda")
    assert cl["ok"], cl["failures"]
