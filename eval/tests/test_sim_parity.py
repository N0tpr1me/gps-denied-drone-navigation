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
from envs.sims.torch_quad import quat_rotate  # noqa: E402
from eval import sim_parity  # noqa: E402


def glue(params, n, dt, device):
    return IsaacQuadSim(params, n, dt, device=device,
                        world=FakePhysxWorld(params, n, dt / params.substeps, device))


def wrong(**changes):
    """A backend that gets one physical parameter wrong."""
    return lambda params, n, dt, device: make_sim("torch", replace(params, **changes), n, dt,
                                                  device=device)


class Wrapped:
    """The mock behind a wrapper; subclasses get one thing wrong."""

    def __init__(self, params, n, dt, device):
        self.sim = make_sim("torch", params, n, dt, device=device)

    def reset(self, *a, **kw):
        self.sim.reset(*a, **kw)

    def set_airframe(self, *a, **kw):
        self.sim.set_airframe(*a, **kw)

    def step(self, thrust, torque, force, moment=None):
        return self.sim.step(thrust, torque, force, moment)


class SwappedTorqueAxes(Wrapped):
    """Applies roll torque about pitch and vice versa (a frame slip)."""

    def step(self, thrust, torque, force, moment=None):
        return self.sim.step(thrust, torque[:, [1, 0, 2]], force, moment)


class IgnoresAirframe(Wrapped):
    """Flies every env as QuadParams whatever ``set_airframe`` says."""

    def set_airframe(self, *a, **kw):
        pass


class IgnoresAirframeMotorLag(Wrapped):
    """Takes the airframe but keeps the nominal motor lag."""

    def set_airframe(self, *a, **kw):
        kw.pop("motor_tau", None)
        self.sim.set_airframe(*a, **kw)


class WorldFrameMoment(Wrapped):
    """Applies the external moment about the world axes instead of the body axes."""

    def step(self, thrust, torque, force, moment=None):
        if moment is not None:
            conj = torch.cat([self.sim.state.quat[:, :1], -self.sim.state.quat[:, 1:]], -1)
            moment = quat_rotate(conj, moment)
        return self.sim.step(thrust, torque, force, moment)


TORCH = sim_parity.factory("torch")


def test_finer_mock_is_within_the_open_loop_tolerances():
    report = sim_parity.open_loop(TORCH, sim_parity.factory("torch-fine"))
    assert report["ok"], report["failures"]


def test_finer_mock_is_within_the_closed_loop_tolerances():
    report = sim_parity.closed_loop("torch", "torch-fine", num_envs=128, seconds=10.0)
    assert report["ok"], report["failures"]


def test_finer_mock_is_within_the_randomised_closed_loop_tolerances():
    report = sim_parity.closed_loop("torch", "torch-fine", num_envs=128, seconds=10.0,
                                    env=sim_parity.RANDOMISED, tol=sim_parity.RANDOMISED_TOL)
    assert report["ok"], report["failures"]


def test_isaac_glue_passes_open_loop_parity_tightly():
    tight = sim_parity.OpenLoopTolerance(pos_m=1e-4, vel_m_s=1e-4, att_rad=1e-4,
                                         rate_rad_s=1e-4)
    report = sim_parity.open_loop(TORCH, glue, tol=tight)
    assert report["ok"], report["failures"]


def test_isaac_glue_passes_the_randomised_closed_loop(monkeypatch):
    """The whole chain the trainer uses: the env draws an airframe per env, hands it to the
    backend through ``set_airframe`` and the wall yaw moment through the fourth ``step``
    argument; the Isaac glue over the PhysX stand-in flies it like the mock."""
    from envs.structure_hold import env as hold_env

    real = hold_env.make_sim

    def make(name, params, n, dt, device="cpu"):
        return glue(params, n, dt, device) if name == "isaac" else real(name, params, n, dt,
                                                                       device=device)

    monkeypatch.setattr(hold_env, "make_sim", make)
    report = sim_parity.closed_loop("torch", "isaac", num_envs=64, seconds=5.0,
                                    env=sim_parity.RANDOMISED, tol=sim_parity.RANDOMISED_TOL)
    assert report["ok"], report["failures"]
    assert report["pos_median_m"] < 1e-3, report


@pytest.mark.parametrize("make, scenario", [
    (wrong(mass=0.8 * 1.05), "climb"),
    (wrong(inertia=(0.0035 * 1.1, 0.0035, 0.0060)), "roll_pulse"),
    (wrong(motor_tau=0.04), "saturated"),
    (SwappedTorqueAxes, "pitch_pulse"),
    (IgnoresAirframe, "other_airframe"),
    (IgnoresAirframeMotorLag, "other_airframe"),
    (WorldFrameMoment, "external_moment"),
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
    cl = sim_parity.closed_loop("torch", "isaac", num_envs=256, seconds=10.0, device="cuda",
                                env=sim_parity.RANDOMISED, tol=sim_parity.RANDOMISED_TOL)
    assert cl["ok"], cl["failures"]
