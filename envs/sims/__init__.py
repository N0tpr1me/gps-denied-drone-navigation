"""Rigid-body quadrotor backends behind one seam.

A backend moves a rigid body and nothing else: ``reset(env_ids, state)`` and
``step(thrust, torque_body, force_ext_world) -> QuadState``. Wind, estimate corruption, the
rate loop, observations and reward live in ``envs/structure_hold`` and never know which
backend is underneath. ``make_sim`` picks the backend with a plain ``if``: ``torch`` is the
CPU mock in ``torch_quad.py``; ``isaac`` is the Isaac Lab backend that arrives with the GPU
server (not implemented yet, so asking for it fails loudly).
"""

from __future__ import annotations

from envs.sims.quad_state import QuadState
from envs.sims.rate_pid import RatePid, RatePidGains
from envs.sims.torch_quad import QuadParams, TorchQuadSim

BACKENDS = ("torch", "isaac")


def make_sim(backend: str, params: QuadParams, num_envs: int, dt: float, device: str = "cpu"):
    if backend == "torch":
        return TorchQuadSim(params, num_envs, dt, device=device)
    if backend == "isaac":
        raise NotImplementedError(
            "the Isaac Lab backend (envs/sims/isaac_quad.py) is written once the GPU server "
            "exists; use --sim torch until then"
        )
    raise ValueError(f"unknown simulator backend {backend!r}; choose one of {BACKENDS}")


__all__ = [
    "BACKENDS",
    "QuadParams",
    "QuadState",
    "RatePid",
    "RatePidGains",
    "TorchQuadSim",
    "make_sim",
]
