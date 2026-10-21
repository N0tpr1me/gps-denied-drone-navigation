"""The one data type that crosses the simulator seam."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch


class QuadState(NamedTuple):
    """Batched rigid-body state, world ENU and body FLU, SI units (docs/conventions.md)."""

    pos: Any  # (N, 3) m, world ENU
    quat: Any  # (N, 4) unit quaternion (w, x, y, z), body to world
    vel: Any  # (N, 3) m/s, world ENU
    ang_vel: Any  # (N, 3) rad/s, body FLU

    @property
    def num_envs(self) -> int:
        return int(self.pos.shape[0])

    def clone(self) -> QuadState:
        return QuadState(*(x.clone() for x in self))

    def finite(self) -> torch.Tensor:
        """(N,) True where every number of that env's state is finite."""
        return (
            torch.isfinite(self.pos).all(1)
            & torch.isfinite(self.quat).all(1)
            & torch.isfinite(self.vel).all(1)
            & torch.isfinite(self.ang_vel).all(1)
        )

    @staticmethod
    def hover(num_envs: int, pos: Any = (0.0, 0.0, 0.0), device: Any = "cpu",
              dtype: torch.dtype = torch.float32) -> QuadState:
        """At rest, level, at ``pos`` ((3,) or (N, 3))."""
        p = torch.as_tensor(pos, dtype=dtype, device=device).expand(num_envs, 3).clone()
        q = torch.zeros(num_envs, 4, dtype=dtype, device=device)
        q[:, 0] = 1.0
        z = torch.zeros(num_envs, 3, dtype=dtype, device=device)
        return QuadState(p, q, z, z.clone())

    @staticmethod
    def stack(states: list[QuadState]) -> QuadState:
        return QuadState(*(torch.cat(parts, 0) for parts in zip(*states, strict=True)))
