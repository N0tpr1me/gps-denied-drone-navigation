"""PyTorch adapter for the disturbance model: the same functions on batched tensors.

Sample with ``core.sample_params`` (NumPy, once per reset), move the result with
``to_torch``, then call ``disturbance_force`` every physics step on the training device. The
maths is the code in ``core``, called with ``torch`` as the array module.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import torch

from envs.disturbance import core
from envs.disturbance.core import DisturbanceParams


def to_torch(
    p: DisturbanceParams, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
) -> DisturbanceParams:
    """Copy NumPy params to tensors. ``has_wall`` stays bool; everything else is ``dtype``."""
    out = {}
    for f in fields(p):
        value = np.asarray(getattr(p, f.name))
        kind = torch.bool if value.dtype == np.bool_ else dtype
        out[f.name] = torch.as_tensor(value, dtype=kind, device=device)
    return DisturbanceParams(**out)


def _like(x: Any, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def wind_velocity(p: DisturbanceParams, t: Any) -> torch.Tensor:
    return core._wind(torch, p, _like(t, p.mean_wind))


def shelter_factor(p: DisturbanceParams, pos: torch.Tensor) -> torch.Tensor:
    return core._shelter(torch, p, pos)


def air_velocity(p: DisturbanceParams, t: Any, pos: torch.Tensor) -> torch.Tensor:
    return core._air(torch, p, _like(t, p.mean_wind), pos)


def drag_force(v_rel: torch.Tensor, k_lin: torch.Tensor, k_quad: torch.Tensor) -> torch.Tensor:
    return core._drag(torch, v_rel, k_lin, k_quad)


def disturbance_force(
    p: DisturbanceParams, t: Any, pos: torch.Tensor, vel: torch.Tensor
) -> torch.Tensor:
    """Aerodynamic force (N, 3) in N, world ENU. See ``core.disturbance_force``."""
    return core._force(torch, p, _like(t, p.mean_wind), pos, vel)
