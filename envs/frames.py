"""Frame conversions at the PX4 boundary (docs/conventions.md).

Our Python code is ENU (world) and FLU (body); PX4 is NED and FRD. These helpers are the
one place for the swap. They work on any array whose last axis is xyz.
"""

from __future__ import annotations

import numpy as np


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    """ENU (east, north, up) to NED (north, east, down), last axis xyz."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    """NED (north, east, down) to ENU (east, north, up), last axis xyz."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def flu_to_frd(v: np.ndarray) -> np.ndarray:
    """FLU (forward, left, up) to FRD (forward, right, down), last axis xyz."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 1], -v[..., 2]], axis=-1)


frd_to_flu = flu_to_frd  # the map is its own inverse
