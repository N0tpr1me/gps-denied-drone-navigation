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


# The axis swaps as matrices: ENU coordinates to NED ones, and FRD body axes to FLU ones.
_ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = (float(c) for c in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)``, w >= 0, of a rotation matrix."""
    r = np.asarray(r, dtype=np.float64)
    w = 0.5 * np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2]))
    x = 0.5 * np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
    y = 0.5 * np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2]))
    z = 0.5 * np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2]))
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def enu_flu_quat_to_ned_frd(q: np.ndarray) -> np.ndarray:
    """An attitude ``(w, x, y, z)`` of an FLU body in the ENU world (Gazebo's pose) as the
    FRD body in the NED world (what PX4 and MAVLink take)."""
    return matrix_to_quat(_ENU_TO_NED @ quat_to_matrix(q) @ _FRD_TO_FLU)
