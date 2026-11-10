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


# WGS84, the ellipsoid GPS receivers and Gazebo's spherical coordinates use.
WGS84_A = 6378137.0  # semi-major axis, m
WGS84_E2 = 6.69437999014e-3  # first eccentricity squared


def _ecef(lat_deg, lon_deg, alt_m) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)  # prime vertical radius
    return np.stack([(n + alt_m) * np.cos(lat) * np.cos(lon),
                     (n + alt_m) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + alt_m) * np.sin(lat)], axis=-1)


def geodetic_to_enu(lat_deg, lon_deg, alt_m, origin) -> np.ndarray:
    """Latitude and longitude (deg) and altitude (m) to ENU metres about ``origin`` (lat,
    lon, alt), last axis xyz. Exact on the WGS84 ellipsoid (through earth-centred
    coordinates), so it holds at a real site's distances as well as at a simulator's few
    metres. ``alt_m`` and the origin's altitude must be the same kind (both above the
    ellipsoid, or both above mean sea level: over a site the two differ by a constant)."""
    lat_deg, lon_deg, alt_m = (np.asarray(v, dtype=np.float64) for v in (lat_deg, lon_deg, alt_m))
    lat0, lon0 = np.radians(float(origin[0])), np.radians(float(origin[1]))
    d = _ecef(lat_deg, lon_deg, alt_m) - _ecef(float(origin[0]), float(origin[1]),
                                               float(origin[2]))
    sl, cl, so, co = np.sin(lat0), np.cos(lat0), np.sin(lon0), np.cos(lon0)
    east = -so * d[..., 0] + co * d[..., 1]
    north = -sl * co * d[..., 0] - sl * so * d[..., 1] + cl * d[..., 2]
    up = cl * co * d[..., 0] + cl * so * d[..., 1] + sl * d[..., 2]
    return np.stack([east, north, up], axis=-1)
