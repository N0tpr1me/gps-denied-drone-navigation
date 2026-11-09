"""The stereo rig: two pinhole cameras on the body, looking forward at the structure.

Camera frame is OpenCV's (x right, y down, z forward), as Basalt and Isaac Sim's
``opencv_pinhole`` model use. Pixel (0, 0) is the centre of the top-left pixel. The rig is
mounted on the body (FLU) ``lever_m`` ahead of the IMU, the two cameras ``baseline_m``
apart along body y (cam0 on the left), pitched down by ``pitch_deg`` so the floor and the
wall's foot are in view during take-off as well as the wall during the hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RigConfig:
    width: int = 640
    height: int = 480
    hfov_deg: float = 90.0
    baseline_m: float = 0.08
    lever_m: float = 0.10  # cameras ahead of the IMU (body x)
    pitch_deg: float = 10.0  # down from the body x axis

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RigConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown rig keys: {sorted(unknown)}")
        return cls(**data)

    @property
    def fx(self) -> float:
        return self.width / 2.0 / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    def body_from_cam(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(R, p): camera ``i``'s axes and centre in the body frame (T_body_cam)."""
        # OpenCV axes in FLU when level: x right = -y_b, y down = -z_b, z forward = x_b.
        r0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        p = math.radians(self.pitch_deg)
        # Pitch down about body y (nose down is +pitch in FLU about +y).
        ry = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0],
                       [-math.sin(p), 0.0, math.cos(p)]])
        side = 0.5 * self.baseline_m * (1.0 if i == 0 else -1.0)
        return ry @ r0, np.array([self.lever_m, side, 0.0])

    def pixel_rays(self, supersample: int = 1) -> np.ndarray:
        """(H, W, S, 3) unnormalised camera-frame ray directions, ``S = supersample**2``
        sub-pixel samples per pixel on a regular grid."""
        k = supersample
        offs = (np.arange(k) + 0.5) / k - 0.5
        u = np.arange(self.width, dtype=np.float64)
        v = np.arange(self.height, dtype=np.float64)
        uu = (u[None, :, None, None] + offs[None, None, None, :]).repeat(k, 2)
        vv = (v[:, None, None, None] + offs[None, None, :, None]).repeat(k, 3)
        uu = np.broadcast_to(uu, (self.height, self.width, k, k))
        vv = np.broadcast_to(vv, (self.height, self.width, k, k))
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fx
        d = np.stack([x, y, np.ones_like(x)], -1)
        return d.reshape(self.height, self.width, k * k, 3)

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) camera-frame points -> (N, 2) pixel coordinates and (N,) depth z."""
        z = pts_cam[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * pts_cam[:, 0] / safe + self.cx
        v = self.fx * pts_cam[:, 1] / safe + self.cy
        return np.stack([u, v], 1), z


def world_from_cam(rig: RigConfig, i: int, pos: np.ndarray,
                   rot_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera ``i``'s rotation and centre in the world for a body pose (pos, R_wb)."""
    r_bc, p_bc = rig.body_from_cam(i)
    return rot_wb @ r_bc, pos + rot_wb @ p_bc
