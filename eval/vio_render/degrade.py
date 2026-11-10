"""Fog, dust and low light on a rendered frame, and the camera that turns light into pixels.

Applied the same way to the NumPy renderer's output and to Isaac Sim's (linear radiance and
range per pixel), so the two renderers differ only in how the clean scene looks.

- **Fog** (Koschmieder's law): each pixel keeps ``exp(-beta * range)`` of the surface's
  light and gains airlight ``A * (1 - exp(-beta * range))``, ``beta = 3.912 / visibility``
  (the meteorological visibility, where contrast falls to 2 %). Contrast falls with range,
  so the wall at 1.5 m stays usable long after the far floor has gone grey.
- **Dust**: particles in the air around the drone, in 3D, so both cameras and successive
  frames see the same grains (a VIO is fooled by dust because it is consistent, not because
  it is noise). They drift with the wind, wander, settle, and are drawn as soft discs,
  bigger and blurrier when close (out of focus), behind the surface hidden by it.
  Fine dust also hazes the image like a thin fog (``haze_visibility_m``).
- **Low light**: the scene's light scaled by ``light_level``. The camera (``CameraModel``)
  exposes for it: longer exposure up to a limit (more motion blur), then analogue gain
  (more noise); photon shot noise, read noise and 8-bit quantisation make the image.

Everything random comes from generators seeded per condition and frame, so a dataset is
reproducible bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from eval.vio_render.camera import RigConfig

APERTURE_M = 1.25e-3  # a 2.5 mm f/2 VIO lens
FOCUS_M = 2.0


@dataclass(frozen=True)
class Condition:
    name: str
    light_level: float = 1.0  # scales the scene's light (1 = a lit hall, 0.01 = very dark)
    fog_visibility_m: float | None = None  # None = no fog
    fog_airlight: float = 0.75  # fog brightness at light_level 1
    dust_per_m3: float = 0.0  # visible grains per cubic metre around the drone
    dust_radius_mm: tuple[float, float] = (0.3, 1.5)
    dust_brightness: float = 0.9  # a grain's radiance when lit (light_level 1)
    dust_settle_mps: float = 0.05
    dust_wander_mps: float = 0.15
    haze_visibility_m: float | None = None  # fine dust's fog, None = none
    # Dust that arrives during the flight (kicked up, blown in): none before dust_start_s
    # (flight time), then the full amount over dust_ramp_s. Default: there from the start.
    dust_start_s: float = 0.0
    dust_ramp_s: float = 0.0

    def dust_share(self, t: float) -> float:
        """How much of the dust (grains and haze) is in the air at flight time ``t``."""
        if t < self.dust_start_s:
            return 0.0
        if self.dust_ramp_s <= 0:
            return 1.0
        return min(1.0, (t - self.dust_start_s) / self.dust_ramp_s)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown condition keys: {sorted(unknown)}")
        data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**data)


@dataclass(frozen=True)
class CameraModel:
    """A global-shutter VIO camera (OV9281 / OV7251 class), auto exposure and gain."""

    full_well_e: float = 10000.0
    # Electrons per ms of exposure for a white surface at light_level 1.
    sensitivity_e_per_ms: float = 4000.0
    exposure_ms: tuple[float, float] = (0.1, 20.0)
    max_gain: float = 16.0
    read_noise_e: float = 4.0
    target_mean: float = 0.4  # auto exposure aims the mean pixel at this share of full scale
    motion_blur: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraModel:
        data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**data)

    def exposure(self, mean_radiance: float) -> tuple[float, float]:
        """(exposure ms, gain) that bring the mean to ``target_mean`` of full scale."""
        want_e = self.target_mean * self.full_well_e
        per_ms = max(mean_radiance, 1e-6) * self.sensitivity_e_per_ms
        exp_ms = float(np.clip(want_e / per_ms, *self.exposure_ms))
        gain = float(np.clip(want_e / (per_ms * exp_ms), 1.0, self.max_gain))
        return exp_ms, gain

    def capture(self, radiance: np.ndarray, exp_ms: float, gain: float,
                rng: np.random.Generator) -> np.ndarray:
        """Radiance (relative) -> 8-bit image with shot noise, read noise and gain."""
        e = np.clip(radiance, 0.0, None) * self.sensitivity_e_per_ms * exp_ms
        e = np.minimum(e, self.full_well_e)
        # Shot noise as its normal approximation (std sqrt(electrons)): the same to within
        # a grey level above ~10 electrons, and several times faster than Poisson draws.
        z = rng.standard_normal((2, *e.shape), dtype=np.float32)
        e = e + np.sqrt(e) * z[0] + self.read_noise_e * z[1]
        dn = e * gain / self.full_well_e * 255.0
        return np.clip(np.rint(dn), 0, 255).astype(np.uint8)


def fog(radiance: np.ndarray, rng_m: np.ndarray, visibility_m: float,
        airlight: float) -> np.ndarray:
    beta = 3.912 / visibility_m
    t = np.exp(-beta * np.where(np.isfinite(rng_m), rng_m, 1e6))
    return radiance * t + airlight * (1.0 - t)


def motion_blur(img: np.ndarray, du: float, dv: float) -> np.ndarray:
    """Average the image along the pixel motion (du, dv) during the exposure."""
    length = math.hypot(du, dv)
    if length < 0.5:  # under half a pixel: no visible smear
        return img
    n = int(math.ceil(length))
    out = np.zeros_like(img)
    for k in range(n + 1):
        s = k / n - 0.5
        out += _shift(img, s * du, s * dv)
    return out / (n + 1)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Bilinear shift by (dx, dy) pixels, edges repeated."""
    h, w = img.shape
    x = np.clip(np.arange(w) - dx, 0, w - 1.001)
    y = np.clip(np.arange(h) - dy, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[None, :], (y - y0)[:, None]
    a = img[y0][:, x0]
    b = img[y0][:, x0 + 1]
    c = img[y0 + 1][:, x0]
    d = img[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


class DustCloud:
    """Grains in a box around the drone that drift with the air.

    Each grain has a fixed position at t = 0, a drift (the mean wind plus its own wander,
    plus settling) and is wrapped into a ``box_m`` cube centred on the camera, so the
    density around the drone stays constant however far it flies; a grain leaving one side
    comes back on the other, far from the cameras.
    """

    def __init__(self, cond: Condition, wind_mps: tuple[float, float], seed: int,
                 box_m: float = 4.0):
        self.cond = cond
        self.box = box_m
        n = int(round(cond.dust_per_m3 * box_m**3))
        rng = np.random.default_rng([seed, 0xD5])
        self.p0 = rng.uniform(-box_m / 2, box_m / 2, (n, 3))
        drift = np.zeros((n, 3))
        drift[:, :2] = np.asarray(wind_mps, dtype=float)
        drift += cond.dust_wander_mps * rng.standard_normal((n, 3))
        drift[:, 2] -= cond.dust_settle_mps
        self.v = drift
        self.radius = rng.uniform(*cond.dust_radius_mm, n) * 1e-3
        self.albedo = rng.uniform(0.5, 1.0, n)
        self.order = rng.random(n)  # grain i is in the air once the dust share passes it

    def positions(self, t: float, centre: np.ndarray) -> np.ndarray:
        p = self.p0 + self.v * t
        return centre + np.mod(p - centre + self.box / 2, self.box) - self.box / 2

    def draw(self, radiance: np.ndarray, rng_m: np.ndarray, rig: RigConfig, rot_wc: np.ndarray,
             centre: np.ndarray, t: float, light: float) -> np.ndarray:
        if self.p0.shape[0] == 0:
            return radiance
        pts = (self.positions(t, centre) - centre) @ rot_wc  # world -> camera
        front = (pts[:, 2] > 0.05) & (self.order < self.cond.dust_share(t))
        pts, rad_m, alb = pts[front], self.radius[front], self.albedo[front]
        uv, z = rig.project(pts)
        # Apparent size: the grain itself plus the lens's blur circle (focused at FOCUS_M,
        # aperture APERTURE_M), at least a third of a pixel; close grains are soft discs.
        size_px = rig.fx * rad_m / z
        defocus_px = rig.fx * APERTURE_M * np.abs(1.0 / z - 1.0 / FOCUS_M)
        sigma = np.maximum(np.hypot(size_px, 0.5 * defocus_px), 0.35)
        h, w = radiance.shape
        u, v = uv[:, 0], uv[:, 1]
        vis = (u > -10) & (u < w + 10) & (v > -10) & (v < h + 10)
        u, v, z, sigma, alb, size_px = u[vis], v[vis], z[vis], sigma[vis], alb[vis], size_px[vis]
        if u.size == 0:
            return radiance
        r = np.clip(np.ceil(3 * sigma).astype(int), 1, 12)
        out = radiance.copy()
        # Coverage at the disc's centre: the grain's share of its blur disc. A grain lit
        # by the scene's light has radiance ``brightness * light * albedo`` and covers
        # what is behind it by that share ("over" compositing, grains added one by one
        # against the clean frame, which is close enough for sparse dust).
        peak = np.clip((size_px / sigma) ** 2, 0.02, 1.0)
        grain = self.cond.dust_brightness * light * alb
        for rad in np.unique(r):
            sel = r == rad
            off = np.arange(-rad, rad + 1)
            ox, oy = np.meshgrid(off, off)
            cu = np.rint(u[sel]).astype(int)[:, None, None] + ox[None]
            cv = np.rint(v[sel]).astype(int)[:, None, None] + oy[None]
            du = cu - u[sel][:, None, None]
            dv = cv - v[sel][:, None, None]
            s2 = (sigma[sel] ** 2)[:, None, None]
            alpha = peak[sel][:, None, None] * np.exp(-(du * du + dv * dv) / (2 * s2))
            g = np.broadcast_to(grain[sel][:, None, None], alpha.shape)
            zz = np.broadcast_to(z[sel][:, None, None], alpha.shape)
            inside = (cu >= 0) & (cu < w) & (cv >= 0) & (cv < h)
            cu, cv, alpha, g, zz = cu[inside], cv[inside], alpha[inside], g[inside], zz[inside]
            # Hidden behind the surface (range vs depth: depth <= range, so a grain at the
            # surface's depth is hidden a little early at the image edge; harmless).
            seen = zz < rng_m[cv, cu]
            cu, cv, alpha, g = cu[seen], cv[seen], alpha[seen], g[seen]
            np.add.at(out, (cv, cu), alpha * (g - radiance[cv, cu]))
        return np.clip(out, 0.0, None)


def apply(cond: Condition, radiance: np.ndarray, rng_m: np.ndarray, *,
          dust: DustCloud | None, rig: RigConfig, rot_wc: np.ndarray, centre: np.ndarray,
          t: float) -> np.ndarray:
    """Scene radiance at light 1 -> radiance reaching the lens under the condition."""
    light = cond.light_level
    out = radiance * light
    share = cond.dust_share(t)
    if cond.haze_visibility_m is not None and share > 0:
        out = fog(out, rng_m, cond.haze_visibility_m / share, 0.6 * cond.fog_airlight * light)
    if cond.fog_visibility_m is not None:
        out = fog(out, rng_m, cond.fog_visibility_m, cond.fog_airlight * light)
    if dust is not None:
        out = dust.draw(out, rng_m, rig, rot_wc, centre, t, light)
    return out


def blur_px(rig: RigConfig, gyro_body: np.ndarray, r_bc: np.ndarray,
            exp_ms: float) -> tuple[float, float]:
    """Pixel motion during the exposure from the body's rotation (translation is small at
    hover next to a wall): the camera-frame rate times focal length and exposure."""
    w = r_bc.T @ gyro_body  # rad/s in the camera frame
    t = exp_ms * 1e-3
    # A rotation about camera y moves the image along u, about x along v (small angles).
    return float(-w[1] * rig.fx * t), float(w[0] * rig.fx * t)
