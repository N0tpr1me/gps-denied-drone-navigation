"""The NumPy renderer: ray cast the scene's rectangles, return linear radiance and range.

The stand-in for Isaac Sim (``isaac_backend.py``) that runs anywhere: no lights beyond one
directional light and an ambient share (Lambert, no shadows), textures sampled bilinearly,
trilinear mip-mapping against aliasing on far and oblique surfaces (each ray reads the
texture level whose texel matches the pixel's footprint there), optionally
``supersample**2`` rays per pixel on top.
Radiance is relative: 1 is a white surface fully lit at ``light_level`` 1. Range is the
distance along the ray to the first surface (``inf`` where a ray hits nothing), the input
the fog and dust in ``degrade.py`` need.
"""

from __future__ import annotations

import numpy as np

from eval.vio_render.camera import RigConfig
from eval.vio_render.scene import Scene

BACKGROUND = 0.05  # radiance where a ray leaves the scene (a dark hall beyond)


def _bilinear(tex: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sample ``tex`` at fractions ``a`` (columns) and ``b`` (rows from the bottom)."""
    h, w = tex.shape
    x = np.clip(a * np.float32(w - 1), 0, w - 1.001)
    y = np.clip((np.float32(1.0) - b) * np.float32(h - 1), 0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = x - x0
    fy = y - y0
    t00 = tex[y0, x0]
    t01 = tex[y0, x0 + 1]
    t10 = tex[y0 + 1, x0]
    t11 = tex[y0 + 1, x0 + 1]
    return (t00 * (1 - fx) + t01 * fx) * (1 - fy) + (t10 * (1 - fx) + t11 * fx) * fy


def mip_pyramid(tex: np.ndarray, levels: int = 6) -> list[np.ndarray]:
    """``tex`` and its 2x box-filtered reductions, finest first."""
    out = [tex.astype(np.float32)]
    for _ in range(levels - 1):
        t = out[-1]
        h, w = (t.shape[0] // 2) * 2, (t.shape[1] // 2) * 2
        if h < 4 or w < 4:
            break
        t = t[:h, :w]
        out.append(0.25 * (t[0::2, 0::2] + t[1::2, 0::2] + t[0::2, 1::2] + t[1::2, 1::2]))
    return out


def _trilinear(mips: list[np.ndarray], a: np.ndarray, b: np.ndarray,
               level: np.ndarray) -> np.ndarray:
    """Sample a mip pyramid at fractional ``level`` (0 = finest), blending two levels."""
    level = np.clip(level, 0.0, len(mips) - 1.0)
    lo = np.floor(level).astype(np.int64)
    frac = level - lo
    out = np.zeros_like(a)
    for k in range(len(mips)):
        sel = lo == k
        if sel.any():
            out[sel] += (1 - frac[sel]) * _bilinear(mips[k], a[sel], b[sel])
        up = sel & (frac > 0) & (k + 1 < len(mips))
        if up.any():
            out[up] += frac[up] * _bilinear(mips[k + 1], a[up], b[up])
    return out


class RasterRenderer:
    def __init__(self, scene: Scene, rig: RigConfig, supersample: int = 1):
        self.scene = scene
        self.rig = rig
        self.k = supersample
        rays = rig.pixel_rays(supersample)
        # float32 throughout: twice as fast, and a millimetre at 10 m is far below a pixel.
        self._rays = rays.reshape(-1, 3).astype(np.float32)
        self._norm = np.linalg.norm(self._rays, axis=1)
        self._shade = {q.name: scene.shade(q) for q in scene.quads}
        self._mips = {name: mip_pyramid(t) for name, t in scene.textures.items()}
        # Angle one (sub-)pixel spans, for the texture footprint.
        self._pix_rad = 1.0 / (rig.fx * supersample)

    def render(self, rot_wc: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Radiance and range, both (H, W) float32, for a camera pose in the world."""
        d = self._rays @ rot_wc.T.astype(np.float32)  # world directions (camera z = 1)
        centre = np.asarray(centre, dtype=np.float32)
        n_rays = d.shape[0]
        best_t = np.full(n_rays, np.inf, dtype=np.float32)
        rad = np.full(n_rays, BACKGROUND, dtype=np.float32)
        for q in self.scene.quads:
            o = np.asarray(q.origin, dtype=np.float32)
            eu = np.asarray(q.edge_u, dtype=np.float32)
            ev = np.asarray(q.edge_v, dtype=np.float32)
            nrm = q.normal.astype(np.float32)
            denom = d @ nrm
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((o - centre) @ nrm) / denom
            ok = np.isfinite(t) & (t > 1e-6) & (t < best_t)
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            hit = centre + t[idx, None] * d[idx]
            rel = hit - o
            a = rel @ eu / (eu @ eu)
            b = rel @ ev / (ev @ ev)
            inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
            idx, a, b = idx[inside], a[inside], b[inside]
            best_t[idx] = t[idx]
            # Footprint of the pixel on the surface, in texels: range times the pixel's
            # angle, stretched by the obliquity.
            dn = np.abs(denom[idx]) / self._norm[idx]
            foot_m = t[idx] * self._norm[idx] * self._pix_rad / np.maximum(dn, 0.15)
            tex = self.scene.textures[q.texture]
            texel_m = float(np.linalg.norm(eu)) / tex.shape[1]
            level = np.log2(np.maximum(foot_m / texel_m, 1.0))
            rad[idx] = _trilinear(self._mips[q.texture], a, b, level) * self._shade[q.name]
        rng = best_t * self._norm
        h, w, s = self.rig.height, self.rig.width, self.k * self.k
        rad = rad.reshape(h, w, s).mean(2)
        # Range per pixel: the nearest sub-sample (a pixel on an edge belongs to the front).
        rng = rng.reshape(h, w, s).min(2)
        return rad.astype(np.float32), rng.astype(np.float32)
