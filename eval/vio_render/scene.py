"""The structure as textured rectangles, shared by the NumPy renderer and the Isaac backend.

World frame is ENU (docs/conventions.md). The structure frame of Structure Hold has x from
the drone to the surface, so the wall stands in the plane ``x = 0`` with the drone at
negative x looking along +x. A rectangle (``Quad``) is a corner plus two edge vectors; its
texture is a grey image laid over it, ``u`` along the first edge and ``v`` along the second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Quad:
    name: str
    origin: tuple[float, float, float]  # m, one corner
    edge_u: tuple[float, float, float]  # m, first edge (texture u runs along it)
    edge_v: tuple[float, float, float]  # m, second edge (texture v)
    texture: str  # key into Scene.textures

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.edge_u, self.edge_v)
        return n / np.linalg.norm(n)


@dataclass
class Scene:
    quads: list[Quad]
    textures: dict[str, np.ndarray]  # name -> (H, W) float32 reflectance in [0, 1]
    # Directional light (unit vector the light travels along) and the ambient share.
    light_dir: tuple[float, float, float] = (0.3, 0.2, -0.93)
    ambient: float = 0.35
    meta: dict[str, Any] = field(default_factory=dict)

    def shade(self, quad: Quad) -> float:
        """Lambert factor of a quad: ambient plus the diffuse part of the directional light."""
        light = -np.asarray(self.light_dir, dtype=float)
        light /= np.linalg.norm(light)
        return float(self.ambient + (1.0 - self.ambient) * max(0.0, abs(quad.normal @ light)))


def value_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random field in [0, 1]: uniforms on a grid ``cell`` texels apart, bilinear
    in between (square cells whatever the texture's aspect)."""
    h, w = shape
    gy, gx = int(np.ceil(h / cell)) + 2, int(np.ceil(w / cell)) + 2
    grid = rng.random((gy, gx))
    ys = np.arange(h) / cell
    xs = np.arange(w) / cell
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def concrete_texture(shape: tuple[int, int], seed: int, contrast: float = 0.6) -> np.ndarray:
    """Concrete-like reflectance around a mean of 0.5 with the given peak-to-peak
    ``contrast``: broad stains, mottling down to a few texels, sand grain, pores (dark
    spots of 1 to 4 texels, the corners a VIO's feature detector finds) and a few
    formwork seams. Built at a texel of about 4 mm, the pixel footprint at a 1.5 m hold."""
    rng = np.random.default_rng(seed)
    h, w = shape
    tex = np.zeros(shape, dtype=np.float64)
    total = 0.0
    for cell, amp in ((512, 0.6), (128, 0.6), (32, 0.7), (8, 0.8), (3, 0.8)):
        tex += amp * (value_noise(shape, cell, rng) - 0.5)
        total += amp
    tex /= total
    tex += 0.10 * (rng.random(shape) - 0.5)  # sand grain, per texel
    # Pores: dark round spots, a few per 100 cm^2 at 4 mm texels.
    n_pores = int(h * w * 0.004)
    py, px = rng.integers(0, h, n_pores), rng.integers(0, w, n_pores)
    pr = rng.uniform(0.6, 2.2, n_pores)
    depth = rng.uniform(0.15, 0.4, n_pores)
    for r in np.unique(np.ceil(pr).astype(int)):
        sel = np.ceil(pr).astype(int) == r
        off = np.arange(-r, r + 1)
        oy, ox = np.meshgrid(off, off, indexing="ij")
        disc = (oy * oy + ox * ox) <= (r + 0.25) ** 2
        for dy, dx in zip(oy[disc], ox[disc], strict=True):
            np.subtract.at(tex, (np.clip(py[sel] + dy, 0, h - 1), np.clip(px[sel] + dx, 0, w - 1)),
                           depth[sel])
    # Seams between pours: thin horizontal dark lines.
    for row in rng.integers(0, h, size=max(1, h // 400)):
        tex[row : row + max(1, h // 500)] -= 0.2
    tex = (tex - tex.mean()) / (tex.std() + 1e-9)  # unit spread
    tex = 0.5 + 0.5 * contrast * np.clip(tex / 2.5, -1.0, 1.0)
    return tex.astype(np.float32)


def blank_patch(tex: np.ndarray, u0: float, u1: float, contrast: float = 0.03) -> np.ndarray:
    """Flatten the texture between fractions ``u0`` and ``u1`` of its width: a painted or
    freshly plastered stretch with almost nothing to track."""
    out = tex.copy()
    c0, c1 = int(u0 * tex.shape[1]), int(u1 * tex.shape[1])
    patch = out[:, c0:c1]
    out[:, c0:c1] = 0.5 + contrast * (patch - 0.5) / (np.abs(patch - 0.5).max() + 1e-9)
    return out


def wall_scene(width_m: float = 10.0, height_m: float = 5.0, floor_z_m: float = -2.0,
               texels_per_m: float = 256.0, contrast: float = 0.6, seed: int = 0,
               blank: tuple[float, float] | None = None) -> Scene:
    """One vertical wall in the plane x = 0, centred on y = 0, from ``floor_z_m`` up, with
    a concrete floor in front of it. ``blank`` flattens the texture between two fractions
    of the wall's width (0 to 1, left to right as the drone sees it)."""
    wall_shape = (int(height_m * texels_per_m), int(width_m * texels_per_m))
    wall = concrete_texture(wall_shape, seed, contrast)
    if blank is not None:
        wall = blank_patch(wall, *blank)
    floor = concrete_texture((int(width_m * texels_per_m / 2), int(width_m * texels_per_m)),
                             seed + 1, 0.8 * contrast)
    quads = [
        # Seen from negative x: u runs along -y (left to right in the image), v up.
        Quad("wall", (0.0, width_m / 2, floor_z_m), (0.0, -width_m, 0.0), (0.0, 0.0, height_m),
             "wall"),
        Quad("floor", (-width_m / 2, width_m / 2, floor_z_m), (0.0, -width_m, 0.0),
             (width_m / 2, 0.0, 0.0), "floor"),
    ]
    return Scene(quads, {"wall": wall, "floor": floor},
                 meta={"kind": "wall", "width_m": width_m, "height_m": height_m,
                       "floor_z_m": floor_z_m, "contrast": contrast, "blank": blank})


def tunnel_scene(length_m: float = 30.0, width_m: float = 4.0, height_m: float = 3.0,
                 texels_per_m: float = 192.0, contrast: float = 0.5, seed: int = 0) -> Scene:
    """A box tunnel along y with the drone looking at its right wall (x = 0): the end wall
    is ahead at the far end, floor and ceiling close above and below. The hold itself is
    the same as at the wall; what changes is how much of the image is near and oblique."""
    long_shape = (int(height_m * texels_per_m), int(length_m * texels_per_m))
    flat_shape = (int(width_m * texels_per_m), int(length_m * texels_per_m))
    tex = {
        "right": concrete_texture(long_shape, seed, contrast),
        "left": concrete_texture(long_shape, seed + 1, contrast),
        "floor": concrete_texture(flat_shape, seed + 2, 0.8 * contrast),
        "ceiling": concrete_texture(flat_shape, seed + 3, 0.7 * contrast),
    }
    z0 = -height_m / 2
    quads = [
        Quad("right", (0.0, length_m / 2, z0), (0.0, -length_m, 0.0), (0.0, 0.0, height_m),
             "right"),
        Quad("left", (-width_m, -length_m / 2, z0), (0.0, length_m, 0.0), (0.0, 0.0, height_m),
             "left"),
        Quad("floor", (-width_m, length_m / 2, z0), (0.0, -length_m, 0.0), (width_m, 0.0, 0.0),
             "floor"),
        Quad("ceiling", (0.0, length_m / 2, z0 + height_m), (0.0, -length_m, 0.0),
             (-width_m, 0.0, 0.0), "ceiling"),
    ]
    return Scene(quads, tex, meta={"kind": "tunnel", "length_m": length_m, "width_m": width_m,
                                   "height_m": height_m, "contrast": contrast})


def build_scene(cfg: dict[str, Any]) -> Scene:
    """A scene from its config block: ``kind: wall | tunnel`` plus that builder's keys."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", "wall")
    if kind == "wall":
        if cfg.get("blank") is not None:
            cfg["blank"] = tuple(cfg["blank"])
        return wall_scene(**cfg)
    if kind == "tunnel":
        return tunnel_scene(**cfg)
    raise ValueError(f"unknown scene kind {kind!r}; choose wall or tunnel")
