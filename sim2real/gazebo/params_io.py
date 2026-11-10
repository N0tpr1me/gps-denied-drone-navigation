"""Write one env's :class:`envs.disturbance.DisturbanceParams` to the text file the Gazebo
plugin reads, and read it back.

Format (``structure-hold-wind-params v1``): one field per line, ``<name> <count> <values>``,
2-D fields row-major, floats as Python's ``repr`` so they round-trip exactly through C++'s
``strtod``. The reader returns params for a single env (``num_envs == 1``), so the same
``disturbance_force`` the training env calls can be evaluated on a Gazebo trace.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.disturbance import DisturbanceParams

HEADER = "structure-hold-wind-params v1"

# Field name -> trailing shape per env (after the leading num_envs axis). Order is the file
# order; the C++ reader is keyed by name so it does not depend on it.
_FIELDS: dict[str, tuple[int, ...] | None] = {
    "mean_wind": (3,),
    "gust_start": None,  # (K,)
    "gust_duration": None,  # (K,)
    "gust_wind": None,  # (K, 3)
    "turb_amp": None,  # (3, M)
    "turb_omega": None,  # (M,)
    "turb_phase": None,  # (3, M)
    "k_lin": (),
    "k_quad": (),
    "wall_point": (3,),
    "wall_normal": (3,),
    "has_wall": (),
    "wall_decay": (),
    "wall_floor": (),
    "wall_deflect": (),
    "wall_tangent": (3,),
    "wall_recirc": (),
    "wall_recirc_length": (),
    "wake_amp": None,  # (3, M), or (3, 0) with the flow off
    "wake_phase": None,  # (3, M), or (3, 0)
    "wake_length": (),
    "shed_amp": (),
    "shed_omega": (),
    "shed_phase": (),
    "shed_dir": (3,),
    "has_surface2": (),
    "surface2_point": (3,),
    "surface2_normal": (3,),
    "surface2_decay": (),
    "surface2_floor": (),
    "surface2_shed_amp": (),
    "surface2_shed_omega": (),
    "surface2_shed_phase": (),
    "surface2_wake_length": (),
}
# Fields added with the near-wall flow (docs/near-wall-flow.md). Files written before it
# lack them; they read as the flow switched off.
_OPTIONAL = {"wall_deflect": 0.0, "wall_tangent": 0.0, "wall_recirc": 0.0,
             "wall_recirc_length": 1.0, "wake_length": 1.0,
             "shed_amp": 0.0, "shed_omega": 0.0, "shed_phase": 0.0, "shed_dir": 0.0,
             # The second surface (docs/bridge-holds.md): files written before it read as
             # no second surface.
             "has_surface2": 0.0, "surface2_point": 0.0, "surface2_normal": 0.0,
             "surface2_decay": 1.0, "surface2_floor": 1.0, "surface2_shed_amp": 0.0,
             "surface2_shed_omega": 0.0, "surface2_shed_phase": 0.0,
             "surface2_wake_length": 1.0}


def write_params(path: str | Path, p: DisturbanceParams, env: int = 0) -> None:
    """Write env ``env`` of ``p`` to ``path``."""
    if not 0 <= env < p.num_envs:
        raise IndexError(f"env {env} out of range for {p.num_envs} envs")
    lines = [HEADER]
    for name in _FIELDS:
        values = np.asarray(getattr(p, name))[env]
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        lines.append(f"{name} {flat.size} " + " ".join(repr(float(v)) for v in flat))
    Path(path).write_text("\n".join(lines) + "\n")


def read_params(path: str | Path) -> DisturbanceParams:
    """Read a params file back as a one-env :class:`DisturbanceParams`."""
    text = Path(path).read_text().splitlines()
    if not text or text[0] != HEADER:
        raise ValueError(f"{path}: expected header '{HEADER}'")
    fields: dict[str, np.ndarray] = {}
    for line in text[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        name, count, *values = line.split()
        if int(count) != len(values):
            raise ValueError(f"{path}: field '{name}' has {len(values)} values, not {count}")
        fields[name] = np.array([float(v) for v in values], dtype=np.float64)
    for name in ("wake_amp", "wake_phase"):
        fields.setdefault(name, np.zeros(0))  # no wake sinusoids: the flow is off
    missing = set(_FIELDS) - set(fields) - set(_OPTIONAL)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")

    k = fields["gust_start"].size
    m = fields["turb_omega"].size
    m_wake = fields["wake_amp"].size // 3  # 0 when the flow is off, else m
    shapes = {
        **{name: shape for name, shape in _FIELDS.items() if shape is not None},
        "gust_start": (k,),
        "gust_duration": (k,),
        "gust_wind": (k, 3),
        "turb_amp": (3, m),
        "turb_omega": (m,),
        "turb_phase": (3, m),
        "wake_amp": (3, m_wake),
        "wake_phase": (3, m_wake),
    }
    arrays = {}
    for name, shape in shapes.items():
        if name not in fields:
            fields[name] = np.full(shape, _OPTIONAL[name]).reshape(-1)
        if fields[name].size != int(np.prod(shape, dtype=int)):
            raise ValueError(f"{path}: field '{name}' does not have shape {shape}")
        arrays[name] = fields[name].reshape((1, *shape))
    arrays["has_wall"] = arrays["has_wall"] != 0.0
    arrays["has_surface2"] = arrays["has_surface2"] != 0.0
    return DisturbanceParams(**arrays)
