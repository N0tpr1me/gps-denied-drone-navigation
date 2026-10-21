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
}


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
    missing = set(_FIELDS) - set(fields)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")

    k = fields["gust_start"].size
    m = fields["turb_omega"].size
    shapes = {
        **{name: shape for name, shape in _FIELDS.items() if shape is not None},
        "gust_start": (k,),
        "gust_duration": (k,),
        "gust_wind": (k, 3),
        "turb_amp": (3, m),
        "turb_omega": (m,),
        "turb_phase": (3, m),
    }
    arrays = {}
    for name, shape in shapes.items():
        if fields[name].size != int(np.prod(shape, dtype=int)):
            raise ValueError(f"{path}: field '{name}' does not have shape {shape}")
        arrays[name] = fields[name].reshape((1, *shape))
    arrays["has_wall"] = arrays["has_wall"] != 0.0
    return DisturbanceParams(**arrays)
