"""PX4 parameter files in QGroundControl's plain-text format, and how they reach SITL.

Format: ``# comment`` lines and ``<vehicle_id>\\t<component_id>\\t<NAME>\\t<value>\\t<type>``
rows (type 6 = INT32, 9 = FLOAT), the same format ``eval/baselines/*.params`` will use. The
PX4 SITL startup script applies any ``PX4_PARAM_<NAME>`` environment variable with
``param set`` before the estimator starts, so :func:`to_environment` is how a file is
applied to a container (no image change, no runtime PARAM_SET after EKF2 is up).
"""

from __future__ import annotations

from pathlib import Path

INT32, FLOAT = 6, 9


def read_params(path: str | Path) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{n}: expected 5 columns, got {len(parts)}")
        _, _, name, value, ptype = parts
        if int(ptype) == INT32:
            out[name] = int(value)
        elif int(ptype) == FLOAT:
            out[name] = float(value)
        else:
            raise ValueError(f"{path}:{n}: unknown parameter type {ptype}")
    return out


def to_environment(params: dict[str, int | float]) -> dict[str, str]:
    """``PX4_PARAM_<NAME>`` variables for ``docker run -e``."""
    return {f"PX4_PARAM_{k}": repr(v) if isinstance(v, float) else str(v)
            for k, v in params.items()}
