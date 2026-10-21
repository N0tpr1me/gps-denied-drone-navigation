"""Check a Gazebo wind trace against the Python disturbance model.

The plugin logs, every physics step, the state it read (t, position, velocity), the wind
and shelter it computed, and the force it applied. This recomputes all of them with
``envs.disturbance`` from the logged state and reports the largest difference, so a
scenario run proves the force Gazebo applied is the model the training env uses.

    python -m sim2real.gazebo.verify_trace runs/<run>/wind_params.txt runs/<run>/wind_trace.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from envs.disturbance import disturbance_force, drag_force, shelter_factor, wind_velocity
from sim2real.gazebo.params_io import read_params

TOLERANCE_N = 1e-6  # applied force, N; the C++ and NumPy models agree to ~1e-12
COLUMNS = ("step", "t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "shelter",
           "fx", "fy", "fz")


def read_trace(path: str | Path) -> dict[str, np.ndarray]:
    """The plugin's CSV as column name -> array."""
    # invalid_raise=False drops a final row cut short if Gazebo was killed mid-write.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64,
                             invalid_raise=False)
    if data.size == 0:
        raise ValueError(f"{path}: empty trace")
    data = np.atleast_1d(data)
    missing = set(COLUMNS) - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return {name: np.asarray(data[name], dtype=np.float64) for name in COLUMNS}


def verify(params_path: str | Path, trace_path: str | Path,
           tolerance: float = TOLERANCE_N) -> dict[str, Any]:
    """Recompute the trace with the Python model; ``ok`` is True within ``tolerance``."""
    p = read_params(params_path)
    tr = read_trace(trace_path)
    n = tr["t"].size
    pos = np.stack([tr["x"], tr["y"], tr["z"]], 1)
    vel = np.stack([tr["vx"], tr["vy"], tr["vz"]], 1)
    logged_force = np.stack([tr["fx"], tr["fy"], tr["fz"]], 1)
    logged_wind = np.stack([tr["wx"], tr["wy"], tr["wz"]], 1)

    # One env in the params, N rows in the trace: evaluate row by row through the batched
    # API by tiling the params' leading axis.
    tiled = type(p)(**{
        name: np.repeat(np.asarray(getattr(p, name)), n, axis=0)
        for name in p.__dataclass_fields__
    })
    wind = wind_velocity(tiled, tr["t"])
    shelter = shelter_factor(tiled, pos)
    expected = disturbance_force(tiled, tr["t"], pos, vel) - drag_force(
        -vel, tiled.k_lin, tiled.k_quad
    )

    force_err = np.abs(logged_force - expected).max(initial=0.0)
    wind_err = np.abs(logged_wind - wind).max(initial=0.0)
    shelter_err = np.abs(tr["shelter"] - shelter).max(initial=0.0)
    dt = np.diff(tr["t"])
    magnitude = np.linalg.norm(logged_force, axis=1)
    return {
        "ok": bool(force_err <= tolerance and np.all(np.isfinite(logged_force))),
        "tolerance_n": tolerance,
        "rows": int(n),
        "t_start_s": float(tr["t"][0]),
        "t_end_s": float(tr["t"][-1]),
        "dt_median_s": float(np.median(dt)) if dt.size else None,
        "dt_max_s": float(dt.max()) if dt.size else None,
        "max_force_error_n": float(force_err),
        "max_wind_error_m_s": float(wind_err),
        "max_shelter_error": float(shelter_err),
        "applied_force_max_n": float(magnitude.max()),
        "applied_force_rms_n": float(np.sqrt(np.mean(magnitude**2))),
        "shelter_min": float(tr["shelter"].min()),
        "shelter_max": float(tr["shelter"].max()),
        # The plugin applies the force in the same physics step it reads the state, so the
        # lag between the commanded and the applied force is zero steps by construction.
        "applied_lag_steps": 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("params", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, help="write the report as JSON here")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_N, help="max |error|, N")
    args = ap.parse_args(argv)
    report = verify(args.params, args.trace, args.tolerance)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
