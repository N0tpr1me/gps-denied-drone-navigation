"""Parity check between ``envs/disturbance/core.py`` and the C++ copy in
``plugin/Disturbance.hh``.

Builds ``plugin/parity_main.cc`` with the system C++ compiler (no Gazebo needed), feeds it
random states for random params, and compares against the NumPy model. Run by pytest where
a compiler exists (CI, the SITL image) and skipped elsewhere; also runnable by hand:

    python -m sim2real.gazebo.parity
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from envs.disturbance import (
    DisturbanceConfig,
    disturbance_force,
    drag_force,
    sample_params,
    shelter_factor,
    wind_velocity,
)
from sim2real.gazebo.params_io import write_params

PLUGIN_DIR = Path(__file__).resolve().parent / "plugin"
PARITY_TOLERANCE = 1e-9


def find_compiler() -> str | None:
    """A usable C++ compiler, or None (``CXX`` wins when set)."""
    for name in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if name and shutil.which(name):
            return name
    return None


def build_parity_binary(out_dir: Path, compiler: str) -> Path:
    exe = out_dir / "disturbance_parity"
    cmd = [compiler, "-std=c++17", "-O2", "-o", str(exe), str(PLUGIN_DIR / "parity_main.cc")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compile failed: {' '.join(cmd)}\n{result.stderr}")
    return exe


def run_parity(exe: Path, seed: int = 0, samples: int = 256,
               work_dir: Path | None = None) -> dict[str, Any]:
    """Compare the C++ binary with the NumPy model on random params and states."""
    rng = np.random.default_rng(seed)
    # The near-wall flow on, at the top of its eval ranges, so every term is checked.
    cfg = DisturbanceConfig(num_gusts=3, gust_start=(0.0, 10.0), gust_duration=(0.5, 3.0),
                            k_lin=(0.0, 0.2), mean_speed=(3.0, 12.0), wall_deflect=(0.2, 0.8),
                            wall_recirc=(0.2, 0.6), wake_ti=(0.1, 0.3), shed_amp=(0.1, 0.4),
                            shed_angle_deg=(0.0, 90.0))
    p = sample_params(cfg, 1, rng, wall_point=(0.3, -0.2, 0.0), wall_normal=(-1.0, 0.1, 0.0))
    work = Path(work_dir or tempfile.mkdtemp())
    params_path = work / "params.txt"
    write_params(params_path, p)

    t = rng.uniform(0.0, 12.0, samples)
    pos = rng.uniform(-4.0, 1.0, (samples, 3))
    vel = rng.normal(0.0, 2.0, (samples, 3))
    # Plain Python floats: repr(np.float64) would print "np.float64(...)".
    rows = np.column_stack([t, pos, vel]).tolist()
    lines = "\n".join(" ".join(repr(v) for v in row) for row in rows)
    result = subprocess.run([str(exe), str(params_path)], input=lines + "\n",
                            capture_output=True, text=True, check=True)
    out = np.array([[float(v) for v in line.split()] for line in result.stdout.splitlines()])
    if out.shape != (samples, 10):
        raise RuntimeError(f"parity binary returned shape {out.shape}, expected ({samples}, 10)")

    tiled = type(p)(**{
        name: np.repeat(np.asarray(getattr(p, name)), samples, axis=0)
        for name in p.__dataclass_fields__
    })
    wind = wind_velocity(tiled, t)
    shelter = shelter_factor(tiled, pos)
    force = disturbance_force(tiled, t, pos, vel)
    applied = force - drag_force(-vel, tiled.k_lin, tiled.k_quad)
    errors = {
        "wind": float(np.abs(out[:, 0:3] - wind).max()),
        "shelter": float(np.abs(out[:, 3] - shelter).max()),
        "force": float(np.abs(out[:, 4:7] - force).max()),
        "applied": float(np.abs(out[:, 7:10] - applied).max()),
    }
    return {"samples": samples, "max_error": errors,
            "ok": max(errors.values()) <= PARITY_TOLERANCE}


def main() -> int:
    compiler = find_compiler()
    if compiler is None:
        print("no C++ compiler found", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_parity_binary(Path(tmp), compiler)
        report = run_parity(exe, work_dir=Path(tmp))
    print(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
