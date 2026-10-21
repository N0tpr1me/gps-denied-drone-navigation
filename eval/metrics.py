"""The metrics every Structure Hold result is scored on, in simulation and in real flight.

One function, :func:`compute_metrics`, turns a :class:`Trace` (time, true position, hold
setpoint, optional commands) into a flat dict of numbers. The same code scores the mock
simulator, Isaac Lab, PX4 SITL ulogs (via :mod:`eval.ulog`) and real flight logs, so a
number in ``experiments/LOG.md`` means the same thing whichever source produced it.
``docs/metrics.md`` repeats these definitions for readers who do not open the code.

Frames and units follow ``docs/conventions.md``: ENU world frame, SI units. The wall is a
plane given by a point on it and the structure-frame x axis (unit normal pointing from the
drone toward the surface).

Definitions (``e(t) = |p(t) - p_sp(t)|``, the 3-D position error in metres):

- ``rms_m``: root mean square of ``e`` over the evaluation window.
- ``p95_m``: 95th percentile of ``e`` over the window (headline number for Gate 2).
- ``max_m``: largest ``e`` in the window.
- ``worst_toward_surface_m``: largest displacement from the setpoint along the wall normal,
  toward the wall, ``max((p - p_sp) . n)``, floored at 0 (moving away never counts).
- ``min_clearance_m``: smallest distance from the vehicle centre to the wall plane.
- ``contacts``: number of separate episodes in which the vehicle touched the wall. A
  sample is in contact when its centre is within ``contact_radius_m`` of the wall plane, or
  when the trace's own ``contact`` flag (from a simulator) is set. Each run of consecutive
  contact samples counts once.
- ``recovery_s``: for each gust ``(start, end)``, the time from ``end`` until ``e`` is back
  within ``recovery_tol_m`` (10 cm) and stays there for ``recovery_hold_s``. 0 if it was
  already inside when the gust ended. ``inf`` if it never settles before the next gust
  starts or the window ends. Reported as ``recovery_max_s``, ``recovery_mean_s`` and the
  per-gust list ``recovery_s``.
- ``rate_roughness_rad_s2``: root mean square of the time derivative of the body-rate
  command vector, ``sqrt(mean(|d omega_cmd / dt|^2))``, in rad/s^2.
- ``thrust_roughness_1_s``: the same for the normalised collective thrust command, in 1/s.

Position metrics are computed on a uniform grid (``grid_dt_s``, default 100 Hz) obtained by
linear interpolation of the position and zero-order hold of the setpoint, so the result does
not depend on the logging rate or on gaps in the log. Command roughness uses the native
command samples (finite differences between consecutive samples, weighted by their time
spacing), because interpolating would smooth out the steps it is meant to measure.

Position must be the TRUE position (simulator state, SITL ground truth or motion capture),
never the flight controller's estimate: when the estimate is corrupted, scoring against it
would hide exactly the error we are trying to measure.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# Headline defaults. Changing one changes every result, so it needs a docs/decisions.md row.
GRID_DT_S = 0.01
RECOVERY_TOL_M = 0.10
RECOVERY_HOLD_S = 1.0
CONTACT_RADIUS_M = 0.25  # centre-to-wall distance at which the props touch (5-inch class)


@dataclass
class Surface:
    """A planar wall: any point on it and the unit normal pointing from the drone toward it."""

    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("surface normal must be non-zero")
        self.normal = n / norm

    def distance(self, pos: np.ndarray) -> np.ndarray:
        """Distance from each position (N, 3) to the plane, positive on the drone's side."""
        return (self.point - pos) @ self.normal


@dataclass
class Trace:
    """One flight or episode, in ENU and SI units.

    ``pos_sp`` may be a single (3,) hold point or (N, 3) sampled at ``t``. Commands have their
    own time base ``cmd_t`` because PX4 logs them on a different topic and rate.
    ``contact`` is an optional per-sample flag from a simulator's contact sensor.
    """

    t: np.ndarray
    pos: np.ndarray
    pos_sp: np.ndarray
    cmd_t: np.ndarray | None = None
    rates_cmd: np.ndarray | None = None
    thrust_cmd: np.ndarray | None = None
    contact: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.pos = np.asarray(self.pos, dtype=float).reshape(-1, 3)
        n = len(self.t)
        if len(self.pos) != n:
            raise ValueError(f"pos has {len(self.pos)} rows, t has {n}")
        if n < 2:
            raise ValueError("trace needs at least two samples")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing")
        sp = np.asarray(self.pos_sp, dtype=float)
        self.pos_sp = np.broadcast_to(sp, (n, 3)).copy() if sp.shape == (3,) else sp
        if self.pos_sp.shape != (n, 3):
            raise ValueError(f"pos_sp must be (3,) or ({n}, 3), got {sp.shape}")
        if not np.all(np.isfinite(self.pos)) or not np.all(np.isfinite(self.pos_sp)):
            raise ValueError("pos and pos_sp must be finite")
        if self.contact is not None:
            self.contact = np.asarray(self.contact, dtype=bool).reshape(-1)
            if len(self.contact) != n:
                raise ValueError("contact must have one flag per sample")
        if self.rates_cmd is not None or self.thrust_cmd is not None:
            if self.cmd_t is None:
                raise ValueError("commands need cmd_t")
            self.cmd_t = np.asarray(self.cmd_t, dtype=float).reshape(-1)
            if self.rates_cmd is not None:
                self.rates_cmd = np.asarray(self.rates_cmd, dtype=float).reshape(-1, 3)
                if len(self.rates_cmd) != len(self.cmd_t):
                    raise ValueError("rates_cmd must have one row per cmd_t sample")
            if self.thrust_cmd is not None:
                self.thrust_cmd = np.asarray(self.thrust_cmd, dtype=float).reshape(-1)
                if len(self.thrust_cmd) != len(self.cmd_t):
                    raise ValueError("thrust_cmd must have one value per cmd_t sample")


def resample(trace: Trace, dt: float, t0: float, t1: float) -> tuple[np.ndarray, ...]:
    """Uniform grid on [t0, t1]: linear interpolation of position, zero-order hold of the
    setpoint (a setpoint is a step, not a ramp) and of the contact flag."""
    tg = t0 + dt * np.arange(int(math.floor((t1 - t0) / dt + 1e-9)) + 1)
    pos = np.stack([np.interp(tg, trace.t, trace.pos[:, i]) for i in range(3)], axis=1)
    idx = np.clip(np.searchsorted(trace.t, tg, side="right") - 1, 0, len(trace.t) - 1)
    contact = trace.contact[idx] if trace.contact is not None else None
    return tg, pos, trace.pos_sp[idx], contact


def count_episodes(flags: np.ndarray) -> int:
    """Number of runs of consecutive True values."""
    f = np.asarray(flags, dtype=bool)
    if f.size == 0:
        return 0
    return int(f[0]) + int(np.count_nonzero(f[1:] & ~f[:-1]))


def recovery_time(
    t: np.ndarray, err: np.ndarray, start: float, until: float, tol: float, hold: float
) -> float:
    """Time from ``start`` until ``err`` is within ``tol`` for ``hold`` seconds without a break.

    Only samples in [start, until] are considered; the hold must also finish by ``until``.
    Returns inf if that never happens.
    """
    sel = (t >= start - 1e-9) & (t <= until + 1e-9)
    ts, inside = t[sel], err[sel] <= tol
    if ts.size == 0:
        return math.inf
    # Start of the current unbroken inside-run at each sample (NaN while outside).
    run_start = np.full(ts.shape, np.nan)
    begin = np.nan
    for i, ok in enumerate(inside):
        if not ok:
            begin = np.nan
        elif math.isnan(begin):
            begin = ts[i]
        run_start[i] = begin
    held = ~np.isnan(run_start) & (ts - run_start >= hold - 1e-9)
    if not held.any():
        return math.inf
    return float(max(run_start[np.argmax(held)] - start, 0.0))


def roughness(t: np.ndarray, u: np.ndarray) -> float:
    """sqrt(time-weighted mean of |du/dt|^2) over consecutive samples."""
    u = u.reshape(len(t), -1)
    keep = np.all(np.isfinite(u), axis=1)
    t, u = t[keep], u[keep]
    if len(t) < 2:
        return math.nan
    dt = np.diff(t)
    ok = dt > 0
    rate_sq = np.sum((np.diff(u, axis=0)[ok] / dt[ok, None]) ** 2, axis=1)
    return float(math.sqrt(np.sum(rate_sq * dt[ok]) / np.sum(dt[ok])))


def compute_metrics(
    trace: Trace,
    surface: Surface | None = None,
    gusts: Sequence[tuple[float, float]] = (),
    window: tuple[float, float] | None = None,
    grid_dt_s: float = GRID_DT_S,
    recovery_tol_m: float = RECOVERY_TOL_M,
    recovery_hold_s: float = RECOVERY_HOLD_S,
    contact_radius_m: float = CONTACT_RADIUS_M,
) -> dict:
    """Score one trace. See the module docstring for every definition.

    ``window`` limits scoring to (t0, t1), for example to skip take-off and settling.
    Without a ``surface``, the wall metrics are NaN and contacts come only from the
    trace's contact flag (0 if it has none).
    """
    t0, t1 = window if window is not None else (trace.t[0], trace.t[-1])
    t0, t1 = max(t0, trace.t[0]), min(t1, trace.t[-1])
    if t1 - t0 < grid_dt_s:
        raise ValueError(f"window ({t0}, {t1}) is shorter than one grid step")
    tg, pos, sp, contact_flag = resample(trace, grid_dt_s, t0, t1)
    err = np.linalg.norm(pos - sp, axis=1)

    out: dict = {
        "duration_s": float(tg[-1] - tg[0]),
        "rms_m": float(np.sqrt(np.mean(err**2))),
        "p95_m": float(np.percentile(err, 95)),
        "max_m": float(err.max()),
    }

    in_contact = np.zeros(len(tg), dtype=bool) if contact_flag is None else contact_flag.copy()
    if surface is not None:
        toward = (pos - sp) @ surface.normal
        clearance = surface.distance(pos)
        out["worst_toward_surface_m"] = float(max(toward.max(), 0.0))
        out["min_clearance_m"] = float(clearance.min())
        in_contact |= clearance <= contact_radius_m
    else:
        out["worst_toward_surface_m"] = math.nan
        out["min_clearance_m"] = math.nan
    out["contacts"] = count_episodes(in_contact)

    ordered = sorted((float(a), float(b)) for a, b in gusts)
    rec = []
    for i, (start, end) in enumerate(ordered):
        if end < start:
            raise ValueError(f"gust ({start}, {end}) ends before it starts")
        until = ordered[i + 1][0] if i + 1 < len(ordered) else tg[-1]
        rec.append(recovery_time(tg, err, end, min(until, tg[-1]), recovery_tol_m,
                                 recovery_hold_s))
    out["recovery_s"] = rec
    out["recovery_max_s"] = max(rec) if rec else math.nan
    out["recovery_mean_s"] = float(np.mean(rec)) if rec else math.nan

    out["rate_roughness_rad_s2"] = math.nan
    out["thrust_roughness_1_s"] = math.nan
    if trace.cmd_t is not None:
        in_win = (trace.cmd_t >= t0) & (trace.cmd_t <= t1)
        ct = trace.cmd_t[in_win]
        if trace.rates_cmd is not None:
            out["rate_roughness_rad_s2"] = roughness(ct, trace.rates_cmd[in_win])
        if trace.thrust_cmd is not None:
            out["thrust_roughness_1_s"] = roughness(ct, trace.thrust_cmd[in_win])
    return out


def _vec3(text: str) -> np.ndarray:
    v = np.array([float(x) for x in text.split(",")])
    if v.shape != (3,):
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return v


def _gust(text: str) -> tuple[float, float]:
    parts = [float(x) for x in text.split(",")]
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected START or START,END in seconds")
    return parts[0], parts[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a PX4 .ulg log with the project metrics.")
    p.add_argument("ulog", help="path to a .ulg file")
    p.add_argument("--wall-point", type=_vec3, help="a point on the wall, ENU, m: x,y,z")
    p.add_argument("--wall-normal", type=_vec3, help="normal from drone toward wall: x,y,z")
    p.add_argument("--hold", type=_vec3, help="hold point (ENU) if the log has no setpoint")
    p.add_argument("--gust", type=_gust, action="append", default=[],
                   help="gust START,END in seconds from log start (repeatable)")
    p.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"))
    p.add_argument("--allow-estimate", action="store_true",
                   help="score against the EKF estimate when the log has no ground truth")
    args = p.parse_args(argv)
    if (args.wall_point is None) != (args.wall_normal is None):
        p.error("--wall-point and --wall-normal go together")

    from eval.ulog import read_ulog

    trace = read_ulog(args.ulog, hold_point=args.hold, allow_estimate=args.allow_estimate)
    surface = Surface(args.wall_point, args.wall_normal) if args.wall_point is not None else None
    result = compute_metrics(trace, surface, args.gust, tuple(args.window) if args.window else None)
    result["position_source"] = trace.meta.get("position_source")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
