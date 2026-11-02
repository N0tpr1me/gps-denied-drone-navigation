"""Fit the wind and near-wall model (``envs/disturbance``) to flight logs.

    python -m training.launch --config eval/configs/wind_fit_gazebo.yaml --device cpu

The build plan's Phase 3 "wind model fit": fly a hold in free air and at several distances
from a wall, in the same wind, and recover from the logs the numbers the disturbance model
is built from. Software only (docs/decisions.md, 2026-10-30): the logs come from a partner's
drone, a fan rig session (docs/fan-rig.md) or Gazebo SITL, where the true wind is known and
the fit can be scored (docs/wind-fit.md).

How, for each hold (a *segment*):

1. **Force from the accelerometer.** The accelerometer measures the specific force
   ``(thrust + aero) / mass`` in the body frame, and thrust is along body z, so the body x
   and y readings times the mass are the aerodynamic force on those axes, whatever the
   attitude or the acceleration. With the attitude, and no vertical aero force (the model
   has no vertical mean wind), that is the horizontal force in the world frame. A calm hold
   (no wind) gives the accelerometer's own offset on x and y, which is taken out first.
2. **Air from the force.** The model's force is drag on the air relative to the drone,
   ``F = (k_lin + k_quad |v|) v``, so each sample's ``v`` follows from ``F`` (one quadratic),
   and the air velocity is ``v`` plus the drone's own velocity.
3. **Statistics of the air.** Mean (speed and heading), turbulence standard deviation per
   horizontal axis after the mean is removed (iteratively clipped at ``clip_sigma``, so a
   gust missed by ``exclude_s`` does not count as turbulence), and the correlation time
   where the along-wind autocorrelation falls to 1/e (the model's Gauss-Markov ``tau``).

Then, across segments:

4. **Drag coefficients.** Free-air holds with a reference wind speed (an anemometer, or the
   scenario's) give ``k_quad`` (and ``k_lin`` with two or more distinct speeds), from the
   mean force along the wind: ``k_lin U + k_quad (U^2 + 1.5 sigma^2)`` (the turbulence term
   is the mean of ``|v| v`` for horizontal turbulence ``sigma`` per axis). Without a
   reference the configured ``k_quad`` is used and every speed scales with
   ``1 / sqrt(k_quad)``; the shelter does not depend on it.
5. **Shelter.** Within one *case* (one fan setting or scenario wind) the free-air holds give
   the free stream ``W``; each wall hold's mean air along ``W``, over ``|W|``, is the shelter
   ``g`` at its mean distance. ``g(d) = 1 - (1 - g0) exp(-d / L)`` is fitted over all cases
   (``L`` on a grid then refined, ``g0`` by least squares for each ``L``), with a
   leave-one-out spread when there are three or more holds.

What it also reports, not fits: the crossflow at each wall hold (mean air across ``W``, a
sign of the windward deflection term) and its turbulence over ``g`` times the free-air value
(more than 1 is wake turbulence). What it cannot see: vertical wind (only horizontal force is
measured), the near-wall yaw torque, and gusts as steps (mask them with ``exclude_s``).

Result: ``fit.json`` in the run folder with every segment's numbers, the fitted model, the
truth and the error when the config gives one (Gazebo), and a proposed ``disturbance`` block
beside the current defaults. Changing the training ranges is a Proposed row in
docs/decisions.md and its own PR (docs/fan-rig.md, calibration step 5).

Frames: PX4 logs are NED/FRD; the rotation uses PX4's own attitude quaternion (FRD to NED)
and the result goes to ENU through ``envs.frames`` (docs/conventions.md).
"""

from __future__ import annotations

import glob
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.disturbance import DisturbanceConfig, sample_params
from envs.disturbance.core import NOMINAL_K_QUAD, disturbance_force
from envs.frames import ned_to_enu
from training.runlib import RunWriter, entry_args, load_yaml

GROUNDTRUTH = "vehicle_local_position_groundtruth"
ESTIMATE = "vehicle_local_position"
ATTITUDE = "vehicle_attitude"
# sensor_combined is the calibrated accelerometer (specific force, FRD) at the IMU rate.
# vehicle_acceleration is used only when it is missing: it is logged at about 20 Hz and has
# EKF2's accelerometer bias taken out, and in a steady wind that estimate soaks up part of
# the aero force (2.6 % of it in the first Gazebo check). The calm hold takes the offset out.
ACCEL_TOPICS = ("sensor_combined", "vehicle_acceleration")
ACCEL_FIELDS = {"vehicle_acceleration": "xyz[{}]", "sensor_combined": "accelerometer_m_s2[{}]"}
TOPICS = [GROUNDTRUTH, ESTIMATE, ATTITUDE, *ACCEL_TOPICS]
GRAVITY = 9.80665
KINDS = ("free", "wall", "calm")
REPO_ROOT = Path(__file__).resolve().parent.parent

Topics = dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class FitParams:
    rate_hz: float = 50.0  # the grid every segment is resampled to
    lowpass_s: float = 0.1  # moving average on the force, against IMU noise and vibration
    settle_s: float = 5.0  # automatic window: skip this long after reaching hold height
    min_segment_s: float = 20.0
    clip_sigma: float = 3.5
    acf_max_lag_s: float = 10.0
    k_lin: float = 0.0  # used when the data cannot fit it (fewer than two reference speeds)
    k_quad: float | None = None  # used without any reference speed; None is the nominal
    fit_k_lin: bool = True  # fit k_lin when two distinct reference speeds exist
    shelter_floor_max: float = 0.9

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FitParams:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown fit keys: {sorted(unknown)}")
        return cls(**d)


# ---------------------------------------------------------------- one segment


@dataclass
class Segment:
    """One hold resampled to a uniform grid. ``f_body`` is the specific force (FRD, m/s^2),
    ``rot`` the attitude as FRD-to-NED matrices; ``vel`` and ``pos`` are ENU."""

    name: str
    case: str
    kind: str  # free, wall or calm
    t: np.ndarray  # (n,) s
    valid: np.ndarray  # (n,) bool: outside every exclude window
    f_body: np.ndarray  # (n, 3)
    rot: np.ndarray  # (n, 3, 3)
    vel: np.ndarray  # (n, 3) m/s
    dist: np.ndarray | None  # (n,) m from the wall, None in free air or calm
    ref_speed: float | None = None  # m/s, free-stream reference (anemometer or scenario)
    true_air: np.ndarray | None = None  # (n, 3) m/s, the simulator's air, when known


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) from quaternions (n, 4) as (w, x, y, z), body to world."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Quaternions (n, 4) as (w, x, y, z) from rotation matrices (n, 3, 3) (small tilts are
    all the synthetic logs need, so the w-largest branch is enough)."""
    w = 0.5 * np.sqrt(np.clip(1.0 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2], 1e-12, None))
    return np.stack([w, (r[:, 2, 1] - r[:, 1, 2]) / (4 * w), (r[:, 0, 2] - r[:, 2, 0]) / (4 * w),
                     (r[:, 1, 0] - r[:, 0, 1]) / (4 * w)], axis=1)


def _seconds(msg: dict[str, np.ndarray]) -> np.ndarray:
    t = np.asarray(msg["timestamp"], dtype=np.int64) * 1e-6
    return t


def _unique(t: np.ndarray, *cols: np.ndarray) -> tuple[np.ndarray, ...]:
    t, first = np.unique(t, return_index=True)
    return (t, *(c[first] for c in cols))


def _box_resample(t_src: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Average the samples falling in each grid cell (cells centred on the grid), then fill
    empty cells by interpolation, so a fast IMU is filtered before it is decimated."""
    dt = grid[1] - grid[0]
    idx = np.floor((t_src - grid[0]) / dt + 0.5).astype(int)
    inside = (idx >= 0) & (idx < grid.size)
    out = np.full((grid.size, x.shape[1]), np.nan)
    counts = np.bincount(idx[inside], minlength=grid.size).astype(float)
    for k in range(x.shape[1]):
        sums = np.bincount(idx[inside], weights=x[inside, k], minlength=grid.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k] = sums / counts
    for k in range(x.shape[1]):
        good = np.isfinite(out[:, k])
        out[:, k] = np.interp(grid, grid[good], out[good, k])
    return out


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    pad = np.pad(x, ((n // 2, n - 1 - n // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, k], kernel, mode="valid") for k in range(x.shape[1])],
                    axis=1)


def auto_window(t: np.ndarray, z_up: np.ndarray, settle_s: float) -> tuple[float, float]:
    """The hold: from ``settle_s`` after first reaching the hold height to 1 s before
    leaving it. The hold height is the median height while above half the highest point."""
    high = z_up > 0.5 * np.nanmax(z_up)
    if not np.any(high):
        raise ValueError("no hold found: the vehicle never climbed")
    hold_z = float(np.median(z_up[high]))
    at = np.flatnonzero(np.abs(z_up - hold_z) < max(0.15, 0.1 * abs(hold_z)))
    if at.size < 2:
        raise ValueError("no hold found at a steady height")
    return float(t[at[0]] + settle_s), float(t[at[-1]] - 1.0)


def segment_from_topics(topics: Topics, spec: dict[str, Any], p: FitParams) -> Segment:
    """Build one segment from PX4 topic arrays (field -> array, as pyulog gives them).

    ``spec`` keys: ``name``, ``kind`` (free, wall, calm), ``case`` (segments in the same
    wind; default ``"default"``), ``window_s`` ([t0, t1] in log seconds; default the
    automatic hold window), ``exclude_s`` (list of [t0, t1], such as the gusts),
    ``distance_m`` (fixed distance from the wall, for logs without position truth),
    ``wall`` ({point, normal} ENU, normal from the wall into the free air, for the distance
    from the logged position), ``ref_speed_m_s`` (free-stream reference).
    """
    kind = spec.get("kind", "free")
    if kind not in KINDS:
        raise ValueError(f"segment {spec.get('name')}: kind must be one of {KINDS}")
    accel_topic = next((a for a in ACCEL_TOPICS if a in topics), None)
    if accel_topic is None or ATTITUDE not in topics:
        raise ValueError(f"log needs {ATTITUDE} and one of {ACCEL_TOPICS}")
    acc = topics[accel_topic]
    fmt = ACCEL_FIELDS[accel_topic]
    ta, fa = _unique(_seconds(acc), np.stack([np.asarray(acc[fmt.format(k)], float)
                                               for k in range(3)], axis=1))
    att = topics[ATTITUDE]
    tq, q = _unique(_seconds(att), np.stack([np.asarray(att[f"q[{k}]"], float)
                                              for k in range(4)], axis=1))
    pos_topic = GROUNDTRUTH if GROUNDTRUTH in topics else ESTIMATE
    if pos_topic not in topics:
        raise ValueError(f"log needs {GROUNDTRUTH} or {ESTIMATE} for the velocity")
    pm = topics[pos_topic]
    tp, pos, vel = _unique(
        _seconds(pm),
        ned_to_enu(np.stack([pm["x"], pm["y"], pm["z"]], axis=1)),
        ned_to_enu(np.stack([pm["vx"], pm["vy"], pm["vz"]], axis=1)),
    )

    if spec.get("window_s") is not None:
        t0, t1 = map(float, spec["window_s"])
    else:
        t0, t1 = auto_window(tp, pos[:, 2], p.settle_s)
    t0, t1 = max(t0, ta[0], tq[0], tp[0]), min(t1, ta[-1], tq[-1], tp[-1])
    if t1 - t0 < p.min_segment_s:
        raise ValueError(f"segment {spec.get('name')}: window {t1 - t0:.1f} s is shorter than "
                         f"min_segment_s {p.min_segment_s}")
    dt = 1.0 / p.rate_hz
    grid = np.arange(t0, t1, dt)
    f_body = _moving_average(_box_resample(ta, fa, grid), round(p.lowpass_s * p.rate_hz))
    qg = np.stack([np.interp(grid, tq, q[:, k]) for k in range(4)], axis=1)
    rot = quat_to_rot(qg)
    vel_g = np.stack([np.interp(grid, tp, vel[:, k]) for k in range(3)], axis=1)
    pos_g = np.stack([np.interp(grid, tp, pos[:, k]) for k in range(3)], axis=1)

    valid = np.ones(grid.size, dtype=bool)
    for a, b in spec.get("exclude_s") or []:
        valid &= ~((grid >= float(a)) & (grid <= float(b)))

    dist = None
    if kind == "wall":
        if spec.get("distance_m") is not None:
            dist = np.full(grid.size, float(spec["distance_m"]))
        elif spec.get("wall") is not None and pos_topic == GROUNDTRUTH:
            w = spec["wall"]
            n = np.asarray(w["normal"], float)
            n = n / np.linalg.norm(n)
            dist = (pos_g - np.asarray(w["point"], float)) @ n
        else:
            raise ValueError(
                f"segment {spec.get('name')}: a wall hold needs distance_m, or wall plus "
                f"position truth ({GROUNDTRUTH}); the EKF's position is not truth")
    ref = spec.get("ref_speed_m_s")
    return Segment(name=str(spec.get("name", "segment")), case=str(spec.get("case", "default")),
                   kind=kind, t=grid, valid=valid, f_body=f_body, rot=rot, vel=vel_g,
                   dist=dist, ref_speed=None if ref is None else float(ref))


# ---------------------------------------------------------------- force and air


def world_force(seg: Segment, mass: float, offset_xy: np.ndarray) -> np.ndarray:
    """Horizontal aero force (n, 3) in N, world ENU (z is 0 by construction).

    Body x and y of the specific force (offset removed) times the mass is the aero force on
    those axes; the body z part is fixed so the force has no vertical world component."""
    fx = mass * (seg.f_body[:, 0] - offset_xy[0])
    fy = mass * (seg.f_body[:, 1] - offset_xy[1])
    r = seg.rot
    fz = -(r[:, 2, 0] * fx + r[:, 2, 1] * fy) / r[:, 2, 2]
    f_ned = np.einsum("nij,nj->ni", r, np.stack([fx, fy, fz], axis=1))
    out = ned_to_enu(f_ned)
    out[:, 2] = 0.0
    return out


def invert_drag(force: np.ndarray, k_lin: float, k_quad: float) -> np.ndarray:
    """Relative air velocity (n, 3) whose drag ``(k_lin + k_quad |v|) v`` is ``force``."""
    mag = np.linalg.norm(force, axis=1)
    if k_quad > 0:
        speed = (-k_lin + np.sqrt(k_lin * k_lin + 4.0 * k_quad * mag)) / (2.0 * k_quad)
    elif k_lin > 0:
        speed = mag / k_lin
    else:
        raise ValueError("k_lin and k_quad cannot both be zero")
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag[:, None] > 0, force / mag[:, None], 0.0)
    return speed[:, None] * unit


def air_velocity(seg: Segment, mass: float, offset_xy: np.ndarray, k_lin: float,
                 k_quad: float) -> np.ndarray:
    """Horizontal air velocity (n, 3), world ENU: relative air plus the drone's velocity."""
    air = invert_drag(world_force(seg, mass, offset_xy), k_lin, k_quad)
    air[:, :2] += seg.vel[:, :2]
    return air


@dataclass
class SegmentStats:
    name: str
    case: str
    kind: str
    duration_s: float
    distance_m: float | None
    mean_air_m_s: list[float]  # (2,) ENU
    speed_m_s: float
    heading_deg: float  # the direction the air moves toward, 0 = +x (as wind_heading_deg)
    sigma_m_s: float  # turbulence per horizontal axis
    tau_s: float | None
    mean_force_n: list[float]  # (2,)
    clipped_frac: float
    ref_speed_m_s: float | None
    true_air: dict[str, Any] | None = None  # the same numbers on the simulator's air


def _clip_mean(x: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean of rows of ``x`` (n, 2) and the mask kept, after iterative clipping at ``k``
    standard deviations (per sample deviation norm, over the 2-axis RMS)."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(5):
        m = x[keep].mean(axis=0)
        dev = np.linalg.norm(x - m, axis=1)
        rms = math.sqrt(float(np.mean(dev[keep] ** 2)) / 2.0)
        new = dev <= k * max(rms, 1e-9) * math.sqrt(2.0)
        if np.array_equal(new, keep):
            break
        keep = new
    return x[keep].mean(axis=0), keep


def correlation_time(x: np.ndarray, valid: np.ndarray, dt: float, max_lag_s: float
                     ) -> float | None:
    """Correlation time of ``x`` (n,): the lag where its autocorrelation, over valid pairs
    only, falls to 1/e, linearly interpolated; None if it never does within ``max_lag_s``.

    Taking out a finite hold's own mean pulls the sample autocorrelation down: it is about
    ``(rho - B) / (1 - B)`` for the true ``rho``, where ``B = (2 tau / T) (1 - (tau / T)
    (1 - exp(-T / tau)))`` is the variance of the mean of a Gauss-Markov process over a hold
    ``T`` long (in units of its variance). So the crossing is searched at ``(1/e - B) /
    (1 - B)`` instead of 1/e, with ``B`` from the previous estimate (three passes). One 60 s
    hold still scatters by about 30 % from one stretch of air to the next."""
    x = np.where(valid, x - x[valid].mean(), 0.0)
    v = valid.astype(float)
    var = float(np.sum(x * x) / max(v.sum(), 1.0))
    if var <= 0:
        return None
    max_lag = min(int(max_lag_s / dt), len(x) - 1)
    r = np.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        n = float(np.sum(v[:-lag] * v[lag:]))
        r[lag] = float(np.sum(x[:-lag] * x[lag:]) / n) / var if n >= 10 else np.nan
    span = float(v.sum()) * dt
    level, tau = 1.0 / math.e, None
    for _ in range(3):
        below = np.flatnonzero(~(r > level))  # NaN counts as "no answer here"
        if below.size == 0 or not np.isfinite(r[below[0]]):
            return tau
        k = int(below[0])
        tau = dt * (k - 1 + (r[k - 1] - level) / max(r[k - 1] - r[k], 1e-12))
        bias = (2 * tau / span) * (1 - (tau / span) * (1 - math.exp(-span / tau)))
        level = (1.0 / math.e - bias) / (1.0 - bias)
    return tau


def air_stats(air: np.ndarray, valid: np.ndarray, p: FitParams) -> dict[str, Any]:
    """Mean, turbulence and correlation time of a horizontal air series (n, 2), over the
    valid samples, after clipping (module docstring, step 3)."""
    mean, keep = _clip_mean(air[valid], p.clip_sigma)
    use = valid.copy()
    use[np.flatnonzero(valid)[~keep]] = False
    dev = air - mean
    sigma = math.sqrt(float(np.mean(np.sum(dev[use] ** 2, axis=1))) / 2.0)
    speed = float(np.linalg.norm(mean))
    along = dev @ (mean / speed) if speed > 1e-6 else dev[:, 0]
    return {"mean": mean, "use": use, "speed": speed, "sigma": sigma,
            "heading": float(math.degrees(math.atan2(mean[1], mean[0])) % 360.0),
            "tau": correlation_time(along, use, 1.0 / p.rate_hz, p.acf_max_lag_s)}


def segment_stats(seg: Segment, mass: float, offset_xy: np.ndarray, k_lin: float,
                  k_quad: float, p: FitParams) -> SegmentStats:
    air = air_velocity(seg, mass, offset_xy, k_lin, k_quad)[:, :2]
    force = world_force(seg, mass, offset_xy)[:, :2]
    a = air_stats(air, seg.valid, p)
    use = a["use"]
    true = None
    if seg.true_air is not None:
        # The same statistics on the simulator's own air over the same samples, so the
        # fit's error is told apart from what a finite hold can show at all.
        ta = air_stats(seg.true_air[:, :2], seg.valid, p)
        err = air[use] - seg.true_air[use, :2]
        true = {"mean_air_m_s": [float(v) for v in ta["mean"]],
                "speed_m_s": ta["speed"], "heading_deg": ta["heading"],
                "sigma_m_s": ta["sigma"], "tau_s": ta["tau"],
                "air_rms_error_m_s": [float(v) for v in np.sqrt(np.mean(err ** 2, axis=0))]}
    return SegmentStats(
        name=seg.name, case=seg.case, kind=seg.kind, duration_s=float(use.sum() / p.rate_hz),
        distance_m=None if seg.dist is None else float(np.mean(seg.dist[use])),
        mean_air_m_s=[float(v) for v in a["mean"]], speed_m_s=a["speed"],
        heading_deg=a["heading"], sigma_m_s=a["sigma"], tau_s=a["tau"],
        mean_force_n=[float(v) for v in force[use].mean(axis=0)],
        clipped_frac=float(1.0 - use.sum() / max(seg.valid.sum(), 1)),
        ref_speed_m_s=seg.ref_speed, true_air=true,
    )


# ---------------------------------------------------------------- across segments


def calm_offset(segments: list[Segment]) -> np.ndarray:
    """The accelerometer's x and y offset (m/s^2): the mean body specific force over calm
    holds (no wind, so no aero force), or zero without any."""
    calm = [s for s in segments if s.kind == "calm"]
    if not calm:
        return np.zeros(2)
    vals = np.concatenate([s.f_body[s.valid, :2] for s in calm])
    return vals.mean(axis=0)


def calibrate_drag(segments: list[Segment], mass: float, offset_xy: np.ndarray,
                   p: FitParams) -> dict[str, Any]:
    """Drag coefficients from free-air holds with a reference speed (module docstring)."""
    refs = [s for s in segments if s.kind == "free" and s.ref_speed]
    k_lin = p.k_lin
    k_quad = p.k_quad if p.k_quad is not None else NOMINAL_K_QUAD
    if not refs:
        return {"k_lin": k_lin, "k_quad": k_quad, "method": "configured",
                "note": "no free-air hold with a reference speed: speeds scale with "
                        "1 / sqrt(k_quad); the shelter does not"}
    speeds = np.array([s.ref_speed for s in refs])
    two = p.fit_k_lin and speeds.max() > 1.2 * speeds.min()
    for _ in range(4):
        rows, rhs = [], []
        for s in refs:
            st = segment_stats(s, mass, offset_xy, k_lin, k_quad, p)
            u = s.ref_speed
            direction = np.asarray(st.mean_air_m_s) / max(st.speed_m_s, 1e-9)
            f_along = float(np.asarray(st.mean_force_n) @ direction)
            rows.append([u, u * u + 1.5 * st.sigma_m_s ** 2])
            rhs.append(f_along)
        a, b = np.asarray(rows), np.asarray(rhs)
        if two:
            sol, *_ = np.linalg.lstsq(a, b, rcond=None)
            k_lin, k_quad = float(sol[0]), float(sol[1])
            if k_lin < 0:
                k_lin = 0.0
                k_quad = float(a[:, 1] @ b / (a[:, 1] @ a[:, 1]))
        else:
            k_quad = float(a[:, 1] @ (b - k_lin * a[:, 0]) / (a[:, 1] @ a[:, 1]))
        if k_quad <= 0:
            raise ValueError("fitted k_quad is not positive: check the reference speeds and "
                             "the calm offset")
    return {"k_lin": k_lin, "k_quad": k_quad,
            "method": "two speeds" if two else "one speed, k_lin fixed",
            "ref_speeds_m_s": sorted({float(v) for v in speeds})}


def shelter_curve(d: np.ndarray, length: float, floor: float) -> np.ndarray:
    return 1.0 - (1.0 - floor) * np.exp(-np.asarray(d, float) / length)


def fit_shelter(d: np.ndarray, g: np.ndarray, w: np.ndarray | None = None,
                floor_max: float = 0.9, floor: float | None = None) -> tuple[float, float]:
    """``(L, g0)`` minimising the weighted squared error of ``shelter_curve`` against
    ``g`` at distances ``d``. ``floor`` fixes ``g0`` (needed with one distance)."""
    d, g = np.asarray(d, float), np.asarray(g, float)
    w = np.ones_like(d) if w is None else np.asarray(w, float)
    y = 1.0 - g

    def best_floor(length: float) -> tuple[float, float]:
        x = np.exp(-d / length)
        if floor is not None:
            c = 1.0 - floor
        else:
            c = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-12))
            c = min(max(c, 1.0 - floor_max), 1.0)
        return float(np.sum(w * (y - c * x) ** 2)), 1.0 - c

    lengths = np.geomspace(0.05, 20.0, 400)
    errs = [best_floor(v)[0] for v in lengths]
    i = int(np.argmin(errs))
    lo, hi = lengths[max(i - 1, 0)], lengths[min(i + 1, len(lengths) - 1)]
    for _ in range(60):  # golden section on the bracket
        a = hi - (hi - lo) / 1.618033988749895
        b = lo + (hi - lo) / 1.618033988749895
        if best_floor(a)[0] < best_floor(b)[0]:
            hi = b
        else:
            lo = a
    length = 0.5 * (lo + hi)
    return float(length), float(best_floor(length)[1])


def fit(segments: list[Segment], mass: float, p: FitParams) -> dict[str, Any]:
    """The whole fit (module docstring, steps 1 to 5)."""
    if not segments:
        raise ValueError("no segments")
    offset = calm_offset(segments)
    drag = calibrate_drag(segments, mass, offset, p)
    k_lin, k_quad = drag["k_lin"], drag["k_quad"]
    stats = [segment_stats(s, mass, offset, k_lin, k_quad, p) for s in segments]

    cases: dict[str, dict[str, Any]] = {}
    for case in sorted({s.case for s in stats if s.kind != "calm"}):
        free = [s for s in stats if s.case == case and s.kind == "free"]
        wall = [s for s in stats if s.case == case and s.kind == "wall"]
        c: dict[str, Any] = {"free": [s.name for s in free], "wall": [s.name for s in wall]}
        if free:
            wts = np.array([s.duration_s for s in free])
            w_vec = np.average([s.mean_air_m_s for s in free], axis=0, weights=wts)
            c["source"] = "free-air holds"
            c["sigma_m_s"] = float(np.average([s.sigma_m_s for s in free], weights=wts))
            taus = [s.tau_s for s in free if s.tau_s is not None]
            c["tau_s"] = float(np.mean(taus)) if taus else None
        else:
            refs = [s.ref_speed_m_s for s in wall if s.ref_speed_m_s]
            if not refs or not wall:
                continue  # nothing to scale the wall holds against
            heading = np.mean([np.asarray(s.mean_air_m_s) / max(s.speed_m_s, 1e-9)
                               for s in wall], axis=0)
            w_vec = float(np.mean(refs)) * heading / np.linalg.norm(heading)
            c["source"] = "reference speed, heading from the wall holds"
            c["sigma_m_s"] = c["tau_s"] = None
        speed = float(np.linalg.norm(w_vec))
        c["mean_speed_m_s"] = speed
        c["heading_deg"] = float(math.degrees(math.atan2(w_vec[1], w_vec[0])) % 360.0)
        unit = w_vec / max(speed, 1e-9)
        true_w = None
        if free and all(s.true_air for s in free):
            true_w = np.average([s.true_air["mean_air_m_s"] for s in free], axis=0,
                                weights=[s.duration_s for s in free])
        holds = []
        for s in wall:
            a = np.asarray(s.mean_air_m_s)
            along = float(a @ unit)
            g = along / max(speed, 1e-9)
            row = {"name": s.name, "distance_m": s.distance_m, "shelter": g,
                   "crossflow_frac": float(abs(a[0] * unit[1] - a[1] * unit[0]) / speed),
                   "duration_s": s.duration_s}
            if c.get("sigma_m_s"):
                row["sigma_over_sheltered"] = s.sigma_m_s / max(abs(g) * c["sigma_m_s"], 1e-9)
            if true_w is not None and s.true_air:
                # The shelter the simulator's own air shows over the same samples.
                tw = float(np.linalg.norm(true_w))
                row["true_shelter"] = float(np.asarray(s.true_air["mean_air_m_s"]) @ true_w
                                            / (tw * tw))
            holds.append(row)
        c["holds"] = holds
        cases[case] = c

    holds = [h for c in cases.values() for h in c["holds"]]
    shelter: dict[str, Any] = {"holds": len(holds)}
    if holds:
        d = np.array([h["distance_m"] for h in holds])
        g = np.array([h["shelter"] for h in holds])
        w = np.array([h["duration_s"] for h in holds])
        distinct = len(np.unique(np.round(d, 2)))
        fixed = 0.0 if distinct < 2 else None
        length, floor = fit_shelter(d, g, w, p.shelter_floor_max, floor=fixed)
        resid = g - shelter_curve(d, length, floor)
        shelter.update(wall_decay_m=length, wall_floor=floor,
                       rms_residual=float(np.sqrt(np.average(resid ** 2, weights=w))),
                       floor_fixed=fixed is not None)
        if len(holds) >= 3 and distinct >= 3:
            loo = [fit_shelter(np.delete(d, i), np.delete(g, i), np.delete(w, i),
                               p.shelter_floor_max) for i in range(len(holds))]
            arr = np.array(loo)
            # Jackknife standard error.
            n = len(holds)
            shelter["wall_decay_se_m"] = float(math.sqrt((n - 1) * np.var(arr[:, 0])))
            shelter["wall_floor_se"] = float(math.sqrt((n - 1) * np.var(arr[:, 1])))

    free_cases = [c for c in cases.values() if c.get("sigma_m_s") is not None]
    turb = {}
    if free_cases:
        turb["sigma_m_s"] = float(np.mean([c["sigma_m_s"] for c in free_cases]))
        turb["intensity"] = float(np.mean([c["sigma_m_s"] / c["mean_speed_m_s"]
                                           for c in free_cases]))
        taus = [c["tau_s"] for c in free_cases if c["tau_s"] is not None]
        turb["tau_s"] = float(np.mean(taus)) if taus else None

    # The same turbulence numbers on the simulator's air over the same samples (Gazebo).
    true_free = [s for s in stats if s.kind == "free" and s.true_air]
    if free_cases and true_free:
        turb["true_air"] = {
            "sigma_m_s": float(np.mean([s.true_air["sigma_m_s"] for s in true_free])),
            "tau_s": float(np.mean([s.true_air["tau_s"] for s in true_free
                                    if s.true_air["tau_s"] is not None] or [np.nan])),
        }

    return {
        "mass_kg": mass,
        "calm_offset_m_s2": [float(v) for v in offset],
        "drag": drag,
        "segments": [asdict(s) for s in stats],
        "cases": cases,
        "shelter": shelter,
        "turbulence": turb,
    }


# ---------------------------------------------------------------- proposal and truth


def proposal(result: dict[str, Any], defaults: DisturbanceConfig | None = None
             ) -> dict[str, Any]:
    """A ``disturbance`` block centred on the fit, beside the defaults: each range is the
    fitted value +- max(2 standard errors, 25 %), and ``covered`` says whether the current
    default range already holds the fitted value."""
    defaults = defaults or DisturbanceConfig()
    out: dict[str, Any] = {"block": {}, "defaults": {}, "covered": {}}

    def put(key: str, value: float | None, se: float | None = None, lo: float = 0.0,
            hi: float = math.inf) -> None:
        if value is None:
            return
        half = max(2.0 * (se or 0.0), 0.25 * abs(value))
        out["block"][key] = [round(max(value - half, lo), 4), round(min(value + half, hi), 4)]
        rng = getattr(defaults, key)
        out["defaults"][key] = list(rng)
        out["covered"][key] = bool(rng[0] <= value <= rng[1])

    sh, tb, dr = result["shelter"], result["turbulence"], result["drag"]
    put("wall_decay", sh.get("wall_decay_m"), sh.get("wall_decay_se_m"), lo=0.05)
    put("wall_floor", sh.get("wall_floor"), sh.get("wall_floor_se"), hi=1.0)
    put("turb_tau", tb.get("tau_s"), lo=0.05)
    if dr["method"] != "configured":
        put("k_quad", dr["k_quad"])
    out["note"] = ("turb_sigma is per site and per wind, so it stays a range set by the "
                   "scenarios; the fitted intensity (sigma / mean speed) is in turbulence")
    return out


TRUTH_KEYS = {"mean_speed": ("cases", "mean_speed_m_s"), "heading_deg": ("cases", "heading_deg"),
              "turb_sigma": ("turbulence", "sigma_m_s"), "turb_tau": ("turbulence", "tau_s"),
              "k_quad": ("drag", "k_quad"), "k_lin": ("drag", "k_lin"),
              "wall_decay": ("shelter", "wall_decay_m"), "wall_floor": ("shelter", "wall_floor")}


def compare_truth(result: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Fitted against true values. ``mean_speed`` and ``heading_deg`` may be a value per
    case ({case: value}) or one value for every case."""
    rows: dict[str, Any] = {}
    for key, value in truth.items():
        if key not in TRUTH_KEYS:
            raise ValueError(f"unknown truth key {key}; known: {sorted(TRUTH_KEYS)}")
        part, name = TRUTH_KEYS[key]
        if part == "cases":
            for case, c in result["cases"].items():
                tv = value.get(case) if isinstance(value, dict) else value
                if tv is None:
                    continue
                fv = c[name]
                err = (fv - tv + 180.0) % 360.0 - 180.0 if key == "heading_deg" else fv - tv
                rows[f"{key}[{case}]"] = {"true": tv, "fit": fv, "error": err}
        else:
            fv = result[part].get(name)
            if fv is None or (part == "drag" and result["drag"].get("method") == "configured"):
                continue  # nothing fitted (a configured drag is not a result)
            rows[key] = {"true": float(value), "fit": fv, "error": fv - float(value),
                         "error_pct": 100.0 * (fv - float(value)) / float(value)
                         if float(value) else None}
    # What the fit can be held to for turbulence: the true air over the same samples (a
    # 60 s hold shows only part of the model's long-run sigma and tau).
    for key, name in (("turb_sigma", "sigma_m_s"), ("turb_tau", "tau_s")):
        tv = result["turbulence"].get("true_air", {}).get(name)
        fv = result["turbulence"].get(name)
        if tv is not None and fv is not None and math.isfinite(tv):
            rows[f"{key}[same air]"] = {"true": tv, "fit": fv, "error": fv - tv,
                                        "error_pct": 100.0 * (fv - tv) / tv}
    return rows


# ---------------------------------------------------------------- synthetic logs


def synthetic_topics(cfg: DisturbanceConfig, seed: int, hold_point: np.ndarray, mass: float,
                     duration_s: float = 60.0, wall_point: Any = None, wall_normal: Any = None,
                     rate_hz: float = 250.0, accel_noise: float = 0.0,
                     accel_offset_frd: Any = (0.0, 0.0, 0.0), sway_m: float = 0.05,
                     t0: float = 10.0) -> Topics:
    """A PX4-shaped log of a drone holding ``hold_point`` (ENU) in the model's air.

    The drone sways on a slow circle of radius ``sway_m`` (so velocity and acceleration are
    not zero) and tilts so that thrust balances gravity, the aero force and its
    acceleration; the accelerometer reads the specific force in FRD plus ``accel_offset_frd``
    and white noise. The air is the model's ``disturbance_force`` exactly, so a fit of
    these logs should return the config's values. No climb: the whole log is the hold."""
    rng = np.random.default_rng(seed + 1)
    p = sample_params(cfg, 1, seed, wall_point=wall_point, wall_normal=wall_normal)
    t = t0 + np.arange(0.0, duration_s, 1.0 / rate_hz)
    w = 2.0 * math.pi / 7.0
    hold = np.asarray(hold_point, float)
    pos = hold + sway_m * np.stack([np.cos(w * t), np.sin(w * t), 0.0 * t], axis=1)
    vel = sway_m * w * np.stack([-np.sin(w * t), np.cos(w * t), 0.0 * t], axis=1)
    acc = -sway_m * w * w * np.stack([np.cos(w * t), np.sin(w * t), 0.0 * t], axis=1)
    force = disturbance_force(p, t - t0, pos, vel)  # one env's params broadcast over time
    force[:, 2] = 0.0  # the fit assumes no vertical aero force; keep the log consistent
    # Thrust (ENU) balances: m a = T + F + m g.
    thrust = mass * acc - force + mass * np.array([0.0, 0.0, GRAVITY])
    z_body_ned = -ned_to_enu(thrust / np.linalg.norm(thrust, axis=1, keepdims=True))
    x_ref = np.array([1.0, 0.0, 0.0])
    x_b = x_ref - (z_body_ned @ x_ref)[:, None] * z_body_ned
    x_b /= np.linalg.norm(x_b, axis=1, keepdims=True)
    y_b = np.cross(z_body_ned, x_b)
    rot = np.stack([x_b, y_b, z_body_ned], axis=2)  # columns: body axes in NED
    spec_force_ned = ned_to_enu((thrust + force) / mass)  # ENU<->NED swap is its own inverse
    f_frd = np.einsum("nji,nj->ni", rot, spec_force_ned)
    f_frd = f_frd + np.asarray(accel_offset_frd, float) + accel_noise * rng.normal(size=f_frd.shape)
    q = rot_to_quat(rot)
    us = np.round(t * 1e6).astype(np.int64)
    pos_ned, vel_ned = ned_to_enu(pos), ned_to_enu(vel)
    return {
        "sensor_combined": {"timestamp": us, **{f"accelerometer_m_s2[{k}]": f_frd[:, k]
                                                for k in range(3)}},
        ATTITUDE: {"timestamp": us, **{f"q[{k}]": q[:, k] for k in range(4)}},
        GROUNDTRUTH: {"timestamp": us, "x": pos_ned[:, 0], "y": pos_ned[:, 1],
                      "z": pos_ned[:, 2], "vx": vel_ned[:, 0], "vy": vel_ned[:, 1],
                      "vz": vel_ned[:, 2]},
    }


# ---------------------------------------------------------------- inputs and entrypoint


def load_topics(path: str | Path) -> Topics:
    """The topics this module needs (first instance of each), with pyulog."""
    try:
        from pyulog import ULog
    except ImportError as e:  # optional dependency: pip install -e .[ulog]
        raise ImportError("reading .ulg needs pyulog: pip install -e .[ulog]") from e
    log = ULog(str(path), message_name_filter_list=TOPICS)
    return {d.name: d.data for d in log.data_list if d.multi_id == 0}


def latest(pattern: str) -> Path:
    """The newest path matching ``pattern`` (relative to the repo root)."""
    pat = pattern if Path(pattern).is_absolute() else str(REPO_ROOT / pattern)
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"nothing matches {pattern}")
    return Path(hits[-1])


def sitl_spec(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """For ``sitl_run`` (a run folder of sim2real.gazebo.run_scenario): the flight log, and
    the spec with the wall and the gust exclusions filled in from its config.yaml. Gazebo's
    sim time is the episode time and PX4's clock in lockstep, so the gust times are log
    times; ``gust_pad_s`` widens each window for the drone's response."""
    run = latest(spec["sitl_run"])
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    out = dict(spec)
    out.setdefault("wall", {"point": [0.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0]})
    pad = float(spec.get("gust_pad_s", 1.0))
    dist = cfg.get("disturbance", {})
    gusts = [[s - pad, s + d + pad] for s, d in zip(dist.get("gust_start_s", []),
                                                   dist.get("gust_duration_s", []), strict=True)]
    out["exclude_s"] = list(spec.get("exclude_s") or []) + gusts
    out["run"] = str(run.relative_to(REPO_ROOT)) if run.is_relative_to(REPO_ROOT) else str(run)
    return run / "flight.ulg", out


def trace_air(path: Path, t: np.ndarray) -> np.ndarray:
    """The air (n, 3) the Gazebo plugin's trace (``wind_trace.csv``) had at times ``t``: the
    free-stream wind times the shelter. Scenarios with near-wall flow terms are not covered
    (the trace does not log them)."""
    from sim2real.gazebo.verify_trace import read_trace

    tr = read_trace(path)
    return np.stack([np.interp(t, tr["t"], tr[w] * tr["shelter"]) for w in ("wx", "wy", "wz")],
                    axis=1)


def load_segments(params: dict[str, Any], p: FitParams) -> tuple[list[Segment],
                                                                 list[dict[str, Any]]]:
    """Segments from the config: each spec gives ``synthetic`` (a model-made log, for the
    smoke run and tests), ``sitl_run`` (a Gazebo run folder) or ``ulog`` (any PX4 log).
    For a Gazebo run, ``ref_speed_m_s: trace`` reads the reference from the plugin's trace,
    as an anemometer at the hold point would, and the trace's air is kept as truth."""
    segments, used = [], []
    for spec in params["segments"]:
        trace = None
        if "synthetic" in spec:
            syn = dict(spec["synthetic"])
            cfg = DisturbanceConfig.from_dict(syn.pop("disturbance", {}))
            topics = synthetic_topics(cfg, mass=mass_of(params), **syn)
        elif "sitl_run" in spec:
            path, spec = sitl_spec(spec)
            topics = load_topics(path)
            trace = path.parent / "wind_trace.csv"
        elif "ulog" in spec:
            topics = load_topics(latest(spec["ulog"]))
        else:
            raise ValueError(f"segment {spec.get('name')}: give synthetic, sitl_run or ulog")
        ref = spec.get("ref_speed_m_s")
        seg = segment_from_topics(topics, {**spec, "ref_speed_m_s": None}, p)
        if trace is not None and trace.exists():
            seg.true_air = trace_air(trace, seg.t)
        if ref == "trace":
            if seg.true_air is None:
                raise ValueError(f"segment {seg.name}: ref_speed_m_s trace needs a Gazebo run")
            seg.ref_speed = float(np.linalg.norm(seg.true_air[seg.valid, :2].mean(axis=0)))
        elif ref is not None:
            seg.ref_speed = float(ref)
        segments.append(seg)
        used.append({**{k: v for k, v in spec.items() if k != "synthetic"},
                     "ref_speed_m_s": seg.ref_speed})
    return segments, used


def mass_of(params: dict[str, Any]) -> float:
    if params.get("mass_kg") is not None:
        return float(params["mass_kg"])
    if params.get("airframe"):
        from envs.structure_hold.airframes import airframe_config
        return float(airframe_config(params["airframe"])["quad"]["mass"])
    raise ValueError("give mass_kg or airframe")


def format_report(result: dict[str, Any]) -> str:
    lines = [f"mass {result['mass_kg']:.3f} kg, calm offset "
             f"{result['calm_offset_m_s2']} m/s^2",
             f"drag ({result['drag']['method']}): k_lin {result['drag']['k_lin']:.4f}, "
             f"k_quad {result['drag']['k_quad']:.5f}", "",
             f"{'segment':28s} {'kind':5s} {'d (m)':>6s} {'speed':>6s} {'head':>6s} "
             f"{'sigma':>6s} {'tau':>5s} {'s':>5s}"]
    for s in result["segments"]:
        d = f"{s['distance_m']:.2f}" if s["distance_m"] is not None else "-"
        tau = f"{s['tau_s']:.2f}" if s["tau_s"] is not None else "-"
        lines.append(f"{s['name']:28s} {s['kind']:5s} {d:>6s} {s['speed_m_s']:6.2f} "
                     f"{s['heading_deg']:6.1f} {s['sigma_m_s']:6.2f} {tau:>5s} "
                     f"{s['duration_s']:5.0f}")
    for case, c in result["cases"].items():
        lines.append(f"\ncase {case}: free stream {c['mean_speed_m_s']:.2f} m/s toward "
                     f"{c['heading_deg']:.1f} deg ({c['source']})")
        for h in c["holds"]:
            extra = (f", sigma/(g sigma_free) {h['sigma_over_sheltered']:.2f}"
                     if "sigma_over_sheltered" in h else "")
            lines.append(f"  {h['name']:26s} d {h['distance_m']:.2f} m  shelter "
                         f"{h['shelter']:.3f}  crossflow {h['crossflow_frac']:.3f}{extra}")
    sh = result["shelter"]
    if "wall_decay_m" in sh:
        lines.append(f"\nshelter: L {sh['wall_decay_m']:.3f} m"
                     + (f" +- {sh['wall_decay_se_m']:.3f}" if "wall_decay_se_m" in sh else "")
                     + f", g0 {sh['wall_floor']:.3f}"
                     + (f" +- {sh['wall_floor_se']:.3f}" if "wall_floor_se" in sh else "")
                     + f", rms residual {sh['rms_residual']:.3f}")
    tb = result["turbulence"]
    if tb:
        tau = f"{tb['tau_s']:.2f} s" if tb.get("tau_s") is not None else "-"
        lines.append(f"turbulence: sigma {tb['sigma_m_s']:.3f} m/s (intensity "
                     f"{tb['intensity']:.3f}), tau {tau}")
    if result.get("truth"):
        lines.append("\nagainst the truth:")
        for k, r in result["truth"].items():
            pct = f" ({r['error_pct']:+.1f} %)" if r.get("error_pct") is not None else ""
            lines.append(f"  {k:24s} true {r['true']:.4f}  fit {r['fit']:.4f}  "
                         f"error {r['error']:+.4f}{pct}")
    prop = result["proposal"]
    lines.append("\nproposed disturbance ranges (current default, covered?):")
    for k, v in prop["block"].items():
        lines.append(f"  {k:12s} {v}  ({prop['defaults'][k]}, {prop['covered'][k]})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    params = cfg["params"]
    if args.smoke and params.get("smoke_segments"):
        params = {**params, "segments": params["smoke_segments"]}
    p = FitParams.from_dict(params.get("fit"))
    writer = RunWriter(args.run_dir)
    try:
        mass = mass_of(params)
        segments, used = load_segments(params, p)
        for i, s in enumerate(segments):
            writer.metric(i + 1, segment=s.name, kind=s.kind, seconds=float(s.valid.sum())
                          / p.rate_hz)
        result = fit(segments, mass, p)
        result["inputs"] = used
        if params.get("truth"):
            result["truth"] = compare_truth(result, params["truth"])
        result["proposal"] = proposal(result)
        (Path(args.run_dir) / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(format_report(result), flush=True)
        headline = {"segments": len(segments),
                    "k_quad": round(result["drag"]["k_quad"], 5)}
        sh = result["shelter"]
        if "wall_decay_m" in sh:
            headline.update(wall_decay_m=round(sh["wall_decay_m"], 3),
                            wall_floor=round(sh["wall_floor"], 3))
        if result["turbulence"]:
            headline["turb_sigma_m_s"] = round(result["turbulence"]["sigma_m_s"], 3)
        if result.get("truth"):
            headline["worst_truth_error_pct"] = round(max(
                abs(r["error_pct"]) for r in result["truth"].values()
                if r.get("error_pct") is not None), 1)
        writer.summary("ok", headline=headline,
                       vs_baseline="truth from the scenario where given; else the current "
                                   "envs/disturbance default ranges (fit.json proposal)")
        return 0
    except Exception as exc:  # recorded in summary.json, then re-raised for the log
        writer.summary("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
