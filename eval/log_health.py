"""Log health check: is PX4 really flying on the external position source?

    python -m eval.log_health flight.ulg [--json report.json]

Reads one PX4 .ulg and answers, in plain English, the questions a drone maker asks when
they wire a VIO (or mocap, or any external position source) into EKF2 for GPS-free flight
(docs/log-health.md):

1. does the vision stream reach PX4, at what rate, with what gaps;
2. is EKF2 fusing it (position, height, velocity), or rejecting it, and is GPS still in use;
3. is it in the right frame: the vision axes against the IMU's, which catches ENU sent as
   NED, a mirrored axis, an upside-down height and a VIO frame that is turned away from the
   compass's north;
4. where the heading comes from (vision yaw or the compass) and, when the vision sends an
   attitude, whether its yaw agrees with EKF2's;
5. how late the vision really is, measured from the log against the IMU, and whether
   ``EKF2_EV_DELAY`` matches it.

Frame and delay are measured against the IMU, not against EKF2's own position: EKF2 is pulled
onto whatever vision it fuses, so comparing vision with EKF2 hides exactly the mistakes this
looks for. The accelerometer, turned into the world frame by EKF2's attitude and integrated,
gives a velocity whose slow drift is filtered out; the vision velocity is filtered the same
way. The lag that lines the two up best is the vision's real delay, counted from the time
PX4 stamped on the sample (``timestamp_sample``), which is the time EKF2 subtracts
``EKF2_EV_DELAY`` from. Which orthogonal map lines up the horizontal axes best gives the
frame: a turn is a heading offset, a reflection is a handedness mistake. Both need the
vehicle to move: the take-off climb is enough for the delay, the horizontal frame needs a
few side-to-side metres or a gusty hover. Without enough motion those checks say so and
skip; they do not guess.

When the log has simulator truth (``vehicle_local_position_groundtruth``, SITL only) the
delay is measured against it too, as a cross-check of the IMU method.

Everything is read in PX4's NED, FRD and microseconds and reported in SI units; only
magnitudes, angles and signs come out, so no ENU conversion is needed (docs/conventions.md).
Needs pyulog (``pip install -e .[ulog]``); the checks themselves take plain arrays
(``check_topics``), so tests build logs without a file.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VISION = "vehicle_visual_odometry"
EV_POS = "estimator_aid_src_ev_pos"
EV_HGT = "estimator_aid_src_ev_hgt"
EV_VEL = "estimator_aid_src_ev_vel"
EV_YAW = "estimator_aid_src_ev_yaw"
FLAGS = "estimator_status_flags"
ESTIMATOR_STATUS = "estimator_status"
IMU = "sensor_combined"
MAG = "vehicle_magnetometer"
ATTITUDE = "vehicle_attitude"
LAND = "vehicle_land_detected"
TRUTH = "vehicle_local_position_groundtruth"
MOCAP = "vehicle_mocap_odometry"
TOPICS = [VISION, EV_POS, EV_HGT, EV_VEL, EV_YAW, FLAGS, ESTIMATOR_STATUS, IMU, ATTITUDE,
          LAND, TRUTH, MAG, MOCAP]
PARAMS = ["EKF2_EV_CTRL", "EKF2_EV_DELAY", "EKF2_HGT_REF", "EKF2_GPS_CTRL", "EKF2_EV_QMIN",
          "EKF2_MAG_TYPE", "EKF2_EV_NOISE_MD"]

# estimator_status.control_mode_flags bits (PX4 v1.17 filter_control_status_u), for logs
# without estimator_status_flags (sim2real/estimate_bridge/ulog_checks.py says why).
CONTROL_MODE_BITS = {"cs_gnss_pos": 2, "cs_mag_hdg": 4, "cs_mag_3d": 5, "cs_baro_hgt": 9,
                     "cs_gps_hgt": 11, "cs_ev_pos": 12, "cs_ev_yaw": 13, "cs_ev_hgt": 14,
                     "cs_ev_vel": 24, "cs_inertial_dead_reckoning": 29, "cs_gnss_vel": 44}
# VehicleOdometry.pose_frame
POSE_FRAMES = {0: "unknown", 1: "NED (north-east-down)", 2: "FRD (forward-right-down, "
               "arbitrary heading)"}
HGT_REF = {0: "barometer", 1: "GPS", 2: "rangefinder", 3: "vision"}
GRAVITY = 9.80665

GRID_HZ = 100.0          # common time base for the IMU and vision velocities
HIGHPASS_S = 4.0         # moving-average window removed from both velocities (IMU drift)
SMOOTH_S = 0.5           # moving average on both, against VIO and accelerometer noise
LAG_RANGE_S = (-0.1, 0.6)
MAX_GAP_S = 0.25         # vision gaps longer than this are cut out of the comparison
JUMP_SPEED_MPS = 5.0     # a vision step faster than this, and than JUMP_RATIO times the
JUMP_RATIO = 3.0         #   median speed around it, is a jump (relocalisation, reset)
MIN_MOTION_Z_MPS = 0.05  # filtered velocity RMS needed on an axis for it to count
MIN_MOTION_XY_MPS = 0.04
MIN_CORRELATION = 0.6    # weaker alignment than this: too little motion, no verdict
SEGMENT_S = 15.0         # level frame fitted per stretch of this length ...
GOOD_SEGMENT_FIT = 0.8   # ... and only the stretches that fit at least this well count

# Thresholds of the verdicts (docs/log-health.md says where each comes from).
FUSED_PASS, FUSED_WARN = 0.95, 0.5
REJECTED_WARN, REJECTED_FAIL = 0.05, 0.2
YAW_WARN_DEG, YAW_FAIL_DEG = 10.0, 20.0
DELAY_WARN_S, DELAY_FAIL_S = 0.03, 0.08
GAP_WARN_S = 0.5

Topics = dict[str, dict[str, np.ndarray]]


@dataclass
class Check:
    name: str
    status: str  # pass, warn, fail, skip (not enough in the log to say)
    says: str    # one or two plain sentences for the integrator
    numbers: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    ulog: str
    verdict: str
    checks: list[Check]

    def worst(self) -> str:
        order = ["pass", "skip", "warn", "fail"]
        return max((c.status for c in self.checks), key=order.index, default="skip")

    def text(self) -> str:
        mark = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "--  "}
        lines = [f"Log health: {self.ulog}", "", self.verdict, ""]
        for c in self.checks:
            lines.append(f"[{mark[c.status]}] {c.name}: {c.says}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"ulog": self.ulog, "verdict": self.verdict, "worst": self.worst(),
                "checks": [asdict(c) for c in self.checks]}


# ---------------------------------------------------------------- reading the log

def load(path: str | Path) -> tuple[Topics, dict[str, float], int]:
    """Topics (instance 0), the parameters this check reads (last value in the log) and the
    log's start time in microseconds."""
    try:
        from pyulog import ULog
    except ImportError as e:  # optional dependency: pip install -e .[ulog]
        raise ImportError("reading .ulg needs pyulog: pip install -e .[ulog]") from e
    log = ULog(str(path), message_name_filter_list=TOPICS)
    topics = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    params = {k: float(v) for k, v in log.initial_parameters.items() if k in PARAMS}
    for _t, k, v in log.changed_parameters:
        if k in PARAMS:
            params[k] = float(v)
    return topics, params, int(log.start_timestamp)


def _t(msg: dict[str, np.ndarray], start_us: int, key: str = "timestamp") -> np.ndarray:
    return (np.asarray(msg[key], dtype=np.int64) - start_us) * 1e-6


def _flags(topics: Topics, start_us: int) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """EKF2 aiding flags (cs_*) and their times, from either topic that carries them."""
    if FLAGS in topics:
        f = topics[FLAGS]
        return {k: np.asarray(v).astype(bool) for k, v in f.items()
                if k.startswith("cs_")}, _t(f, start_us)
    es = topics.get(ESTIMATOR_STATUS)
    if es is not None and "control_mode_flags" in es:
        bits = np.asarray(es["control_mode_flags"]).astype(np.uint64)
        return ({k: ((bits >> np.uint64(b)) & np.uint64(1)).astype(bool)
                 for k, b in CONTROL_MODE_BITS.items()}, _t(es, start_us))
    return None


def airborne_window(topics: Topics, start_us: int) -> tuple[float, float] | None:
    """First to last time the land detector said airborne, or None (never airborne, or no
    land detector in the log: then every check uses the whole log)."""
    land = topics.get(LAND)
    if land is None or "landed" not in land:
        return None
    t = _t(land, start_us)
    up = ~np.asarray(land["landed"]).astype(bool)
    if not np.any(up):
        return None
    idx = np.flatnonzero(up)
    t1 = float(t[idx[-1] + 1]) if idx[-1] + 1 < t.size else float("inf")
    return float(t[idx[0]]), t1


def _in(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    return np.ones(t.size, bool) if window is None else (t >= window[0]) & (t <= window[1])


# ---------------------------------------------------------------- signal helpers

def _movavg(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average along axis 0 (same length; edges average what exists)."""
    if n <= 1:
        return x.copy()
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, [(pad, n - 1 - pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(xp, k, mode="valid")
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def _bandpass(v: np.ndarray) -> np.ndarray:
    hp = v - _movavg(v, int(round(HIGHPASS_S * GRID_HZ)))
    return _movavg(hp, max(1, int(round(SMOOTH_S * GRID_HZ))))


def _quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions (w, x, y, z), body to world, as (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def _circ_mean(a: np.ndarray) -> float:
    return float(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def compass_yaw(topics: Topics, start_us: int, window: tuple[float, float] | None
                ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The compass heading over the log: times, heading (rad) and the EKF-minus-compass
    offset on the ground in the 5 s before take-off (declination plus calibration), which
    is added to the heading so that on the ground it equals EKF2's yaw.

    The heading is the magnetometer levelled with EKF2's roll and pitch (roll and pitch come
    from gravity, not from the vision, so a wrong vision frame cannot bend them much)."""
    mag, att = topics.get(MAG), topics.get(ATTITUDE)
    if mag is None or att is None or "magnetometer_ga[0]" not in mag or window is None:
        return None
    tm = _t(mag, start_us)
    m = np.stack([mag[f"magnetometer_ga[{i}]"] for i in range(3)], 1).astype(float)
    ta = _t(att, start_us)
    ta, first = np.unique(ta, return_index=True)
    q = np.stack([att[f"q[{i}]"] for i in range(4)], 1).astype(float)[first]
    qa = q[np.clip(np.searchsorted(ta, tm), 0, ta.size - 1)]
    yaw = _yaw(qa)
    # level the field: R_tilt = Rz(-yaw) R(q); a level field (B cos h, -B sin h) at heading h
    c, sn = np.cos(-yaw), np.sin(-yaw)
    rz = np.zeros((yaw.size, 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = c, -sn, sn, c, 1.0
    m_level = np.einsum("nij,njk,nk->ni", rz, _quat_to_dcm(qa), m)
    heading = np.arctan2(-m_level[:, 1], m_level[:, 0])
    ground = (tm >= window[0] - 5.0) & (tm < window[0])
    if ground.sum() < 10:
        return None
    zero = _circ_mean(_wrap(yaw[ground] - heading[ground]))
    return tm, np.asarray(_wrap(heading + zero)), zero


def heading_vs_compass(topics: Topics, start_us: int, window: tuple[float, float] | None
                       ) -> dict[str, float] | None:
    """How far EKF2's heading moved away from the compass once airborne, in degrees.

    When EKF2 fuses a vision frame that is turned from north, it turns its own heading to
    make the vision fit, the compass notwithstanding: on the Gazebo test log with a 90 deg
    turned frame EKF2's heading moved 79 deg away from the compass (docs/log-health.md)."""
    cy = compass_yaw(topics, start_us, window)
    if cy is None:
        return None
    tm, heading, zero = cy
    att = topics[ATTITUDE]
    ta = _t(att, start_us)
    ta, first = np.unique(ta, return_index=True)
    q = np.stack([att[f"q[{i}]"] for i in range(4)], 1).astype(float)[first]
    yaw = _yaw(q[np.clip(np.searchsorted(ta, tm), 0, ta.size - 1)])
    flying = _in(tm, window)
    if flying.sum() < 10:
        return None
    drift = _wrap(yaw[flying] - heading[flying])
    return {"drift_deg": math.degrees(_circ_mean(drift)),
            "drift_p95_deg": float(np.degrees(np.percentile(np.abs(drift), 95))),
            "ground_offset_deg": math.degrees(zero)}


def imu_velocity(topics: Topics, start_us: int, grid: np.ndarray,
                 compass: tuple[np.ndarray, np.ndarray, float] | None = None
                 ) -> np.ndarray | None:
    """Velocity from the accelerometer in NED on ``grid`` (drift and offset included; only
    its filtered part is used). The body-frame specific force is turned into the world by
    EKF2's attitude and gravity added back. With ``compass`` (``compass_yaw``) the heading
    of that rotation is the compass's instead of EKF2's, so a heading EKF2 bent to fit a
    wrong vision frame does not hide the mistake."""
    imu, att = topics.get(IMU), topics.get(ATTITUDE)
    if imu is None or att is None:
        return None
    ti = _t(imu, start_us)
    acc = np.stack([imu[f"accelerometer_m_s2[{i}]"] for i in range(3)], 1).astype(float)
    ta = _t(att, start_us)
    q = np.stack([att[f"q[{i}]"] for i in range(4)], 1).astype(float)
    ta, first = np.unique(ta, return_index=True)
    q = q[first]
    # Nearest attitude sample is plenty: attitude changes slowly next to the IMU rate.
    idx = np.clip(np.searchsorted(ta, ti), 1, ta.size - 1)
    idx -= (ti - ta[idx - 1]) < (ta[idx] - ti)
    a_world = np.einsum("nij,nj->ni", _quat_to_dcm(q[idx]), acc)
    if compass is not None:
        tm, heading, _zero = compass
        # turn the level part by (compass heading - EKF yaw) about the vertical
        dyaw = _wrap(np.interp(ti, tm, np.unwrap(heading)) - _yaw(q[idx]))
        c, sn = np.cos(dyaw), np.sin(dyaw)
        n, e = a_world[:, 0].copy(), a_world[:, 1].copy()
        a_world[:, 0], a_world[:, 1] = c * n - sn * e, sn * n + c * e
    a_world[:, 2] += GRAVITY
    # Integrate at the IMU's own rate (trapezoids), then sample on the grid: resampling
    # the acceleration first would lose short spikes and leave steps in the velocity.
    v = np.concatenate([np.zeros((1, 3)), np.cumsum(
        0.5 * (a_world[1:] + a_world[:-1]) * np.diff(ti)[:, None], axis=0)])
    return np.stack([np.interp(grid, ti, v[:, i]) for i in range(3)], 1)


def _central_velocity(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Velocity by central differences (no half-sample lag), one-sided at the ends."""
    v = np.empty_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / (t[2:] - t[:-2])[:, None]
    v[0] = (p[1] - p[0]) / (t[1] - t[0])
    v[-1] = (p[-1] - p[-2]) / (t[-1] - t[-2])
    return v


def _jumps(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Times of vision jumps: steps faster than JUMP_SPEED_MPS and than JUMP_RATIO times
    the median step speed over the 31 samples around (so fast flight is not a jump)."""
    if t.size < 32:
        return np.zeros(0)
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
    padded = np.pad(speed, 15, mode="edge")
    local = np.median(np.lib.stride_tricks.sliding_window_view(padded, 31), axis=1)
    return t[1:][(speed > JUMP_SPEED_MPS) & (speed > JUMP_RATIO * local)]


def _valid_mask(grid: np.ndarray, t: np.ndarray, window: tuple[float, float] | None,
                breaks: np.ndarray | None = None) -> np.ndarray:
    """Grid points covered by samples no further apart than MAX_GAP_S, inside ``window``,
    away from the times in ``breaks`` (vision jumps), with the high-pass window's
    half-width trimmed off every edge."""
    ok = np.zeros(grid.size, bool)
    if t.size < 3:
        return ok
    j = np.clip(np.searchsorted(t, grid), 1, t.size - 1)
    ok = (grid >= t[0]) & (grid <= t[-1]) & ((t[j] - t[j - 1]) <= MAX_GAP_S)
    ok &= _in(grid, window)
    if breaks is not None and breaks.size:
        near = np.searchsorted(grid, breaks)
        for d in (-2, -1, 0, 1, 2):
            ok[np.clip(near + d, 0, grid.size - 1)] = False
    trim = int(round(HIGHPASS_S * GRID_HZ / 2))
    bad = ~ok
    # widen every invalid stretch by the filter half-width (edges of the filter see it)
    bad = np.convolve(bad.astype(float), np.ones(2 * trim + 1), mode="same") > 0
    return ~bad


@dataclass
class Alignment:
    """How the vision velocity lines up with a reference velocity."""

    lag_s: float | None
    score: float          # 0..1: how well the best lag lines them up
    delay_from: str | None  # "vertical" or "level": the motion the lag was measured on
    z_corr: float | None  # vertical correlation at the best lag (negative: height flipped)
    xy_score: float | None  # 0..1: how well the best orthogonal map lines up level motion
    yaw_offset_deg: float | None  # turn from the reference's north to the vision's x axis
    mirrored: bool | None         # horizontal axes are a reflection, not a turn
    xy_motion_mps: float
    z_motion_mps: float
    samples: int


def _z_score(v: np.ndarray, r: np.ndarray) -> float:
    """|correlation| of the vertical velocities (a flipped height still lines up)."""
    den = float(np.sqrt((v[:, 2] @ v[:, 2]) * (r[:, 2] @ r[:, 2])))
    return abs(float(v[:, 2] @ r[:, 2])) / den if den > 0 else -np.inf


def _xy_score(v: np.ndarray, r: np.ndarray) -> float:
    """How well the best orthogonal 2x2 map (any turn, or a reflection) lines the level
    velocities up: the sum of singular values of v^T r over the norms, 1 for a perfect map."""
    den = float(np.sqrt((v[:, :2] ** 2).sum() * (r[:, :2] ** 2).sum()))
    if den <= 0:
        return -np.inf
    return float(np.linalg.svd(v[:, :2].T @ r[:, :2], compute_uv=False).sum()) / den


def _procrustes(v: np.ndarray, r: np.ndarray) -> tuple[float, bool, float]:
    """The orthogonal 2x2 map M with r ~ v @ M (reflection allowed): the turn of the vision x
    axis in the reference frame (deg, a NED yaw), whether M is a reflection, and the fit."""
    u, _s, vt = np.linalg.svd(v[:, :2].T @ r[:, :2])
    m2 = u @ vt
    # r = v @ m2: a vision x-axis unit vector lands on row 0 of m2 in the reference frame
    return (math.degrees(math.atan2(m2[0, 1], m2[0, 0])), bool(np.linalg.det(m2) < 0),
            _xy_score(v, r))


def _level_map(v: np.ndarray, r: np.ndarray) -> tuple[float, bool, float]:
    """``_procrustes`` over SEGMENT_S stretches, keeping the stretches that fit well.

    One fit over the whole flight is spoilt by any stretch where the IMU's world heading is
    off (on the Gazebo ENU-as-NED log EKF2's heading, and with it the vehicle, swung around
    for the first 15 s: fit 0.44 there, 0.98 after). The answer is the majority of the
    well-fitting stretches (circular mean of their turn); with none, the whole-flight fit."""
    n = int(SEGMENT_S * GRID_HZ)
    segs = [_procrustes(v[i:i + n], r[i:i + n]) for i in range(0, len(v) - n // 2, n)
            if np.sqrt((r[i:i + n, :2] ** 2).sum(1).mean() / 2) >= MIN_MOTION_XY_MPS]
    good = [sg for sg in segs if sg[2] >= GOOD_SEGMENT_FIT]
    if not good:
        return _procrustes(v, r)
    mirrored = sum(sg[1] for sg in good) * 2 > len(good)
    same = [sg for sg in good if sg[1] == mirrored]
    yaw = np.radians([sg[0] for sg in same])
    return (math.degrees(_circ_mean(yaw)), mirrored, float(np.median([sg[2] for sg in same])))


def align(t_grid: np.ndarray, ref_vel: np.ndarray, vis_vel: np.ndarray, mask: np.ndarray
          ) -> Alignment:
    """Search the lag that lines ``vis_vel`` up with ``ref_vel`` (both (N, 3) NED on the
    grid, already band-passed), then fit the map between the horizontal axes at that lag.

    The lag comes from the vertical motion when there is enough of it (the take-off climb):
    the vertical accelerometer barely depends on the attitude, and on the Gazebo test logs it
    matched simulator truth within 4 ms, where the level motion, which leans on EKF2's tilt
    to take gravity out, was off by up to 25 ms (docs/log-health.md). Without enough
    vertical motion the level motion is used, allowing any orthogonal map between the axes so
    that a wrong frame does not hide the delay."""
    lags = np.arange(int(round(LAG_RANGE_S[0] * GRID_HZ)), int(round(LAG_RANGE_S[1] * GRID_HZ))
                     + 1)
    z_motion = float(np.sqrt(np.mean(ref_vel[mask, 2] ** 2))) if mask.any() else 0.0
    xy_motion = (float(np.sqrt(np.mean((ref_vel[mask, :2] ** 2).sum(1)) / 2))
                 if mask.any() else 0.0)
    use_z, use_xy = z_motion >= MIN_MOTION_Z_MPS, xy_motion >= MIN_MOTION_XY_MPS
    empty = Alignment(None, 0.0, None, None, None, None, None, xy_motion, z_motion,
                      int(mask.sum()))
    if not (use_z or use_xy):
        return empty
    n = t_grid.size
    score = _z_score if use_z else _xy_score

    def search(ok: np.ndarray) -> np.ndarray:
        scores = np.full(lags.size, -np.inf)
        for k, lag in enumerate(lags):
            # vision sample at grid i was taken at grid i - lag: compare vis[i], ref[i - lag]
            i0, i1 = max(0, lag), min(n, n + lag)
            m = ok[i0:i1] & ok[i0 - lag:i1 - lag]
            if m.sum() < GRID_HZ * 2:
                continue
            scores[k] = score(vis_vel[i0:i1][m], ref_vel[i0 - lag:i1 - lag][m])
        return scores

    scores = search(mask)
    if not np.isfinite(scores).any():
        return empty
    k = int(np.argmax(scores))
    best = float(lags[k])
    if 0 < k < lags.size - 1 and np.isfinite(scores[k - 1]) and np.isfinite(scores[k + 1]):
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        if a - 2 * b + c < 0:
            best += 0.5 * (a - c) / (a - 2 * b + c)
    lag = int(lags[k])
    i0, i1 = max(0, lag), min(n, n + lag)
    m = mask[i0:i1] & mask[i0 - lag:i1 - lag]
    v, r = vis_vel[i0:i1][m], ref_vel[i0 - lag:i1 - lag][m]
    z_corr = (float(np.corrcoef(v[:, 2], r[:, 2])[0, 1]) if use_z else None)
    yaw, mirrored, xy_score = None, None, None
    if use_xy:
        yaw, mirrored, xy_score = _level_map(v, r)
    return Alignment(best / GRID_HZ, float(scores[k]), "vertical" if use_z else "level", z_corr,
                     xy_score, yaw, mirrored, xy_motion, z_motion, int(m.sum()))


def vision_alignment(topics: Topics, start_us: int, window: tuple[float, float] | None,
                     reference: str = "imu", compass: tuple | None = None
                     ) -> Alignment | None:
    """Align the vision velocity with the IMU's (``reference="imu"``, heading from
    ``compass`` when given) or with simulator truth (``"truth"``). The vision sample time is
    ``timestamp_sample``."""
    vis = topics.get(VISION)
    if vis is None or "position[0]" not in vis:
        return None
    tv = _t(vis, start_us, "timestamp_sample" if "timestamp_sample" in vis else "timestamp")
    pv = np.stack([vis[f"position[{i}]"] for i in range(3)], 1).astype(float)
    tv, first = np.unique(tv, return_index=True)
    pv = pv[first]
    fin = np.isfinite(pv).all(1)
    tv, pv = tv[fin], pv[fin]
    if tv.size < 10:
        return None
    grid = np.arange(math.ceil(tv[0] * GRID_HZ), math.floor(tv[-1] * GRID_HZ)) / GRID_HZ
    if grid.size < GRID_HZ * 5:
        return None
    vv = _central_velocity(tv, pv)
    # A VIO jump is a velocity spike no IMU saw; one is enough to drag the lag (by 20 ms on
    # the noisy Gazebo log, docs/log-health.md), so it is cut out with the filter's reach.
    breaks = _jumps(tv, pv)
    vis_vel = np.stack([np.interp(grid, tv, vv[:, i]) for i in range(3)], 1)
    if reference == "imu":
        ref = imu_velocity(topics, start_us, grid, compass)
        mask = _valid_mask(grid, tv, window, breaks)
        if IMU in topics:
            mask &= (grid >= _t(topics[IMU], start_us)[0]) & (grid <= _t(topics[IMU],
                                                                           start_us)[-1])
    else:
        truth = topics.get(TRUTH)
        if truth is None or "vx" not in truth:
            return None
        tt = _t(truth, start_us)
        tt, f2 = np.unique(tt, return_index=True)
        ref = np.stack([np.interp(grid, tt, np.asarray(truth[k], float)[f2])
                        for k in ("vx", "vy", "vz")], 1)
        mask = _valid_mask(grid, tv, window, breaks) & (grid >= tt[0]) & (grid <= tt[-1])
    if ref is None:
        return None
    return align(grid, _bandpass(ref), _bandpass(vis_vel), mask)


# ---------------------------------------------------------------- the checks

def _frac(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def _bits(params: dict[str, float], name: str) -> int | None:
    return int(params[name]) if name in params else None


def check_arrival(topics: Topics, start_us: int, window) -> Check:
    vis = topics.get(VISION)
    if vis is None or len(vis.get("timestamp", [])) == 0:
        # PX4 v1.17 mavlink_receiver: estimator_type VISION, VIO (and UNKNOWN, NAIVE) go to
        # vehicle_visual_odometry, MOCAP to vehicle_mocap_odometry, the rest is dropped.
        if len(topics.get(MOCAP, {}).get("timestamp", [])):
            return Check("Vision reaches PX4", "fail",
                         "No vehicle_visual_odometry, but vehicle_mocap_odometry is there: the "
                         "ODOMETRY messages carry estimator_type MOCAP, which PX4 logs and "
                         "never fuses. Send MAV_ESTIMATOR_TYPE_VIO (or VISION).")
        return Check("Vision reaches PX4", "fail",
                     "No vehicle_visual_odometry in the log: PX4 never accepted a vision "
                     "message. Check the MAVLink link, and that ODOMETRY's estimator_type is "
                     "VIO or VISION (PX4 drops GPS, LIDAR and AUTOPILOT types with a critical "
                     "message), or that the logger profile records the topic.")
    t = np.unique(_t(vis, start_us))
    tw = t[_in(t, window)] if window else t
    if tw.size < 2:
        tw = t
    dt = np.diff(tw)
    rate = float((tw.size - 1) / (tw[-1] - tw[0])) if tw.size > 1 else 0.0
    gaps = dt[dt > GAP_WARN_S]
    nums = {"logged_rate_hz": rate, "samples": int(t.size),
            "gaps_over_0p5s": int(gaps.size), "longest_gap_s": float(dt.max()) if dt.size else 0}
    pos = np.stack([vis[f"position[{i}]"] for i in range(3)], 1) if "position[0]" in vis else None
    if pos is not None and not np.isfinite(pos).all(1).any():
        return Check("Vision reaches PX4", "fail",
                     "Vision messages arrive but every position is NaN, so EKF2 has nothing "
                     "to fuse. PX4 leaves the position empty when ODOMETRY's frame_id is not "
                     "MAV_FRAME_LOCAL_NED, LOCAL_ENU, LOCAL_FRD or LOCAL_FLU, or when the "
                     "position sent is NaN.", nums)
    where = "while airborne" if window else "over the log"
    note = (" The logger thins this topic, so the rate shown can be lower than what was "
            "sent; EKF2's own count is in the fusion check.")
    if gaps.size:
        return Check("Vision reaches PX4", "warn",
                     f"Vision arrives at about {rate:.0f} Hz {where}, but with {gaps.size} "
                     f"gap(s) over 0.5 s (longest {dt.max():.1f} s). EKF2 dead-reckons on the "
                     f"IMU through a gap and drifts.", nums)
    return Check("Vision reaches PX4", "pass",
                 f"Vision arrives at about {rate:.0f} Hz {where}, with no gap over 0.5 s."
                 + note, nums)


def check_fusion(topics: Topics, params: dict[str, float], start_us: int, window) -> Check:
    ev_ctrl = _bits(params, "EKF2_EV_CTRL")
    ctrl_txt = ""
    if ev_ctrl is not None:
        parts = [n for b, n in [(0, "horizontal position"), (1, "height"), (2, "velocity"),
                                (3, "yaw")] if ev_ctrl >> b & 1]
        ctrl_txt = (f" EKF2_EV_CTRL is {ev_ctrl} (" + (", ".join(parts) or "nothing") + ").")
    fl = _flags(topics, start_us)
    aid = topics.get(EV_POS)
    nums: dict[str, Any] = {"EKF2_EV_CTRL": ev_ctrl}
    if fl is not None:
        f, tf = fl
        m = _in(tf, window)
        for key in ("cs_ev_pos", "cs_ev_hgt", "cs_ev_vel", "cs_ev_yaw"):
            if key in f:
                nums[f"{key}_fraction"] = _frac(f[key][m])
    if aid is not None and "fused" in aid:
        ta = _t(aid, start_us)
        m = _in(ta, window)
        nums["ev_pos_samples_fused_fraction"] = _frac(np.asarray(aid["fused"])[m] > 0)
    qmin = params.get("EKF2_EV_QMIN")
    vis = topics.get(VISION)
    if qmin is not None and qmin > 0 and vis is not None and "quality" in vis:
        low = _frac(np.asarray(vis["quality"], float) < qmin)
        nums["quality_below_qmin_fraction"] = low
    active = nums.get("cs_ev_pos_fraction", nums.get("ev_pos_samples_fused_fraction"))
    if active is None:
        return Check("Vision fused", "skip",
                     "The log has neither EKF2's aiding flags nor its vision fusion record."
                     + ctrl_txt, nums)
    if math.isnan(active):
        return Check("Vision fused", "skip",
                     "EKF2's aiding record has no samples in the airborne window." + ctrl_txt,
                     nums)
    pct = 100 * active
    where = "of the airborne time" if window else "of the log"
    if active < FUSED_WARN:
        why = ""
        if ev_ctrl is not None and not ev_ctrl & 1:
            why = " EKF2_EV_CTRL does not ask for vision position, so EKF2 ignores it."
        elif nums.get("quality_below_qmin_fraction", 0) > 0.5:
            why = (f" Most samples carry a quality below EKF2_EV_QMIN ({qmin:g}), and EKF2 "
                   "drops those.")
        return Check("Vision fused", "fail",
                     f"EKF2 used the vision position for only {pct:.0f} % {where}.{why}"
                     + ctrl_txt, nums)
    extra = []
    for key, name in [("cs_ev_hgt", "height"), ("cs_ev_vel", "velocity"), ("cs_ev_yaw", "yaw")]:
        frac = nums.get(f"{key}_fraction")
        if frac is not None and frac > 0.5:
            extra.append(name)
    also = f" It also fuses vision {', '.join(extra)}." if extra else ""
    status = "pass" if active >= FUSED_PASS else "warn"
    tail = ("" if status == "pass"
            else " Something stopped it for the rest; see the rejection check.")
    return Check("Vision fused", status,
                 f"EKF2 used the vision position for {pct:.0f} % {where}.{also}{tail}"
                 + ctrl_txt, nums)


def check_rejection(topics: Topics, start_us: int, window) -> Check:
    aid = topics.get(EV_POS)
    if aid is None or "innovation_rejected" not in aid:
        return Check("Vision consistent with IMU", "skip",
                     "No estimator_aid_src_ev_pos in the log, so EKF2's accept/reject record "
                     "for the vision is not available.")
    ta = _t(aid, start_us)
    m = _in(ta, window)
    rej = np.asarray(aid["innovation_rejected"])[m] > 0
    ratio = np.stack([aid["test_ratio[0]"], aid["test_ratio[1]"]], 1).astype(float)[m]
    innov = np.stack([aid["innovation[0]"], aid["innovation[1]"]], 1).astype(float)[m]
    nums = {"rejected_fraction": _frac(rej),
            "test_ratio_p95": float(np.nanpercentile(ratio.max(1), 95)) if ratio.size else None,
            "innovation_rms_m": (float(np.sqrt(np.nanmean((innov ** 2).sum(1))))
                                 if innov.size else None)}
    r = nums["rejected_fraction"]
    if not math.isfinite(r):
        return Check("Vision consistent with IMU", "skip", "No vision samples in the window.",
                     nums)
    msg = (f"EKF2 rejected {100 * r:.1f} % of the vision positions as inconsistent with "
           f"the IMU (innovation RMS {nums['innovation_rms_m']:.2f} m).")
    if r >= REJECTED_FAIL:
        return Check("Vision consistent with IMU", "fail", msg + " That many rejections "
                     "usually means a wrong frame, a wrong delay or a noise setting far too "
                     "small (EKF2_EVP_NOISE, or the covariance the VIO sends).", nums)
    if r >= REJECTED_WARN:
        return Check("Vision consistent with IMU", "warn", msg + " A few per cent is a "
                     "jumpy VIO or a delay that is slightly off.", nums)
    return Check("Vision consistent with IMU", "pass", msg, nums)


def check_gps(topics: Topics, params: dict[str, float], start_us: int, window) -> Check:
    fl = _flags(topics, start_us)
    gps_ctrl = _bits(params, "EKF2_GPS_CTRL")
    nums: dict[str, Any] = {"EKF2_GPS_CTRL": gps_ctrl}
    if fl is None:
        return Check("GPS off", "skip", "No EKF2 aiding flags in the log.", nums)
    f, tf = fl
    m = _in(tf, window)
    used = np.zeros(int(m.sum()), bool)
    for k in ("cs_gnss_pos", "cs_gnss_vel", "cs_gps", "cs_gps_hgt"):
        if k in f:
            used |= f[k][m]
    nums["gnss_fraction"] = _frac(used)
    if used.any():
        return Check("GPS off", "warn",
                     f"EKF2 also fused GPS for {100 * _frac(used):.0f} % of the time, so this "
                     "log does not show GPS-free flight; with both on, GPS hides vision "
                     "mistakes. Set EKF2_GPS_CTRL to 0 to test the vision alone.", nums)
    return Check("GPS off", "pass", "EKF2 never fused GPS.", nums)


def check_frame(topics: Topics, start_us: int, window, al: Alignment | None,
                compass: dict[str, float] | None = None) -> Check:
    """``compass`` (``heading_vs_compass``, only when the compass is EKF2's heading source)
    adds back the turn EKF2 made to its own heading to fit the vision."""
    vis = topics.get(VISION, {})
    frame = int(np.asarray(vis["pose_frame"])[0]) if "pose_frame" in vis and len(
        vis["pose_frame"]) else None
    nums: dict[str, Any] = {"pose_frame": frame}
    name = POSE_FRAMES.get(frame, f"code {frame}") if frame is not None else "not logged"
    if al is None:
        return Check("Vision frame", "skip", f"Vision frame is {name}. The log lacks the IMU "
                     "or attitude topics needed to check the axes.", nums)
    nums.update({"yaw_offset_deg": al.yaw_offset_deg, "mirrored": al.mirrored,
                 "vertical_correlation": al.z_corr, "level_fit": al.xy_score,
                 "xy_motion_mps": al.xy_motion_mps, "z_motion_mps": al.z_motion_mps})
    problems: list[tuple[str, str]] = []  # (status, what is wrong)
    fine: list[str] = []
    if al.z_corr is not None and abs(al.z_corr) >= MIN_CORRELATION:
        if al.z_corr < 0:
            problems.append(("fail", "vision height moves the wrong way (up is down): the "
                             "stream is probably ENU or another z-up frame sent as NED"))
        else:
            fine.append("height moves the right way")
    if al.yaw_offset_deg is not None and (al.xy_score or 0.0) >= MIN_CORRELATION:
        yaw, ref = al.yaw_offset_deg, "EKF2's north"
        if compass is not None:
            ref = "the compass's north"
            if abs(compass["drift_deg"]) >= YAW_WARN_DEG:
                ref += (f" (EKF2 turned its own heading {compass['drift_deg']:+.0f} deg away "
                        "from the compass to make the vision fit)")
        nums["heading_reference"] = "compass" if compass is not None else "EKF2"
        off = abs(yaw)
        turn = f"turned {yaw:+.0f} deg (clockwise seen from above) from {ref}"
        if al.mirrored:
            problems.append(("fail", "the horizontal axes are mirrored (x and y swapped or one "
                             "flipped), the classic ENU-sent-as-NED mistake; send ENU as "
                             "MAV_FRAME_LOCAL_ENU and PX4 converts it, or convert ENU (x east, "
                             "y north, z up) to NED as (y, x, -z)"))
        elif frame == 1 and off >= YAW_WARN_DEG:
            problems.append(("fail" if off >= YAW_FAIL_DEG else "warn",
                             f"the vision frame is {turn}, while NED tells EKF2 "
                             "its x axis is north; either align the VIO's frame to north "
                             "or send it as MAV_FRAME_LOCAL_FRD with an attitude so EKF2 can "
                             "rotate it"))
        elif frame == 1:
            fine.append(f"its x axis points north within {off:.0f} deg"
                        + (" of the compass" if compass is not None else ""))
        else:
            fine.append(f"its x axis is {turn} (fine for an FRD frame, which EKF2 rotates)")
    if problems:
        status = "fail" if any(st == "fail" for st, _ in problems) else "warn"
        return Check("Vision frame", status, f"Declared {name}. Problem: "
                     + "; and ".join(txt for _, txt in problems) + ".", nums)
    if not fine:
        return Check("Vision frame", "skip",
                     f"Declared {name}. Too little motion in the log to check the axes "
                     f"(filtered speed {al.xy_motion_mps:.2f} m/s level, {al.z_motion_mps:.2f} "
                     "m/s vertical); fly a climb and a few metres side to side.", nums)
    if al.yaw_offset_deg is None:
        fine.append("not enough sideways motion to check the horizontal axes")
    return Check("Vision frame", "pass", f"Declared {name}; checked against the IMU: "
                 + ", ".join(fine) + ".", nums)


def check_heading(topics: Topics, params: dict[str, float], start_us: int, window,
                  compass: dict[str, float] | None = None) -> Check:
    fl = _flags(topics, start_us)
    nums: dict[str, Any] = {"EKF2_MAG_TYPE": params.get("EKF2_MAG_TYPE")}
    if compass is not None:
        nums.update({f"ekf_vs_compass_{k}": v for k, v in compass.items()})
    src = "unknown"
    if fl is not None:
        f, tf = fl
        m = _in(tf, window)
        ev_yaw = _frac(f["cs_ev_yaw"][m]) if "cs_ev_yaw" in f else 0.0
        mag = max(_frac(f[k][m]) if k in f else 0.0 for k in ("cs_mag_hdg", "cs_mag_3d"))
        nums.update({"ev_yaw_fraction": ev_yaw, "mag_fraction": mag})
        src = "vision" if ev_yaw > 0.5 else ("compass" if mag > 0.5 else "none")
    vis = topics.get(VISION, {})
    yaw_msg = ""
    status = "pass"
    if "q[0]" in vis and ATTITUDE in topics:
        qv = np.stack([vis[f"q[{i}]"] for i in range(4)], 1).astype(float)
        ok = np.isfinite(qv).all(1)
        if ok.any():
            tv = _t(vis, start_us, "timestamp_sample" if "timestamp_sample" in vis
                    else "timestamp")[ok]
            att = topics[ATTITUDE]
            ta = _t(att, start_us)
            ya = np.unwrap(_yaw(np.stack([att[f"q[{i}]"] for i in range(4)], 1).astype(float)))
            diff = _wrap(_yaw(qv[ok]) - np.interp(tv, ta, ya))
            mw = _in(tv, window)
            diff = diff[mw] if mw.any() else diff
            mean = math.degrees(math.atan2(np.mean(np.sin(diff)), np.mean(np.cos(diff))))
            nums["vision_minus_ekf_yaw_deg"] = mean
            yaw_msg = f" The vision's yaw differs from EKF2's by {mean:+.1f} deg on average."
            if abs(mean) >= YAW_WARN_DEG and vis.get("pose_frame") is not None and int(
                    np.asarray(vis["pose_frame"])[0]) == 1:
                status = "warn" if abs(mean) < YAW_FAIL_DEG else "fail"
                yaw_msg += (" In a NED frame those should agree; a VIO frame that is not "
                            "aligned to north does this.")
        else:
            yaw_msg = " The vision sends no attitude."
    if src == "none":
        return Check("Heading", "fail", "EKF2 had no heading source (neither vision "
                     "yaw nor the compass) for most of the flight, so its heading drifts and "
                     "with it every position." + yaw_msg, nums)
    if src == "compass" and compass is not None and abs(compass["drift_deg"]) >= YAW_WARN_DEG:
        return Check("Heading", "fail" if abs(compass["drift_deg"]) >= YAW_FAIL_DEG
                     else "warn",
                     f"Heading should come from the compass, but once airborne EKF2's heading "
                     f"moved {compass['drift_deg']:+.0f} deg away from it. Either the vision "
                     "frame is turned from north and EKF2 turned its heading to fit it (see "
                     "the frame check), or the compass is disturbed in flight (motor current, "
                     "steel nearby)." + yaw_msg, nums)
    if src == "compass":
        return Check("Heading", status, "Heading comes from the compass. Near steel "
                     "structures and indoors the compass can be off by tens of degrees, and a "
                     "heading error turns vision position into the wrong direction; if the "
                     "VIO's yaw is good, fusing it (EKF2_EV_CTRL bit 3) avoids that."
                     + yaw_msg, nums)
    if src == "vision":
        return Check("Heading", status, "Heading comes from the vision's yaw."
                     + yaw_msg, nums)
    return Check("Heading", "skip", "No EKF2 aiding flags in the log." + yaw_msg, nums)


def check_delay(topics: Topics, params: dict[str, float], start_us: int, al: Alignment | None,
                truth: Alignment | None, frame_ok: bool = True) -> Check:
    """``frame_ok`` False (the frame check failed): the number is reported but not judged,
    because EKF2 fusing a wrong frame bends its attitude, which the IMU method leans on (on
    the Gazebo ENU-as-NED log it read 144 ms against 88 ms from truth)."""
    ev_delay = params.get("EKF2_EV_DELAY")
    vis = topics.get(VISION, {})
    nums: dict[str, Any] = {"EKF2_EV_DELAY_ms": ev_delay}
    stamp_txt = ""
    if "timestamp_sample" in vis and len(vis["timestamp_sample"]):
        d = (np.asarray(vis["timestamp"], np.int64)
             - np.asarray(vis["timestamp_sample"], np.int64)) * 1e-6
        nums["stamp_minus_sample_mean_s"] = float(np.mean(d))
        if np.all(np.abs(d) < 1e-3):
            stamp_txt = (" PX4 stamps the samples on arrival (no clock sync with the VIO), so "
                         "EKF2_EV_DELAY has to carry the whole latency.")
    if truth is not None and truth.lag_s is not None:
        nums["delay_vs_truth_s"] = truth.lag_s
    if al is None or al.lag_s is None or al.score < MIN_CORRELATION:
        motion = "" if al is None else (f" (filtered speed {al.z_motion_mps:.2f} m/s vertical, "
                                        f"{al.xy_motion_mps:.2f} m/s level)")
        return Check("Vision delay", "skip", "Not enough motion in the log to measure the "
                     f"delay{motion}; a climb of a metre or two is enough." + stamp_txt, nums)
    nums.update({"measured_delay_s": al.lag_s, "alignment_score": al.score,
                 "measured_on": al.delay_from})
    meas_ms = 1000 * al.lag_s
    if not frame_ok:
        return Check("Vision delay", "skip", f"Measured about {meas_ms:.0f} ms, but not judged: "
                     "with the frame wrong EKF2's attitude is bent too, and this measurement "
                     "leans on it. Fix the frame and fly again." + stamp_txt, nums)
    check_txt = ""
    if "delay_vs_truth_s" in nums:
        check_txt = (f" Against simulator truth it is {1000 * nums['delay_vs_truth_s']:.0f} "
                     "ms.")
    if ev_delay is None:
        return Check("Vision delay", "warn", f"The vision is about {meas_ms:.0f} ms old when "
                     "PX4 stamps it, but EKF2_EV_DELAY is not in the log to compare."
                     + check_txt + stamp_txt, nums)
    err = al.lag_s - ev_delay / 1000
    nums["delay_error_s"] = err
    base = (f"The vision is about {meas_ms:.0f} ms old when PX4 stamps it; EKF2_EV_DELAY is "
            f"{ev_delay:.0f} ms.")
    if abs(err) >= DELAY_WARN_S:
        fix = (f" Set EKF2_EV_DELAY to about {max(0.0, meas_ms):.0f} ms; a mismatch makes EKF2 "
               "fuse each sample at the wrong moment, which shows as lag and wobble when the "
               "drone moves.")
        status = "fail" if abs(err) >= DELAY_FAIL_S else "warn"
        return Check("Vision delay", status, base + fix + check_txt + stamp_txt, nums)
    return Check("Vision delay", "pass", base + " They match." + check_txt + stamp_txt, nums)


def check_height(topics: Topics, params: dict[str, float], start_us: int, window) -> Check:
    ref = _bits(params, "EKF2_HGT_REF")
    nums: dict[str, Any] = {"EKF2_HGT_REF": ref}
    name = HGT_REF.get(ref, f"code {ref}") if ref is not None else None
    # What EKF2 actually fused for height, whatever the parameter asked for.
    fused = []
    fl = _flags(topics, start_us)
    if fl is not None:
        f, tf = fl
        m = _in(tf, window)
        for key, src in [("cs_baro_hgt", "barometer"), ("cs_ev_hgt", "vision"),
                         ("cs_gps_hgt", "GPS"), ("cs_rng_hgt", "rangefinder")]:
            if key in f:
                nums[f"{key}_fraction"] = _frac(f[key][m])
                if nums[f"{key}_fraction"] > 0.5:
                    fused.append(src)
    used = f" EKF2 fused height from: {', '.join(fused)}." if fused else ""
    if name is None:
        return Check("Height reference", "skip", "EKF2_HGT_REF is not in the log." + used, nums)
    if ref == 3:
        return Check("Height reference", "warn", "Height reference is the vision. When the VIO "
                     "stops, EKF2's height stops with it and the drone can sink without "
                     "noticing (docs/field-faults.md); the barometer as the reference with "
                     "vision height fused as an aid holds height through a VIO loss." + used,
                     nums)
    if ref == 1:
        # PX4 v1.17 falls back GPS -> barometer -> vision -> rangefinder when the reference
        # is not running (Ekf::checkHeightSensorRefFallback), so this flies on the barometer
        # until a GPS fix appears; then GPS becomes the height reference.
        if "GPS" in fused:
            what = ("On this flight GPS height was fused, so GPS was the height reference. "
                    "Set 0 for the barometer.")
        else:
            what = ("With no GPS EKF2 falls back to the barometer, so this flight was on the "
                    "barometer, but the moment a GPS fix appears GPS becomes the height "
                    "reference. Set 0 for the barometer.")
        return Check("Height reference", "warn", "EKF2_HGT_REF is 1, which is GPS (0 is the "
                     "barometer). " + what + used, nums)
    return Check("Height reference", "pass", f"Height reference is the {name}." + used, nums)


def check_topics(topics: Topics, params: dict[str, float], start_us: int,
                 ulog: str = "<topics>") -> Report:
    window = airborne_window(topics, start_us)
    al = vision_alignment(topics, start_us, window, "imu")
    # With the compass as EKF2's heading source, the IMU is turned with the compass heading
    # instead (compass_yaw), so the frame is judged against magnetic north, not against a
    # heading EKF2 may have bent to fit the vision.
    fl = _flags(topics, start_us)
    compass = None
    if fl is not None and ("cs_mag_hdg" in fl[0] or "cs_mag_3d" in fl[0]):
        f, tf = fl
        m = _in(tf, window)
        mag = np.zeros(int(m.sum()), bool)
        for k in ("cs_mag_hdg", "cs_mag_3d"):
            if k in f:
                mag |= f[k][m]
        ev_yaw = f["cs_ev_yaw"][m] if "cs_ev_yaw" in f else np.zeros(int(m.sum()), bool)
        if _frac(mag) > 0.5 and not _frac(ev_yaw) > 0.5:
            compass = heading_vs_compass(topics, start_us, window)
    cy = compass_yaw(topics, start_us, window) if compass is not None else None
    if cy is not None:
        al = vision_alignment(topics, start_us, window, "imu", cy)
    truth = vision_alignment(topics, start_us, window, "truth") if TRUTH in topics else None
    checks = [check_arrival(topics, start_us, window)]
    if checks[0].status != "fail":
        checks += [
            check_fusion(topics, params, start_us, window),
            check_rejection(topics, start_us, window),
            check_gps(topics, params, start_us, window),
            check_frame(topics, start_us, window, al, compass),
            check_heading(topics, params, start_us, window, compass),
        ]
        checks += [
            check_delay(topics, params, start_us, al, truth,
                        frame_ok=checks[-2].status != "fail"),
            check_height(topics, params, start_us, window),
        ]
    else:
        checks.append(check_gps(topics, params, start_us, window))
    bad = [c.name for c in checks if c.status == "fail"]
    warn = [c.name for c in checks if c.status == "warn"]
    if bad:
        verdict = ("PROBLEM: the external position source is not being used correctly. "
                   "Fix first: " + ", ".join(bad) + ".")
    elif warn:
        verdict = ("MOSTLY OK: EKF2 is flying on the external position source, with things to "
                   "look at: " + ", ".join(warn) + ".")
    else:
        verdict = "OK: EKF2 is flying on the external position source, in the right frame."
    if window is None:
        verdict += " (No airborne time found, so the whole log was used.)"
    return Report(ulog, verdict, checks)


def check_log(path: str | Path) -> Report:
    topics, params, start_us = load(path)
    return check_topics(topics, params, start_us, str(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ulog", type=Path, help="PX4 flight log (.ulg)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON here")
    args = ap.parse_args(argv)
    report = check_log(args.ulog)
    print(report.text())
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2, default=_jsonable) + "\n")
    return 1 if report.worst() == "fail" else 0


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x)}")


if __name__ == "__main__":
    raise SystemExit(main())
