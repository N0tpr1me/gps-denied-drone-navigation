"""Read a PX4 .ulg log into an ENU :class:`eval.metrics.Trace`.

PX4 logs in NED (world) and FRD (body); this module is the PX4 boundary for logs, so the
conversion to ENU / FLU happens here (through ``envs.frames``) and nowhere else in ``eval/``.

Topics used:

- position: ``vehicle_local_position_groundtruth`` (logged by SITL), else, for real flights,
  ``vehicle_mocap_odometry`` (motion capture over MAVLink ``ODOMETRY`` with frame MOCAP, or
  ``ATT_POS_MOCAP``; logged, never fused). Outdoors the caller hands in RTK truth instead
  (``truth``, from ``eval.rtk_truth``), which like mocap is in a frame of its own. The
  EKF's ``vehicle_local_position`` is used only with ``allow_estimate=True``, because
  scoring a corrupted-estimate run against the estimate hides the error being measured.
- setpoint: ``vehicle_local_position_setpoint`` (x, y, z), held between samples, unless the
  caller gives a fixed ``hold_point``. Samples before the first finite setpoint, or where it
  is NaN (rate-only modes), are dropped.

Mocap truth is in the mocap system's own frame, not in the EKF's local frame that the
setpoint is in: the origins differ and so may the headings. So a mocap trace needs one of

- ``hold_point``: the hold point in the mocap frame (ENU after this module's conversion),
  for example the point measured 1 m off the rig's wall; or
- ``align_window``: a stretch of the log (seconds) over which the EKF estimate is fitted to
  mocap by a yaw rotation and an offset, which then carries the setpoint into the mocap
  frame. Use the take-off and the move to the hold point, before engaging, so the estimate
  error during the hold still counts. The window needs at least ``MIN_ALIGN_SPREAD_M`` of
  horizontal motion, or the yaw is a guess.

The wall for ``eval.metrics`` is then given in the mocap frame as well.
- commands: ``vehicle_rates_setpoint`` (roll, pitch, yaw rates and ``thrust_body``).

Gazebo truth is in the EKF's local frame horizontally, but not always vertically: on
barometer height EKF2 sets its origin from the first barometer sample, so its height can
read a few decimetres on the pad before the motors start, and the setpoint lives in that
frame. ``gazebo_reference`` is how Gazebo holds are scored since 2026-11-10
(docs/decisions.md): that one offset, measured on the ground, comes out of the setpoint, and
the hold is scored against the setpoint at the start of the scored window, one fixed point,
whatever the autopilot does to its setpoint afterwards. A partner's flight gets the same
two things from ``align_window`` and ``eval/partner_session.py``.

Times are seconds since the log's start timestamp.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from envs import frames
from eval.metrics import Trace, compute_metrics

GROUNDTRUTH = "vehicle_local_position_groundtruth"
MOCAP = "vehicle_mocap_odometry"
ESTIMATE = "vehicle_local_position"
SETPOINT = "vehicle_local_position_setpoint"
RATES = "vehicle_rates_setpoint"
TOPICS = [GROUNDTRUTH, MOCAP, ESTIMATE, SETPOINT, RATES]

# VehicleOdometry.pose_frame: 1 is NED, 2 is FRD (z down, arbitrary heading). Both are
# right-handed z-down world frames, so the NED to ENU flip gives a z-up frame either way.
POSE_FRAMES_Z_DOWN = (1, 2)
MIN_ALIGN_SPREAD_M = 0.5  # horizontal RMS spread about the mean, over the align window
LIFT_M = 0.2  # truth height above its first sample that counts as off the ground
GROUND_SPAN_S = (3.0, 0.5)  # the ground offset is the mean from 3 s to 0.5 s before lift-off

Topics = dict[str, dict[str, np.ndarray]]


def ned_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """NED (north, east, down) to ENU (east, north, up), stacked as (N, 3)."""
    return frames.ned_to_enu(np.stack([x, y, z], axis=1))


def frd_to_flu(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """FRD (forward, right, down) to FLU (forward, left, up), stacked as (N, 3)."""
    return frames.frd_to_flu(np.stack([x, y, z], axis=1))


def fit_yaw_offset(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Yaw ``psi`` and offset ``o`` with ``dst ~ Rz(psi) src + o``, by least squares (N, 3)."""
    sc, dc = src - src.mean(axis=0), dst - dst.mean(axis=0)
    psi = math.atan2(float(np.sum(sc[:, 0] * dc[:, 1] - sc[:, 1] * dc[:, 0])),
                     float(np.sum(sc[:, 0] * dc[:, 0] + sc[:, 1] * dc[:, 1])))
    offset = dst.mean(axis=0) - rotate_z(src.mean(axis=0)[None], psi)[0]
    return psi, offset


def rotate_z(v: np.ndarray, psi: float) -> np.ndarray:
    """Rotate (N, 3) vectors by ``psi`` about +z (counter-clockwise from above)."""
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)


def trace_from_topics(
    topics: Topics,
    start_us: int,
    hold_point: np.ndarray | None = None,
    allow_estimate: bool = False,
    align_window: tuple[float, float] | None = None,
    truth: tuple[str, np.ndarray, np.ndarray] | None = None,
) -> Trace:
    """Build a Trace from PX4 topic arrays (field name -> array, as pyulog gives them).

    ``truth`` is a truth track from outside this module: its name, log seconds (N,) and ENU
    positions (N, 3) in a frame of its own, as mocap's is (RTK, ``eval.rtk_truth``). It is
    used whatever else the log holds."""
    if truth is not None:
        source, pos_msg = truth[0], {}
    elif GROUNDTRUTH in topics:
        source, pos_msg = GROUNDTRUTH, topics[GROUNDTRUTH]
    elif MOCAP in topics:
        source, pos_msg = MOCAP, topics[MOCAP]
    elif allow_estimate and ESTIMATE in topics:
        source, pos_msg = ESTIMATE, topics[ESTIMATE]
    else:
        raise ValueError(
            f"log has no {GROUNDTRUTH} or {MOCAP}; pass allow_estimate=True to score against "
            "the EKF estimate (only meaningful when the estimate is not corrupted)"
        )

    def seconds(msg: dict[str, np.ndarray], key: str = "timestamp") -> np.ndarray:
        return (np.asarray(msg[key], dtype=np.int64) - int(start_us)) * 1e-6

    meta: dict = {"position_source": source}
    own_frame = truth is not None or source == MOCAP  # not the frame the setpoint is in
    if own_frame and hold_point is None and align_window is None:
        name = "mocap" if source == MOCAP else source
        raise ValueError(
            f"{name} truth is in the {name} frame, not the setpoint's: pass hold_point (in the "
            f"{name} frame) or align_window (seconds, to fit the estimate to {name})"
        )
    if truth is not None:
        t, pos = np.asarray(truth[1], dtype=float), np.asarray(truth[2], dtype=float)
    elif source == MOCAP:
        frames = np.unique(np.asarray(pos_msg.get("pose_frame", [1]), dtype=int))
        if not set(frames.tolist()) <= set(POSE_FRAMES_Z_DOWN):
            raise ValueError(f"{MOCAP} pose_frame {frames.tolist()} is not NED (1) or FRD (2)")
        # timestamp_sample is when mocap saw the drone, on PX4's clock after timesync;
        # timestamp is when PX4 published it, one link delay later.
        stamp = "timestamp_sample" if np.all(np.asarray(
            pos_msg.get("timestamp_sample", [0])) > 0) else "timestamp"
        t = seconds(pos_msg, stamp)
        pos = ned_to_enu(pos_msg["position[0]"], pos_msg["position[1]"], pos_msg["position[2]"])
    else:
        t = seconds(pos_msg)
        pos = ned_to_enu(pos_msg["x"], pos_msg["y"], pos_msg["z"])
    # Duplicate or out-of-order stamps break interpolation; keep the first of each.
    t, first = np.unique(t, return_index=True)
    pos = pos[first]

    if hold_point is not None:
        sp = np.broadcast_to(np.asarray(hold_point, dtype=float), pos.shape).copy()
    elif SETPOINT in topics:
        sp_msg = topics[SETPOINT]
        sp_t = seconds(sp_msg)
        sp_all = ned_to_enu(sp_msg["x"], sp_msg["y"], sp_msg["z"])
        idx = np.searchsorted(sp_t, t, side="right") - 1
        sp = np.full(pos.shape, np.nan)
        sp[idx >= 0] = sp_all[idx[idx >= 0]]
        if own_frame:
            psi, offset, rms = align_estimate(topics, t, pos, seconds, align_window)
            sp = rotate_z(sp, psi) + offset
            meta.update(align_window_s=[float(v) for v in align_window],
                        align_yaw_deg=math.degrees(psi), align_offset_m=offset.tolist(),
                        align_rms_m=rms)
    else:
        raise ValueError(f"log has no {SETPOINT}; pass hold_point")

    keep = np.all(np.isfinite(sp), axis=1) & np.all(np.isfinite(pos), axis=1)
    if keep.sum() < 2:
        raise ValueError("fewer than two samples with both a position and a finite setpoint")
    t, pos, sp = t[keep], pos[keep], sp[keep]

    cmd_t = rates = thrust = None
    if RATES in topics:
        r = topics[RATES]
        cmd_t = seconds(r)
        rates = frd_to_flu(r["roll"], r["pitch"], r["yaw"])
        # thrust_body is normalised, in FRD, so upward collective thrust is -z.
        thrust = -np.asarray(r["thrust_body[2]"], dtype=float)

    return Trace(t=t, pos=pos, pos_sp=sp, cmd_t=cmd_t, rates_cmd=rates, thrust_cmd=thrust,
                 meta=meta)


def align_samples(topics: Topics, t: np.ndarray, pos: np.ndarray, seconds,
                  window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """The EKF estimate at the truth's sample times inside ``window`` (N, 3), the truth
    there (N, 3), and how far the estimate moves horizontally (RMS about its mean, m)."""
    if ESTIMATE not in topics:
        raise ValueError(f"align_window needs {ESTIMATE} in the log")
    est = topics[ESTIMATE]
    et, first = np.unique(seconds(est), return_index=True)
    ep = ned_to_enu(est["x"], est["y"], est["z"])[first]
    inside = (t >= window[0]) & (t <= window[1]) & (t >= et[0]) & (t <= et[-1])
    if inside.sum() < 10:
        raise ValueError(f"align window {window} holds {inside.sum()} truth samples; need 10")
    src = np.stack([np.interp(t[inside], et, ep[:, k]) for k in range(3)], axis=1)
    spread = float(np.sqrt(np.mean(np.sum((src[:, :2] - src[:, :2].mean(axis=0)) ** 2, axis=1))))
    return src, pos[inside], spread


def fit_residual(src: np.ndarray, dst: np.ndarray, psi: float, offset: np.ndarray) -> float:
    resid = dst - (rotate_z(src, psi) + offset)
    return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def align_estimate(topics: Topics, t: np.ndarray, pos: np.ndarray, seconds,
                   window: tuple[float, float]) -> tuple[float, np.ndarray, float]:
    """Fit the EKF estimate to the truth (``t``, ``pos``) over ``window``: yaw, offset and
    the residual RMS."""
    src, dst, spread = align_samples(topics, t, pos, seconds, window)
    if spread < MIN_ALIGN_SPREAD_M:
        raise ValueError(f"align window moves {spread:.2f} m horizontally (RMS); the yaw fit "
                         f"needs {MIN_ALIGN_SPREAD_M} m, so widen it or pass hold_point")
    psi, offset = fit_yaw_offset(src, dst)
    return psi, offset, fit_residual(src, dst, psi, offset)


def load_topics(path: str | Path) -> tuple[Topics, int]:
    """Read the topics this module needs (first instance of each) with pyulog."""
    try:
        from pyulog import ULog
    except ImportError as e:  # optional dependency: pip install -e .[ulog]
        raise ImportError("reading .ulg needs pyulog: pip install -e .[ulog]") from e
    log = ULog(str(path), message_name_filter_list=TOPICS)
    topics: Topics = {}
    for d in log.data_list:
        if d.multi_id == 0:
            topics[d.name] = d.data
    return topics, int(log.start_timestamp)


def read_ulog(
    path: str | Path,
    hold_point: np.ndarray | None = None,
    allow_estimate: bool = False,
    align_window: tuple[float, float] | None = None,
) -> Trace:
    topics, start_us = load_topics(path)
    trace = trace_from_topics(topics, start_us, hold_point, allow_estimate, align_window)
    trace.meta["ulog"] = str(path)
    return trace


def ground_offset(t: np.ndarray, up_truth: np.ndarray, up_est: np.ndarray,
                  lift_m: float = LIFT_M, span_s: tuple[float, float] = GROUND_SPAN_S) -> float:
    """Estimated height minus true height (m, up positive), averaged over the stretch just
    before lift-off. NaN when the flight never lifts or the stretch is not in the log."""
    lifted = np.flatnonzero(up_truth - up_truth[0] > lift_m)
    if lifted.size == 0:
        return math.nan
    t_lift = float(t[lifted[0]])
    m = (t >= t_lift - span_s[0]) & (t <= t_lift - span_s[1]) & np.isfinite(up_est)
    return float(np.mean(up_est[m] - up_truth[m])) if m.any() else math.nan


def heights(topics: Topics, start_us: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log seconds, Gazebo's true height and EKF2's height at the same instants (m, up)."""

    def seconds(msg: dict[str, np.ndarray]) -> np.ndarray:
        return (np.asarray(msg["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6

    truth, est = topics[GROUNDTRUTH], topics[ESTIMATE]
    t = seconds(truth)
    order = np.argsort(t)
    t = t[order]
    up_truth = ned_to_enu(truth["x"], truth["y"], truth["z"])[order, 2]
    te = seconds(est)
    eo = np.argsort(te)
    up_est = np.interp(t, te[eo], ned_to_enu(est["x"], est["y"], est["z"])[eo, 2])
    return t, up_truth, up_est


def gazebo_reference(trace: Trace, topics: Topics, start_us: int,
                     window: tuple[float, float] | None, fixed: bool = True) -> Trace:
    """``trace`` again with the setpoint a Gazebo hold is scored against.

    The height EKF2 read on the ground before lift-off, minus Gazebo's, comes out of the
    setpoint (it is where the estimator's frame starts, there with the motors off, not hold
    error; whatever the estimate does after lift-off stays in). With ``fixed`` the setpoint
    is the one in force at the start of ``window``, for the whole trace: stock PX4 moves
    its setpoint when EKF2 resets, and a drone that then sits on the moved setpoint has
    still left the place it was holding. A surface pass keeps its moving reference
    (``fixed=False``): there the reference moves because it was told to.

    ``meta`` gains ``ground_offset_m`` (None when the log has no stretch on the ground, and
    then nothing is taken out) and ``setpoint_moved_m``, how far the logged setpoint
    travelled from its first value inside the window."""
    offset = math.nan
    if GROUNDTRUTH in topics and ESTIMATE in topics:
        offset = ground_offset(*heights(topics, start_us))
    sp = trace.pos_sp.copy()
    moved = None
    if window is not None:
        # The setpoint in force at the window's start, held as eval.metrics.resample holds it.
        t0 = max(float(window[0]), float(trace.t[0]))
        first = sp[max(int(np.searchsorted(trace.t, t0, side="right")) - 1, 0)].copy()
        inside = (trace.t >= t0) & (trace.t <= window[1])
        if inside.any():
            moved = float(np.max(np.linalg.norm(sp[inside] - first, axis=1)))
        if fixed:
            sp[:] = first
    if math.isfinite(offset):
        sp[:, 2] -= offset
    meta = {**trace.meta, "ground_offset_m": offset if math.isfinite(offset) else None,
            "setpoint_moved_m": moved, "setpoint": "fixed" if fixed else "moving"}
    return dataclasses.replace(trace, pos_sp=sp, meta=meta)


def reference_numbers(scored: Trace, logged: Trace,
                      window: tuple[float, float] | None) -> dict[str, float | None]:
    """What ``gazebo_reference`` did, for a flight's metrics: the height offset it took out,
    how far the logged setpoint moved, and ``p95_logged_setpoint_m``, the p95 against the
    setpoint exactly as logged (the number every Gazebo table before 2026-11-10 carries)."""
    return {"ground_offset_m": scored.meta.get("ground_offset_m"),
            "setpoint_moved_m": scored.meta.get("setpoint_moved_m"),
            "p95_logged_setpoint_m": compute_metrics(logged, window=window)["p95_m"]}
