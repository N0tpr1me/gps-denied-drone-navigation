"""Read a PX4 .ulg log into an ENU :class:`eval.metrics.Trace`.

PX4 logs in NED (world) and FRD (body); this module is the PX4 boundary for logs, so the
conversion to ENU / FLU happens here and nowhere else in ``eval/``.

Topics used:

- position: ``vehicle_local_position_groundtruth`` (logged by SITL) or, for real flights
  with motion capture, whatever the caller maps in. The EKF's ``vehicle_local_position``
  is used only with ``allow_estimate=True``, because scoring a corrupted-estimate run
  against the estimate hides the error being measured.
- setpoint: ``vehicle_local_position_setpoint`` (x, y, z), held between samples, unless the
  caller gives a fixed ``hold_point``. Samples before the first finite setpoint, or where it
  is NaN (rate-only modes), are dropped.
- commands: ``vehicle_rates_setpoint`` (roll, pitch, yaw rates and ``thrust_body``).

Times are seconds since the log's start timestamp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.metrics import Trace

GROUNDTRUTH = "vehicle_local_position_groundtruth"
ESTIMATE = "vehicle_local_position"
SETPOINT = "vehicle_local_position_setpoint"
RATES = "vehicle_rates_setpoint"
TOPICS = [GROUNDTRUTH, ESTIMATE, SETPOINT, RATES]

Topics = dict[str, dict[str, np.ndarray]]


def ned_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """NED (north, east, down) to ENU (east, north, up), stacked as (N, 3)."""
    return np.stack([y, x, -z], axis=1).astype(float)


def frd_to_flu(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """FRD (forward, right, down) to FLU (forward, left, up), stacked as (N, 3)."""
    return np.stack([x, -y, -z], axis=1).astype(float)


def trace_from_topics(
    topics: Topics,
    start_us: int,
    hold_point: np.ndarray | None = None,
    allow_estimate: bool = False,
) -> Trace:
    """Build a Trace from PX4 topic arrays (field name -> array, as pyulog gives them)."""
    if GROUNDTRUTH in topics:
        source, pos_msg = GROUNDTRUTH, topics[GROUNDTRUTH]
    elif allow_estimate and ESTIMATE in topics:
        source, pos_msg = ESTIMATE, topics[ESTIMATE]
    else:
        raise ValueError(
            f"log has no {GROUNDTRUTH}; pass allow_estimate=True to score against the EKF "
            "estimate (only meaningful when the estimate is not corrupted)"
        )

    def seconds(msg: dict[str, np.ndarray]) -> np.ndarray:
        return (np.asarray(msg["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6

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
                 meta={"position_source": source})


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
    path: str | Path, hold_point: np.ndarray | None = None, allow_estimate: bool = False
) -> Trace:
    topics, start_us = load_topics(path)
    trace = trace_from_topics(topics, start_us, hold_point, allow_estimate)
    trace.meta["ulog"] = str(path)
    return trace
