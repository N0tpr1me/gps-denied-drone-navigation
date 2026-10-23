"""Export what the Structure Hold module saw in a flight, tick by tick, from its ulog.

    python -m sim2real.px4.tools.export_traces --label structure_hold_v6 \\
        runs/20261023-00*_sitl_hold_* --out runs/gazebo-traces

Writes two CSVs per run folder (one that holds ``flight.ulg``) into ``--out``:

- ``<label>_<scenario>_<profile>_seed<N>_ticks.csv``: one row per module tick over the
  first engaged hold (``structure_hold_status.state == ENGAGED``), with every other topic
  sampled as the module would have read it at that tick (last sample at or before the
  tick): true position and velocity (``vehicle_local_position_groundtruth``), the EKF's
  ``vehicle_local_position`` with its valid flags, ``vehicle_attitude`` q,
  ``vehicle_angular_velocity``, ``hover_thrust_estimate``, and the module's own status
  (setpoint, yaw setpoint, position error, stale and age, action, rates and thrust
  setpoints, fallback and yaw-guard flags).
- ``<label>_<scenario>_<profile>_seed<N>_odometry.csv``: every ``vehicle_visual_odometry``
  sample from 1 s before to 1 s after the hold (timestamp, timestamp_sample, position,
  velocity, q, reset counter).

Everything is as logged: NED world, FRD body, microsecond stamps since boot. ``t`` is
seconds since the first engaged tick. Columns that are not in the log are left out and
listed in ``<name>_columns.json`` next to the CSVs, so the reader knows what is missing
rather than guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

STATUS = "structure_hold_status"
STATE_ENGAGED = 2

# topic -> (column prefix, fields as pyulog names them)
TOPICS: dict[str, tuple[str, list[str]]] = {
    "vehicle_local_position_groundtruth": (
        "truth", ["x", "y", "z", "vx", "vy", "vz"]),
    "vehicle_local_position": (
        "est", ["x", "y", "z", "vx", "vy", "vz", "xy_valid", "z_valid", "v_xy_valid",
                "v_z_valid", "heading", "delta_xy[0]", "delta_xy[1]", "delta_z",
                "xy_reset_counter", "z_reset_counter"]),
    "vehicle_attitude": ("att", ["q[0]", "q[1]", "q[2]", "q[3]"]),
    "vehicle_angular_velocity": ("gyro", ["xyz[0]", "xyz[1]", "xyz[2]"]),
    "hover_thrust_estimate": ("hover", ["hover_thrust", "hover_thrust_var", "valid"]),
    "vehicle_rates_setpoint": (
        "rates_sp", ["roll", "pitch", "yaw", "thrust_body[0]", "thrust_body[1]",
                     "thrust_body[2]"]),
}
STATUS_FIELDS = [
    "state", "trip_reason", "trip_count", "controller", "fallback_active", "yaw_guarded",
    "odometry_fix", "setpoint_ned[0]", "setpoint_ned[1]", "setpoint_ned[2]", "yaw_setpoint",
    "position_error_ned[0]", "position_error_ned[1]", "position_error_ned[2]",
    "estimate_age_s", "stale_s", "action[0]", "action[1]", "action[2]", "action[3]",
    "rates_setpoint[0]", "rates_setpoint[1]", "rates_setpoint[2]", "thrust_setpoint",
    "loop_time_us",
]
ODOMETRY = "vehicle_visual_odometry"
ODOMETRY_FIELDS = [
    "timestamp", "timestamp_sample", "position[0]", "position[1]", "position[2]",
    "velocity[0]", "velocity[1]", "velocity[2]", "q[0]", "q[1]", "q[2]", "q[3]",
    "reset_counter", "quality",
]

Topics = dict[str, dict[str, np.ndarray]]


def load(path: Path) -> Topics:
    from pyulog import ULog  # optional dependency: pip install -e .[ulog]

    log = ULog(str(path), message_name_filter_list=[STATUS, ODOMETRY, *TOPICS])
    topics: Topics = {}
    for d in log.data_list:
        if d.multi_id == 0:
            topics[d.name] = {k: np.asarray(v) for k, v in d.data.items()}
    return topics


def first_engaged_window(status: dict[str, np.ndarray]) -> np.ndarray:
    """Indices of the first contiguous run of ENGAGED status samples."""
    engaged = np.asarray(status["state"]) == STATE_ENGAGED
    if not engaged.any():
        raise ValueError("the module was never engaged in this log")
    start = int(np.argmax(engaged))
    after = np.where(~engaged[start:])[0]
    end = start + int(after[0]) if after.size else engaged.size
    return np.arange(start, end)


def sample_at(topic: dict[str, np.ndarray], fields: list[str], ticks_us: np.ndarray,
              prefix: str, missing: list[str]) -> dict[str, np.ndarray]:
    """Last sample of each field at or before every tick (what a module read would see)."""
    ts = np.asarray(topic["timestamp"], dtype=np.int64)
    idx = np.searchsorted(ts, ticks_us, side="right") - 1
    ok = idx >= 0
    out: dict[str, np.ndarray] = {}
    for f in fields:
        name = f"{prefix}_{f}"
        if f not in topic:
            missing.append(name)
            continue
        col = np.full(ticks_us.size, np.nan)
        col[ok] = np.asarray(topic[f], dtype=float)[idx[ok]]
        out[name] = col
    # how stale the sampled row was, so the reader can see the topic's rate and delay
    age = np.full(ticks_us.size, np.nan)
    age[ok] = (ticks_us[ok] - ts[idx[ok]]) * 1e-6
    out[f"{prefix}_sample_age_s"] = age
    return out


def export(run: Path, label: str, out_dir: Path) -> dict[str, Any]:
    ulog = run / "flight.ulg"
    summary_path = run / "hold_summary.json"
    meta: dict[str, Any] = {}
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        meta = {k: s.get(k) for k in ("scenario", "profile", "seed", "commit", "module_params")}
    scenario = meta.get("scenario") or run.name
    profile = meta.get("profile") or "unknown"
    seed = meta.get("seed")
    stem = f"{label}_{scenario}_{profile}_seed{seed}"
    topics = load(ulog)
    if STATUS not in topics:
        raise ValueError(f"{ulog}: no {STATUS} topic")
    status = topics[STATUS]
    win = first_engaged_window(status)
    ticks_us = np.asarray(status["timestamp"], dtype=np.int64)[win]
    t0 = int(ticks_us[0])

    missing: list[str] = []
    cols: dict[str, np.ndarray] = {
        "tick": np.arange(win.size),
        "t": (ticks_us - t0) * 1e-6,
        "timestamp_us": ticks_us,
    }
    for f in STATUS_FIELDS:
        if f in status:
            cols[f"sh_{f}"] = np.asarray(status[f], dtype=float)[win]
        else:
            missing.append(f"sh_{f}")
    for topic, (prefix, fields) in TOPICS.items():
        if topic not in topics:
            missing.extend(f"{prefix}_{f}" for f in fields)
            continue
        cols.update(sample_at(topics[topic], fields, ticks_us, prefix, missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    ticks_path = out_dir / f"{stem}_ticks.csv"
    write_csv(ticks_path, cols)

    odom_rows = 0
    odom_path = out_dir / f"{stem}_odometry.csv"
    if ODOMETRY in topics:
        od = topics[ODOMETRY]
        ts = np.asarray(od["timestamp"], dtype=np.int64)
        keep = (ts >= ticks_us[0] - 1_000_000) & (ts <= ticks_us[-1] + 1_000_000)
        ocols: dict[str, np.ndarray] = {"t": (ts[keep] - t0) * 1e-6}
        for f in ODOMETRY_FIELDS:
            if f in od:
                ocols[f] = np.asarray(od[f], dtype=float if f.startswith(("p", "v", "q"))
                                      else np.int64)[keep]
            else:
                missing.append(f"odometry_{f}")
        write_csv(odom_path, ocols)
        odom_rows = int(keep.sum())
    else:
        missing.append("odometry_*")

    tick_dt = np.diff(cols["t"])
    info = {
        "run": str(run), "label": label, **meta,
        "ticks": int(win.size), "hold_s": float(cols["t"][-1]),
        "tick_hz": float(1.0 / np.median(tick_dt)) if tick_dt.size else None,
        "odometry_rows": odom_rows,
        "files": [ticks_path.name, odom_path.name if odom_rows else None],
        "columns": list(cols), "missing": missing,
    }
    (out_dir / f"{stem}_columns.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def write_csv(path: Path, cols: dict[str, np.ndarray]) -> None:
    names = list(cols)
    n = len(cols[names[0]])
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(n):
            w.writerow([fmt(cols[k][i]) for k in names])


def fmt(v: Any) -> str:
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    x = float(v)
    if np.isnan(x):
        return ""
    return f"{x:.6g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run folders holding flight.ulg")
    ap.add_argument("--label", required=True, help="policy name for the file names")
    ap.add_argument("--out", type=Path, default=Path("runs/gazebo-traces"))
    args = ap.parse_args(argv)
    rc = 0
    for run in args.runs:
        if not (run / "flight.ulg").exists():
            print(f"{run}: no flight.ulg, skipped")
            continue
        try:
            info = export(run, args.label, args.out)
        except Exception as exc:  # keep going through the batch, report at the end
            print(f"{run}: FAILED {type(exc).__name__}: {exc}")
            rc = 1
            continue
        print(f"{run.name}: {info['ticks']} ticks at {info['tick_hz']:.1f} Hz over "
              f"{info['hold_s']:.1f} s, {info['odometry_rows']} odometry rows"
              + (f", missing {info['missing']}" if info["missing"] else ""))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
