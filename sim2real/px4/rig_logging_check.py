"""Check, in PX4 SITL (SIH), that the fan rig's wind and mocap streams land in the ulog.

The rig's logger sends wind sensor speeds and the gust switch to PX4 over MAVLink, and the
mocap system streams the true pose as ``ODOMETRY`` (estimator type MOCAP); both are scored
from the flight log (docs/fan-rig.md). This flight checks that path on PX4 v1.17.0 with the
rig's logging profile, ``SDLOG_PROFILE`` 179, before any hardware exists:

1. start PX4 SIH from a built tree with ``SDLOG_PROFILE`` 179 (logging from boot);
2. stream for ``--seconds`` per phase, at the rig's 20 Hz:

   - ``burst``: four ``NAMED_VALUE_FLOAT`` keys (three sensors and the gust switch) sent
     back to back each tick (expected to lose samples: ``debug_key_value`` has a uORB
     queue of one, so only the last key of a burst survives until the logger reads it;
     reported under ``findings``, not as a check);
   - ``spaced``: the same four keys, one every 12.5 ms;
   - ``array``: one ``DEBUG_FLOAT_ARRAY`` per tick holding all four values;

   and mocap ``ODOMETRY`` at 30 Hz throughout, on a known circle;
3. stop PX4, read the ulog and count what arrived: every sample of each key and array,
   the mocap rate, and that ``eval.ulog`` reads the mocap topic as truth.

In lockstep a stalled simulator can let two messages of one stream land in the same PX4
step, and a queue-of-one topic then keeps only the second, so a few per cent can go missing
in SIH that a real-time flight controller would keep; hence ``MIN_FRACTION``.

Usage (PX4 tree built with ``make px4_sitl_default``; needs pymavlink and pyulog)::

    python -m sim2real.px4.rig_logging_check <PX4_DIR> [--out DIR]

Writes ``summary.json`` and ``flight.ulg`` to the out folder; exits 0 when every check
passed. SIH runs in lockstep, so rates are in PX4 time, which is what the log is in too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAVLINK_URL = "udpin:0.0.0.0:14540"
PROFILE = 179
RATE_HZ = 20.0
MOCAP_HZ = 30.0
PHASES = ("burst", "spaced", "array")
KEYS = ("s1", "s2", "s3", "gust")  # debug_key_value keeps 9 characters of a key
ARRAY_ID = 7
MIN_FRACTION = 0.95  # a stream passes when at least this share of what was sent is logged
# The logger keeps vehicle_mocap_odometry at most every 30 ms (logged_topics.cpp, profile bit
# 7), so mocap is judged on its logged rate, not on the share of what was sent.
MIN_MOCAP_HZ = 25.0


def key_name(phase: str, key: str) -> str:
    return f"{phase[0]}_{key}"


def values(t: float) -> list[float]:
    """Sensor speeds (m/s) and the gust switch at time ``t``: easy to recognise in the log."""
    return [4.0 + math.sin(t), 2.4 + 0.5 * math.sin(t), 1.0, float(int(t) % 2)]


def mocap_ned(t: float) -> tuple[float, float, float]:
    """A 1 m circle at 1.5 m height in the mocap frame, NED."""
    return math.cos(0.5 * t), math.sin(0.5 * t), -1.5


def stream(seconds: float) -> dict[str, int]:
    """Send every phase in turn with mocap throughout; return how many of each were sent."""
    from pymavlink import mavutil

    m = mavutil.mavlink_connection(MAVLINK_URL, source_system=1, source_component=191)
    if m.wait_heartbeat(timeout=60) is None:
        raise RuntimeError("no heartbeat from PX4 on 14540")
    mav = m.mav
    sent = {key_name(p, k): 0 for p in ("burst", "spaced") for k in KEYS}
    sent.update(array=0, mocap=0)
    t0 = time.monotonic()
    next_mocap = t0
    for phase in PHASES:
        start = time.monotonic()
        tick = 0
        while time.monotonic() - start < seconds:
            now = time.monotonic()
            due = start + tick / RATE_HZ
            if now >= due:
                v = values(now - t0)
                boot_ms = int((now - t0) * 1e3)
                if phase == "burst":
                    for k, x in zip(KEYS, v, strict=True):
                        mav.named_value_float_send(boot_ms, key_name(phase, k).encode(), x)
                        sent[key_name(phase, k)] += 1
                elif phase == "spaced":
                    for i, (k, x) in enumerate(zip(KEYS, v, strict=True)):
                        if i:
                            time.sleep(1.0 / RATE_HZ / len(KEYS))
                        mav.named_value_float_send(boot_ms, key_name(phase, k).encode(), x)
                        sent[key_name(phase, k)] += 1
                else:
                    data = v + [0.0] * (58 - len(v))
                    mav.debug_float_array_send(int((now - t0) * 1e6), b"wind", ARRAY_ID, data)
                    sent["array"] += 1
                tick += 1
            if now >= next_mocap:
                n, e, d = mocap_ned(now - t0)
                nan9 = [math.nan] * 21
                mav.odometry_send(
                    int((now - t0) * 1e6), mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    mavutil.mavlink.MAV_FRAME_BODY_FRD, n, e, d, [1.0, 0.0, 0.0, 0.0],
                    math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, nan9, nan9,
                    0, mavutil.mavlink.MAV_ESTIMATOR_TYPE_MOCAP, 100)
                sent["mocap"] += 1
                next_mocap += 1.0 / MOCAP_HZ
            time.sleep(0.001)
    m.close()
    return sent


def decode_chars(msg: dict[str, np.ndarray], field: str, size: int = 10) -> list[str]:
    """pyulog splits a char[N] into ``field[0]`` .. ``field[N-1]``; join each row back."""
    cols = [np.asarray(msg[f"{field}[{i}]"]).astype(np.uint8) for i in range(size)]
    rows = np.stack(cols, axis=1)
    return [bytes(r).split(b"\0", 1)[0].decode(errors="replace") for r in rows]


def analyse(topics: dict[str, dict[str, np.ndarray]], sent: dict[str, int]) -> dict:
    """Counts per stream, the logged share of what was sent, and pass/fail checks."""
    logged: dict[str, int] = dict.fromkeys(sent, 0)
    kv = topics.get("debug_key_value")
    if kv is not None:
        names, counts = np.unique(decode_chars(kv, "key"), return_counts=True)
        for n, c in zip(names, counts, strict=True):
            if n in logged:
                logged[n] = int(c)
    arr = topics.get("debug_array")
    if arr is not None:
        logged["array"] = int(np.sum(np.asarray(arr["id"]) == ARRAY_ID))
    mocap = topics.get("vehicle_mocap_odometry")
    mocap_hz = 0.0
    if mocap is not None:
        logged["mocap"] = len(mocap["timestamp"])
        span = (int(mocap["timestamp"][-1]) - int(mocap["timestamp"][0])) * 1e-6
        mocap_hz = (logged["mocap"] - 1) / span if span > 0 else 0.0
    share = {k: (logged[k] / sent[k] if sent[k] else math.nan) for k in sent}
    checks = {
        "debug_key_value_logged": kv is not None,
        "debug_array_logged": arr is not None,
        "mocap_logged": mocap is not None,
    }
    checks["spaced_keys_complete"] = all(
        share[key_name("spaced", k)] >= MIN_FRACTION for k in KEYS)
    checks["array_complete"] = share["array"] >= MIN_FRACTION
    checks["mocap_rate_ok"] = mocap_hz >= MIN_MOCAP_HZ
    # Not a pass condition: debug_key_value has a queue of one, so keys sent back to back
    # overwrite each other before the logger reads them. Recorded so a change shows.
    findings = {"burst_keys_complete": all(
        share[key_name("burst", k)] >= MIN_FRACTION for k in KEYS)}
    return {"sent": sent, "logged": logged, "share": share, "mocap_logged_hz": mocap_hz,
            "checks": checks, "findings": findings}


def mocap_truth_check(path: Path) -> dict:
    """eval.ulog reads the mocap topic as truth, in ENU (NED circle -> ENU circle at +1.5 m)."""
    from eval.ulog import GROUNDTRUTH, load_topics, trace_from_topics

    topics, start_us = load_topics(path)
    topics.pop(GROUNDTRUTH, None)  # SIH logs its own truth; a real drone has only mocap
    tr = trace_from_topics(topics, start_us, hold_point=[0.0, 0.0, 1.5])
    radius = np.hypot(tr.pos[:, 0], tr.pos[:, 1])
    return {
        "position_source": tr.meta["position_source"],
        "samples": len(tr.t),
        "radius_m": [float(radius.min()), float(radius.max())],
        "height_m": [float(tr.pos[:, 2].min()), float(tr.pos[:, 2].max())],
        "ok": tr.meta["position_source"] == "vehicle_mocap_odometry"
        and bool(np.allclose(radius, 1.0, atol=1e-3) and np.allclose(tr.pos[:, 2], 1.5)),
    }


def run(px4_dir: Path, out: Path, seconds: float) -> dict:
    build = px4_dir / "build" / "px4_sitl_default"
    rootfs = out / "rootfs"
    rootfs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PX4_SIM_MODEL="sihsim_quadx", HEADLESS="1",
               PX4_PARAM_SDLOG_PROFILE=str(PROFILE), PX4_PARAM_SDLOG_MODE="1")
    with open(out / "px4.log", "w") as log:
        px4 = subprocess.Popen([str(build / "bin" / "px4"), "-d", str(build / "etc")],
                               cwd=rootfs, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        time.sleep(5)  # let EKF2 and the MAVLink instances come up
        sent = stream(seconds)
        time.sleep(1)
    finally:
        px4.send_signal(signal.SIGINT)
        try:
            px4.wait(timeout=20)
        except subprocess.TimeoutExpired:
            px4.kill()
            px4.wait()
    logs = sorted((rootfs / "log").rglob("*.ulg"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise RuntimeError("PX4 wrote no ulog")
    ulog = out / "flight.ulg"
    ulog.write_bytes(logs[-1].read_bytes())

    from pyulog import ULog

    log = ULog(str(ulog), message_name_filter_list=[
        "debug_key_value", "debug_array", "vehicle_mocap_odometry", "vehicle_visual_odometry"])
    topics = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    profile = log.initial_parameters.get("SDLOG_PROFILE")
    summary = analyse(topics, sent)
    summary["sdlog_profile"] = profile
    summary["checks"]["profile_179"] = profile == PROFILE
    # Mocap must not reach the estimator's vision input: EKF2 fuses vehicle_visual_odometry.
    summary["checks"]["mocap_not_in_visual_odometry"] = "vehicle_visual_odometry" not in topics
    summary["mocap_truth"] = mocap_truth_check(ulog)
    summary["checks"]["mocap_read_as_truth"] = summary["mocap_truth"]["ok"]
    summary["passed"] = all(summary["checks"].values())
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seconds", type=float, default=10.0, help="per phase")
    args = ap.parse_args(argv)
    out = args.out or REPO / "runs" / time.strftime("%Y%m%d-%H%M%S_rig_logging_check")
    out.mkdir(parents=True, exist_ok=True)
    summary = run(args.px4_dir, out, args.seconds)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
