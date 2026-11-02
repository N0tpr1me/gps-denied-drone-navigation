"""Replay a bridge trace through ``envs.estimate`` and check PX4 was fed exactly that.

The bridge logs every truth sample it consumed and the estimate it produced. Re-running
the corruptor on the logged truth with the run's profile (config, seed, dt) must give the
same estimate to floating-point precision, and no row inside a dropout may have been
published. That proves the stream PX4 fused is the training model with that seed.

    python -m sim2real.estimate_bridge.verify_trace runs/<run>/estimate_config.yaml \
        runs/<run>/estimate_trace.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sim2real.estimate_bridge.bridge import read_trace
from sim2real.estimate_bridge.config import EstimateProfile, load_profile

TOLERANCE_M = 1e-9


def verify(profile: EstimateProfile, trace_path: str | Path,
           tolerance: float = TOLERANCE_M) -> dict[str, Any]:
    tr = read_trace(trace_path)
    n = tr["t"].size
    report: dict[str, Any] = {"samples": int(n), "tolerance": tolerance}
    if n == 0:
        return {**report, "ok": False, "error": "empty trace"}

    truth = np.stack([tr["tx"], tr["ty"], tr["tz"]], 1)
    est = np.stack([tr["ex"], tr["ey"], tr["ez"]], 1)
    est_v = np.stack([tr["evx"], tr["evy"], tr["evz"]], 1)
    gps = np.stack([tr["gx"], tr["gy"], tr["gz"]], 1)
    max_pos = max_vel = max_gps = 0.0
    stale_mismatch = gps_mismatch = 0
    # The trace holds the corruptor's steps only; with a divisor > 1 the skipped truth
    # samples are not logged, so the replay feeds each logged row as one step.
    replay = profile.bridge(dt=profile.dt * profile.divisor, divisor=1)
    for i in range(n):
        s = replay.update(tr["t"][i], truth[i], engaged=bool(tr["engaged"][i]))
        max_pos = max(max_pos, float(np.abs(s.est_pos - est[i]).max()))
        max_vel = max(max_vel, float(np.abs(s.est_vel - est_v[i]).max()))
        stale_mismatch += int(s.stale != bool(tr["stale"][i]))
        # The scripted GPS (gps.py): same ticks, same fix, same noise.
        gps_mismatch += int(s.gps.fix_type != int(tr["gps_fix"][i]))
        if s.gps.sent and int(tr["gps_fix"][i]) > 0:
            max_gps = max(max_gps, float(np.abs(s.gps.pos - gps[i]).max()))

    published_while_stale = int(np.sum((tr["published"] > 0) & (tr["stale"] > 0)))
    unpublished_fresh = int(np.sum((tr["published"] == 0) & (tr["stale"] == 0)))
    spacing = np.diff(tr["t"])
    report.update({
        "max_pos_error_m": max_pos,
        "max_vel_error_mps": max_vel,
        "stale_flag_mismatches": stale_mismatch,
        "gps_mismatches": gps_mismatch,
        "max_gps_error_m": max_gps,
        "gps_samples": int(np.sum(tr["gps_fix"] > 0)),
        "published_while_stale": published_while_stale,
        "unpublished_fresh": unpublished_fresh,
        "published": int(tr["published"].sum()),
        "stale": int(tr["stale"].sum()),
        "jumps": int(tr["jumped"].sum()),
        "dropouts": int(tr["dropout_started"].sum()),
        "duration_s": float(tr["t"][-1] - tr["t"][0]),
        "spacing_mean_s": float(spacing.mean()) if spacing.size else None,
        "spacing_max_s": float(spacing.max()) if spacing.size else None,
        "spacing_irregular": int(np.sum(np.abs(spacing - profile.dt * profile.divisor)
                                        > 0.5 * profile.dt)) if spacing.size else 0,
        "fault_anchor": profile.fault_anchor,
        "anchor_t": replay.anchor_t,
        "anchor_step": replay.anchor_step,
        "engaged_samples": int(tr["engaged"].sum()),
    })
    report["ok"] = bool(
        max_pos <= tolerance and max_vel <= tolerance and stale_mismatch == 0
        and published_while_stale == 0 and unpublished_fresh == 0
        and gps_mismatch == 0 and max_gps <= tolerance
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile", help="profile YAML (the run's estimate_config.yaml) or name")
    ap.add_argument("trace", help="estimate_trace.csv")
    args = ap.parse_args(argv)
    report = verify(load_profile(args.profile), args.trace)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
