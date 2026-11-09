"""Split a Gazebo batch's hold error into horizontal and vertical, and show how much of the
vertical part is a height offset the estimator already had on the ground.

``eval.metrics`` scores the TRUE position against the setpoint, and the setpoint lives in
EKF2's local frame. Horizontally that frame is the truth frame (the vision estimate starts
unbiased, ``bias_from_anchor``). Vertically it need not be: on barometer height EKF2 puts
its origin at the first barometer sample it sees, so its height on the ground can read a
few decimetres before the motors start. PX4 then climbs 2.5 m above home and the truth is
2.5 m, but the setpoint says 2.5 m plus that offset, and the scorer counts the constant
difference as hold error for the whole flight, for tuned PX4 and Structure Hold alike. The
offset is where the frame starts, not how well anything holds: it is there with the motors
off (``docs/baseline.md``, "What moved tuned PX4 from 0.43 to 0.51 m").

Per flight this prints the 95th percentile of the error as scored, of its horizontal and
vertical parts, the estimator's height minus the truth on the ground before lift-off, and
the vertical and total p95 with that one number taken out. Whatever the estimate does after
lift-off (drift, jumps, a frozen height) stays in: that is real hold error. A diagnostic
read-out of flown batches, not a run and not the scorer; the headline numbers stay
``eval.metrics``' until ``docs/decisions.md`` says otherwise.

    python -m eval.height_split runs/pooled_tuned_seeds1-8.json runs/pooled_v22_seeds1-8.json
    python -m eval.height_split --per-seed runs/<batch>/batch.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import GRID_DT_S, resample
from eval.ulog import ESTIMATE, GROUNDTRUTH, load_topics, ned_to_enu, trace_from_topics

REPO_ROOT = Path(__file__).resolve().parents[1]
LIFT_M = 0.2             # truth height above its first sample that counts as off the ground
GROUND_SPAN_S = (3.0, 0.5)  # the offset is the mean from 3 s to 0.5 s before lift-off
FIELDS = ("p95_m", "h_p95_m", "v_p95_m", "ground_offset_m", "v_p95_no_offset_m",
          "p95_no_offset_m")


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


def split_error(err: np.ndarray, offset_m: float) -> dict[str, float]:
    """``err`` is truth minus setpoint, (n, 3) ENU, over the scored window. The setpoint is
    ``offset_m`` too high in the truth frame, so the error with the offset out is
    ``err_z + offset_m``."""
    h = np.hypot(err[:, 0], err[:, 1])
    out = {
        "p95_m": float(np.percentile(np.linalg.norm(err, axis=1), 95)),
        "h_p95_m": float(np.percentile(h, 95)),
        "v_p95_m": float(np.percentile(np.abs(err[:, 2]), 95)),
        "ground_offset_m": float(offset_m),
        "v_p95_no_offset_m": math.nan,
        "p95_no_offset_m": math.nan,
    }
    if math.isfinite(offset_m):
        v = err[:, 2] + offset_m
        out["v_p95_no_offset_m"] = float(np.percentile(np.abs(v), 95))
        out["p95_no_offset_m"] = float(np.percentile(np.hypot(h, v), 95))
    return out


def flight_split(run: Path) -> dict[str, float]:
    """One flight folder of either runner: the stock hover (``metrics.json``, setpoint from
    the log) or a Structure Hold flight (``hold.json``, the module's hold point)."""
    hold_file = run / "hold.json"
    if hold_file.exists():
        hold = json.loads(hold_file.read_text())
        window, hold_point = hold["window_s"], np.asarray(hold["hold_point_enu"], float)
    else:
        window, hold_point = json.loads((run / "metrics.json").read_text())["window_s"], None
    topics, start_us = load_topics(run / "flight.ulg")
    trace = trace_from_topics(topics, start_us, hold_point=hold_point)
    _, pos, sp, _ = resample(trace, GRID_DT_S, max(window[0], trace.t[0]),
                             min(window[1], trace.t[-1]))

    def seconds(msg: dict[str, np.ndarray]) -> np.ndarray:
        return (np.asarray(msg["timestamp"], dtype=np.int64) - start_us) * 1e-6

    truth, est = topics[GROUNDTRUTH], topics[ESTIMATE]
    t = seconds(truth)
    order = np.argsort(t)
    t = t[order]
    up_truth = ned_to_enu(truth["x"], truth["y"], truth["z"])[order, 2]
    te = seconds(est)
    eo = np.argsort(te)
    up_est = np.interp(t, te[eo], ned_to_enu(est["x"], est["y"], est["z"])[eo, 2])
    return split_error(pos - sp, ground_offset(t, up_truth, up_est))


def batch_split(batch_file: Path) -> dict[str, Any]:
    batch = json.loads(batch_file.read_text())
    rows = []
    for r in batch["rows"]:
        if not r.get("flown") or r.get("p95_m") is None:
            continue
        s = flight_split(REPO_ROOT / r["run"])
        # The same number the batch table carries, or this read-out is of something else.
        if abs(s["p95_m"] - r["p95_m"]) > 1e-3:
            raise ValueError(f"{r['run']}: p95 {s['p95_m']:.4f} m here, {r['p95_m']:.4f} m "
                             "in the batch")
        rows.append({"scenario": r["scenario"], "seed": r["seed"], **s})
    return {"label": batch["label"], "params": batch.get("params"), "rows": rows}


def table(split: dict[str, Any], per_seed: bool) -> str:
    lines = [f"`{split['label']}` (`{split['params']}`), mean over seeds; m", "",
             "| scenario | n | p95 as scored | horizontal p95 | vertical p95 | height offset "
             "on the ground, mean (min to max) | vertical p95, offset out | p95, offset out |",
             "|---|---|---|---|---|---|---|---|"]
    scenarios = list(dict.fromkeys(r["scenario"] for r in split["rows"]))
    for sc in scenarios:
        rows = [r for r in split["rows"] if r["scenario"] == sc]
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in FIELDS}
        off = [r["ground_offset_m"] for r in rows]
        lines.append(
            f"| {sc} | {len(rows)} | {mean['p95_m']:.3f} | {mean['h_p95_m']:.3f} "
            f"| {mean['v_p95_m']:.3f} | {mean['ground_offset_m']:+.3f} "
            f"({np.nanmin(off):+.3f} to {np.nanmax(off):+.3f}) "
            f"| {mean['v_p95_no_offset_m']:.3f} | {mean['p95_no_offset_m']:.3f} |")
    if per_seed:
        lines += ["", "| scenario | seed | p95 as scored | horizontal p95 | vertical p95 "
                  "| height offset on the ground | vertical p95, offset out | p95, offset out |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in split["rows"]:
            lines.append(
                f"| {r['scenario']} | {r['seed']} | {r['p95_m']:.3f} | {r['h_p95_m']:.3f} "
                f"| {r['v_p95_m']:.3f} | {r['ground_offset_m']:+.3f} "
                f"| {r['v_p95_no_offset_m']:.3f} | {r['p95_no_offset_m']:.3f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("batches", nargs="+", type=Path,
                    help="batch.json or pooled batch files of eval.baselines.run_baseline")
    ap.add_argument("--per-seed", action="store_true", help="also print every flight")
    ap.add_argument("--json", type=Path, help="also write every flight's numbers here")
    args = ap.parse_args(argv)
    splits = [batch_split(b) for b in args.batches]
    print("\n\n".join(table(s, args.per_seed) for s in splits))
    if args.json:
        args.json.write_text(json.dumps(splits, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
