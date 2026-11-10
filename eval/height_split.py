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
lift-off (drift, jumps, a frozen height) stays in: that is real hold error.

The last column scores against a FIXED hold point, the setpoint at the start of the scored
window, with the offset out. Structure Hold's hold point is fixed already, so nothing
changes for it. Stock PX4 moves its setpoint when EKF2 resets (it holds the jumped
estimate rather than fly back), and the stock runner scores against that moving setpoint;
the hold runner and ``eval/partner_session.py`` score against a fixed point. The column
puts both controllers on the fixed one.

A diagnostic read-out of flown batches, not a run and not the scorer. Since 2026-11-10
(``docs/decisions.md``) the Gazebo runners score the last column themselves
(``eval.ulog.gazebo_reference``): "p95 as scored" here is then the number against the
setpoint as logged, the one every table before that date carries, and this read-out shows
what each of the two steps did to it.

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
from eval.ulog import ground_offset, heights, load_topics, trace_from_topics

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("p95_m", "h_p95_m", "v_p95_m", "ground_offset_m", "v_p95_no_offset_m",
          "p95_no_offset_m", "p95_fixed_no_offset_m", "setpoint_moved_m")
SMOOTH_S = 5.0           # "slow" drift of the height estimate: the range of its 5 s mean
EST_FIELDS = ("est_mean_m", "est_std_m", "est_drift_m", "est_from_ground_m")


def height_estimate_error(t: np.ndarray, up_truth: np.ndarray, up_est: np.ndarray,
                          window: tuple[float, float], offset_m: float,
                          smooth_s: float = SMOOTH_S) -> dict[str, float]:
    """EKF2's height minus the true height over the scored window: its mean, its spread, how
    far its ``smooth_s`` running mean travels (the slow drift a controller follows without
    seeing it) and how far the mean sits from the offset on the ground (what the estimate
    did after lift-off). This is the error a retrained policy cannot remove: it holds the
    estimate, and the estimate is what moved."""
    tg = np.arange(window[0], window[1], GRID_DT_S)
    e = np.interp(tg, t, up_est) - np.interp(tg, t, up_truth)
    n = max(1, int(round(smooth_s / GRID_DT_S)))
    slow = np.convolve(e, np.ones(n) / n, mode="valid") if e.size >= n else e
    return {
        "est_mean_m": float(e.mean()),
        "est_std_m": float(e.std()),
        "est_drift_m": float(slow.max() - slow.min()),
        "est_from_ground_m": float(e.mean() - offset_m) if math.isfinite(offset_m)
        else math.nan,
    }


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


def fixed_point_error(pos: np.ndarray, sp: np.ndarray, offset_m: float) -> dict[str, float]:
    """p95 of the true position against the setpoint at the START of the window (one fixed
    point for the whole hold) with the ground offset out, and how far the setpoint itself
    travelled from there (0 for Structure Hold; a metre and more when stock PX4 follows an
    EKF2 reset)."""
    err = pos - sp[0]
    out = {"p95_fixed_no_offset_m": math.nan,
           "setpoint_moved_m": float(np.max(np.linalg.norm(sp - sp[0], axis=1)))}
    if math.isfinite(offset_m):
        out["p95_fixed_no_offset_m"] = float(np.percentile(
            np.linalg.norm(err + np.array([0.0, 0.0, offset_m]), axis=1), 95))
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
    t, up_truth, up_est = heights(topics, start_us)
    offset = ground_offset(t, up_truth, up_est)
    return {**split_error(pos - sp, offset), **fixed_point_error(pos, sp, offset),
            **height_estimate_error(t, up_truth, up_est, (float(window[0]), float(window[1])),
                                    offset)}


def batch_split(batch_file: Path) -> dict[str, Any]:
    batch = json.loads(batch_file.read_text())
    rows = []
    for r in batch["rows"]:
        if not r.get("flown") or r.get("p95_m") is None:
            continue
        s = flight_split(REPO_ROOT / r["run"])
        # The same number the batch carries against the setpoint as logged (its p95 itself
        # in a batch scored before 2026-11-10), or this read-out is of something else.
        logged = r.get("p95_logged_setpoint_m")
        logged = r["p95_m"] if logged is None or not math.isfinite(logged) else logged
        if abs(s["p95_m"] - logged) > 1e-3:
            raise ValueError(f"{r['run']}: p95 {s['p95_m']:.4f} m here, {logged:.4f} m "
                             "in the batch")
        rows.append({"scenario": r["scenario"], "seed": r["seed"], **s})
    return {"label": batch["label"], "params": batch.get("params"), "rows": rows}


def table(split: dict[str, Any], per_seed: bool) -> str:
    lines = [f"`{split['label']}` (`{split['params']}`), mean over seeds; m", "",
             "| scenario | n | p95, setpoint as logged | horizontal p95 | vertical p95 "
             "| height offset on the ground, mean (min to max) | vertical p95, offset out "
             "| p95, offset out "
             "| p95, offset out, fixed hold point | setpoint moved, worst |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    scenarios = list(dict.fromkeys(r["scenario"] for r in split["rows"]))
    for sc in scenarios:
        rows = [r for r in split["rows"] if r["scenario"] == sc]
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in FIELDS}
        off = [r["ground_offset_m"] for r in rows]
        lines.append(
            f"| {sc} | {len(rows)} | {mean['p95_m']:.3f} | {mean['h_p95_m']:.3f} "
            f"| {mean['v_p95_m']:.3f} | {mean['ground_offset_m']:+.3f} "
            f"({np.nanmin(off):+.3f} to {np.nanmax(off):+.3f}) "
            f"| {mean['v_p95_no_offset_m']:.3f} | {mean['p95_no_offset_m']:.3f} "
            f"| {mean['p95_fixed_no_offset_m']:.3f} "
            f"| {max(r['setpoint_moved_m'] for r in rows):.2f} |")
    lines += ["", "EKF2 height minus true height over the scored hold, mean over seeds; m", "",
              "| scenario | mean | std within a hold | slow drift (range of the "
              f"{SMOOTH_S:.0f} s mean), mean (worst) | mean minus the offset on the ground |",
              "|---|---|---|---|---|"]
    for sc in scenarios:
        rows = [r for r in split["rows"] if r["scenario"] == sc]
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in EST_FIELDS}
        lines.append(
            f"| {sc} | {mean['est_mean_m']:+.3f} | {mean['est_std_m']:.3f} "
            f"| {mean['est_drift_m']:.3f} ({max(r['est_drift_m'] for r in rows):.3f}) "
            f"| {mean['est_from_ground_m']:+.3f} |")
    if per_seed:
        lines += ["", "| scenario | seed | p95, setpoint as logged | horizontal p95 | vertical p95 "
                  "| height offset on the ground | vertical p95, offset out | p95, offset out "
                  "| p95, offset out, fixed hold point | setpoint moved |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in split["rows"]:
            lines.append(
                f"| {r['scenario']} | {r['seed']} | {r['p95_m']:.3f} | {r['h_p95_m']:.3f} "
                f"| {r['v_p95_m']:.3f} | {r['ground_offset_m']:+.3f} "
                f"| {r['v_p95_no_offset_m']:.3f} | {r['p95_no_offset_m']:.3f} "
                f"| {r['p95_fixed_no_offset_m']:.3f} | {r['setpoint_moved_m']:.2f} |")
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
