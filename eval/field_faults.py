"""Score Gazebo flights flown on a field-fault profile (``docs/field-faults.md``).

A field fault (``sim2real/estimate_bridge/field_faults.py``) is a blunt, scripted estimate
failure: a VIO reset to its origin, a restart, a long silence, a frozen estimate, a slow
drift; with GPS on, a scripted GPS fault (``gps.py``: RTK float, a jump, a glitch, a loss
and the hand-over back) counts the same way. The benchmark's p95 and its "the hold must
run to the end" crash rule do not fit them: when the estimate snaps by metres, handing
back is the right answer, not a failure. What matters is what the vehicle physically did,
so this reads the TRUE position from each run's bridge trace over the window after the
fault and reports:

- ``clearance_before_m``: closest to the wall in the 5 s before the fault;
- ``min_clearance_m``: closest to the wall in the window (the x500's props touch at
  ``CONTACT_M``, 0.40 m, as ``eval/gazebo_check.py`` scores it), ``contact`` if under it;
- ``max_horizontal_m`` / ``max_drop_m`` / ``max_climb_m``: the largest move away from where
  the vehicle was when the fault began;
- ``min_height_m``: lowest point above the takeoff height in the window;
- ``airborne_end``: still more than 0.5 m up at the end of the window;
- ``trip``: the module's first watchdog trip (hold runner only, from px4.log); "none" when the
  only one is the test sequence's scripted ``test_fault`` after the hold.

Works for both runners' folders (``run_gazebo`` for the module, ``run_estimate`` for stock
PX4), so the two can be read side by side, as every result here must be.

    python -m eval.field_faults runs/<flight 1> runs/<flight 2> ... [--window-s 25]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim2real.estimate_bridge.bridge import read_trace
from sim2real.estimate_bridge.config import EstimateProfile
from sim2real.gazebo.scenario import Scenario, load_scenario

CONTACT_M = 0.40  # x500 centre to wall when the props touch (eval/gazebo_check.py)
WINDOW_S = 25.0
_TRIP = re.compile(r"watchdog trip: (\w+)")


def wall_clearance(scenario: Scenario, pos_enu: np.ndarray) -> np.ndarray:
    """Distance from each position to the wall plane along its normal, m (positive: in front)."""
    n = np.asarray(scenario.wall_normal_toward, dtype=np.float64)
    return (np.asarray(scenario.wall_point, dtype=np.float64) - pos_enu) @ n


def first_trip(px4_log: Path) -> str | None:
    """The module's first watchdog trip reason in the PX4 console log, or "none" when the
    only trip is the test sequence's own scripted one after the hold (``test_fault``):
    refusals and that trip are not the fault's doing."""
    if not px4_log.exists():
        return None
    for line in px4_log.read_text(errors="replace").splitlines():
        m = _TRIP.search(line)
        if m:
            return "none" if m.group(1) == "test_fault" else m.group(1)
    return "none"


def score_run(run: Path, window_s: float = WINDOW_S) -> dict[str, Any]:
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    profile = EstimateProfile.from_dict(cfg["profile"])
    faults = [*profile.field_faults, *profile.gps_faults]
    if not faults:
        raise ValueError(f"{run}: profile {profile.name} has no field or GPS faults")
    scenario = load_scenario(cfg.get("scenario", cfg.get("name")))
    verify = json.loads((run / "estimate_verify.json").read_text())
    # run_gazebo names its folders *_sitl_hold_*; run_estimate (stock PX4) *_sitl_estimate_*.
    runner = "hold" if "_sitl_hold_" in run.name or (run / "hold_summary.json").exists() \
        else "stock"
    row: dict[str, Any] = {"run": run.name, "runner": runner, "profile": profile.name,
                           "faults": [f.kind for f in faults], "gps_on": profile.gps is not None,
                           "seed": profile.seed, "scenario": scenario.name}
    anchor_t = verify.get("anchor_t")
    if anchor_t is None:
        return {**row, "error": "the fault anchor never fired"}
    tr = read_trace(run / "estimate_trace.csv")
    t = tr["t"]
    truth = np.stack([tr["tx"], tr["ty"], tr["tz"]], 1)
    t_fault = float(anchor_t) + min(f.at_s for f in faults)
    i0 = int(np.searchsorted(t, t_fault))
    if i0 >= t.size:
        return {**row, "error": "the trace ends before the fault"}
    before = (t >= t_fault - 5.0) & (t < t_fault)
    win = (t >= t_fault) & (t <= t_fault + window_s)
    ground = float(truth[0, 2])
    clear = wall_clearance(scenario, truth)
    d = truth[win] - truth[i0]
    min_clear = float(clear[win].min())
    end_i = int(np.flatnonzero(win)[-1])
    row.update({
        "t_fault_s": round(t_fault, 2),
        "window_s": round(float(t[end_i] - t_fault), 2),
        "clearance_before_m": round(float(clear[before].min()), 3) if before.any() else None,
        "min_clearance_m": round(min_clear, 3),
        "contact": bool(min_clear < CONTACT_M),
        "max_horizontal_m": round(float(np.hypot(d[:, 0], d[:, 1]).max()), 3),
        "max_drop_m": round(float(max(0.0, -d[:, 2].min())), 3),
        "max_climb_m": round(float(max(0.0, d[:, 2].max())), 3),
        "min_height_m": round(float(truth[win, 2].min() - ground), 3),
        "airborne_end": bool(truth[end_i, 2] - ground > 0.5),
        "trip": first_trip(run / "px4.log") if runner == "hold" else None,
    })
    return row


def table(rows: list[dict[str, Any]]) -> str:
    cols = ["profile", "runner", "seed", "clearance_before_m", "min_clearance_m", "contact",
            "max_horizontal_m", "max_drop_m", "max_climb_m", "min_height_m", "airborne_end",
            "trip"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['profile']} | {r['runner']} | {r['seed']} | error: {r['error']} |")
            continue
        out.append("| " + " | ".join("n/a" if r.get(c) is None else str(r[c]) for c in cols)
                   + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run folders flown on a field profile")
    ap.add_argument("--window-s", type=float, default=WINDOW_S)
    ap.add_argument("--json", type=Path, help="also write the rows here")
    args = ap.parse_args(argv)
    rows = [score_run(r, args.window_s) for r in args.runs]
    print(table(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
