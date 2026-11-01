"""Score a design partner's flight session: tuned PX4 against Structure Hold on a real drone.

    python -m eval.partner_session <session folder>/session.yaml [--out DIR]

The partner flies the protocol in ``docs/partner-pack.md`` and sends one folder: the .ulg
files and a ``session.yaml`` run sheet (template: ``sim2real/partner/session_template.yaml``).
Each run is one 60 s hold, flown either by tuned stock PX4 in Hold mode or by Structure
Hold. This scores every run with the same code as Gazebo and the mock (``eval.metrics`` on
the TRUE position from motion capture, via ``eval.ulog``) and prints the same tables as
``eval.baselines.run_baseline``: per case, the mean p95 of Structure Hold against tuned PX4.

Per run, from the log alone:

- the scored window is the first stretch in the controller's mode (``vehicle_status``
  ``nav_state``: 23 EXTERNAL1 for Structure Hold, 4 AUTO_LOITER for tuned PX4 in Hold mode,
  2 POSCTL with ``baseline_mode: position``), minus the first ``settle_s`` and cut at
  ``hold_s``, as ``sim2real.px4.run_gazebo`` cuts the Gazebo hold;
- the hold point is the setpoint at the window start (``structure_hold_status`` for the
  module, whose hold is rate-only; ``vehicle_local_position_setpoint`` for stock PX4), in
  the EKF's frame. Mocap is in its own frame, so the EKF estimate is fitted to mocap by a
  yaw and an offset over the take-off and the move to the hold point (from arming to the
  mode switch, or the run's ``align`` window), and the hold point is carried through that
  fit, as ``eval.ulog`` does for setpoints;
- a hold that does not last ``hold_s - settle_s - 1`` seconds ended early (a watchdog
  trip, a failsafe or the pilot); it counts in the crash column with any wall contact,
  the Gate 2 rule (crash judged on the 60 s hold only), and its trip reason is reported.

Runs marked ``aborted`` in the sheet are listed with their reason and not scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval.baselines.run_baseline import aggregate, compare, table
from eval.metrics import Surface, compute_metrics
from eval.ulog import (
    GROUNDTRUTH,
    MOCAP,
    TOPICS,
    Topics,
    align_estimate,
    ned_to_enu,
    rotate_z,
    trace_from_topics,
)

VEHICLE_STATUS = "vehicle_status"
STATUS = "structure_hold_status"
SETPOINT = "vehicle_local_position_setpoint"
# vehicle_status nav_state and arming_state on PX4 v1.17.0 (msg/VehicleStatus.msg).
NAV_POSCTL, NAV_AUTO_LOITER, NAV_EXTERNAL1 = 2, 4, 23
ARMING_STATE_ARMED = 2
CONTROLLERS = ("tuned_px4", "structure_hold")
TRIP_NAMES = {0: "none", 1: "test_fault", 2: "estimate_invalid", 3: "estimate_stale",
              4: "tilt", 5: "position_error", 6: "estimate_reset"}
HOLD_S, SETTLE_S = 60.0, 2.0


def seconds(msg: dict[str, np.ndarray], start_us: int) -> np.ndarray:
    return (np.asarray(msg["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6


def mode_stretch(status: dict[str, np.ndarray], start_us: int, nav_state: int
                 ) -> tuple[float, float, int] | None:
    """(start, end, index after the end) of the first stretch in ``nav_state``, log seconds.
    The end is the first sample in another mode, or the last sample of the log."""
    t = seconds(status, start_us)
    inside = np.asarray(status["nav_state"]) == nav_state
    if not inside.any():
        return None
    a = int(np.argmax(inside))
    rest = np.flatnonzero(~inside[a:])
    b = a + int(rest[0]) if rest.size else len(t)
    return float(t[a]), float(t[b] if b < len(t) else t[-1]), b


def armed_at(status: dict[str, np.ndarray], start_us: int) -> float | None:
    armed = np.asarray(status["arming_state"]) == ARMING_STATE_ARMED
    return float(seconds(status, start_us)[int(np.argmax(armed))]) if armed.any() else None


def setpoint_at(topics: Topics, start_us: int, controller: str, t0: float) -> np.ndarray:
    """The hold point (ENU, EKF frame) at log second ``t0``."""
    if controller == "structure_hold":
        msg, keys = topics[STATUS], [f"setpoint_ned[{k}]" for k in range(3)]
    else:
        msg, keys = topics[SETPOINT], ["x", "y", "z"]
    t = seconds(msg, start_us)
    after = np.flatnonzero(t >= t0 - 1e-9)
    if not after.size:
        raise ValueError(f"no setpoint sample at or after {t0:.2f} s")
    i = int(after[0])
    sp = ned_to_enu(*(np.asarray(msg[k][i:i + 1], dtype=float) for k in keys))[0]
    if not np.all(np.isfinite(sp)):
        raise ValueError(f"setpoint at {t[i]:.2f} s is not finite")
    return sp


def mocap_fit(topics: Topics, start_us: int, window: tuple[float, float]
              ) -> tuple[float, np.ndarray, float]:
    """Yaw, offset and residual that carry the EKF frame into the mocap frame."""
    mc = topics[MOCAP]
    stamp = "timestamp_sample" if np.all(np.asarray(mc.get("timestamp_sample", [0])) > 0) \
        else "timestamp"
    t = (np.asarray(mc[stamp], dtype=np.int64) - int(start_us)) * 1e-6
    pos = ned_to_enu(mc["position[0]"], mc["position[1]"], mc["position[2]"])
    t, first = np.unique(t, return_index=True)
    return align_estimate(topics, t, pos[first],
                          lambda m: seconds(m, start_us), window)


def score_run(topics: Topics, start_us: int, run: dict[str, Any], session: dict[str, Any]
              ) -> dict[str, Any]:
    """One table row in ``eval.baselines.run_baseline``'s format, plus what happened."""
    controller = run["controller"]
    if controller not in CONTROLLERS:
        raise ValueError(f"controller {controller!r}: expected one of {CONTROLLERS}")
    hold_s = float(session.get("hold_s", HOLD_S))
    settle_s = float(session.get("settle_s", SETTLE_S))
    nav = NAV_EXTERNAL1 if controller == "structure_hold" else (
        NAV_POSCTL if session.get("baseline_mode") == "position" else NAV_AUTO_LOITER)
    vs = topics[VEHICLE_STATUS]
    stretch = mode_stretch(vs, start_us, nav)
    if stretch is None:
        raise ValueError(f"the log never enters nav_state {nav} ({controller})")
    m0, m1, after = stretch
    if run.get("window"):
        window = (float(run["window"][0]), float(run["window"][1]))
    else:
        window = (m0 + settle_s, min(m1, m0 + hold_s))
    if window[1] <= window[0]:
        raise ValueError(f"the {controller} stretch is {m1 - m0:.1f} s, shorter than the settle")
    full = window[1] - window[0] >= hold_s - settle_s - 1.0

    hold_point = setpoint_at(topics, start_us, controller, window[0])
    row: dict[str, Any] = {"file": run["file"], "controller": controller}
    if GROUNDTRUTH not in topics and MOCAP in topics:
        align = run.get("align") or session.get("align")
        if not align or align == "auto":
            t_arm = armed_at(vs, start_us)
            align = (t_arm if t_arm is not None else 0.0, m0)
        psi, offset, rms = mocap_fit(topics, start_us, (float(align[0]), float(align[1])))
        hold_point = rotate_z(hold_point[None], psi)[0] + offset
        row.update(align_window_s=[float(align[0]), float(align[1])],
                   align_yaw_deg=math.degrees(psi), align_rms_m=rms)
    trace = trace_from_topics(topics, start_us, hold_point=hold_point,
                              allow_estimate=bool(session.get("allow_estimate", False)))
    wall = session.get("wall") or {}
    surface = Surface(wall["point"], wall["normal"]) if wall else None
    gusts = [(float(a), float(b)) for a, b in run.get("gusts", [])]
    radius = session.get("contact_radius_m")
    extra = {"contact_radius_m": float(radius)} if radius else {}
    metrics = compute_metrics(trace, surface, gusts, window, **extra)

    trip = None
    if controller == "structure_hold" and STATUS in topics:
        st = topics[STATUS]
        t = seconds(st, start_us)
        near = (t >= window[1] - 0.5) & (t <= window[1] + 2.0)
        if near.any():
            trip = TRIP_NAMES.get(int(np.asarray(st["trip_reason"])[near].max()), "unknown")
    next_mode = int(np.asarray(vs["nav_state"])[after]) if after < len(vs["nav_state"]) else None
    tv = seconds(vs, start_us)
    in_window = (tv >= window[0]) & (tv <= window[1])
    failsafe = "failsafe" in vs and bool(np.asarray(vs["failsafe"])[in_window].any())
    contacts = int(metrics.get("contacts", 0))
    row.update({
        "scenario": run["case"], "seed": int(run["run"]), "run": run["file"],
        "window_s": [float(window[0]), float(window[1])],
        "hold_point_enu": [float(v) for v in hold_point],
        "position_source": trace.meta.get("position_source"),
        "ended_early": not full, "next_nav_state": next_mode, "trip_reason": trip,
        "contacts": contacts, "crash": contacts > 0 or not full,
        "failsafe": failsafe, "failsafe_in_hover": failsafe, "post_hold_fail": False,
        "metrics": metrics,
    })
    for key in ("p95_m", "rms_m", "max_m", "worst_toward_surface_m", "min_clearance_m",
                "recovery_max_s"):
        row[key] = float(metrics[key]) if key in metrics else math.nan
    return row


def load_session_topics(path: Path) -> tuple[Topics, int]:
    try:
        from pyulog import ULog
    except ImportError as e:
        raise ImportError("reading .ulg needs pyulog: pip install -e .[ulog]") from e
    log = ULog(str(path), message_name_filter_list=TOPICS + [VEHICLE_STATUS, STATUS])
    return {d.name: d.data for d in log.data_list if d.multi_id == 0}, int(log.start_timestamp)


def score_session(sheet: Path, loader=load_session_topics) -> dict[str, Any]:
    session = yaml.safe_load(sheet.read_text())
    folder = sheet.parent
    rows, aborted, errors = [], [], []
    for run in session["runs"]:
        if run.get("aborted"):
            aborted.append({k: run.get(k) for k in ("file", "controller", "case", "run",
                                                    "reason")})
            continue
        try:
            topics, start_us = loader(folder / run["file"])
            rows.append(score_run(topics, start_us, run, session))
        except (ValueError, KeyError, OSError) as e:
            errors.append({"file": run.get("file"), "error": str(e)})
    batches = {}
    for c in CONTROLLERS:
        rs = [r for r in rows if r["controller"] == c]
        if rs:
            batches[c] = {"label": c, "params": session.get(f"{c}_params", "see session.yaml"),
                          "commit": session.get("firmware", "unknown"), "rows": rs,
                          "aggregate": aggregate(rs)}
    return {"session": {k: v for k, v in session.items() if k != "runs"},
            "batches": batches, "aborted": aborted, "errors": errors}


def report(result: dict[str, Any]) -> str:
    s = result["session"]
    out = [f"# Partner session: {s.get('drone', '?')} at {s.get('site', '?')}, "
           f"{s.get('date', '?')}", ""]
    for batch in result["batches"].values():
        out += [table(batch), ""]
    b = result["batches"]
    if len(b) == 2:
        out += [compare(b["tuned_px4"], b["structure_hold"]), ""]
    early = [r for bt in b.values() for r in bt["rows"] if r["ended_early"]]
    for r in early:
        out.append(f"- ended early: {r['file']} ({r['controller']}, {r['scenario']} run "
                   f"{r['seed']}), window {r['window_s'][0]:.1f} to {r['window_s'][1]:.1f} s, "
                   f"trip {r['trip_reason']}, next nav_state {r['next_nav_state']}")
    for a in result["aborted"]:
        out.append(f"- aborted, not scored: {a['file']} ({a['controller']}, {a['case']} run "
                   f"{a['run']}): {a.get('reason') or 'no reason given'}")
    for e in result["errors"]:
        out.append(f"- not scored: {e['file']}: {e['error']}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sheet", type=Path, help="the session's session.yaml")
    p.add_argument("--out", type=Path, help="output folder (default: <session>/scored)")
    args = p.parse_args(argv)
    result = score_session(args.sheet)
    out = args.out or args.sheet.parent / "scored"
    out.mkdir(parents=True, exist_ok=True)
    (out / "session_summary.json").write_text(json.dumps(result, indent=2, default=str))
    text = report(result)
    (out / "report.md").write_text(text)
    print(text)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
