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

Outdoors there is no mocap. A sheet with ``truth: rtk`` is scored on the RTK receiver's
samples in the log instead (``eval.rtk_truth``): RTK-fixed samples only, turned into ENU
metres about the first surveyed wall point (or ``rtk.origin``), with the estimate fitted to
them over the same take-off window and the receiver's stamping lag measured there too. The
wall is ``wall.survey``: points on its face measured with the RTK rover. Each run says how
much of its hold had a fixed solution (``truth_coverage``; the rest is left out of the
numbers, and a hold under ``RTK_MIN_COVERAGE`` is not scored) and whether EKF2 fused GPS
(``truth_independent``): if it did, the truth is the estimate's own source and the report
says so on that run instead of scoring it quietly. ``eval.rtk_check`` checks all of this
on Gazebo flights, where the simulator's truth is in the same log (``docs/rtk-truth.md``).
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
from eval.partner_report import COVERAGE_WARN, git_commit, render
from eval.rtk_truth import (
    RTK_TOPICS,
    SENSOR_GPS,
    covered,
    fit_delay,
    fit_frame,
    gps_fused,
    read_rtk,
    wall_from_survey,
)
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
STATE_HANDBACK = 3  # structure_hold_status.state
CONTROLLERS = ("tuned_px4", "structure_hold")
TRIP_NAMES = {0: "none", 1: "test_fault", 2: "estimate_invalid", 3: "estimate_stale",
              4: "tilt", 5: "position_error", 6: "estimate_reset", 7: "landed", 8: "range_lost",
              9: "estimate_frozen", 10: "estimate_runaway"}
HOLD_S, SETTLE_S = 60.0, 2.0
# RTK truth: a hold with a fixed solution for less of its time than RTK_MIN_COVERAGE is not
# scored (its numbers would describe some other part of the hold); under the report's
# COVERAGE_WARN it is scored and listed.
RTK_MIN_COVERAGE = 0.5


def is_rtk(session: dict[str, Any]) -> bool:
    return str(session.get("truth", "")).strip().lower() == "rtk"


def seconds(msg: dict[str, np.ndarray], start_us: int) -> np.ndarray:
    return (np.asarray(msg["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6


def mode_stretch(status: dict[str, np.ndarray], start_us: int, nav_state: int,
                 min_s: float = 0.0) -> tuple[float, float, int] | None:
    """(start, end, index after the end) of the first stretch in ``nav_state`` that starts
    once the vehicle is armed and lasts longer than ``min_s``, log seconds. The end is the
    first sample in another mode, or the last sample of the log. Shorter stretches are
    refused engages (the module refuses on the ground or without a usable range and hands
    straight back), and a stretch before arming is the mode the drone sat in on the ground
    (Hold, in PX4's SITL), not holds."""
    t = seconds(status, start_us)
    inside = np.asarray(status["nav_state"]) == nav_state
    a, first = 0, None
    if "arming_state" in status:
        armed = np.flatnonzero(np.asarray(status["arming_state"]) == ARMING_STATE_ARMED)
        if armed.size:
            a = int(armed[0])
            # A stretch already running at arming started on the ground: skip it.
            while a < len(t) and inside[a]:
                a += 1
    while a < len(t):
        start = np.flatnonzero(inside[a:])
        if not start.size:
            break
        a += int(start[0])
        rest = np.flatnonzero(~inside[a:])
        b = a + int(rest[0]) if rest.size else len(t)
        stretch = float(t[a]), float(t[b] if b < len(t) else t[-1]), b
        first = first or stretch
        if stretch[1] - stretch[0] > min_s:
            return stretch
        a = b
    # Only blips: return the first, so the caller reports how short it was.
    return first


def armed_at(status: dict[str, np.ndarray], start_us: int) -> float | None:
    armed = np.asarray(status["arming_state"]) == ARMING_STATE_ARMED
    return float(seconds(status, start_us)[int(np.argmax(armed))]) if armed.any() else None


def armed_span(status: dict[str, np.ndarray], start_us: int) -> tuple[float, float]:
    """From arming to the last armed sample, log seconds (the whole log without the flag)."""
    t = seconds(status, start_us)
    armed = np.flatnonzero(np.asarray(status.get("arming_state", [])) == ARMING_STATE_ARMED)
    return (float(t[armed[0]]), float(t[armed[-1]])) if armed.size else (float(t[0]),
                                                                         float(t[-1]))


def align_window(run: dict[str, Any], session: dict[str, Any], status: dict[str, np.ndarray],
                 start_us: int, m0: float) -> tuple[float, float]:
    """Where the estimate is fitted to the truth: the run's or the session's ``align``, or
    from arming to the mode switch."""
    align = run.get("align") or session.get("align")
    if not align or align == "auto":
        t_arm = armed_at(status, start_us)
        align = (t_arm if t_arm is not None else 0.0, m0)
    return float(align[0]), float(align[1])


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
    stretch = mode_stretch(vs, start_us, nav, min_s=settle_s + 1.0)
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
    wall = session.get("wall") or {}
    truth, extra = None, {}
    if is_rtk(session):
        cfg = session.get("rtk") or {}
        survey = wall.get("survey")
        if wall and not survey:
            raise ValueError("with truth: rtk the wall is wall.survey, points on its face "
                             "measured with the RTK rover (latitude, longitude, altitude)")
        rtk = read_rtk(topics, start_us, cfg.get("origin") or (survey[0] if survey else None),
                       str(cfg.get("altitude", "msl")))
        align = align_window(run, session, vs, start_us, m0)
        delay = cfg.get("delay_s", "auto")
        delay = fit_delay(topics, start_us, rtk, align) if delay == "auto" else float(delay)
        rtk = rtk.delayed(delay)
        psi, offset, rms, yaw_fitted = fit_frame(topics, start_us, rtk, align)
        hold_point = rotate_z(hold_point[None], psi)[0] + offset
        cover = covered(rtk, *window)
        if cover < RTK_MIN_COVERAGE:
            raise ValueError(f"RTK fixed for {100 * cover:.0f} % of the hold, under the "
                             f"{100 * RTK_MIN_COVERAGE:.0f} % a score needs")
        flown = armed_span(vs, start_us)
        row.update(truth="rtk", align_window_s=list(align), align_yaw_deg=math.degrees(psi),
                   align_rms_m=rms, align_yaw_fitted=yaw_fitted, rtk_delay_s=delay,
                   rtk_origin=list(rtk.origin), rtk_rate_hz=1.0 / rtk.period_s,
                   truth_coverage=cover,
                   truth_coverage_flight=covered(rtk, *flown))
        # Truth must be logged, not fused: with GPS in EKF2 the estimate follows the same
        # receiver, so a wrong solution is in both, and the moments it loses its fix (left
        # out of the score) are the estimate's worst. Not a reason to drop the run, but
        # never one to pass over in silence.
        fused = gps_fused(topics, start_us, (flown[0], window[1]))
        row["truth_independent"] = None if fused is None else not fused
        truth, extra = (rtk.source, rtk.t, rtk.pos), {"max_gap_s": rtk.max_gap_s}
        surface = None
        if survey:
            point, normal, fit = wall_from_survey(survey, rtk.origin, hold_point)
            surface = Surface(point, normal)
            row["wall_fit_rms_m"] = fit
    else:
        if GROUNDTRUTH not in topics and MOCAP in topics:
            align = align_window(run, session, vs, start_us, m0)
            psi, offset, rms = mocap_fit(topics, start_us, align)
            hold_point = rotate_z(hold_point[None], psi)[0] + offset
            row.update(align_window_s=list(align), align_yaw_deg=math.degrees(psi),
                       align_rms_m=rms)
        surface = Surface(wall["point"], wall["normal"]) if wall else None
    trace = trace_from_topics(topics, start_us, hold_point=hold_point, truth=truth,
                              allow_estimate=bool(session.get("allow_estimate", False)))
    gusts = [(float(a), float(b)) for a, b in run.get("gusts", [])]
    radius = session.get("contact_radius_m")
    if radius:
        extra["contact_radius_m"] = float(radius)
    metrics = compute_metrics(trace, surface, gusts, window, **extra)

    trip = None
    if controller == "structure_hold" and STATUS in topics:
        st = topics[STATUS]
        t = seconds(st, start_us)
        near = (t >= window[1] - 0.5) & (t <= window[1] + 2.0)
        # trip_reason is the LAST trip or refusal and clears on disarm only, so a refused
        # engage earlier in the flight (code 7, "landed") is still there during the hold.
        # The hold's own trip is the reason on the first handback sample inside the window
        # (the module enters STATE_HANDBACK as it trips).
        hb = near & (np.asarray(st["state"]) == STATE_HANDBACK) if "state" in st else near
        if hb.any():
            trip = TRIP_NAMES.get(int(np.asarray(st["trip_reason"])[hb][0]), "unknown")
        elif near.any():  # no handback sample logged (a thinned topic): the last code seen
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
    log = ULog(str(path), message_name_filter_list=TOPICS + RTK_TOPICS + [VEHICLE_STATUS,
                                                                           STATUS])
    topics = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    # A drone may carry two receivers; the RTK one is not always the first.
    topics.update({f"{SENSOR_GPS}#{d.multi_id}": d.data for d in log.data_list
                   if d.name == SENSOR_GPS and d.multi_id > 0})
    return topics, int(log.start_timestamp)


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
    b = result["batches"]
    rtk = [r for bt in b.values() for r in bt["rows"] if r.get("truth") == "rtk"]
    if rtk:
        cover = [r["truth_coverage"] for r in rtk]
        out += [f"Truth: RTK GPS, RTK-fixed samples only (`{rtk[0]['position_source']}`), "
                f"logged and not fused. Fixed for {100 * float(np.mean(cover)):.0f} % of the "
                f"scored hold time (worst hold {100 * min(cover):.0f} %); the rest is left "
                "out of every number.", ""]
        fused = sum(r["truth_independent"] is False for r in rtk)
        if fused:
            out += [f"**WARNING: EKF2 fused GPS on {fused} of {len(rtk)} scored holds.** On "
                    "those the RTK truth is not independent of the drone's own estimate: "
                    "a wrong solution is in both and does not show, and the moments the "
                    "receiver loses its fix, when the estimate suffers most, are the moments "
                    "left out of the score. Read their hold error as a best case. They are "
                    "listed below.", ""]
    for batch in b.values():
        out += [table(batch), ""]
    if len(b) == 2:
        out += [compare(b["tuned_px4"], b["structure_hold"]), ""]
    early = [r for bt in b.values() for r in bt["rows"] if r["ended_early"]]
    for r in early:
        out.append(f"- ended early: {r['file']} ({r['controller']}, {r['scenario']} run "
                   f"{r['seed']}), window {r['window_s'][0]:.1f} to {r['window_s'][1]:.1f} s, "
                   f"trip {r['trip_reason']}, next nav_state {r['next_nav_state']}")
    for r in rtk:
        who = f"{r['file']} ({r['controller']}, {r['scenario']} run {r['seed']})"
        if r["truth_coverage"] < COVERAGE_WARN:
            out.append(f"- RTK fixed for {100 * r['truth_coverage']:.0f} % of the hold: {who}; "
                       "the rest of that hold is not scored")
        if r["truth_independent"] is False:
            out.append(f"- GPS was fused: {who}: the RTK truth is not independent of the "
                       "estimate on this hold")
        elif r["truth_independent"] is None:
            out.append(f"- could not check whether GPS was fused: {who} (the log has no "
                       "estimator_status_flags)")
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
    # The one-page report reads the same numbers, as session_summary.json holds them.
    plain = json.loads(json.dumps(result, default=str))
    (out / "benchmark_report.html").write_text(render(plain, git_commit()))
    print(text)
    print("one-page report:", out / "benchmark_report.html")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
