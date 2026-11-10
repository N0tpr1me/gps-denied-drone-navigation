"""Fly the Structure Hold module in Gazebo, GPS-free on the corrupted estimate, and score it.

The SIH flights (sitl_test.sh) prove the mode plumbing; this is the flight that judges the
policy: the x500 in the wall-and-slab world with wind (sim2real/gazebo), the VIO-like
estimate with dropouts through EKF2 (sim2real/estimate_bridge), and the acceptance
sequence of sitl_test.py (engage, hold, disengage, re-engage, watchdog trip, handback,
latched refusal, land) run inside the SITL image. Needs Docker and the image from
``sim2real/docker/run.sh build`` (rebuild after any change under sim2real/px4).

    python -m sim2real.px4.run_gazebo calm_0ms --profile clean --hold-s 60
    python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default --hold-s 60
    python -m sim2real.px4.run_gazebo wind_8ms --param SH_FB_EN=1        # fallback on
    python -m sim2real.px4.run_gazebo wind_8ms --param SH_CTRL=0         # cascade, regression
    python -m sim2real.px4.run_gazebo calm_0ms --rangefinder             # wall guard on

Writes ``runs/<time>_sitl_hold_<scenario>_<profile>_<commit>/`` with the same files as
``sim2real.estimate_bridge.run_estimate`` (traces, their verification, ``ekf.json``) plus:

- ``summary.json``: sitl_test.py's checks, with the hold checks re-judged on truth
  (``estimate_checks`` keeps sitl_test's verdicts on PX4's estimate);
- ``hold.json``: from the ulog, over the first engaged hold: ``eval.metrics`` on the TRUE
  position against the hold point the module captured, the share of ticks in the fallback
  and under the yaw guard, the stale time seen, EKF2's dead-reckoning share;
- ``hold_summary.json``: every check and number in one place, for experiments/LOG.md.

``--rangefinder`` has the bridge send a forward rangefinder against the scenario's wall
(``bridge.Rangefinder``), which the x500 model lacks, so the module's wall guard
(``SH_WALL_MIN``) acts; ``hold.json`` then reports what the guard did in each engaged
stretch, and for every flight the closest the truth came to the wall after the scored hold
(the re-engage, trip and landing, where seed 3's strikes happened) and over the landing
itself (``landing``), and over the guarded hover the trip hands back into (``hover``), and
for the re-engage where the truth was and how fast it was closing when the module took over
(``reengage``; compare closest approaches only next to that entry state). The
sequence lands through the module's Structure Land mode (``--land structure``, the default,
which keeps the wall guard down to touchdown) or stock Land (``--land stock``, the sequence
before 2026-10-28).

A scenario with a ``pass`` block (``wall1m_pass_wind_8ms``; docs/surface-passes.md "In
Gazebo") flies its first hold as a surface pass: the module's ``SH_PASS_*`` are set from the
scenario, the hold lasts the scenario's ``hover_s`` and starts at the scenario's
``engage_at_s`` on PX4's clock, and the score is made against the moving hold point (the one
captured at engage plus the pass offset the module logs), with the pass numbers of
``eval/pass_score.py`` in ``hold.json`` under ``pass``. Stock PX4 flies the same pass through
``sim2real.estimate_bridge.run_estimate``.

Exit code 0 when the flight passed its checks, both traces verify, GPS was never used and
EKF2 fused the vision stream (a profile with a ``gps`` block flies with GPS on instead, and
EKF2 must have used it: ``run_estimate.gps_as_expected``). For the benchmark,
``eval.baselines.run_baseline --runner hold`` flies this over the baseline's scenarios and
seeds and tabulates it next to the tuned-PX4 rows.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.frames import ned_to_enu
from eval.metrics import Surface, compute_metrics
from eval.pass_score import moving_trace, pass_numbers
from eval.ulog import gazebo_reference, reference_numbers, trace_from_topics
from eval.ulog import load_topics as load_trace_topics
from sim2real.estimate_bridge.bridge import Rangefinder
from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.params import to_environment
from sim2real.estimate_bridge.run_estimate import DEFAULT_PARAMS, gps_as_expected, px4_params_for
from sim2real.estimate_bridge.ulog_checks import analyse
from sim2real.estimate_bridge.verify_trace import verify as verify_estimate
from sim2real.gazebo.params_io import write_params
from sim2real.gazebo.run_scenario import (
    DEFAULT_IMAGE,
    WATCH_IMAGE,
    WORLD,
    git_commit,
    isolation_options,
    make_run_dir,
    params_summary,
    watch_options,
    watch_url,
)
from sim2real.gazebo.scenario import load_scenario, sample_scenario
from sim2real.gazebo.verify_trace import verify as verify_wind

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
STATUS_TOPIC = "structure_hold_status"
# structure_hold_status.state (msg/StructureHoldStatus.msg); sitl_test.py has the same
# values but needs pymavlink, which the host and CI do not have.
STATE_STANDBY, STATE_ENGAGED, STATE_HANDBACK, STATE_LANDING, STATE_HOVER = 1, 2, 3, 4, 5


def docker_command(image: str, out: Path, pose: str, altitude: float, hold_s: float,
                   settle_s: float, drift_tol: float, params: list[str], name: str,
                   env: dict[str, str], watch: bool = False, land: str = "structure",
                   cpuset: str | None = None, world: str = WORLD,
                   flight_args: list[str] | None = None) -> list[str]:
    """``flight_args``: more sitl_test.py options (a scenario's surface pass)."""
    cmd = ["docker", "run", "--rm", "--name", name, *isolation_options(name, cpuset),
           "-v", f"{out}:/out", "-v", f"{REPO_ROOT}:/repo:ro",
           "-e", f"PX4_GZ_WORLD={world}", "-e", f"PX4_GZ_MODEL_POSE={pose}",
           "-e", "STRUCTURE_HOLD_WIND_PARAMS=/out/wind_params.txt",
           "-e", "STRUCTURE_HOLD_WIND_TRACE=/out/wind_trace.csv",
           "-e", "ESTIMATE_BRIDGE_PROFILE=/out/estimate_config.yaml"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    if watch:
        cmd += watch_options()
    cmd += [image, "hold-test", "--altitude", str(altitude), "--hold-s", str(hold_s),
            "--settle-s", str(settle_s), "--drift-tol", str(drift_tol), "--land", land]
    for p in params:
        cmd += ["--param", p]
    return cmd + list(flight_args or [])


# ---------------------------------------------------------------------------------------
# The hold, from the module's own status topic


def load_status(ulog_path: str | Path) -> tuple[dict[str, np.ndarray] | None, int]:
    from pyulog import ULog

    log = ULog(str(ulog_path), message_name_filter_list=[STATUS_TOPIC])
    for d in log.data_list:
        if d.name == STATUS_TOPIC and d.multi_id == 0:
            return d.data, int(log.start_timestamp)
    return None, int(log.start_timestamp)


def hold_window(status: dict[str, np.ndarray], start_us: int, settle_s: float = 2.0,
                which: int = 0) -> tuple[float, float] | None:
    """(t0, t1) in log seconds of the ``which``-th engaged stretch, trimmed by ``settle_s``
    at the start so the engage transient is not scored."""
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    engaged = np.asarray(status["state"]) == STATE_ENGAGED
    edges = np.flatnonzero(np.diff(np.concatenate(([False], engaged, [False])).astype(int)))
    starts, ends = edges[0::2], edges[1::2]
    if which >= starts.size:
        return None
    t0, t1 = float(t[starts[which]]) + settle_s, float(t[ends[which] - 1])
    return (t0, t1) if t1 > t0 else None


def hold_point_enu(status: dict[str, np.ndarray], start_us: int,
                   window: tuple[float, float]) -> np.ndarray:
    """The hold point the module captured at engage (NED in the status, ENU here)."""
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    i = int(np.flatnonzero(t >= window[0] - 1e-9)[0])
    ned = np.array([status[f"setpoint_ned[{k}]"][i] for k in range(3)], dtype=float)
    return ned_to_enu(ned)


def pass_reference_enu(status: dict[str, np.ndarray], start_us: int,
                       window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """The moving hold point of a surface pass over the first engaged stretch: (times,
    points in ENU), or None for a build without the pass fields or a hold without a pass.

    The point is the one the module held at the window's start plus what the pass has moved
    it since (``pass_offset_ned``), not the logged setpoint itself: the wall guard and the
    keep-out move that too, and a hold point pushed off the wall must still count as error,
    as it does for a plain hold."""
    if "pass_offset_ned[0]" not in status:
        return None
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    offset = np.stack([np.asarray(status[f"pass_offset_ned[{k}]"], dtype=float)
                       for k in range(3)], axis=1)
    a, b = engaged_stretches(status)[0]
    if not np.any(np.linalg.norm(offset[a:b], axis=1) > 0):
        return None
    i = int(np.flatnonzero(t >= window[0] - 1e-9)[0])
    ref = hold_point_enu(status, start_us, window) + ned_to_enu(offset[a:b] - offset[i])
    return t[a:b], ref


def hold_stats(status: dict[str, np.ndarray], start_us: int,
               window: tuple[float, float]) -> dict[str, Any]:
    """What the module did during the window: fallback and guard shares, stale time, actions."""
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    m = (t >= window[0]) & (t <= window[1]) & (np.asarray(status["state"]) == STATE_ENGAGED)
    n = int(m.sum())
    if n == 0:
        return {"ticks": 0}
    fb = np.asarray(status["fallback_active"])[m].astype(bool)
    guarded = np.asarray(status["yaw_guarded"])[m].astype(bool)
    stale = np.asarray(status["stale_s"])[m].astype(float)
    err = np.stack([np.asarray(status[f"position_error_ned[{k}]"])[m] for k in range(3)], 1)
    actions = np.stack([np.asarray(status[f"action[{k}]"])[m] for k in range(4)], 1)
    # Fallback episodes: runs of consecutive fallback ticks.
    starts = np.flatnonzero(np.diff(np.concatenate(([0], fb.astype(int)))) == 1)
    return {
        "ticks": n,
        "controller": int(np.asarray(status["controller"])[m][0]),
        "fallback_share": float(fb.mean()),
        "fallback_episodes": int(starts.size),
        "yaw_guard_share": float(guarded.mean()),
        "stale_max_s": float(stale.max()),
        "stale_share_over_0p5s": float((stale > 0.5).mean()),
        "estimate_error_xy_max_m": float(np.hypot(err[:, 0], err[:, 1]).max()),
        "action_abs_mean": [float(v) for v in np.abs(actions).mean(0)],
        "action_saturated_share": float((np.abs(actions[:, :3]) > 0.99).mean()),
        "loop_time_max_us": int(np.asarray(status["loop_time_us"])[m].max()),
    }


def engaged_stretches(status: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """[start, end) indices of every engaged stretch in the status topic."""
    engaged = np.asarray(status["state"]) == STATE_ENGAGED
    edges = np.flatnonzero(np.diff(np.concatenate(([False], engaged, [False])).astype(int)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=True))


def wall_guard_stats(status: dict[str, np.ndarray], start_us: int) -> list[dict[str, Any]]:
    """Per engaged stretch: how often a forward range was used, the closest wall and hold
    point clearance it saw, and how far the guard moved the hold point back. Empty for a
    build without the guard's status fields."""
    if "standoff_shift_m" not in status:
        return []
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    wall = np.asarray(status["wall_distance_m"], dtype=float)
    clear = np.asarray(status["hold_clearance_m"], dtype=float)
    shift = np.asarray(status["standoff_shift_m"], dtype=float)
    out = []
    for a, b in engaged_stretches(status):
        ok = wall[a:b] >= 0
        out.append({
            "t_s": [float(t[a]), float(t[b - 1])],
            "range_share": float(ok.mean()),
            "wall_min_m": float(wall[a:b][ok].min()) if ok.any() else None,
            "clearance_min_m": float(clear[a:b][ok].min()) if ok.any() else None,
            "shift_m": float(shift[a:b].max()),
        })
    return out


def reengage_entries(t: np.ndarray, pos: np.ndarray, surface: Surface,
                     status: dict[str, np.ndarray], start_us: int,
                     window_s: float = 3.0, lead_s: float = 0.2) -> list[dict[str, Any]]:
    """Per engaged stretch after the first (the sequence's re-engage): where the truth was
    when the module took over, how fast it was closing on the wall then, and the closest it
    came in the next ``window_s``.

    The re-engage is timed on the wall clock after 5 s of stock Position mode chasing a
    jumped estimate, so it catches the vehicle at a different place and speed in every
    flight. The closest approach after it is only comparable between controllers next to
    this entry state: ``overrun_m`` (entry clearance minus the closest) is what the module
    let the vehicle travel after it took over."""
    ts = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    d = surface.distance(pos)
    out = []
    for a, b in engaged_stretches(status)[1:]:
        t0 = float(ts[a])
        if t0 - lead_s < t[0] or t0 > t[-1]:
            continue
        entry = float(np.interp(t0, t, d))
        closing = (float(np.interp(t0 - lead_s, t, d)) - entry) / lead_s
        t1 = min(t0 + window_s, float(ts[b - 1]))
        inside = (t >= t0) & (t <= t1)
        closest = min(entry, float(d[inside].min())) if inside.any() else entry
        out.append({"t_s": t0, "entry_clearance_m": entry, "closing_mps": closing,
                    "closest_m": closest, "overrun_m": entry - closest})
    return out


def landing_stretch(status: dict[str, np.ndarray], start_us: int,
                    state: int = STATE_LANDING) -> dict[str, Any] | None:
    """The Structure Land stretch (the last one; ``state=STATE_HOVER`` for Structure Hover
    after the trip): its window in log seconds and how far the wall guard moved the xy
    setpoint back. None when the flight never entered that mode."""
    landing = np.asarray(status["state"]) == state
    if not landing.any():
        return None
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    edges = np.flatnonzero(np.diff(np.concatenate(([False], landing, [False])).astype(int)))
    a, b = int(edges[-2]), int(edges[-1])
    shift = np.asarray(status.get("standoff_shift_m", np.zeros(len(t))), dtype=float)
    return {"t_s": [float(t[a]), float(t[b - 1])], "shift_m": float(shift[a:b].max())}


def score_hold(ulog: Path, sc, wind, altitude: float, settle_s: float
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """EKF report and eval.metrics on the TRUE position over the first hold, against the
    module's hold point with the estimator's height offset on the ground taken out
    (``eval.ulog.gazebo_reference``)."""
    status, start_us = load_status(ulog)
    if status is None:
        raise ValueError(f"{ulog} has no {STATUS_TOPIC}")
    window = hold_window(status, start_us, settle_s)
    if window is None:
        raise ValueError("no engaged stretch in the status topic")
    ekf = analyse(ulog, altitude, window)
    topics, start_us2 = load_trace_topics(ulog)
    hold_point = hold_point_enu(status, start_us, window)
    logged = trace_from_topics(topics, start_us2, hold_point=hold_point)
    moving = pass_reference_enu(status, start_us, window)
    if moving is not None:
        logged = moving_trace(logged, *moving)
    trace = gazebo_reference(logged, topics, start_us2, window, fixed=moving is None)
    surface = Surface(sc.wall_point, sc.wall_normal_toward)
    surface2 = (Surface(sc.surface2_point, sc.surface2_normal_toward)
                if sc.surface2_normal_toward is not None else None)
    gusts = [(float(s), float(s) + float(d))
             for s, d in zip(wind.gust_start[0], wind.gust_duration[0], strict=True)]
    metrics = compute_metrics(trace, surface, gusts, window, surface2=surface2)
    metrics.update(reference_numbers(trace, logged, window))
    hold = {
        "window_s": list(window),
        # The module's own hold point, as logged (the score moves it by the ground offset).
        "hold_point_enu": [float(v) for v in hold_point],
        "gusts_s": gusts,
        "module": hold_stats(status, start_us, window),
        "metrics": metrics,
        "wall_guard": wall_guard_stats(status, start_us),
    }
    # The hold checks on truth: sitl_test.py judges drift on PX4's estimate, which a
    # corrupted profile (vio_at_engage) jumps by design.
    held = (trace.t >= window[0]) & (trace.t <= window[1])
    if held.any():
        err = trace.pos[held] - trace.pos_sp[held]  # the hold point, moving or not
        hold["truth_drift"] = {"xy_max_m": float(np.hypot(err[:, 0], err[:, 1]).max()),
                               "z_max_m": float(np.abs(err[:, 2]).max())}
    if moving is not None:
        hold["pass"] = pass_numbers(trace, window, sc.wall_normal_toward)
        state = np.asarray(status["pass_state"])
        hold["pass"]["state_last"] = int(state[engaged_stretches(status)[0][1] - 1])
    after = trace.t > window[1]
    if after.any():
        d = surface.distance(trace.pos[after])
        i = int(np.argmin(d))
        hold["after_hold"] = {"min_clearance_m": float(d[i]), "t_s": float(trace.t[after][i])}
    hold["reengage"] = reengage_entries(trace.t, trace.pos, surface, status, start_us)
    for key, state in (("hover", STATE_HOVER), ("landing", STATE_LANDING)):
        stretch = landing_stretch(status, start_us, state)
        if stretch is not None:
            during = (trace.t >= stretch["t_s"][0]) & (trace.t <= stretch["t_s"][1])
            if during.any():
                stretch["min_clearance_m"] = float(surface.distance(trace.pos[during]).min())
            hold[key] = stretch
    return ekf, hold


# ---------------------------------------------------------------------------------------


def judge_hold_on_truth(summary: dict[str, Any], hold: dict[str, Any], drift_tol: float) -> None:
    """Replace sitl_test.py's estimate-based hold checks with the same tolerance on truth.
    The estimate's own verdicts stay in the summary under ``estimate_checks``."""
    drift = hold.get("truth_drift")
    checks = summary.get("checks")
    if drift is None or checks is None or "error" in summary:
        return  # a sequence that raised (sitl_test.py "error") failed, whatever the truth says
    keys = ("position_held", "altitude_held")
    flown = hold.get("pass")
    if flown is not None:
        keys += ("pass_standoff_held", "pass_reached_end")
    summary["estimate_checks"] = {k: checks[k] for k in keys if k in checks}
    checks["position_held"] = drift["xy_max_m"] <= drift_tol
    checks["altitude_held"] = drift["z_max_m"] <= drift_tol
    if flown is not None:
        # The same two pass checks on truth: the standoff kept, and the vehicle at the end
        # of the whole length (the point's travel is the module's; the vehicle's is ours).
        length = abs(float((summary.get("pass") or {}).get("length_m", 0.0)))
        if "pass_standoff_held" in checks:
            checks["pass_standoff_held"] = flown["standoff_max_m"] <= drift_tol
        if "pass_reached_end" in checks:
            checks["pass_reached_end"] = (abs(flown["travel_m"] - length) <= 0.05
                                          and flown["end_error_m"] <= drift_tol)
    summary["passed"] = all(checks.values())


def assemble(report: dict[str, Any]) -> dict[str, Any]:
    """The pass line: the flight's own checks, both traces, GPS as the profile says (off,
    or on and fused), vision fused."""
    report["ok"] = bool(report.get("flight_ok") and report.get("estimate_ok")
                        and report.get("wind_ok") and report.get("gps_ok")
                        and report.get("vision_fused"))
    return report


def rescore(out: Path, drift_tol: float = 1.0) -> dict[str, Any]:
    """Score a flown run folder again from its ulog (after a change to the scoring):
    rewrite ``hold.json`` and the hold, metrics and truth-judged checks in
    ``hold_summary.json``, return the summary."""
    config = yaml.safe_load((out / "config.yaml").read_text())
    sc = dataclasses.replace(load_scenario(config["scenario"]), seed=config["scenario_seed"])
    report = json.loads((out / "hold_summary.json").read_text())
    _, hold = score_hold(out / "flight.ulg", sc, sample_scenario(sc), sc.hover.altitude, 2.0)
    (out / "hold.json").write_text(json.dumps(hold, indent=2) + "\n")
    report["hold"] = hold
    report["metrics"] = hold["metrics"]
    flight = report.get("flight")
    if flight is not None and "checks" in flight:
        # The estimate's own verdicts back first, or judging again would file the truth's
        # verdicts of the first scoring as the estimate's.
        flight["checks"].update(flight.get("estimate_checks", {}))
        judge_hold_on_truth(flight, hold, drift_tol)
        report["flight_ok"] = bool(flight.get("passed"))
        (out / "summary.json").write_text(json.dumps(flight, indent=2) + "\n")
    assemble(report)
    (out / "hold_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", help="wall scenario name (sim2real/gazebo/scenarios) or path")
    ap.add_argument("--profile", default="vio_default", help="estimate profile name or path")
    ap.add_argument("--seed", type=int, help="override the profile's seed")
    ap.add_argument("--scenario-seed", type=int, help="override the scenario's (wind) seed")
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS,
                    help="PX4 .params file (the tuned baseline's for a fair comparison)")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="module parameter set before the flight (SH_FB_EN=1, SH_CTRL=0, ...)")
    ap.add_argument("--image", help=f"SITL image (default {DEFAULT_IMAGE}, or {WATCH_IMAGE} "
                                     "with --watch)")
    ap.add_argument("--watch", action="store_true",
                    help="watch the flight live in a browser (needs run.sh build-watch)")
    ap.add_argument("--hold-s", type=float,
                    help="time in Structure Hold (default 60 s; a scenario with a pass: its "
                         "hover_s)")
    ap.add_argument("--settle-s", type=float, default=5.0, help="stock-mode hover times")
    ap.add_argument("--drift-tol", type=float, default=1.0,
                    help="hold drift tolerance, judged on truth here (sitl_test applies it "
                         "to PX4's estimate; that verdict stays in estimate_checks)")
    ap.add_argument("--rangefinder", action="store_true",
                    help="bridge sends a forward rangefinder against the wall (wall guard)")
    ap.add_argument("--land", choices=("structure", "stock"), default="structure",
                    help="end the sequence in Structure Land (wall guard to touchdown) or "
                         "stock Land")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--container-name", default="px4-sitl-hold")
    ap.add_argument("--cpuset", help="host CPUs for this flight's container, e.g. 0-3 "
                                     "(parallel batches: eval.baselines.run_baseline --jobs)")
    args = ap.parse_args(argv)
    if args.image is None:
        args.image = WATCH_IMAGE if args.watch else DEFAULT_IMAGE

    sc = load_scenario(args.scenario)
    if args.scenario_seed is not None:
        sc = dataclasses.replace(sc, seed=args.scenario_seed)
    if args.hold_s is None:
        args.hold_s = sc.hover.hover_s if sc.surface_pass is not None else 60.0
    flight_args: list[str] = []
    if sc.surface_pass is not None:
        # The estimate's dropouts hold the pass back (PassPath, hold), so it may finish
        # later than the trapezoid says; the hold after it is there to absorb that.
        flight_args = [*sc.surface_pass.flight_args(), "--pass-slack-s",
                       str(max(args.hold_s - sc.surface_pass.start_s - 12.0, 2.0))]
    prof = load_profile(args.profile)
    if args.seed is not None:
        prof = prof.with_seed(args.seed)
    prof = prof.with_anchor_altitude(sc.hover.altitude)  # only an "altitude" anchor changes
    if args.rangefinder:
        prof = dataclasses.replace(prof, rangefinder=Rangefinder(sc.wall_point,
                                                                 sc.wall_normal_toward))
    px4_params = px4_params_for(prof, args.params)
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = make_run_dir(args.runs_dir, f"{stamp}_sitl_hold_{sc.name}_{prof.name}_{commit}")

    wind = sample_scenario(sc)
    write_params(out / "wind_params.txt", wind)
    (out / "estimate_config.yaml").write_text(yaml.safe_dump(prof.to_dict(), sort_keys=False))
    config = {
        "scenario": args.scenario, "name": sc.name, "scenario_seed": sc.seed,
        "profile": prof.to_dict(), "px4_params": str(args.params), "module_params": args.param,
        "commit": commit, "image": args.image, "world": sc.world, "spawn": sc.spawn.__dict__,
        "wall_clearance_m": sc.wall_clearance, "altitude_m": sc.hover.altitude,
        "hold_s": args.hold_s, "settle_s": args.settle_s, "disturbance": params_summary(wind),
    }
    if sc.surface_pass is not None:
        config["pass"] = dataclasses.asdict(sc.surface_pass)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = docker_command(args.image, out, sc.spawn.px4_pose(), sc.hover.altitude, args.hold_s,
                         args.settle_s, args.drift_tol, args.param, args.container_name,
                         to_environment(px4_params), watch=args.watch, land=args.land,
                         cpuset=args.cpuset, world=sc.world, flight_args=flight_args)
    print("run folder:", out)
    print("$", " ".join(cmd), flush=True)
    if args.watch:
        print("watch live:", watch_url(), "(the view appears once Gazebo is up)", flush=True)
    flight_status = subprocess.run(cmd).returncode

    report: dict[str, Any] = {
        "scenario": sc.name, "profile": prof.name, "seed": prof.seed, "commit": commit,
        "module_params": args.param, "land": args.land, "run": str(out.relative_to(REPO_ROOT)),
        "flight_exit_code": flight_status,
        # For run_baseline's rows: how long the scored hold was meant to last.
        "hold_s": args.hold_s,
    }
    ulog = out / "flight.ulg"
    summary_path = out / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if ulog.exists() and "error" not in summary:
            from sim2real.px4.sitl_test import check_ulog  # needs pymavlink (in [dev])

            check_ulog(ulog, summary)  # the status-topic checks sitl_test.sh adds
            summary["passed"] = all(summary["checks"].values())
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        report["flight"] = summary
        report["flight_ok"] = bool(summary.get("passed"))
    else:
        report["flight_ok"] = False
        report["flight_error"] = "no summary.json written by sitl_test.py"

    if (out / "estimate_trace.csv").exists():
        report["estimate"] = verify_estimate(prof, out / "estimate_trace.csv")
        (out / "estimate_verify.json").write_text(json.dumps(report["estimate"], indent=2) + "\n")
        report["estimate_ok"] = report["estimate"]["ok"]
    else:
        report["estimate_ok"] = False
        report["estimate_error"] = "no estimate_trace.csv written by the bridge"

    if (out / "wind_trace.csv").exists():
        report["wind"] = verify_wind(out / "wind_params.txt", out / "wind_trace.csv")
        (out / "wind_verify.json").write_text(json.dumps(report["wind"], indent=2) + "\n")
        report["wind_ok"] = report["wind"]["ok"]
    else:
        report["wind_ok"] = False

    if ulog.exists():
        try:
            ekf, hold = score_hold(ulog, sc, wind, sc.hover.altitude, 2.0)
            (out / "ekf.json").write_text(json.dumps(ekf, indent=2) + "\n")
            (out / "hold.json").write_text(json.dumps(hold, indent=2) + "\n")
            report["ekf"] = {k: ekf[k] for k in ("gps_free", "vision_fused", "estimate_error",
                                                 "vision", "modes", "window_s", "aiding")}
            report["hold"] = hold
            report["metrics"] = hold["metrics"]  # where run_baseline's rows read it
            report["gps_free"] = ekf["gps_free"]
            report["gps_ok"] = gps_as_expected(prof, ekf)
            report["vision_fused"] = ekf["vision_fused"]
            if "flight" in report:
                judge_hold_on_truth(report["flight"], hold, args.drift_tol)
                report["flight_ok"] = bool(report["flight"].get("passed"))
                summary_path.write_text(json.dumps(report["flight"], indent=2) + "\n")
        except Exception as exc:  # the flight already happened; record why scoring failed
            report["ulog_error"] = f"{type(exc).__name__}: {exc}"
            report["gps_free"] = report["gps_ok"] = report["vision_fused"] = False
    else:
        report["ulog_error"] = "no flight.ulg"
        report["gps_free"] = report["gps_ok"] = report["vision_fused"] = False

    assemble(report)
    (out / "hold_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
