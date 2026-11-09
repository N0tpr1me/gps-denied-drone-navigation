"""Fly a wall-and-slab scenario GPS-free on a corrupted estimate, then score it.

    python -m sim2real.estimate_bridge.run_estimate calm_0ms --profile clean --hover-s 60
    python -m sim2real.estimate_bridge.run_estimate wind_8ms --profile vio_default --seed 1

Writes ``runs/<time>_sitl_estimate_<scenario>_<profile>_<commit>/`` (ignored by git):

- ``config.yaml``: scenario, profile, seed, PX4 params, commit, image
- ``estimate_config.yaml``: the profile the bridge ran (replayable by ``verify_trace``)
- ``wind_params.txt``, ``wind_trace.csv``, ``wind_verify.json``: as ``sim2real.gazebo``
- ``estimate_trace.csv``, ``bridge.log``: what the bridge fed PX4, and its console
- ``summary.json``, ``flight.ulg``, ``px4.log``: the hover test (its altitude and drift
  numbers come from PX4's own, corrupted, estimate; the ulog scores below use truth)
- ``estimate_verify.json``: the trace replayed through ``envs.estimate``
- ``ekf.json``: EKF2 aiding, estimate-vs-truth error, vision gaps, modes (``ulog_checks``)
- ``metrics.json``: ``eval.metrics`` on the TRUE position over the hover window
- ``estimate_summary.json``: pass/fail of every check, for experiments/LOG.md

A scenario with a ``pass`` block (docs/surface-passes.md "In Gazebo") is flown by
``sim2real/px4/pass_flight.py`` instead of the hover test: stock PX4 in Offboard on
position, velocity and acceleration setpoints along the same path the module flies. The
scored window is then the Offboard stretch (2 s after the switch, as the hold runner trims
its engage) and the reference is PX4's own position setpoint, moving; ``metrics.json`` gains
the pass numbers of ``eval/pass_score.py`` under ``pass``. The flight's tracking checks are
re-judged on truth, as the hold runner's are.

Needs Docker, the image from ``sim2real/docker/run.sh build`` and ``pip install -e .[dev]``.
Exit code 0 when the hover passed, both traces verify, GPS was never used and EKF2 fused
the vision stream. A profile with a ``gps`` block (``gps.py``) flies with GPS on instead:
``gps_on.params`` is laid over the PX4 set, and the check becomes that EKF2 did fuse it
(``gps_ok`` either way; ``gps_free`` stays what it was).
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

from eval.metrics import Surface, compute_metrics
from eval.pass_score import moving_trace, pass_numbers
from eval.ulog import SETPOINT, read_ulog, trace_from_topics
from eval.ulog import load_topics as load_trace_topics
from eval.ulog import ned_to_enu as ulog_ned_to_enu
from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.params import read_params, to_environment
from sim2real.estimate_bridge.ulog_checks import STATUS, analyse, hover_window, load_topics
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
DEFAULT_PARAMS = HERE / "ekf2_ev.params"
GPS_ON_PARAMS = HERE / "gps_on.params"
VISION_YAW_PARAMS = HERE / "vision_yaw.params"
NAV_OFFBOARD = 14  # vehicle_status.nav_state (ulog_checks.NAV_STATE)
PASS_SETTLE_S = 2.0  # the switch transient left out of the scored pass, as the hold runner's
PASS_DRIFT_TOL_M = 1.0  # run_gazebo's --drift-tol, on truth


def px4_params_for(profile, params_path: Path) -> dict[str, int | float]:
    """The PX4 set of a flight: the file given, with ``gps_on.params`` over it when the
    profile carries the scripted GPS (the bridge is then the receiver EKF2 fuses), and
    ``vision_yaw.params`` when it carries a scripted heading error (EKF2 takes its heading
    from the vision stream's attitude)."""
    params = read_params(params_path)
    if profile.gps is not None:
        params.update(read_params(GPS_ON_PARAMS))
    if getattr(profile, "vision_yaw", None) is not None:
        params.update(read_params(VISION_YAW_PARAMS))
    return params


def gps_as_expected(profile, ekf: dict[str, Any]) -> bool:
    """GPS-free flights must never have used GNSS; GPS-on flights must have."""
    used = not ekf["gps_free"]
    return used if profile.gps is not None else not used


def docker_command(image: str, out: Path, pose: str, altitude: float, hover_s: float,
                   name: str, env: dict[str, str], watch: bool = False,
                   cpuset: str | None = None, world: str = WORLD,
                   pass_args: list[str] | None = None) -> list[str]:
    """``pass_args``: the scenario's surface pass as pass_flight.py options; the flight is
    then that script (Offboard for ``hover_s``) instead of the hover test."""
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
    if pass_args is not None:
        return cmd + [image, "with-sitl", "python3", "/repo/sim2real/px4/pass_flight.py",
                      "--out", "/out/summary.json", "--altitude", str(altitude),
                      "--hold-s", str(hover_s), *pass_args]
    cmd += [image, "hover-test", "--altitude", str(altitude), "--hover-s", str(hover_s),
            "--local-position-only"]
    return cmd


def offboard_window(topics, start_us: int, settle_s: float = PASS_SETTLE_S
                    ) -> tuple[float, float] | None:
    """(t0, t1) in log seconds of the first Offboard stretch, trimmed by ``settle_s`` at the
    start: the stock pass flight's scored window, as the first engaged stretch is the
    module's."""
    if STATUS not in topics:
        return None
    t = (np.asarray(topics[STATUS]["timestamp"], dtype=np.int64) - start_us) * 1e-6
    on = np.asarray(topics[STATUS]["nav_state"], dtype=int) == NAV_OFFBOARD
    edges = np.flatnonzero(np.diff(np.concatenate(([False], on, [False])).astype(int)))
    if edges.size == 0:
        return None
    # The stretch ends at the first status sample after it (the commanded Land).
    a, b = int(edges[0]), int(edges[1])
    t0, t1 = float(t[a]) + settle_s, float(t[min(b, t.size - 1)])
    return (t0, t1) if t1 > t0 else None


def score_pass_ulog(ulog: Path, sc, params, altitude: float
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """As ``score_ulog`` for the stock pass flight: the Offboard stretch against PX4's own
    position setpoint, which moves (linear between its 100 ms log samples)."""
    topics, start_us = load_topics(ulog)
    window = offboard_window(topics, start_us)
    if window is None:
        raise ValueError("the flight never entered Offboard")
    ekf = analyse(ulog, altitude, window)
    trace_topics, start_us2 = load_trace_topics(ulog)
    if SETPOINT not in trace_topics:
        raise ValueError(f"log has no {SETPOINT}")
    sp = trace_topics[SETPOINT]
    sp_t = (np.asarray(sp["timestamp"], dtype=np.int64) - start_us2) * 1e-6
    sp_pos = ulog_ned_to_enu(sp["x"], sp["y"], sp["z"])
    # Only the Offboard setpoints: before the switch the setpoint is stock Hold's.
    flown = (sp_t >= window[0] - PASS_SETTLE_S) & (sp_t <= window[1])
    hold_point = sp_pos[flown][np.all(np.isfinite(sp_pos[flown]), axis=1)][0]
    trace = moving_trace(trace_from_topics(trace_topics, start_us2, hold_point=hold_point),
                         sp_t[flown], sp_pos[flown])
    surface = Surface(sc.wall_point, sc.wall_normal_toward)
    surface2 = (Surface(sc.surface2_point, sc.surface2_normal_toward)
                if sc.surface2_normal_toward is not None else None)
    gusts = [(float(s), float(s) + float(d))
             for s, d in zip(params.gust_start[0], params.gust_duration[0], strict=True)]
    metrics = compute_metrics(trace, surface, gusts, window, surface2=surface2)
    metrics["window_s"] = list(window)
    metrics["gusts_s"] = gusts
    metrics["pass"] = pass_numbers(trace, window, sc.wall_normal_toward)
    return ekf, metrics


def judge_pass_on_truth(summary: dict[str, Any], metrics: dict[str, Any],
                        drift_tol: float = PASS_DRIFT_TOL_M) -> None:
    """Replace pass_flight.py's tracking checks, made on PX4's estimate, with the same
    tolerance on truth (``run_gazebo.judge_hold_on_truth`` does it for the module). The
    estimate's own verdicts stay in the summary under ``estimate_checks``."""
    checks = summary.get("checks")
    flown = metrics.get("pass")
    if checks is None or flown is None or "error" in summary:
        return
    keys = ("pass_tracked", "pass_reached_end")
    summary["estimate_checks"] = {k: checks[k] for k in keys if k in checks}
    length = abs(float((summary.get("pass") or {}).get("length_m", 0.0)))
    checks["pass_tracked"] = flown["error_max_m"] <= drift_tol
    checks["pass_reached_end"] = (abs(flown["travel_m"] - length) <= 0.05
                                  and flown["end_error_m"] <= drift_tol)
    summary["passed"] = all(checks.values())


def score_ulog(ulog: Path, sc, params, altitude: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """EKF health report and eval.metrics on the true position, over the hover window."""
    topics, start_us = load_topics(ulog)
    window = hover_window(topics, start_us, altitude)
    ekf = analyse(ulog, altitude, window)
    trace = read_ulog(ulog)
    surface = Surface(sc.wall_point, sc.wall_normal_toward)
    surface2 = (Surface(sc.surface2_point, sc.surface2_normal_toward)
                if sc.surface2_normal_toward is not None else None)
    gusts = [(float(s), float(s) + float(d))
             for s, d in zip(params.gust_start[0], params.gust_duration[0], strict=True)]
    metrics = compute_metrics(trace, surface, gusts, window, surface2=surface2)
    metrics["window_s"] = list(window) if window else None
    metrics["gusts_s"] = gusts
    return ekf, metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", help="wall scenario name (sim2real/gazebo/scenarios) or path")
    ap.add_argument("--profile", default="vio_default", help="estimate profile name or path")
    ap.add_argument("--seed", type=int, help="override the profile's seed")
    ap.add_argument("--scenario-seed", type=int, help="override the scenario's (wind) seed")
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS, help="PX4 .params file")
    ap.add_argument("--image", help=f"SITL image (default {DEFAULT_IMAGE}, or {WATCH_IMAGE} "
                                     "with --watch)")
    ap.add_argument("--watch", action="store_true",
                    help="watch the flight live in a browser (needs run.sh build-watch)")
    ap.add_argument("--hover-s", type=float, help="override the scenario's hover time")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--container-name", default="px4-sitl-estimate")
    ap.add_argument("--cpuset", help="host CPUs for this flight's container, e.g. 0-3 "
                                     "(parallel batches: eval.baselines.run_baseline --jobs)")
    args = ap.parse_args(argv)
    if args.image is None:
        args.image = WATCH_IMAGE if args.watch else DEFAULT_IMAGE

    sc = load_scenario(args.scenario)
    if args.scenario_seed is not None:
        sc = dataclasses.replace(sc, seed=args.scenario_seed)
    prof = load_profile(args.profile)
    if args.seed is not None:
        prof = prof.with_seed(args.seed)
    prof = prof.with_anchor_altitude(sc.hover.altitude)  # "altitude" anchor: the hover window
    px4_params = px4_params_for(prof, args.params)
    hover_s = args.hover_s if args.hover_s is not None else sc.hover.hover_s
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = make_run_dir(args.runs_dir, f"{stamp}_sitl_estimate_{sc.name}_{prof.name}_{commit}")

    wind = sample_scenario(sc)
    write_params(out / "wind_params.txt", wind)
    (out / "estimate_config.yaml").write_text(yaml.safe_dump(prof.to_dict(), sort_keys=False))
    config = {
        "scenario": args.scenario, "name": sc.name, "scenario_seed": sc.seed,
        "profile": prof.to_dict(), "px4_params": px4_params, "commit": commit,
        "image": args.image, "world": sc.world, "spawn": sc.spawn.__dict__,
        "wall_clearance_m": sc.wall_clearance, "altitude_m": sc.hover.altitude,
        "hover_s": hover_s, "disturbance": params_summary(wind),
    }
    flies_pass = sc.surface_pass is not None
    if flies_pass:
        config["pass"] = dataclasses.asdict(sc.surface_pass)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = docker_command(args.image, out, sc.spawn.px4_pose(), sc.hover.altitude, hover_s,
                         args.container_name, to_environment(px4_params), watch=args.watch,
                         cpuset=args.cpuset, world=sc.world,
                         pass_args=sc.surface_pass.flight_args() if flies_pass else None)
    print("run folder:", out)
    print("$", " ".join(cmd), flush=True)
    if args.watch:
        print("watch live:", watch_url(), "(the view appears once Gazebo is up)", flush=True)
    hover_status = subprocess.run(cmd).returncode

    report: dict[str, Any] = {
        "scenario": sc.name, "profile": prof.name, "seed": prof.seed, "commit": commit,
        "run": str(out.relative_to(REPO_ROOT)), "hover_exit_code": hover_status,
        "hover_ok": hover_status == 0,
    }
    if (out / "summary.json").exists():
        report["hover"] = json.loads((out / "summary.json").read_text())
    if flies_pass:
        # For run_baseline's rows: the mode the scored window is flown in, and its length.
        report["scored_mode"] = "Offboard"
        report["hold_s"] = hover_s

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

    ulog = out / "flight.ulg"
    if ulog.exists():
        try:
            score = score_pass_ulog if flies_pass else score_ulog
            ekf, metrics = score(ulog, sc, wind, sc.hover.altitude)
            (out / "ekf.json").write_text(json.dumps(ekf, indent=2) + "\n")
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if flies_pass and "hover" in report:
                judge_pass_on_truth(report["hover"], metrics)
                report["hover_ok"] = bool(report["hover"].get("passed"))
                (out / "summary.json").write_text(json.dumps(report["hover"], indent=2) + "\n")
            report["ekf"] = {k: ekf[k] for k in ("gps_free", "vision_fused", "estimate_error",
                                                 "vision", "modes", "window_s")}
            report["ekf"]["aiding"] = ekf["aiding"]
            report["metrics"] = metrics
            report["gps_free"] = ekf["gps_free"]
            report["gps_ok"] = gps_as_expected(prof, ekf)
            report["vision_fused"] = ekf["vision_fused"]
        except Exception as exc:  # the flight already happened; record why scoring failed
            report["ulog_error"] = f"{type(exc).__name__}: {exc}"
            report["gps_free"] = report["gps_ok"] = report["vision_fused"] = False
    else:
        report["ulog_error"] = "no flight.ulg"
        report["gps_free"] = report["gps_ok"] = report["vision_fused"] = False

    report["ok"] = bool(report["hover_ok"] and report["estimate_ok"] and report["wind_ok"]
                        and report["gps_ok"] and report["vision_fused"])
    (out / "estimate_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
