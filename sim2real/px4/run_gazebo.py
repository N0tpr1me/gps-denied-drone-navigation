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

- ``summary.json``: sitl_test.py's checks (drift numbers there are PX4's own estimate);
- ``hold.json``: from the ulog, over the first engaged hold: ``eval.metrics`` on the TRUE
  position against the hold point the module captured, the share of ticks in the fallback
  and under the yaw guard, the stale time seen, EKF2's dead-reckoning share;
- ``hold_summary.json``: every check and number in one place, for experiments/LOG.md.

``--rangefinder`` has the bridge send a forward rangefinder against the scenario's wall
(``bridge.Rangefinder``), which the x500 model lacks, so the module's wall guard
(``SH_WALL_MIN``) acts; ``hold.json`` then reports what the guard did in each engaged
stretch, and for every flight the closest the truth came to the wall after the scored hold
(the re-engage, trip and landing, where seed 3's strikes happened) and over the landing
itself. The sequence lands through the module's Structure Land mode (``--land structure``,
the default, which keeps the wall guard down to touchdown) or stock Land (``--land stock``,
the sequence before 2026-10-28).

Exit code 0 when the flight passed its checks, both traces verify, GPS was never used and
EKF2 fused the vision stream. For the benchmark, ``eval.baselines.run_baseline --runner
hold`` flies this over the baseline's scenarios and seeds and tabulates it next to the
tuned-PX4 rows.
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
from eval.ulog import load_topics as load_trace_topics
from eval.ulog import trace_from_topics
from sim2real.estimate_bridge.bridge import Rangefinder
from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.params import read_params, to_environment
from sim2real.estimate_bridge.run_estimate import DEFAULT_PARAMS
from sim2real.estimate_bridge.ulog_checks import analyse
from sim2real.estimate_bridge.verify_trace import verify as verify_estimate
from sim2real.gazebo.params_io import write_params
from sim2real.gazebo.run_scenario import (
    DEFAULT_IMAGE,
    WATCH_IMAGE,
    WORLD,
    git_commit,
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
STATE_STANDBY, STATE_ENGAGED, STATE_HANDBACK, STATE_LANDING = 1, 2, 3, 4


def docker_command(image: str, out: Path, pose: str, altitude: float, hold_s: float,
                   settle_s: float, drift_tol: float, params: list[str], name: str,
                   env: dict[str, str], watch: bool = False, land: str = "structure"
                   ) -> list[str]:
    cmd = ["docker", "run", "--rm", "--name", name,
           "-v", f"{out}:/out", "-v", f"{REPO_ROOT}:/repo:ro",
           "-e", f"PX4_GZ_WORLD={WORLD}", "-e", f"PX4_GZ_MODEL_POSE={pose}",
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
    return cmd


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


def landing_stretch(status: dict[str, np.ndarray], start_us: int) -> dict[str, Any] | None:
    """The Structure Land stretch (the last one): its window in log seconds and how far the
    wall guard moved the xy setpoint back. None when the flight landed through stock Land."""
    landing = np.asarray(status["state"]) == STATE_LANDING
    if not landing.any():
        return None
    t = (np.asarray(status["timestamp"], dtype=np.int64) - start_us) * 1e-6
    edges = np.flatnonzero(np.diff(np.concatenate(([False], landing, [False])).astype(int)))
    a, b = int(edges[-2]), int(edges[-1])
    shift = np.asarray(status.get("standoff_shift_m", np.zeros(len(t))), dtype=float)
    return {"t_s": [float(t[a]), float(t[b - 1])], "shift_m": float(shift[a:b].max())}


def score_hold(ulog: Path, sc, wind, altitude: float, settle_s: float
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """EKF report and eval.metrics on the TRUE position over the first hold."""
    status, start_us = load_status(ulog)
    if status is None:
        raise ValueError(f"{ulog} has no {STATUS_TOPIC}")
    window = hold_window(status, start_us, settle_s)
    if window is None:
        raise ValueError("no engaged stretch in the status topic")
    ekf = analyse(ulog, altitude, window)
    topics, start_us2 = load_trace_topics(ulog)
    hold_point = hold_point_enu(status, start_us, window)
    trace = trace_from_topics(topics, start_us2, hold_point=hold_point)
    surface = Surface(sc.wall_point, sc.wall_normal_toward)
    gusts = [(float(s), float(s) + float(d))
             for s, d in zip(wind.gust_start[0], wind.gust_duration[0], strict=True)]
    metrics = compute_metrics(trace, surface, gusts, window)
    hold = {
        "window_s": list(window),
        "hold_point_enu": [float(v) for v in hold_point],
        "gusts_s": gusts,
        "module": hold_stats(status, start_us, window),
        "metrics": metrics,
        "wall_guard": wall_guard_stats(status, start_us),
    }
    after = trace.t > window[1]
    if after.any():
        d = surface.distance(trace.pos[after])
        i = int(np.argmin(d))
        hold["after_hold"] = {"min_clearance_m": float(d[i]), "t_s": float(trace.t[after][i])}
    landing = landing_stretch(status, start_us)
    if landing is not None:
        during = (trace.t >= landing["t_s"][0]) & (trace.t <= landing["t_s"][1])
        if during.any():
            landing["min_clearance_m"] = float(surface.distance(trace.pos[during]).min())
        hold["landing"] = landing
    return ekf, hold


# ---------------------------------------------------------------------------------------


def assemble(report: dict[str, Any]) -> dict[str, Any]:
    """The pass line: the flight's own checks, both traces, GPS-free, vision fused."""
    report["ok"] = bool(report.get("flight_ok") and report.get("estimate_ok")
                        and report.get("wind_ok") and report.get("gps_free")
                        and report.get("vision_fused"))
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
    ap.add_argument("--hold-s", type=float, default=60.0, help="time in Structure Hold")
    ap.add_argument("--settle-s", type=float, default=5.0, help="stock-mode hover times")
    ap.add_argument("--drift-tol", type=float, default=1.0,
                    help="sitl_test's hold tolerance on PX4's estimate (the truth score is "
                         "in hold.json)")
    ap.add_argument("--rangefinder", action="store_true",
                    help="bridge sends a forward rangefinder against the wall (wall guard)")
    ap.add_argument("--land", choices=("structure", "stock"), default="structure",
                    help="end the sequence in Structure Land (wall guard to touchdown) or "
                         "stock Land")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--container-name", default="px4-sitl-hold")
    args = ap.parse_args(argv)
    if args.image is None:
        args.image = WATCH_IMAGE if args.watch else DEFAULT_IMAGE

    sc = load_scenario(args.scenario)
    if args.scenario_seed is not None:
        sc = dataclasses.replace(sc, seed=args.scenario_seed)
    prof = load_profile(args.profile)
    if args.seed is not None:
        prof = prof.with_seed(args.seed)
    prof = prof.with_anchor_altitude(sc.hover.altitude)  # only an "altitude" anchor changes
    if args.rangefinder:
        prof = dataclasses.replace(prof, rangefinder=Rangefinder(sc.wall_point,
                                                                 sc.wall_normal_toward))
    px4_params = read_params(args.params)
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.runs_dir / f"{stamp}_sitl_hold_{sc.name}_{prof.name}_{commit}"
    out.mkdir(parents=True)

    wind = sample_scenario(sc)
    write_params(out / "wind_params.txt", wind)
    (out / "estimate_config.yaml").write_text(yaml.safe_dump(prof.to_dict(), sort_keys=False))
    config = {
        "scenario": args.scenario, "name": sc.name, "scenario_seed": sc.seed,
        "profile": prof.to_dict(), "px4_params": str(args.params), "module_params": args.param,
        "commit": commit, "image": args.image, "world": WORLD, "spawn": sc.spawn.__dict__,
        "wall_clearance_m": sc.wall_clearance, "altitude_m": sc.hover.altitude,
        "hold_s": args.hold_s, "settle_s": args.settle_s, "disturbance": params_summary(wind),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = docker_command(args.image, out, sc.spawn.px4_pose(), sc.hover.altitude, args.hold_s,
                         args.settle_s, args.drift_tol, args.param, args.container_name,
                         to_environment(px4_params), watch=args.watch, land=args.land)
    print("run folder:", out)
    print("$", " ".join(cmd), flush=True)
    if args.watch:
        print("watch live:", watch_url(), "(the view appears once Gazebo is up)", flush=True)
    flight_status = subprocess.run(cmd).returncode

    report: dict[str, Any] = {
        "scenario": sc.name, "profile": prof.name, "seed": prof.seed, "commit": commit,
        "module_params": args.param, "land": args.land, "run": str(out.relative_to(REPO_ROOT)),
        "flight_exit_code": flight_status,
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
            report["vision_fused"] = ekf["vision_fused"]
        except Exception as exc:  # the flight already happened; record why scoring failed
            report["ulog_error"] = f"{type(exc).__name__}: {exc}"
            report["gps_free"] = report["vision_fused"] = False
    else:
        report["ulog_error"] = "no flight.ulg"
        report["gps_free"] = report["vision_fused"] = False

    assemble(report)
    (out / "hold_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
