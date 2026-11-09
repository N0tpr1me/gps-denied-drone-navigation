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

import yaml

from eval.metrics import Surface, compute_metrics
from eval.ulog import read_ulog
from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.params import read_params, to_environment
from sim2real.estimate_bridge.ulog_checks import analyse, hover_window, load_topics
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
                   cpuset: str | None = None) -> list[str]:
    cmd = ["docker", "run", "--rm", "--name", name, *isolation_options(name, cpuset),
           "-v", f"{out}:/out", "-v", f"{REPO_ROOT}:/repo:ro",
           "-e", f"PX4_GZ_WORLD={WORLD}", "-e", f"PX4_GZ_MODEL_POSE={pose}",
           "-e", "STRUCTURE_HOLD_WIND_PARAMS=/out/wind_params.txt",
           "-e", "STRUCTURE_HOLD_WIND_TRACE=/out/wind_trace.csv",
           "-e", "ESTIMATE_BRIDGE_PROFILE=/out/estimate_config.yaml"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    if watch:
        cmd += watch_options()
    cmd += [image, "hover-test", "--altitude", str(altitude), "--hover-s", str(hover_s),
            "--local-position-only"]
    return cmd


def score_ulog(ulog: Path, sc, params, altitude: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """EKF health report and eval.metrics on the true position, over the hover window."""
    topics, start_us = load_topics(ulog)
    window = hover_window(topics, start_us, altitude)
    ekf = analyse(ulog, altitude, window)
    trace = read_ulog(ulog)
    surface = Surface(sc.wall_point, sc.wall_normal_toward)
    gusts = [(float(s), float(s) + float(d))
             for s, d in zip(params.gust_start[0], params.gust_duration[0], strict=True)]
    metrics = compute_metrics(trace, surface, gusts, window)
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
        "image": args.image, "world": WORLD, "spawn": sc.spawn.__dict__,
        "wall_clearance_m": sc.wall_clearance, "altitude_m": sc.hover.altitude,
        "hover_s": hover_s, "disturbance": params_summary(wind),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = docker_command(args.image, out, sc.spawn.px4_pose(), sc.hover.altitude, hover_s,
                         args.container_name, to_environment(px4_params), watch=args.watch,
                         cpuset=args.cpuset)
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
            ekf, metrics = score_ulog(ulog, sc, wind, sc.hover.altitude)
            (out / "ekf.json").write_text(json.dumps(ekf, indent=2) + "\n")
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
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
