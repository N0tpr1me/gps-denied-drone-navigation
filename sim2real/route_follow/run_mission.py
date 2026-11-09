"""Fly stock PX4 Mission mode through Gazebo route worlds, GPS-free, and score it on truth.

The Route Follow baseline in SITL (docs/route-follow.md, "How it is scored"): the x500 takes
off from the floor of a tunnel or corridor (``world.py``), flies the route's waypoints as a
PX4 mission at the route's speed on the corrupted vision estimate through EKF2 (the estimate
bridge, a ``route_*`` profile), holds on the last waypoint and lands. The flight is scored
from the ulog on the TRUE position against the route's exact walls (``walls.py``), the same
measurement the mock makes:

- ``cross_track_p95_m`` / ``cross_track_max_m``: distance from the centreline;
- ``track_p95_m``: distance from the mock's moving target, started when the mission starts
  (PX4 plans its own speed profile, so this one is for reference; cross-track is the
  headline, as in the mock);
- ``min_clearance_m`` and ``contact``: closest approach to any wall, floor or ceiling, and
  whether it came inside ``--contact-radius-m`` (0.40 m, the x500, as the mock's configs);
- ``completed``: the mission reached its last item and the vehicle ended within 0.5 m of the
  last waypoint (truth), with no contact; ``crash``: a contact, or a mission PX4 never
  finished (a failsafe or a timeout), as the mock counts contacts and lost targets. Ending
  more than 0.5 m off the last waypoint because the estimate drifted is "not completed", not
  a crash, as in the mock;
- ``time_s`` against ``planned_s``: how long PX4 took to reach the last waypoint, against the
  mock target's speed plan.

The scored window runs from the switch into Mission mode to the commanded landing, so the
take-off from the floor (where the clearance is zero) is not judged.

    python -m sim2real.route_follow.run_mission --label tuned_mission \\
        --params eval/baselines/px4_tuned_v2.params --routes tunnel_bend --seeds 1
    python -m sim2real.route_follow.run_mission --label tuned_mission --jobs auto \\
        --params eval/baselines/px4_tuned_v2.params --seeds 1 2 3 4 5 6 7 8   # every route

Each flight writes ``runs/<time>_sitl_route_<route>_<profile>_<commit>/`` (``config.yaml``,
``world.sdf``, ``mission.json``, the wind and estimate files and their verification as
``sim2real.estimate_bridge.run_estimate``, ``summary.json`` from the flight, ``flight.ulg``,
``ekf.json``, ``route_metrics.json`` and ``route_summary.json``); the batch writes
``runs/<time>_route_mission_<label>_<commit>/`` with ``batch.json``, ``table.md`` and
``log_lines.md`` (one experiments/LOG.md line per flight, to paste after review).
``--jobs`` flies several at once on their own CPUs, as ``eval.baselines.run_baseline``.
Needs Docker and the SITL image (``sim2real/docker/run.sh build``); the world is generated
per flight and mounted into the image, so no image rebuild is needed for a new route.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval.baselines.run_baseline import cpu_slots, host_label
from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.gps import enu_to_lat_lon_alt
from sim2real.estimate_bridge.params import read_params, to_environment
from sim2real.estimate_bridge.verify_trace import verify as verify_estimate
from sim2real.gazebo.params_io import write_params
from sim2real.gazebo.run_scenario import (
    DEFAULT_IMAGE,
    git_commit,
    isolation_options,
    make_run_dir,
    params_summary,
)
from sim2real.gazebo.verify_trace import verify as verify_wind
from sim2real.route_follow.world import (
    RouteWorld,
    load_route_world,
    route_world_names,
    world_sdf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PX4_WORLDS = "/opt/px4/Tools/simulation/gz/worlds"  # PX4_GZ_WORLDS in the SITL image
# The world origin (wall_slab.sdf's spherical coordinates, which the route worlds keep, and
# run_bridge's --origin-*), so the mission's latitudes and longitudes land on the route.
ORIGIN = (47.397971057728974, 8.546163739800146, 0.0)
CONTACT_RADIUS_M = 0.40  # the x500, as eval/configs/route_follow_*.yaml
COMPLETE_TOL_M = 0.5  # eval/route_follow.py
ACCEPTANCE_M = 0.5  # each waypoint's acceptance radius: a tunnel plan, not PX4's 2 m default
DEFAULT_PROFILE = "route_vio"
METRICS = ("cross_track_p95_m", "cross_track_max_m", "track_p95_m", "min_clearance_m",
           "end_error_m", "time_s")
# Gazebo's real-time factor, sampled beside the flight (``<run>/rtf_samples.txt``, lines of
# "<utc time> <rtf>", written by whatever watches the containers, e.g. every 15 s). Numbers
# are comparable only at real time (the GPU server's load dropped Gazebo to 0.5 to 0.8 and
# moved tuned PX4's error by a fifth), so a flight whose worst sample is under this is flagged.
RTF_FLAG = 0.9


def rtf_of(run_dir: Path) -> dict[str, Any]:
    """Mean, worst and count of a flight's real-time-factor samples (NaN, 0 when none)."""
    f = run_dir / "rtf_samples.txt"
    vals = []
    if f.exists():
        for line in f.read_text().splitlines():
            parts = line.split()
            try:
                vals.append(float(parts[-1]))
            except (IndexError, ValueError):
                continue
    return {"rtf_mean": statistics.fmean(vals) if vals else math.nan,
            "rtf_min": min(vals) if vals else math.nan, "rtf_n": len(vals)}


def mission_json(rw: RouteWorld) -> dict[str, Any]:
    """What ``mission_flight.py`` flies: the waypoints as latitude, longitude and height
    above home (the take-off point on the floor, z = 0)."""
    wps = []
    for p in rw.waypoints:
        lat, lon, _ = enu_to_lat_lon_alt(p, *ORIGIN)
        wps.append({"lat": lat, "lon": lon, "rel_alt_m": float(p[2]),
                    "enu": [float(v) for v in p]})
    return {"route": rw.name, "waypoints": wps, "speed_mps": rw.speed,
            "acceptance_m": ACCEPTANCE_M, "takeoff_alt_m": rw.takeoff_alt_m,
            "end_hold_s": rw.end_hold_s, "planned_s": rw.walls.duration}


def mission_window(modes: dict[str, Any] | None) -> tuple[float, float] | None:
    """From the switch into Mission mode to the commanded Land, from ``ulog_checks.modes``.
    PX4 drops to Hold when the mission completes, so the end hold on the last waypoint is
    inside the window, as the mock's last ``end_hold_s``; a flight that never reaches the
    Land ends at the first mode after the mission that is neither Mission nor Hold."""
    seq = (modes or {}).get("sequence") or []
    start = next((m["t"] for m in seq if m["mode"] == "Mission"), None)
    if start is None:
        return None
    after = [m for m in seq if m["t"] > start]
    end = next((m["t"] for m in after if m["mode"] == "Land"), None)
    if end is None:
        end = next((m["t"] for m in after if m["mode"] not in ("Mission", "Hold")), None)
    return (float(start), float(end)) if end is not None else None


def score_route(t: np.ndarray, pos: np.ndarray, rw: RouteWorld, window: tuple[float, float],
                mission_finished: bool, contact_radius_m: float = CONTACT_RADIUS_M
                ) -> dict[str, Any]:
    """The mock's per-episode numbers for one flight (module docstring), on the TRUE
    position ``pos`` (N, 3) ENU at times ``t`` (N,), over ``window``."""
    w = rw.walls
    inside = (t >= window[0]) & (t <= window[1])
    if not np.any(inside):
        raise ValueError("no true position samples in the mission window")
    tt, pp = t[inside], pos[inside]
    xt = w.distance_to_route(pp)
    clear = w.clearance(pp)
    target = w.point_at(w.target_s(tt - window[0]))
    track = np.linalg.norm(pp - target, axis=-1)
    end = w.waypoints[-1]
    dist_end = np.linalg.norm(pp - end, axis=-1)
    reached = np.flatnonzero(dist_end <= COMPLETE_TOL_M)
    contact = bool(np.any(clear < contact_radius_m))
    end_error = float(dist_end[-1])
    completed = bool(mission_finished and end_error <= COMPLETE_TOL_M and not contact)
    return {
        "window_s": [float(window[0]), float(window[1])],
        "cross_track_p95_m": float(np.percentile(xt, 95)),
        "cross_track_max_m": float(xt.max()),
        "track_p95_m": float(np.percentile(track, 95)),
        "min_clearance_m": float(clear.min()),
        "contact": contact,
        "contact_s": float(np.sum(np.diff(tt, prepend=tt[0])[clear < contact_radius_m])),
        "contact_radius_m": contact_radius_m,
        "end_error_m": end_error,
        "progress_m": float(w.progress(pp).max()),
        "length_m": w.length,
        "time_s": float(tt[reached[0]] - window[0]) if len(reached) else math.nan,
        "planned_s": w.duration,
        "mission_finished": bool(mission_finished),
        "completed": completed,
        "crash": contact or not mission_finished,
    }


def docker_command(image: str, out: Path, rw: RouteWorld, name: str, env: dict[str, str],
                   cpuset: str | None = None) -> list[str]:
    cmd = ["docker", "run", "--rm", "--name", name, *isolation_options(name, cpuset),
           "-v", f"{out}:/out", "-v", f"{REPO_ROOT}:/repo:ro",
           "-v", f"{out / 'world.sdf'}:{PX4_WORLDS}/{rw.world_name}.sdf:ro",
           "-e", f"PX4_GZ_WORLD={rw.world_name}", "-e", f"PX4_GZ_MODEL_POSE={rw.spawn_pose}",
           "-e", "STRUCTURE_HOLD_WIND_PARAMS=/out/wind_params.txt",
           "-e", "STRUCTURE_HOLD_WIND_TRACE=/out/wind_trace.csv",
           "-e", "ESTIMATE_BRIDGE_PROFILE=/out/estimate_config.yaml"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [image, "with-sitl", "python3", "/repo/sim2real/route_follow/mission_flight.py",
                  "--mission", "/out/mission.json", "--out", "/out/summary.json"]


def score_flight(out: Path, rw: RouteWorld, contact_radius_m: float) -> dict[str, Any]:
    """Read the flight's ulog: EKF health over the mission and the route metrics."""
    from eval.ulog import read_ulog
    from sim2real.estimate_bridge.ulog_checks import analyse

    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json"
                                                                 ).exists() else {}
    ekf = analyse(out / "flight.ulg", rw.takeoff_alt_m)
    window = mission_window(ekf.get("modes"))
    ekf_win = analyse(out / "flight.ulg", rw.takeoff_alt_m, window) if window else ekf
    (out / "ekf.json").write_text(json.dumps(ekf_win, indent=2) + "\n")
    if window is None:
        raise ValueError("the log never shows Mission mode followed by another mode")
    trace = read_ulog(out / "flight.ulg")
    metrics = score_route(trace.t, trace.pos, rw, window,
                          bool(summary.get("mission_finished")), contact_radius_m)
    (out / "route_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"ekf": ekf_win, "metrics": metrics}


def fly_one(rw: RouteWorld, profile: str, seed: int, params_path: Path, image: str,
            runs_dir: Path, container: str, cpuset: str | None,
            contact_radius_m: float) -> dict[str, Any]:
    """One flight: write its files, fly it in the image, verify and score it."""
    prof = load_profile(profile).with_seed(seed).with_anchor_altitude(rw.takeoff_alt_m)
    if prof.gps is not None or prof.rangefinder is not None:
        raise ValueError("route profiles fly GPS-free with no wall rangefinder")
    px4_params = read_params(params_path)
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = make_run_dir(runs_dir, f"{stamp}_sitl_route_{rw.name}_{prof.name}_{commit}")
    wind = rw.wind()
    write_params(out / "wind_params.txt", wind)
    (out / "estimate_config.yaml").write_text(yaml.safe_dump(prof.to_dict(), sort_keys=False))
    (out / "world.sdf").write_text(world_sdf(rw))
    mission = mission_json(rw)
    (out / "mission.json").write_text(json.dumps(mission, indent=2) + "\n")
    config = {"route": rw.name, "kind": rw.kind, "scenario_seed": rw.seed,
              "profile": prof.to_dict(), "px4_params": px4_params,
              "params_file": str(params_path), "commit": commit, "image": image,
              "world": rw.world_name, "spawn": rw.spawn_pose,
              "waypoints": rw.waypoints.tolist(), "speed_mps": rw.speed,
              "contact_radius_m": contact_radius_m, "disturbance": params_summary(wind)}
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    cmd = docker_command(image, out, rw, container, to_environment(px4_params), cpuset)
    print("run folder:", out, flush=True)
    log = out / "runner.log"
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        status = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    report: dict[str, Any] = {
        "route": rw.name, "profile": prof.name, "seed": seed, "commit": commit,
        "run": str(out.relative_to(REPO_ROOT)) if out.is_relative_to(REPO_ROOT) else str(out),
        "flight_exit_code": status}
    if (out / "summary.json").exists():
        report["flight"] = json.loads((out / "summary.json").read_text())
    if (out / "estimate_trace.csv").exists():
        report["estimate_ok"] = bool(verify_estimate(prof, out / "estimate_trace.csv")["ok"])
    else:
        report["estimate_ok"] = False
    if (out / "wind_trace.csv").exists():
        report["wind_ok"] = bool(verify_wind(out / "wind_params.txt", out / "wind_trace.csv")["ok"])
    else:
        report["wind_ok"] = False
    if (out / "flight.ulg").exists():
        try:
            scored = score_flight(out, rw, contact_radius_m)
            report["metrics"] = scored["metrics"]
            ekf = scored["ekf"]
            report["gps_free"] = bool(ekf["gps_free"])
            report["vision_fused"] = bool(ekf["vision_fused"])
            report["failsafe_fraction"] = (ekf.get("modes") or {}).get("failsafe_fraction")
            report["modes"] = (ekf.get("modes") or {}).get("sequence")
        except Exception as exc:  # the flight happened; record why scoring failed
            report["score_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["score_error"] = "no flight.ulg"
    m = report.get("metrics") or {}
    report["ok"] = bool(status == 0 and report["estimate_ok"] and report["wind_ok"]
                        and report.get("gps_free") and report.get("vision_fused") and m)
    (out / "route_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def row_of(r: dict[str, Any]) -> dict[str, Any]:
    m = r.get("metrics") or {}
    flown = "never became armable" not in str((r.get("flight") or {}).get("error", ""))
    row = {"route": r["route"], "seed": r["seed"], "run": r["run"], "ok": r["ok"],
           "flown": flown, "crash": bool(m.get("crash", flown)),
           "contact": bool(m.get("contact", False)), "completed": bool(m.get("completed")),
           "gps_free": bool(r.get("gps_free")), "vision_fused": bool(r.get("vision_fused")),
           "planned_s": m.get("planned_s", math.nan),
           "error": (r.get("flight") or {}).get("error") or r.get("score_error")}
    for k in METRICS:
        v = m.get(k)
        row[k] = float(v) if v is not None else math.nan
    row.update(rtf_of(REPO_ROOT / r["run"]))
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(r["route"] for r in rows):
        rs = [r for r in rows if r["route"] == name]
        ok = [r for r in rs if not r["crash"]]
        agg: dict[str, Any] = {"n": len(rs), "crashes": sum(r["crash"] for r in rs),
                               "contacts": sum(r["contact"] for r in rs),
                               "completed": sum(r["completed"] for r in rs)}
        for k in METRICS:
            # Error metrics over the flights that did not crash, as eval/route_follow.py.
            src = rs if k == "min_clearance_m" else ok
            vals = [r[k] for r in src if not math.isnan(r[k])]
            agg[f"{k}_mean"] = statistics.fmean(vals) if vals else math.nan
            worst = min if k == "min_clearance_m" else max
            agg[f"{k}_worst"] = worst(vals) if vals else math.nan
        rtf = [r["rtf_min"] for r in rs if not math.isnan(r.get("rtf_min", math.nan))]
        agg["rtf_min"] = min(rtf) if rtf else math.nan
        out[name] = agg
    return out


def fmt(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table(batch: dict[str, Any]) -> str:
    lines = [f"Stock PX4 Mission mode, `{batch['label']}` (`{batch['params']}`), profile "
             f"`{batch['profile']}`, commit {batch['commit']}", "",
             "| route | seed | cross-track p95 m | worst m | min clearance m | done | crash "
             "| time / planned s | RTF mean / min |", "|---|---|---|---|---|---|---|---|---|"]
    for r in batch["rows"]:
        lines.append(f"| {r['route']} | {r['seed']} | {fmt(r['cross_track_p95_m'])} "
                     f"| {fmt(r['cross_track_max_m'])} | {fmt(r['min_clearance_m'])} "
                     f"| {'yes' if r['completed'] else 'no'} | {'yes' if r['crash'] else 'no'} "
                     f"| {fmt(r['time_s'], 1)} / {fmt(r['planned_s'], 1)} "
                     f"| {fmt(r.get('rtf_mean', math.nan))} / "
                     f"{fmt(r.get('rtf_min', math.nan))} |")
    lines += ["", "| route | cross-track p95 mean / worst m | min clearance worst m | done "
              "| crashes | RTF min |", "|---|---|---|---|---|---|"]
    for name, a in batch["aggregate"].items():
        lines.append(f"| {name} | {fmt(a['cross_track_p95_m_mean'])} / "
                     f"{fmt(a['cross_track_p95_m_worst'])} | {fmt(a['min_clearance_m_worst'])} "
                     f"| {a['completed']}/{a['n']} | {a['crashes']}/{a['n']} "
                     f"| {fmt(a.get('rtf_min', math.nan))} |")
    slow = [r for r in batch["rows"] if r.get("rtf_min", math.nan) < RTF_FLAG]
    if slow:
        lines += ["", f"Below real time (a sample under {RTF_FLAG}): " + ", ".join(
            f"{r['route']} seed {r['seed']} ({fmt(r['rtf_min'])})" for r in slow)]
    return "\n".join(lines) + "\n"


def log_line(r: dict[str, Any], batch: dict[str, Any], when: str) -> str:
    status = ("not flown (never armable)" if not r["flown"] else "crash" if r["crash"]
              else "ok" if r["ok"] else "flown, check failed")
    result = (f"truth: cross-track p95 {fmt(r['cross_track_p95_m'])} m, worst "
              f"{fmt(r['cross_track_max_m'])} m, min clearance {fmt(r['min_clearance_m'])} m, "
              f"{'contact' if r['contact'] else 'no contact'}, "
              f"{'finished' if r['completed'] else 'not finished'} in {fmt(r['time_s'], 1)} s "
              f"(target {fmt(r['planned_s'], 1)} s)"
              + (f", Gazebo RTF {fmt(r['rtf_mean'])} (min {fmt(r['rtf_min'])})"
                 if r.get("rtf_n") else ""))
    return (f"| {when} | {r['run']} | {batch['commit']} | sim2real/route_follow/run_mission "
            f"{batch['label']} (stock PX4 Mission mode): {r['route']} seed {r['seed']}, "
            f"{batch['profile']} estimate, `{batch['params']}`, GPS off | {batch['host']} "
            f"| {status} | {result} | {{vs}} |")


def rescore(batch_dir: Path, contact_radius_m: float) -> dict[str, Any]:
    """Score every flight of a batch again from its ulog (after a change to the scoring)
    and rewrite the batch's ``batch.json``, ``table.md`` and ``log_lines.md``."""
    batch = json.loads((batch_dir / "batch.json").read_text())
    rows, lines = [], []
    for old in batch["rows"]:
        out = REPO_ROOT / old["run"]
        report = json.loads((out / "route_summary.json").read_text())
        rw = load_route_world(old["route"])  # the walls; the wind seed plays no part here
        if (out / "flight.ulg").exists():
            scored = score_flight(out, rw, contact_radius_m)
            report["metrics"] = scored["metrics"]
            (out / "route_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        row = row_of(report)
        rows.append(row)
        stamp = out.name.split("_", 1)[0]
        when = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
        lines.append((row, when))
    batch["rows"] = rows
    batch["aggregate"] = aggregate(rows)
    (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (batch_dir / "table.md").write_text(table(batch))
    (batch_dir / "log_lines.md").write_text(
        "\n".join(log_line(r, batch, w) for r, w in lines) + "\n")
    print(table(batch))
    return batch


def pool(batch_dirs: list[Path], out: Path) -> dict[str, Any]:
    """Merge batches of one label, profile and parameter set flown in parts (seed halves,
    each under the one-hour rule) into one batch: rows re-read from each flight's
    ``route_summary.json`` (so the real-time factor is picked up), routes in their first
    batch's order, then seeds."""
    batches = [json.loads((d / "batch.json").read_text()) for d in batch_dirs]
    for key in ("label", "profile", "params"):
        vals = {b[key] for b in batches}
        if len(vals) != 1:
            raise ValueError(f"pooled batches must share one {key}, got {sorted(vals)}")
    rows = [row_of(json.loads((REPO_ROOT / r["run"] / "route_summary.json").read_text()))
            for b in batches for r in b["rows"]]
    pairs = [(r["route"], r["seed"]) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("pooled batches repeat a (route, seed) pair")
    routes = list(dict.fromkeys(name for b in batches for name in b["routes"]))
    rows.sort(key=lambda r: (routes.index(r["route"]), r["seed"]))
    pooled = {**batches[0], "commit": ",".join(dict.fromkeys(b["commit"] for b in batches)),
              "routes": routes, "seeds": sorted({r["seed"] for r in rows}), "rows": rows,
              "aggregate": aggregate(rows), "pooled_from": [str(d) for d in batch_dirs]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.json").write_text(json.dumps(pooled, indent=2) + "\n")
    (out / "table.md").write_text(table(pooled))
    print(table(pooled))
    return pooled


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = make_run_dir(args.runs_dir, f"{stamp}_route_mission_{args.label}_{commit}")
    params = args.params.resolve()
    params_rel = params.relative_to(REPO_ROOT) if params.is_relative_to(REPO_ROOT) else params
    slots = cpu_slots(args.jobs, args.cpus_per_job)
    batch: dict[str, Any] = {"label": args.label, "params": str(params_rel), "commit": commit,
                             "profile": args.profile, "routes": list(args.routes),
                             "seeds": list(args.seeds), "host": host_label(len(slots)),
                             "jobs": len(slots), "rows": []}
    flights = [(name, seed) for name in args.routes for seed in args.seeds]
    order = {f: i for i, f in enumerate(flights)}
    free: queue.Queue[tuple[int, str | None]] = queue.Queue()
    for i, cpuset in enumerate(slots):
        free.put((i, cpuset))
    done: dict[tuple[str, int], tuple[dict[str, Any], str]] = {}
    lock = threading.Lock()

    def one(flight: tuple[str, int]) -> None:
        name, seed = flight
        rw = load_route_world(name)
        rw = rw.with_seed(seed) if args.scenario_seed_from_seed else rw
        slot, cpuset = free.get()
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            report = fly_one(rw, args.profile, seed, params, args.image, args.runs_dir,
                             f"px4-route-{stamp}-{slot}", cpuset, args.contact_radius_m)
        except Exception as exc:
            print(f"{name} seed {seed}: {type(exc).__name__}: {exc}", flush=True)
            return
        finally:
            free.put((slot, cpuset))
        row = row_of(report)
        print(f"{name} seed {seed}: cross-track p95 {fmt(row['cross_track_p95_m'])} m, min "
              f"clearance {fmt(row['min_clearance_m'])} m, done {row['completed']}, crash "
              f"{row['crash']}" + (f" ({row['error']})" if row["error"] else ""), flush=True)
        with lock:
            done[flight] = (row, when)
            batch["rows"] = [done[k][0] for k in sorted(done, key=order.__getitem__)]
            batch["aggregate"] = aggregate(batch["rows"])
            (out / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for fut in [pool.submit(one, f) for f in flights]:
            fut.result()
    keys = sorted(done, key=order.__getitem__)
    batch["rows"] = [done[k][0] for k in keys]
    batch["aggregate"] = aggregate(batch["rows"])
    (out / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (out / "table.md").write_text(table(batch))
    (out / "log_lines.md").write_text(
        "\n".join(log_line(done[k][0], batch, done[k][1]) for k in keys) + "\n")
    print("batch folder:", out)
    print(table(batch))
    return batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="name of the batch, e.g. tuned_mission")
    ap.add_argument("--params", type=Path, help="PX4 .params file")
    ap.add_argument("--rescore", type=Path, metavar="BATCH_DIR",
                    help="score a flown batch again from its ulogs and exit")
    ap.add_argument("--routes", nargs="+", default=None,
                    help=f"route worlds (default: all of {route_world_names()})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="estimate profile name or path")
    ap.add_argument("--no-scenario-seed", dest="scenario_seed_from_seed", action="store_false",
                    help="keep each route file's wind seed instead of the flight's seed")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--contact-radius-m", type=float, default=CONTACT_RADIUS_M)
    ap.add_argument("--jobs", default="1", help="flights at once, or 'auto'")
    ap.add_argument("--cpus-per-job", type=int, default=4)
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--pool", nargs="+", type=Path, metavar="BATCH_DIR",
                    help="merge flown batches of one label (e.g. seed halves) and exit")
    ap.add_argument("--out", type=Path, help="where --pool writes the pooled batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every route's world and mission and print the plan; no Docker")
    args = ap.parse_args(argv)
    if args.rescore:
        rescore(args.rescore, args.contact_radius_m)
        return 0
    if args.pool:
        if not args.out:
            ap.error("--pool needs --out")
        pool(args.pool, args.out)
        return 0
    if not args.label or not args.params:
        ap.error("--label and --params are required unless --rescore is given")
    args.routes = args.routes or route_world_names()
    worlds = [load_route_world(name) for name in args.routes]  # fail before any flight
    load_profile(args.profile)
    read_params(args.params)
    if args.dry_run:
        for rw in worlds:
            sdf, m = world_sdf(rw), mission_json(rw)
            print(f"{rw.name}: {rw.kind}, {rw.walls.length:.1f} m in {len(m['waypoints'])} "
                  f"waypoints at {rw.speed:.1f} m/s, target {m['planned_s']:.1f} s, "
                  f"world {len(sdf) // 1024} KB, seeds {args.seeds}, profile {args.profile}")
        return 0
    if args.jobs != "auto":
        try:
            args.jobs = int(args.jobs)
        except ValueError:
            ap.error("--jobs takes a number or 'auto'")
    try:
        cpu_slots(args.jobs, args.cpus_per_job)
    except ValueError as exc:
        ap.error(str(exc))
    batch = run_batch(args)
    return 0 if batch["rows"] and all(r["ok"] for r in batch["rows"]) else 1


if __name__ == "__main__":
    sys.exit(main())
