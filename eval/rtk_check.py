"""Check the RTK scorer where the truth is known: Gazebo flights with an RTK receiver logged.

    python -m eval.rtk_check runs/<flight folder> [runs/<flight folder> ...] [--out DIR]

An outdoor partner flight is scored on its RTK receiver (``eval.rtk_truth``), and nobody can
say afterwards how right that score was. In Gazebo somebody can: fly the benchmark hold with
the estimate bridge's scripted receiver logged but not fused (``rtk_truth_*`` profiles,
``sim2real/estimate_bridge/gps.py``) and the log holds both the receiver's samples and the
simulator's truth. This scores each such flight through ``eval.partner_session``, the path a
partner's session takes, with the wall as two surveyed points, four ways:

``rtk``
    on the receiver as logged: what a partner's report would say.
``gazebo``
    the same, with every receiver sample moved onto Gazebo's truth (a perfect receiver that
    keeps the real one's fix types). ``rtk`` minus this is what the receiver's own error
    did to the score (``difference``).
``gazebo_whole_hold``
    the perfect receiver with a fixed solution all flight. This minus ``gazebo`` is what
    the moments without a fix hid (``hidden_by_lost_fix``): nothing RTK can know.
``benchmark``
    Gazebo's truth against the hold point read as world coordinates, with no frame fit,
    as the benchmark's runners read it (for Structure Hold this is the runner's own score,
    ``sim2real.px4.run_gazebo``; the tuned-PX4 runner follows the setpoint as it moves
    instead of the hold point at the window start, so its own number differs again). It
    differs from the three above by the offset between EKF2's frame and the world at
    take-off (its barometer height above all), which the partner path fits out over the
    take-off and the benchmark counts.

The flight folders are those of ``eval.baselines.run_baseline`` (``sim2real.px4.run_gazebo``
for Structure Hold, ``sim2real.estimate_bridge.run_estimate`` for tuned PX4). Every score
uses the runner's own scored window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.frames import geodetic_to_enu
from eval.partner_session import load_session_topics, score_run
from eval.rtk_truth import FIX_RTK_FIXED, SELECTED_GPS, SENSOR_GPS, read_rtk
from eval.ulog import GROUNDTRUTH, ned_to_enu
from sim2real.estimate_bridge.gps import WORLD_ORIGIN, enu_to_lat_lon_alt
from sim2real.gazebo.scenario import load_scenario

KEYS = ("p95_m", "rms_m", "max_m", "worst_toward_surface_m", "min_clearance_m")
SURVEY_HALF_M = 3.0  # the two surveyed points sit this far either side of the wall point


def survey_points(wall_point, normal) -> list[list[float]]:
    """Two points on the scenario's wall as an RTK rover would report them: latitude,
    longitude and altitude, either side of the wall point along its foot."""
    n = np.asarray(normal, dtype=float)
    along = np.cross([0.0, 0.0, 1.0], n)
    along /= np.linalg.norm(along)
    return [list(enu_to_lat_lon_alt(np.asarray(wall_point, dtype=float) + s * along,
                                    *WORLD_ORIGIN)) for s in (-SURVEY_HALF_M, SURVEY_HALF_M)]


def flight_window(folder: Path) -> tuple[str, list[float]]:
    """The controller that flew and the window its runner scored."""
    if (folder / "hold.json").exists():
        return "structure_hold", json.loads((folder / "hold.json").read_text())["window_s"]
    return "tuned_px4", json.loads((folder / "metrics.json").read_text())["window_s"]


def receiver_error(topics, start_us: int, window) -> dict[str, float]:
    """How far the receiver's fixed samples are from Gazebo's truth inside the window, m."""
    rtk = read_rtk(topics, start_us, WORLD_ORIGIN)
    gt = topics[GROUNDTRUTH]
    tg = (np.asarray(gt["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6
    pg = ned_to_enu(gt["x"], gt["y"], gt["z"])
    keep = (rtk.t >= window[0]) & (rtk.t <= window[1])
    err = rtk.pos[keep] - np.stack([np.interp(rtk.t[keep], tg, pg[:, k]) for k in range(3)], 1)
    return {"horizontal_rms_m": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
            "vertical_rms_m": float(np.sqrt(np.mean(err[:, 2] ** 2))),
            "max_m": float(np.linalg.norm(err, axis=1).max())}


def perfect_receiver(topics, start_us: int, always_fixed: bool = False):
    """The log with its RTK receiver's samples moved onto Gazebo's truth at their own
    stamps: same rate, same fix types (or RTK fixed throughout), no error."""
    source = read_rtk(topics, start_us, WORLD_ORIGIN).source
    msg = dict(topics[source])
    t = (np.asarray(msg["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6
    gt = topics[GROUNDTRUTH]
    tg, first = np.unique((np.asarray(gt["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6,
                          return_index=True)
    pg = ned_to_enu(gt["x"], gt["y"], gt["z"])[first]
    llh = np.array([enu_to_lat_lon_alt(p, *WORLD_ORIGIN) for p in
                    np.stack([np.interp(t, tg, pg[:, k]) for k in range(3)], axis=1)])
    msg.update({"latitude_deg": llh[:, 0], "longitude_deg": llh[:, 1],
                "altitude_msl_m": llh[:, 2], "altitude_ellipsoid_m": llh[:, 2]})
    if always_fixed:
        msg["fix_type"] = np.full(len(t), FIX_RTK_FIXED)
    others = {k: v for k, v in topics.items()
              if k != SELECTED_GPS and not k.startswith(SENSOR_GPS)}
    return {**others, SENSOR_GPS: msg}


def check_topics(topics, start_us: int, controller: str, window, wall_point, wall_normal
                 ) -> dict[str, Any]:
    """One flight scored on its RTK receiver and on Gazebo's truth, over the same window."""
    run = {"file": "flight.ulg", "controller": controller, "case": "check", "run": 0,
           "window": list(window)}
    sheet = {"truth": "rtk", "wall": {"survey": survey_points(wall_point, wall_normal)}}
    on_rtk = score_run(topics, start_us, run, sheet)
    same = score_run(perfect_receiver(topics, start_us), start_us, run, sheet)
    whole = score_run(perfect_receiver(topics, start_us, always_fixed=True), start_us, run,
                      sheet)
    bench = score_run(topics, start_us, run, {
        "truth": "Gazebo", "wall": {"point": list(wall_point), "normal": list(wall_normal)}})
    out: dict[str, Any] = {
        "controller": controller, "window_s": list(window),
        "rtk": {k: on_rtk[k] for k in KEYS}, "gazebo": {k: same[k] for k in KEYS},
        "gazebo_whole_hold": {k: whole[k] for k in KEYS},
        "benchmark": {k: bench[k] for k in KEYS},
        "difference": {k: on_rtk[k] - same[k] for k in KEYS},
        "hidden_by_lost_fix": {k: whole[k] - same[k] for k in KEYS},
        "receiver_error": receiver_error(topics, start_us, window),
        "contacts": {"rtk": on_rtk["contacts"], "gazebo_whole_hold": whole["contacts"]},
        # EKF2's frame against the world at take-off: the hold point as the partner path
        # carries it, minus the setpoint read as world coordinates (east, north, up).
        "ekf_frame_offset_m": [float(a - b - c) for a, b, c in zip(
            whole["hold_point_enu"], bench["hold_point_enu"],
            geodetic_to_enu(*WORLD_ORIGIN, whole["rtk_origin"]), strict=True)],
    }
    out.update({k: on_rtk[k] for k in ("truth_coverage", "truth_coverage_flight",
                                       "truth_independent", "align_rms_m", "align_yaw_deg",
                                       "rtk_delay_s", "rtk_rate_hz", "position_source")})
    return out


def check_flight(folder: Path) -> dict[str, Any]:
    config = yaml.safe_load((folder / "config.yaml").read_text())
    sc = load_scenario(config["scenario"])
    controller, window = flight_window(folder)
    topics, start_us = load_session_topics(folder / "flight.ulg")
    out = check_topics(topics, start_us, controller, window, sc.wall_point,
                       sc.wall_normal_toward)
    out.update(run=folder.name, scenario=sc.name, seed=int(config.get("scenario_seed", 0)),
               profile=(config.get("profile") or {}).get("name"))
    return out


def table(rows: list[dict[str, Any]]) -> str:
    out = ["| flight | profile | p95 m: RTK / Gazebo truth, same samples / whole hold "
           "| closest to wall m: RTK / Gazebo truth | RTK fixed | GPS fused "
           "| fit residual m | receiver off by m (horizontal rms) | p95 m with no frame fit |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fused = {True: "no", False: "YES", None: "unknown"}[r["truth_independent"]]
        out.append(
            f"| {r['controller']} seed {r['seed']} | {r['profile']} "
            f"| {r['rtk']['p95_m']:.3f} / {r['gazebo']['p95_m']:.3f} / "
            f"{r['gazebo_whole_hold']['p95_m']:.3f} "
            f"| {r['rtk']['min_clearance_m']:.3f} / "
            f"{r['gazebo_whole_hold']['min_clearance_m']:.3f} "
            f"| {100 * r['truth_coverage']:.0f} % | {fused} | {r['align_rms_m']:.3f} "
            f"| {r['receiver_error']['horizontal_rms_m']:.3f} "
            f"| {r['benchmark']['p95_m']:.3f} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", type=Path, nargs="+", help="flight folders under runs/")
    p.add_argument("--out", type=Path, help="write rtk_check.json and rtk_check.md here")
    args = p.parse_args(argv)
    rows = [check_flight(f) for f in args.folders]
    text = table(rows)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "rtk_check.json").write_text(json.dumps(rows, indent=2) + "\n")
        (args.out / "rtk_check.md").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
