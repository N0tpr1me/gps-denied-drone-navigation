"""Fly one wall-and-slab scenario in the SITL image and check the applied wind.

    python -m sim2real.gazebo.run_scenario wind_4ms          # a name in scenarios/
    python -m sim2real.gazebo.run_scenario path/to/file.yaml --hover-s 60

Writes ``runs/<time>_sitl_wind_<scenario>_<commit>/`` (ignored by git):

- ``config.yaml``: the resolved scenario, seed, commit and image
- ``wind_params.txt``: the episode's disturbance, as the plugin read it
- ``summary.json``, ``flight.ulg``, ``px4.log``: from the hover test (sim2real/docker)
- ``wind_trace.csv``: the plugin's per-step state, wind and applied force
- ``wind_verify.json``: the trace rechecked against the Python model
- ``wind_summary.json``: pass/fail of both, for experiments/LOG.md

Needs the image from ``sim2real/docker/run.sh build`` and Docker running. Exit code 0 when
the hover passes and the trace matches the model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sim2real.gazebo.params_io import write_params
from sim2real.gazebo.scenario import load_scenario, sample_scenario
from sim2real.gazebo.verify_trace import verify

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "structure-hold/px4-sitl:v1.17.0"
# sim2real/docker/Dockerfile.watch: the same image plus a browser view of Gazebo on WATCH_PORT.
WATCH_IMAGE = DEFAULT_IMAGE + "-watch"
WATCH_PORT = 6080
WORLD = "wall_slab"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def watch_options() -> list[str]:
    """``docker run`` options that stream Gazebo's GUI to http://localhost:WATCH_PORT."""
    return ["-p", f"{WATCH_PORT}:{WATCH_PORT}", "-e", "WATCH=1"]


def watch_url() -> str:
    return f"http://localhost:{WATCH_PORT}/vnc.html?autoconnect=1&resize=scale"


def docker_command(image: str, out: Path, pose: str, altitude: float, hover_s: float,
                   name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "-v", f"{out}:/out",
        "-e", f"PX4_GZ_WORLD={WORLD}",
        "-e", f"PX4_GZ_MODEL_POSE={pose}",
        "-e", "STRUCTURE_HOLD_WIND_PARAMS=/out/wind_params.txt",
        "-e", "STRUCTURE_HOLD_WIND_TRACE=/out/wind_trace.csv",
        image, "hover-test", "--altitude", str(altitude), "--hover-s", str(hover_s),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", help="scenario name or YAML path")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--hover-s", type=float, help="override the scenario's hover time")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--container-name", default="px4-sitl-wind")
    args = ap.parse_args(argv)

    sc = load_scenario(args.scenario)
    hover_s = args.hover_s if args.hover_s is not None else sc.hover.hover_s
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.runs_dir / f"{stamp}_sitl_wind_{sc.name}_{commit}"
    out.mkdir(parents=True)

    params = sample_scenario(sc)
    write_params(out / "wind_params.txt", params)
    config = {
        "scenario": args.scenario, "name": sc.name, "seed": sc.seed, "commit": commit,
        "image": args.image, "world": WORLD, "spawn": sc.spawn.__dict__,
        "wall_clearance_m": sc.wall_clearance, "altitude_m": sc.hover.altitude,
        "hover_s": hover_s, "disturbance": params_summary(params),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = docker_command(args.image, out, sc.spawn.px4_pose(), sc.hover.altitude, hover_s,
                         args.container_name)
    print("run folder:", out)
    print("$", " ".join(cmd), flush=True)
    hover_status = subprocess.run(cmd).returncode

    report = {"scenario": sc.name, "commit": commit, "run": str(out.relative_to(REPO_ROOT)),
              "hover_exit_code": hover_status, "hover_ok": hover_status == 0}
    summary_path = out / "summary.json"
    if summary_path.exists():
        report["hover"] = json.loads(summary_path.read_text())
    trace_path = out / "wind_trace.csv"
    if trace_path.exists():
        report["wind"] = verify(out / "wind_params.txt", trace_path)
        report["wind_ok"] = report["wind"]["ok"]
    else:
        report["wind_ok"] = False
        report["wind_error"] = "no wind_trace.csv written by the plugin"
    report["ok"] = bool(report["hover_ok"] and report["wind_ok"])
    (out / "wind_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def params_summary(p) -> dict:
    """The human-readable part of the sampled params, for config.yaml."""
    return {
        "mean_wind_m_s": [float(v) for v in p.mean_wind[0]],
        "gust_start_s": [float(v) for v in p.gust_start[0]],
        "gust_duration_s": [float(v) for v in p.gust_duration[0]],
        "gust_speed_m_s": [float(v) for v in (p.gust_wind[0] ** 2).sum(-1) ** 0.5],
        "turb_sigma_m_s": float(((p.turb_amp[0, 0] ** 2).sum() / 2.0) ** 0.5),
        "k_lin": float(p.k_lin[0]), "k_quad": float(p.k_quad[0]),
        "wall_decay_m": float(p.wall_decay[0]), "wall_floor": float(p.wall_floor[0]),
    }


if __name__ == "__main__":
    sys.exit(main())
