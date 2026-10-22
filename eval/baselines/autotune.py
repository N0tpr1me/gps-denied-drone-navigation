"""Run PX4's attitude autotune in Gazebo SITL, GPS-free, and save the gains it found.

    python -m eval.baselines.autotune
    python -m eval.baselines.autotune --params eval/baselines/px4_tuned_v1.params  # re-check

Flies ``autotune_open_calm`` (still air, 4 m from the wall) on the clean vision estimate
with the EKF2 settings of ``--params`` plus ``MC_AT_EN 1`` and ``MC_AT_APPLY 1``, through the
image's ``with-sitl`` entrypoint running ``autotune_flight.py``. Still air and a clean
estimate because the autotune identifies the attitude dynamics; wind and estimate
corruption would only add noise to that fit (PX4's guide says to tune in calm conditions).

Writes ``runs/<time>_px4_autotune_<commit>/`` with ``autotune.json`` (gains before and
after), ``autotuned.params`` (the changed gains in .params format), ``flight.ulg`` and the
usual wind and estimate files. Exit code 0 when the autotune changed the gains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.params import FLOAT, read_params, to_environment
from sim2real.estimate_bridge.run_estimate import DEFAULT_PARAMS
from sim2real.gazebo.params_io import write_params
from sim2real.gazebo.run_scenario import DEFAULT_IMAGE, WORLD, git_commit
from sim2real.gazebo.scenario import load_scenario, sample_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOTUNE_PARAMS = {"MC_AT_EN": 1, "MC_AT_APPLY": 1}


def params_text(gains: dict[str, float], header: str) -> str:
    rows = [f"1\t1\t{k}\t{v:.6g}\t{FLOAT}" for k, v in sorted(gains.items())]
    return "".join(f"# {line}\n" for line in header.splitlines()) + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS,
                    help="base PX4 .params (EKF2 on vision, GPS off)")
    ap.add_argument("--scenario", default="autotune_open_calm")
    ap.add_argument("--profile", default="clean")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    args = ap.parse_args(argv)

    sc, prof = load_scenario(args.scenario), load_profile(args.profile)
    px4_params = {**read_params(args.params), **AUTOTUNE_PARAMS}
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / "runs" / f"{stamp}_px4_autotune_{commit}"
    out.mkdir(parents=True)
    write_params(out / "wind_params.txt", sample_scenario(sc))
    (out / "estimate_config.yaml").write_text(yaml.safe_dump(prof.to_dict(), sort_keys=False))
    (out / "config.yaml").write_text(yaml.safe_dump(
        {"scenario": args.scenario, "profile": prof.to_dict(), "px4_params": px4_params,
         "commit": commit, "image": args.image, "base_params": str(args.params)},
        sort_keys=False))

    cmd = ["docker", "run", "--rm", "--name", "px4-sitl-autotune",
           "--entrypoint", "/repo/sim2real/docker/entrypoint.sh",
           "-v", f"{out}:/out", "-v", f"{REPO_ROOT}:/repo:ro",
           "-e", f"PX4_GZ_WORLD={WORLD}", "-e", f"PX4_GZ_MODEL_POSE={sc.spawn.px4_pose()}",
           "-e", "STRUCTURE_HOLD_WIND_PARAMS=/out/wind_params.txt",
           "-e", "STRUCTURE_HOLD_WIND_TRACE=/out/wind_trace.csv",
           "-e", "ESTIMATE_BRIDGE_PROFILE=/out/estimate_config.yaml"]
    for k, v in to_environment(px4_params).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [args.image, "with-sitl", "python3", "/repo/eval/baselines/autotune_flight.py",
            "--out", "/out/autotune.json", "--altitude", str(sc.hover.altitude)]
    print("run folder:", out)
    print("$", " ".join(cmd), flush=True)
    status = subprocess.run(cmd).returncode

    result_path = out / "autotune.json"
    if not result_path.exists():
        print("no autotune.json written", file=sys.stderr)
        return 1
    result = json.loads(result_path.read_text())
    if result.get("changed"):
        header = (f"PX4 autotune result, run {out.relative_to(REPO_ROOT)}, commit {commit}\n"
                  f"{args.scenario}, profile {args.profile}, base {args.params}")
        (out / "autotuned.params").write_text(params_text(result["changed"], header))
    print(json.dumps(result, indent=2))
    return 0 if status == 0 and result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
