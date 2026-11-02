"""Software parts list (SBOM) for Structure Hold, as CycloneDX 1.6 JSON.

A buyer's security or procurement reviewer asks "what is in it, who made each part, under
which licence". This writes that list, grouped by where each part runs (the `dronenav:role`
property): on the drone, in the evaluation kit, in training, in development, or only in the
offline research that tuned the simulator. docs/assurance-pack.md explains it in words.

    python -m sim2real.assurance.make_sbom              # rewrite sim2real/assurance/sbom.cdx.json
    python -m sim2real.assurance.make_sbom --check      # exit 1 if that file is stale
    python -m sim2real.assurance.make_sbom --release runs/sbom_<date>.cdx.json

The checked-in file lists declared parts only (no time, commit or hashes), so it changes only
when a part changes and the test in this folder can insist it is current. `--release` adds
what a delivery needs: the time, the git commit, the SHA-256 of every file that goes onto the
drone, and the versions actually installed in the environment that made it.

Plain standard library on purpose: it runs anywhere the repo is checked out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sbom.cdx.json"
MODULE_DIR = ROOT / "sim2real" / "px4" / "structure_hold"
POLICY_HEADER = MODULE_DIR / "PolicyWeights.hpp"
STATUS_MSG = ROOT / "sim2real" / "px4" / "msg" / "StructureHoldStatus.msg"

# Pins read from the files that actually use them, so the list cannot drift from the build.
SITL_DOCKERFILE = ROOT / "sim2real" / "docker" / "Dockerfile"
BASALT_DOCKERFILE = ROOT / "eval" / "vio_basalt" / "Dockerfile"
RPG_CONFIG = ROOT / "eval" / "configs" / "vio_fit_rpg_euroc.yaml"

OWNER = "Zain Ali"
PROPRIETARY = {"license": {"name": "Proprietary, all rights reserved (LICENSE)"}}

ROLES = {
    "onboard": "compiled into the drone's PX4 firmware",
    "evaluation": "evaluation kit and scoring tools (desktop or server, never on the drone)",
    "training": "policy training only (desktop or server)",
    "development": "tests and lint only",
    "research": "offline fit of the simulator's estimate-error model; not delivered",
}


def _match(path: Path, pattern: str) -> str:
    m = re.search(pattern, path.read_text(), re.M)
    if not m:
        raise SystemExit(f"{path}: no match for {pattern!r}")
    return m.group(1)


def pins() -> dict[str, str]:
    """Versions and commits as the build files pin them."""
    return {
        "px4": _match(SITL_DOCKERFILE, r"^ARG PX4_TAG=(\S+)"),
        "mavsdk": _match(SITL_DOCKERFILE, r"^ARG MAVSDK_VERSION=(\S+)"),
        "basalt": _match(BASALT_DOCKERFILE, r"^ARG BASALT_COMMIT=([0-9a-f]{40})"),
        "rpg": _match(RPG_CONFIG, r"commit: ([0-9a-f]{40})"),
        "policy": _match(POLICY_HEADER, r'NAME = "([^"]+)"'),
    }


def onboard_files() -> list[Path]:
    """Every file that goes into the flight controller firmware."""
    return sorted(p for p in MODULE_DIR.iterdir() if p.is_file()) + [STATUS_MSG]


def _lic(spdx: str) -> list[dict]:
    return [{"license": {"id": spdx}}]


def _part(ref: str, name: str, role: str, *, version: str | None = None,
          constraint: str | None = None, supplier: str, licenses: list[dict],
          kind: str = "library", purl: str | None = None, vcs: str | None = None,
          note: str | None = None) -> dict:
    part: dict = {"type": kind, "bom-ref": ref, "name": name}
    if version:
        part["version"] = version
    part["supplier"] = {"name": supplier}
    part["licenses"] = licenses
    if purl:
        part["purl"] = purl
    if note:
        part["description"] = note
    if vcs:
        part["externalReferences"] = [{"type": "vcs", "url": vcs}]
    props = [{"name": "dronenav:role", "value": role},
             {"name": "dronenav:where", "value": ROLES[role]}]
    if constraint:
        props.append({"name": "dronenav:version-constraint", "value": constraint})
    part["properties"] = props
    return part


def _pypi(name: str, role: str, constraint: str | None, supplier: str, spdx: str,
          version: str | None = None, note: str | None = None) -> dict:
    return _part(f"pypi-{name.lower()}", name, role, version=version, constraint=constraint,
                 supplier=supplier, licenses=_lic(spdx),
                 purl=f"pkg:pypi/{name.lower()}" + (f"@{version}" if version else ""),
                 note=note)


def components(p: dict[str, str]) -> list[dict]:
    px4 = p["px4"]
    return [
        # On the drone: our module and the PX4 it is compiled into. Nothing else.
        _part("structure-hold-module", "Structure Hold PX4 module", "onboard",
              supplier=OWNER, licenses=[PROPRIETARY], kind="application",
              note="C++14 source in sim2real/px4/structure_hold/ plus the "
                   "StructureHoldStatus uORB message. Uses only PX4's own libraries "
                   "(uORB, px4_work_queue, mathlib, mavlink_log) and the C++ standard "
                   "headers; no heap, no ONNX runtime, no network code of its own."),
        _part("structure-hold-policy", p["policy"], "onboard", supplier=OWNER,
              licenses=[PROPRIETARY], kind="machine-learning-model",
              note="64x64 MLP weights compiled in as PolicyWeights.hpp, generated from "
                   f"training/policies/{p['policy']}.onnx. Trained by us with PPO on our own "
                   "simulator only; no third-party or pre-trained weights, no recorded data."),
        _part("px4-autopilot", "PX4-Autopilot", "onboard", version=px4,
              supplier="Dronecode Foundation (PX4 project)", licenses=_lic("BSD-3-Clause"),
              kind="framework", purl=f"pkg:github/PX4/PX4-Autopilot@{px4}",
              vcs="https://github.com/PX4/PX4-Autopilot",
              note="The host flight stack. The drone maker builds and ships it, so PX4's "
                   "own parts (NuttX and its submodules) belong in the maker's firmware SBOM."),
        # Evaluation kit: the SITL image, the estimate bridge and the scoring scripts.
        _part("dronenav", "dronenav (simulator, scoring and SITL tools)", "evaluation",
              supplier=OWNER, licenses=[PROPRIETARY], kind="application",
              note="The Python in envs/, eval/, training/ and sim2real/."),
        _pypi("numpy", "evaluation", ">=1.24", "NumPy developers", "BSD-3-Clause"),
        _pypi("PyYAML", "evaluation", ">=6.0", "PyYAML developers", "MIT"),
        _pypi("pyulog", "evaluation", ">=1.0", "PX4 project", "BSD-3-Clause"),
        _pypi("pymavlink", "evaluation", ">=2.4.41", "ArduPilot / MAVLink project",
              "LGPL-3.0-only", note="Host side only; not linked into the module."),
        _pypi("mavsdk", "evaluation", None, "Dronecode Foundation (MAVSDK project)",
              "BSD-3-Clause", version=p["mavsdk"], note="Inside the SITL image."),
        _part("gazebo-harmonic", "Gazebo Harmonic (gz-sim8)", "evaluation",
              supplier="Open Source Robotics Foundation", licenses=_lic("Apache-2.0"),
              kind="application", vcs="https://github.com/gazebosim/gz-sim",
              note="The SITL simulator; our wind plugin links gz-sim8 and gz-plugin2."),
        _part("ubuntu-base-image", "Ubuntu 24.04 container base", "evaluation",
              version="24.04", supplier="Canonical", kind="operating-system",
              licenses=[{"license": {"name": "Various (Ubuntu distribution licences)"}}],
              note="Base of the SITL Docker image (sim2real/docker/Dockerfile)."),
        # Training only.
        _pypi("torch", "training", ">=2.6", "PyTorch Foundation", "BSD-3-Clause"),
        _pypi("rsl-rl-lib", "training", "==5.5.1", "ETH Zurich Robotic Systems Lab",
              "BSD-3-Clause", version="5.5.1"),
        _pypi("tensordict", "training", None, "PyTorch Foundation", "MIT",
              note="Pulled in by rsl-rl-lib."),
        _pypi("onnx", "training", ">=1.16", "LF AI & Data (ONNX project)", "Apache-2.0"),
        _pypi("onnxruntime", "training", ">=1.17", "Microsoft", "MIT",
              note="Checks the exported policy on the desktop; not on the drone."),
        # Development only.
        _pypi("pytest", "development", ">=8.0", "pytest developers", "MIT"),
        _pypi("ruff", "development", ">=0.6", "Astral", "MIT"),
        # Research only: fitted numbers came out, no code or data went in.
        _part("basalt", "Basalt VIO", "research", version=p["basalt"],
              supplier="Technical University of Munich", licenses=_lic("BSD-3-Clause"),
              kind="application", vcs="https://gitlab.com/VladyslavUsenko/basalt",
              note="Run offline on EuRoC in eval/vio_basalt/ to measure real VIO error."),
        _part("rpg-trajectory-evaluation", "rpg_trajectory_evaluation", "research",
              version=p["rpg"], supplier="University of Zurich, Robotics and Perception Group",
              licenses=_lic("MIT"), kind="data",
              vcs="https://github.com/uzh-rpg/rpg_trajectory_evaluation",
              note="Its published example VIO results were the first fit's input."),
        _part("euroc-mav-dataset", "EuRoC MAV dataset", "research", supplier="ETH Zurich",
              licenses=[{"license": {"name": "Dataset owner's terms; not redistributed"}}],
              kind="data", note="Read offline; only fitted error statistics are kept "
                                "(docs/vio-fit.md). Not in the repo, not in the policy."),
    ]


def build(p: dict[str, str] | None = None) -> dict:
    p = p or pins()
    parts = components(p)
    onboard = [c["bom-ref"] for c in parts if c["properties"][0]["value"] == "onboard"]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "firmware",
                "bom-ref": "structure-hold",
                "name": "Structure Hold",
                "version": f"{p['policy']} on PX4 {p['px4']}",
                "supplier": {"name": OWNER},
                "licenses": [PROPRIETARY],
                "description": "PX4 flight mode that holds a quadrotor 1 to 3 m from a "
                               "structure without GPS. Confidential.",
            },
            "authors": [{"name": OWNER}],
            "manufacturer": {"name": OWNER},
            "properties": [{"name": "dronenav:classification", "value": "Confidential"}],
        },
        "components": parts,
        "dependencies": [{"ref": "structure-hold", "dependsOn": onboard},
                         {"ref": "structure-hold-module",
                          "dependsOn": ["px4-autopilot", "structure-hold-policy"]}],
    }


def release(bom: dict) -> dict:
    """Stamp a delivery copy: time, commit, file hashes, installed versions."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
    bom["metadata"]["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bom["metadata"]["properties"] += [{"name": "dronenav:git-commit", "value": commit},
                                      {"name": "dronenav:git-clean", "value": str(not dirty)}]
    files = []
    for path in onboard_files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"type": "file", "bom-ref": f"file-{path.name}",
                      "name": str(path.relative_to(ROOT)),
                      "hashes": [{"alg": "SHA-256", "content": digest}]})
    for c in bom["components"]:
        if c["bom-ref"] == "structure-hold-module":
            c["components"] = files
        if c.get("purl", "").startswith("pkg:pypi/") and "version" not in c:
            try:
                c["version"] = metadata.version(c["name"])
            except metadata.PackageNotFoundError:
                continue
            c["purl"] += f"@{c['version']}"
    return bom


def dump(bom: dict) -> str:
    return json.dumps(bom, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if sbom.cdx.json is stale")
    mode.add_argument("--release", type=Path, metavar="OUT",
                      help="write a stamped delivery copy (time, commit, hashes) to OUT")
    args = ap.parse_args(argv)
    text = dump(build())
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is stale: python -m sim2real.assurance.make_sbom")
            return 1
        return 0
    if args.release:
        args.release.parent.mkdir(parents=True, exist_ok=True)
        args.release.write_text(dump(release(build())))
        print(f"wrote {args.release}")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
