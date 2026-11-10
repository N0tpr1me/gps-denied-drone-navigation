"""Write the module's parameters as a PX4 YAML module config, for PX4 v1.18.

Up to PX4 v1.17 the build reads parameter definitions from C files in a module's folder
(``structure_hold_params.c``, the source of truth here: the module's limits, the
integration guide and the safety manual are all pinned to it). PX4 v1.18 dropped that and
reads only YAML module configs, so the same definitions are kept beside it as
``structure_hold_params.yaml``, which the module's CMakeLists.txt hands to PX4 only on a
tree that no longer reads the C file. This script generates the YAML from the C file;
``--check`` (and ``tests/test_params_yaml.py``) fails when the two differ.

    python -m sim2real.px4.tools.params_yaml            # rewrite the YAML
    python -m sim2real.px4.tools.params_yaml --check    # exit 1 if it is stale
    python -m sim2real.px4.tools.params_yaml --against <PX4 build dir>/parameters.json

``--against`` compares what a PX4 build made of the definitions (the metadata a ground
station shows) with the C file, whichever of the two formats that build read.

The layout follows PX4's own ``Tools/migrate_c_params.py`` (v1.18), and the result was
checked against it and against the parameter metadata each PX4 build generates
(``docs/px4-versions.md``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

MODULE = Path(__file__).resolve().parents[1] / "structure_hold"
PARAMS_C = MODULE / "structure_hold_params.c"
PARAMS_YAML = MODULE / "structure_hold_params.yaml"

HEADER = (
    "# Generated from structure_hold_params.c by sim2real/px4/tools/params_yaml.py; do not\n"
    "# edit by hand. PX4 v1.18 reads parameter definitions from YAML only; v1.16 and v1.17\n"
    "# read the C file and never see this one (CMakeLists.txt).\n"
)
TAGS = {"boolean", "reboot_required", "group", "min", "max", "unit", "decimal", "value"}
# A comment block directly above its PARAM_DEFINE (the block itself holds no "*/", so the
# file's opening comment is not swallowed into the first parameter).
BLOCK = re.compile(
    r"/\*\*((?:(?!\*/).)*)\*/\s*PARAM_DEFINE_(FLOAT|INT32)\((\w+),\s*([-\w.]+?)f?\);", re.S)


def _number(text: str) -> int | float:
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def _definition(block: str, kind: str, default: str) -> tuple[str, dict]:
    """One parameter's group and YAML entry from its comment block."""
    lines = [re.sub(r"^\s*\* ?", "", line).rstrip() for line in block.strip("\n").split("\n")]
    text = [line for line in lines if not line.startswith("@")]
    tags: dict[str, list[str]] = {}
    for line in lines:
        if line.startswith("@"):
            tag, _, value = line[1:].partition(" ")
            if tag not in TAGS:
                raise ValueError(f"unknown tag @{tag}: teach params_yaml.py about it first")
            tags.setdefault(tag, []).append(value.strip())

    # The first paragraph is the short description, the rest the long one.
    paragraphs = "\n".join(text).strip().split("\n\n", 1)
    description = {"short": " ".join(paragraphs[0].split())}
    if len(paragraphs) > 1 and paragraphs[1].strip():
        description["long"] = paragraphs[1].strip()

    entry: dict = {"description": description}
    if "boolean" in tags:
        entry["type"] = "boolean"
    elif "value" in tags:
        entry["type"] = "enum"
        entry["values"] = {int(v.split(" ", 1)[0]): v.split(" ", 1)[1] for v in tags["value"]}
    else:
        entry["type"] = "float" if kind == "FLOAT" else "int32"
    entry["default"] = float(default) if kind == "FLOAT" else int(default)
    if "unit" in tags:
        entry["unit"] = tags["unit"][0]
    for key in ("min", "max", "decimal"):
        if key in tags:
            entry[key] = _number(tags[key][0])
    if tags.get("reboot_required") == ["true"]:
        entry["reboot_required"] = True
    return tags["group"][0], entry


def build(params_c: Path = PARAMS_C) -> dict:
    """The module config as a dict: every PARAM_DEFINE of the C file, in its order."""
    groups: dict[str, dict] = {}
    for block, kind, name, default in BLOCK.findall(params_c.read_text()):
        group, entry = _definition(block, kind, default)
        groups.setdefault(group, {})[name] = entry
    return {"module_name": "structure_hold",
            "parameters": [{"group": g, "definitions": d} for g, d in groups.items()]}


def render(config: dict) -> str:
    class Dumper(yaml.SafeDumper):
        pass

    def text(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
        style = "|" if "\n" in data else None  # long descriptions keep their line breaks
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    Dumper.add_representer(str, text)
    return HEADER + yaml.dump(config, Dumper=Dumper, sort_keys=False, width=100,
                              allow_unicode=True)


def differences(parameters_json: Path, params_c: Path = PARAMS_C) -> list[str]:
    """Where a PX4 build's generated SH_ parameter metadata departs from the C file."""
    built = {p["name"]: p for p in json.loads(parameters_json.read_text())["parameters"]
             if p["name"].startswith("SH_")}
    ours = {n: d for g in build(params_c)["parameters"] for n, d in g["definitions"].items()}
    out = [f"{n}: not in the build" for n in ours if n not in built]
    out += [f"{n}: in the build, not in {params_c.name}" for n in built if n not in ours]
    for name, d in ours.items():
        b = built.get(name)
        if b is None:
            continue
        desc = d["description"]
        want = {
            "type": "Float" if d["type"] == "float" else "Int32",
            "default": d["default"],
            "min": d.get("min"),
            "max": d.get("max"),
            "units": d.get("unit"),
            "decimalPlaces": d.get("decimal"),
            "rebootRequired": d.get("reboot_required", False),
            "shortDesc": desc["short"],
            "longDesc": " ".join(desc.get("long", "").split()) or None,
        }
        got = {k: b.get(k) for k in want}
        got["rebootRequired"] = b.get("rebootRequired", False)
        got["longDesc"] = " ".join(b.get("longDesc", "").split()) or None
        if d["type"] == "enum":
            want["values"] = d["values"]
            got["values"] = {v["value"]: v["description"] for v in b.get("values", [])}
        out += [f"{name}.{k}: build {got[k]!r}, {params_c.name} {want[k]!r}"
                for k in want if got[k] != want[k]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if stale")
    ap.add_argument("--against", type=Path, metavar="PARAMETERS_JSON",
                    help="compare a PX4 build's parameters.json with the C file; exit 1 if "
                         "they differ")
    args = ap.parse_args()
    if args.against:
        diffs = differences(args.against)
        print("\n".join(diffs) if diffs else
              f"{args.against}: every SH_ parameter as in {PARAMS_C.name}")
        return 1 if diffs else 0
    fresh = render(build())
    if args.check:
        if not PARAMS_YAML.exists() or PARAMS_YAML.read_text() != fresh:
            print(f"{PARAMS_YAML.name} is stale: run python -m sim2real.px4.tools.params_yaml",
                  file=sys.stderr)
            return 1
        return 0
    PARAMS_YAML.write_text(fresh)
    print(f"wrote {PARAMS_YAML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
