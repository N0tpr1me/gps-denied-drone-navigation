"""The software parts list and the assurance pack stay true to the code.

A buyer reads sbom.cdx.json and docs/assurance-pack.md as statements about what we ship, so
these tests fail when a dependency, a pin, the released policy or the module's includes
change without the paperwork following.
"""

from __future__ import annotations

import json
import re

import pytest

from sim2real.assurance import make_sbom as sb

ROOT = sb.ROOT
PACK = (ROOT / "docs" / "assurance-pack.md").read_text()


def _refs(bom: dict) -> dict[str, dict]:
    return {c["bom-ref"]: c for c in bom["components"]}


def test_checked_in_sbom_is_current() -> None:
    assert sb.main(["--check"]) == 0, "run: python -m sim2real.assurance.make_sbom"


def test_sbom_shape() -> None:
    bom = json.loads(sb.OUT.read_text())
    assert bom["bomFormat"] == "CycloneDX" and bom["specVersion"] == "1.6"
    refs = _refs(bom)
    assert len(refs) == len(bom["components"]), "bom-refs must be unique"
    for c in bom["components"]:
        assert c["supplier"]["name"] and c["licenses"], c["name"]
        role = c["properties"][0]
        assert role["name"] == "dronenav:role" and role["value"] in sb.ROLES, c["name"]
    for dep in bom["dependencies"]:
        assert all(d in refs for d in dep["dependsOn"])


def test_only_our_code_and_px4_go_on_the_drone() -> None:
    onboard = {ref for ref, c in _refs(sb.build()).items()
               if c["properties"][0]["value"] == "onboard"}
    assert onboard == {"structure-hold-module", "structure-hold-policy", "px4-autopilot"}


def test_module_uses_only_px4_and_the_standard_library() -> None:
    # A new library linked into the module must be added to the SBOM's onboard parts.
    cmake = (sb.MODULE_DIR / "CMakeLists.txt").read_text()
    depends = re.search(r"DEPENDS\s+(.*?)\)", cmake, re.S).group(1).split()
    assert set(depends) <= {"px4_work_queue", "mathlib"}, depends
    local = {p.name for p in sb.MODULE_DIR.iterdir()}
    px4_prefixes = ("uORB/", "px4_platform_common/", "commander/", "systemlib/",
                    "lib/", "drivers/drv_hrt.h", "parameters/", "matrix/", "mathlib/")
    std = {"cstdint", "cmath", "cstring", "cstdlib", "cstddef", "cfloat", "climits",
           "algorithm", "array", "limits", "type_traits", "math.h", "stdint.h", "float.h"}
    for path in sb.MODULE_DIR.iterdir():
        if path.suffix not in {".hpp", ".cpp", ".c", ".h"}:
            continue
        for inc in re.findall(r'^\s*#include\s+[<"]([^>"]+)[>"]', path.read_text(), re.M):
            ok = inc in local or inc in std or inc.startswith(px4_prefixes)
            assert ok, f"{path.name} includes {inc}: add it to the SBOM, then here"


def test_every_declared_python_dependency_is_listed() -> None:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = list(project["dependencies"])
    for group in project["optional-dependencies"].values():
        declared += group
    names = {re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in declared}
    listed = {c["name"].lower() for c in sb.build()["components"]
              if c.get("purl", "").startswith("pkg:pypi/")}
    assert names <= listed, names - listed


def test_pins_match_the_build_files() -> None:
    p = sb.pins()
    refs = _refs(sb.build(p))
    assert refs["px4-autopilot"]["version"] == p["px4"] == "v1.17.0"  # Decided pin
    assert refs["pypi-mavsdk"]["version"] == p["mavsdk"]
    assert refs["basalt"]["version"] == p["basalt"]
    assert refs["structure-hold-policy"]["name"] == p["policy"]
    assert (ROOT / "training" / "policies" / f"{p['policy']}.onnx").exists()


def test_release_copy_hashes_every_onboard_file(tmp_path) -> None:
    bom = sb.release(sb.build())
    files = _refs(bom)["structure-hold-module"]["components"]
    names = {f["name"] for f in files}
    assert "sim2real/px4/structure_hold/StructureHold.cpp" in names
    assert "sim2real/px4/msg/StructureHoldStatus.msg" in names
    assert all(len(f["hashes"][0]["content"]) == 64 for f in files)
    assert bom["metadata"]["timestamp"].endswith("Z")


def test_pack_is_confidential_and_claims_no_certification() -> None:
    head = PACK[:600]
    assert "Confidential" in head and "All rights reserved" in head
    assert "holds no certification" in PACK


def test_pack_points_at_files_that_exist() -> None:
    for path in re.findall(r"`((?:docs|sim2real|eval|training|\.github)/[\w./-]+)`", PACK):
        assert (ROOT / path).exists(), path
