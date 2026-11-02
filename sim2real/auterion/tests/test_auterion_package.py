"""Checks on the AuterionOS app package that run without Docker, ROS or an Auterion account.

The app compiles in the onboard module's PX4-free headers; stage_app.sh copies them next to
the app for `auterion-cli app build`, and docker/Dockerfile.sitl copies the same list. A
header added to an #include chain and forgotten in either place would only fail inside a
Docker build, so these tests follow the includes and compare the lists.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
AUT = HERE.parent
REPO = AUT.parent.parent
MODULE = AUT.parent / "px4" / "structure_hold"
APP = AUT / "structure_hold_app"

INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"', re.M)


def _local_includes(start: Path) -> set[str]:
    """Module header names reached from ``start`` through quoted #includes."""
    seen: set[str] = set()
    todo = [start]
    while todo:
        text = todo.pop().read_text()
        for name in INCLUDE.findall(text):
            base = Path(name).name
            if (APP / "include" / name).exists():
                todo.append(APP / "include" / name)
            elif base not in seen:
                seen.add(base)
                todo.append(MODULE / base)
    return seen


def _stage_headers() -> set[str]:
    text = (AUT / "stage_app.sh").read_text()
    return set(re.search(r"HEADERS=\(([^)]*)\)", text).group(1).split())


def _sitl_headers() -> set[str]:
    text = (AUT / "docker" / "Dockerfile.sitl").read_text()
    return set(re.findall(r"sim2real/px4/structure_hold/(\w+\.hpp)", text))


def test_module_header_lists_cover_includes():
    needed = _local_includes(APP / "src" / "structure_hold_app.cpp")
    assert needed, "the app should include module headers"
    assert needed <= _stage_headers()
    assert needed <= _sitl_headers()
    assert _stage_headers() == _sitl_headers()


def test_manifest():
    m = yaml.safe_load((AUT / "package" / "auterion-app.yml").read_text())
    assert m["auterion-api-version"] >= 3  # the SDK and px4_ros2 packages need API v3+
    assert m["target-platform"] == "skynode"
    assert m["app-name"] == "structure-hold"
    assert list(m["services"].values())[0]["build"] == "."


def test_dockerfile_pins_match_doc():
    dockerfile = (AUT / "package" / "Dockerfile").read_text()
    doc = (REPO / "docs" / "auterion-app.md").read_text()
    ros2 = re.search(r"ARG PX4_ROS2_CPP=([\d.]+)-", dockerfile).group(1)
    msgs = re.search(r"ARG PX4_MSGS=([\d.]+)-", dockerfile).group(1)
    assert "FROM auterion/app-base:v2" in dockerfile
    # The doc's compatibility table names the AuterionOS row the pins come from.
    row = next(line for line in doc.splitlines() if line.startswith("| 4.6.2"))
    assert ros2 in row and msgs in row


def test_stage_app(tmp_path):
    out = tmp_path / "app"
    subprocess.run(["bash", str(AUT / "stage_app.sh"), str(out)], check=True,
                   capture_output=True)
    assert (out / "auterion-app.yml").is_file()
    assert (out / "Dockerfile").is_file()
    assert (out / "structure_hold_app" / "CMakeLists.txt").is_file()
    assert (out / "structure_hold_app" / "src" / "structure_hold_app.cpp").is_file()
    for h in _stage_headers():
        assert (out / "structure_hold_app" / "module" / h).is_file(), h
