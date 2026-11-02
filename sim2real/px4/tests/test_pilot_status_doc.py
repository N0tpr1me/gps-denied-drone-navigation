"""docs/pilot-status.md lists the live array exactly as PilotStatus.hpp fills it.

A flight-test engineer reads the array by index in the MAVLink Inspector, so a table that
drifts from the code would point them at the wrong number.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEADER = (ROOT / "sim2real" / "px4" / "structure_hold" / "PilotStatus.hpp").read_text()
DOC = (ROOT / "docs" / "pilot-status.md").read_text()


def _enum() -> dict[str, int]:
    found = dict(re.findall(r"^\s*LIVE_(\w+) = (\d+),", HEADER, re.M))
    found.pop("COUNT", None)
    return {name.lower(): int(i) for name, i in found.items()}


def _doc_rows() -> dict[str, int]:
    return {name: int(i) for i, name in re.findall(r"^\| (\d+) \| `(\w+)` \|", DOC, re.M)}


def test_live_array_table_matches_the_header() -> None:
    enum = _enum()
    assert len(enum) >= 10
    assert _doc_rows() == enum


def test_live_count_follows_the_last_index() -> None:
    count = int(re.search(r"LIVE_COUNT = (\d+)", HEADER).group(1))
    assert count == max(_enum().values()) + 1
