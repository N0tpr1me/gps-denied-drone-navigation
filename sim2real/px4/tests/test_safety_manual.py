"""docs/safety-manual.md matches the module it signs off.

The manual is a signed document, so the pilot messages it lists (section 4) and the
parameter defaults it recommends against (section 6) are copies of the module's code; this
keeps the copies from drifting after a signature.
"""

from __future__ import annotations

import re
from pathlib import Path

from sim2real.px4.tests.test_param_limits import _params_c

ROOT = Path(__file__).resolve().parents[3]
MANUAL = (ROOT / "docs" / "safety-manual.md").read_text()
MODULE_CPP = (ROOT / "sim2real" / "px4" / "structure_hold" / "StructureHold.cpp").read_text()

# Placeholders the manual writes for a value the module fills in.
_PLACEHOLDER = re.compile(r"<[^>]+>|\b[XYN]\b")


def _section(title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## )", MANUAL, re.M | re.S)
    assert m, title
    return m.group(1)


def _manual_messages() -> list[str]:
    rows = re.findall(r"^\| (`.+?`) \| (?:critical|warning|info) \|",
                      _section("4. What the pilot sees, and what to do"), re.M)
    return [msg for row in rows for msg in re.findall(r"`([^`]+)`", row)]


def _module_strings() -> list[str]:
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"', MODULE_CPP)
    return [re.sub(r"%[-+ #0-9.]*[sdfiu]", "\0", s) for s in literals]


def _pattern(message: str) -> re.Pattern:
    parts = _PLACEHOLDER.split(message)
    return re.compile(".+".join(re.escape(p) for p in parts), re.S)


def test_manual_lists_messages() -> None:
    assert len(_manual_messages()) >= 10


def test_every_pilot_message_is_in_the_module() -> None:
    strings = _module_strings()
    for message in _manual_messages():
        pat = _pattern(message)
        bare = message.removeprefix("Structure Hold: ")
        assert any(pat.fullmatch(s) or s == bare for s in strings), message


def test_recommended_defaults_match_the_module() -> None:
    params = _params_c()
    rows = re.findall(r"^\| `(SH_\w+)` \| (-?[\d.]+)\b",
                      _section("6. What a reviewer should set or decide"), re.M)
    assert len(rows) >= 8
    for name, default in rows:
        assert name in params, name
        assert float(default) == params[name]["default"], name
