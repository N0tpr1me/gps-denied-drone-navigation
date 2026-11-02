"""docs/integration-guide.md's parameter table matches structure_hold_params.c.

The guide is the interface spec a drone maker reads, so its defaults and ranges are a copy
of the module's metadata; this keeps the copy from drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

from sim2real.px4.tests.test_param_limits import _params_c

GUIDE = Path(__file__).resolve().parents[3] / "docs" / "integration-guide.md"


def _guide_rows() -> dict[str, tuple[float, str]]:
    rows = re.findall(r"^\| `(SH_\w+)` \| (-?[\d.]+) \| ([^|]+) \|", GUIDE.read_text(), re.M)
    return {name: (float(default), rng.strip()) for name, default, rng in rows}


def _range(text: str) -> tuple[float, float]:
    lo, hi = re.fullmatch(r"(-?[\d.]+) (?:to|or) (-?[\d.]+)", text).groups()
    return float(lo), float(hi)


def test_every_parameter_is_in_the_guide() -> None:
    params, rows = _params_c(), _guide_rows()
    assert set(params) == set(rows), (sorted(set(params) - set(rows)),
                                      sorted(set(rows) - set(params)))


def test_guide_defaults_and_ranges_match() -> None:
    params = _params_c()
    for name, (default, rng) in _guide_rows().items():
        p = params[name]
        assert default == p["default"], name
        if "min" in p:
            assert _range(rng) == (p["min"], p["max"]), name


def test_guide_links_the_safety_manual() -> None:
    assert "docs/safety-manual.md" in GUIDE.read_text()
