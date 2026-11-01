"""The module's own parameter limits (ParamLimits.hpp) match structure_hold_params.c.

PX4 does not enforce @min / @max, so the module clamps every parameter itself; this keeps
its numbers from drifting away from the metadata ground stations show.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "structure_hold"

# Read by the start script (SH_EN) or deliberately left alone (any nonzero SH_TEST_FAULT trips).
NOT_SANITIZED = {"SH_EN", "SH_TEST_FAULT"}


def _params_c() -> dict[str, dict]:
    text = (MODULE / "structure_hold_params.c").read_text()
    out: dict[str, dict] = {}
    for block, kind, name, default in re.findall(
            r"/\*\*(.*?)\*/\s*PARAM_DEFINE_(FLOAT|INT32)\((\w+),\s*([-\w.]+?)f?\);", text, re.S):
        entry: dict = {"kind": kind, "default": float(default)}
        for key in ("min", "max"):
            m = re.search(rf"@{key}\s+([-\d.]+)", block)
            if m:
                entry[key] = float(m.group(1))
        if "@boolean" in block:
            entry["min"], entry["max"] = 0.0, 1.0
        values = [float(v) for v in re.findall(r"@value\s+(-?\d+)", block)]
        if values and "min" not in entry:
            entry["min"], entry["max"] = min(values), max(values)
        out[name] = entry
    return out


def _limits_hpp() -> dict[str, tuple[str, float, float, float]]:
    text = (MODULE / "ParamLimits.hpp").read_text()
    num = r"([-\d.]+)f?"
    pattern = rf"static constexpr (FloatRange|IntRange) (\w+)\{{{num}, {num}, {num}\}};"
    return {name: (kind, float(lo), float(hi), float(d))
            for kind, name, lo, hi, d in re.findall(pattern, text)}


def test_every_module_parameter_has_limits() -> None:
    params = _params_c()
    limits = _limits_hpp()
    assert len(params) >= 25
    missing = set(params) - set(limits) - NOT_SANITIZED
    assert not missing, f"no ParamLimits.hpp entry for {sorted(missing)}"


def test_limits_match_metadata() -> None:
    params = _params_c()
    for name, (kind, lo, hi, default) in _limits_hpp().items():
        if name == "HOVER_THRUST":  # PX4's, not ours
            continue
        p = params[name]
        assert (kind == "FloatRange") == (p["kind"] == "FLOAT"), name
        assert (lo, hi, default) == (p["min"], p["max"], p["default"]), name


def test_every_limit_is_applied_in_the_module() -> None:
    cpp = (MODULE / "StructureHold.cpp").read_text()
    for name in _limits_hpp():
        if name == "HOVER_THRUST":
            assert "sanitizeHoverThrust(" in cpp
            continue
        assert f"param_limits::{name}, \"{name}\"" in cpp, name
