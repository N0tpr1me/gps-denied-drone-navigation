"""The per-factor read-out (eval/environment_factors.py): the table against the reference
cell, parts joined, a tiny run on the PD and v22, and the shipped config's cells loading."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from envs.structure_hold.config import StructureHoldConfig  # noqa: E402
from eval import environment_factors as ef  # noqa: E402
from training.runlib import RunWriter, merge  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "eval/configs/environment_factors.yaml"


def report(cells: dict[str, tuple[float, float]]) -> dict:
    def ctl(p95: float, crash: float) -> dict:
        return {"p95_m_mean": p95, "crash_rate": crash, "episodes": 10,
                "worst_toward_surface_m": 0.1}

    return {"cells": [{"name": n, "label": n.title(), "cell": {"group": "g"},
                       "controllers": {"v22": ctl(a, 0.0), "pd": ctl(b, 0.1)}}
                      for n, (a, b) in cells.items()]}


def test_rows_are_changes_from_each_controllers_own_reference():
    rows = ef.factor_rows(report({"nominal": (0.20, 0.30), "rain": (0.25, 0.30)}), "nominal")
    rain = rows[1]["controllers"]
    assert rain["v22"]["change"] == pytest.approx(0.25) and rain["pd"]["change"] == 0.0
    text = ef.markdown(rows, ["v22", "pd"], "nominal")
    assert "| Rain | 0.250 (+25%) | 0.0% | 0.300 (+0%) | 10.0% |" in text
    assert "| Nominal | 0.200 | 0.0% | 0.300 | 10.0% |" in text
    # Without the reference flown the changes are blank, not wrong.
    alone = ef.factor_rows(report({"rain": (0.25, 0.30)}), "nominal")
    assert math.isnan(alone[0]["controllers"]["v22"]["change"])
    assert "(+" not in ef.markdown(alone, ["v22", "pd"], "nominal")


def test_collect_joins_parts_and_keeps_the_first_reference(tmp_path):
    for part, cells in (("a", {"nominal": (0.2, 0.3), "rain": (0.25, 0.3)}),
                        ("b", {"nominal": (0.2, 0.3), "glass": (0.4, 0.3)})):
        (tmp_path / part).mkdir()
        rows = ef.factor_rows(report(cells), "nominal")
        (tmp_path / part / "factors.json").write_text(json.dumps(
            {"reference": "nominal", "controllers": ["v22", "pd"], "rows": rows}))
    text = ef.collect([str(tmp_path / "a"), str(tmp_path / "b")], str(tmp_path / "out.md"))
    assert text.count("| Nominal |") == 1 and "| Rain |" in text and "| Glass |" in text
    assert (tmp_path / "out.md").read_text() == text


def test_every_shipped_cell_builds_an_env_config_and_switches_something_on():
    p = yaml.safe_load(CONFIG.read_text())["params"]
    names = [c["name"] for c in p["cells"]]
    assert len(names) == len(set(names)) and p["reference"] in names
    base = StructureHoldConfig.from_dict(p["base_env"])
    assert base.fusion.baro and not base.ambient.active
    for cell in p["cells"]:
        cfg = StructureHoldConfig.from_dict(merge(p["base_env"], cell.get("env")))
        if cell["name"] != p["reference"]:
            assert cfg != base, cell["name"]
    for s in p["seeds"]:
        assert not 9000 <= s <= 9999  # not Gate 1's held-out block


def test_tiny_run_writes_the_table_and_only_keeps_the_reference(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"]["only"] = ["rain", "range_glass"]
    cfg["params"]["num_envs"] = 3
    cfg["params"]["base_env"]["episode_length_s"] = 0.6
    writer = RunWriter(tmp_path)
    try:
        out = ef.run(cfg, "cpu", writer)
    finally:
        writer.close()
    data = json.loads((tmp_path / "factors.json").read_text())
    assert [r["name"] for r in data["rows"]] == ["nominal", "rain", "range_glass"]
    assert data["controllers"] == ["v22", "v24", "pd"]
    assert data["rows"][1]["group"] == "air"
    text = (tmp_path / "factors.md").read_text()
    assert "rangefinder on glass" in text and "| v22 p95 (m) | v22 crash |" in text
    assert out["headline"]["cells"] == 3 and "not Gazebo" in out["vs_baseline"]
    with pytest.raises(ValueError, match="unknown cells"):
        ef.run({**cfg, "params": {**cfg["params"], "only": ["nope"]}}, "cpu", writer)
