"""The hold datasheet: cell names, the envelope rule and the table, on made-up reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eval import hold_datasheet

REPO = Path(__file__).resolve().parents[2]


def cell(name: str, v22: tuple[float, float, int], pd: tuple[float, float, int],
         n: int = 512) -> dict:
    def ctl(p95: float, crash: float, k: int) -> dict:
        from eval.gate1 import wilson

        return {"episodes": n, "p95_m_mean": p95, "p95_ci": [p95 - 0.01, p95 + 0.01],
                "crash_rate": crash, "crash_ci": list(wilson(k, n)),
                "worst_toward_surface_m": 0.3}

    return {"name": name, "label": name, "controllers": {"v22": ctl(*v22), "pd": ctl(*pd)}}


def report(cells: list[dict]) -> dict:
    return {"gate_controller": "v22", "cells": cells}


def test_cell_names_carry_the_sweep_coordinates():
    assert hold_datasheet.parse_cell("into_w10_d1_vio") == {
        "direction": "into", "wind": 10, "distance": 1, "estimate": "vio"}
    assert hold_datasheet.parse_cell("lee_w04_d3_dropout3")["estimate"] == "dropout3"
    assert hold_datasheet.parse_cell("calm_clean") is None


def test_envelope_is_the_last_wind_before_the_first_break():
    into = report([
        cell("into_w00_d1_vio", (0.15, 0.0, 0), (0.30, 0.0, 0)),
        cell("into_w04_d1_vio", (0.18, 0.0, 0), (0.35, 0.0, 0)),
        cell("into_w08_d1_vio", (0.20, 0.0, 0), (0.45, 0.02, 10)),  # the PD crashes 2 %
        cell("into_w12_d1_vio", (0.40, 0.0, 0), (0.70, 0.10, 51)),
        cell("into_w14_d1_vio", (0.55, 0.05, 26), (0.90, 0.30, 154)),  # v22 past both limits
    ])
    along = report([
        cell("along_w04_d1_vio", (0.17, 0.0, 0), (0.33, 0.0, 0)),
        cell("along_w08_d1_vio", (0.19, 0.0, 0), (0.40, 0.0, 0)),
    ])
    ds = hold_datasheet.build([into, along])
    env = {(e["direction"], e["distance"], e["estimate"]): e["controllers"]
           for e in ds["envelope"]}
    assert env[("into", 1, "vio")]["v22"] == {"holds_to_mps": 12, "breaks_at_mps": 14,
                                            "winds_flown": [0, 4, 8, 12, 14]}
    assert env[("into", 1, "vio")]["pd"]["holds_to_mps"] == 4
    assert env[("into", 1, "vio")]["pd"]["breaks_at_mps"] == 8
    # The calm cell was flown once and counts for every direction; along never broke.
    assert env[("along", 1, "vio")]["v22"] == {"holds_to_mps": 8, "breaks_at_mps": None,
                                             "winds_flown": [0, 4, 8]}
    md = hold_datasheet.markdown(ds)
    assert "| wind into the wall | 1 m | vio_default | 12 m/s | 14 m/s | 4 m/s | 8 m/s |" in md
    assert "not within 8 m/s" in md
    assert "| 14 m/s | 0.55 m, 5.0% / 0.90 m, 30.0% |" in md


def test_tighter_limits_shrink_the_envelope():
    rep = report([cell("into_w04_d1_vio", (0.18, 0.0, 0), (0.35, 0.0, 0)),
                  cell("into_w08_d1_vio", (0.30, 0.0, 0), (0.45, 0.0, 0))])
    loose = hold_datasheet.build([rep])["envelope"][0]["controllers"]["v22"]
    tight = hold_datasheet.build([rep], p95_limit_m=0.25)["envelope"][0]["controllers"]["v22"]
    assert loose["holds_to_mps"] == 8 and tight["holds_to_mps"] == 4


def test_a_cell_that_is_not_a_sweep_cell_is_refused():
    with pytest.raises(ValueError, match="not a sweep cell"):
        hold_datasheet.build([report([cell("calm_clean", (0.1, 0, 0), (0.2, 0, 0))])])


def test_cli_writes_markdown_and_json(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps(report([
        cell("into_w04_d1_vio", (0.18, 0.0, 0), (0.35, 0.0, 0))])))
    assert hold_datasheet.main([str(run), "--out", str(tmp_path / "ds.md")]) == 0
    assert (tmp_path / "ds.md").read_text().startswith("# Hold datasheet")
    assert json.loads((tmp_path / "ds.json").read_text())["controllers"] == ["v22", "pd"]


def test_sweep_configs_cover_the_grid_and_name_their_cells():
    seen = set()
    for key, winds in (("into", {0, 4, 8, 10, 12, 14}), ("along", {4, 8, 10, 12, 14}),
                       ("lee", {4, 8, 10, 12, 14})):
        cfg = yaml.safe_load((REPO / f"eval/configs/hold_datasheet_{key}.yaml").read_text())
        assert cfg["entrypoint"] == "eval.gate1" and cfg["expected_hours"] <= 1.0
        assert all(s not in range(9000, 10000) for s in cfg["params"]["seeds"])
        cells = cfg["params"]["cells"]
        coords = [hold_datasheet.parse_cell(c["name"]) for c in cells]
        assert all(coords), key
        assert {c["wind"] for c in coords} == winds
        assert {c["distance"] for c in coords} == {1, 3}
        assert {c["estimate"] for c in coords} == set(hold_datasheet.ESTIMATE_ORDER)
        assert len(cells) == len(winds) * 2 * 4
        for c, co in zip(cells, coords, strict=True):
            assert c["env"]["disturbance"]["mean_speed"] == co["wind"]
            assert c["env"]["hold_distance"] == co["distance"]
            # Inside the released envelope the run still judges v22 against the PD.
            assert c["envelope"] == (co["wind"] <= 8 and co["estimate"] in ("clean", "vio"))
            seen.add(c["name"])
        assert "smoke" in cfg
    assert len(seen) == 128
