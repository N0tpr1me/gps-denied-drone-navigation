"""Failure rates per fault: the drift offsets, the bound wording, the collected table, and
a tiny run on the PD alone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")

from eval import fault_rates  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_drift_walks_then_holds():
    off = fault_rates.drift_offsets({"start_s": 1.0, "duration_s": 2.0, "rate_mps": 0.5,
                                     "direction": [-2.0, 0.0, 0.0]}, 5.0, 0.1)
    assert off.shape == (50, 3)
    assert np.all(off[:10] == 0.0)
    assert off[20, 0] == pytest.approx(-0.5, abs=1e-6)  # 1 s into the walk at 0.5 m/s
    assert off[-1, 0] == pytest.approx(-1.0, abs=1e-6)  # held at 2 s x 0.5 m/s
    assert np.all(off[:, 1:] == 0.0)


def test_one_in_rounds_the_bound_down():
    assert fault_rates.one_in(1 / 5432.1) == "1 in 5,000"
    assert fault_rates.one_in(0.25) == "1 in 4.0"
    assert fault_rates.one_in(0.0) == "-"


def test_describe_counts_crashes_and_contacts():
    rows = [{"crash": False, "reason": "timeout", "p95_m": 0.2, "toward_wall_m": 0.1}] * 98
    rows += [{"crash": True, "reason": "contact", "p95_m": 0.9, "toward_wall_m": 0.7},
             {"crash": True, "reason": "tilt", "p95_m": 0.5, "toward_wall_m": 0.2}]
    d = fault_rates.describe(rows)
    assert d["episodes"] == 100 and d["crashes"] == 2 and d["contacts"] == 1
    assert d["crash_rate"] == pytest.approx(0.02) and d["crash_ci"][1] > 0.02
    assert d["crash_reasons"] == {"contact": 1, "tilt": 1}
    assert d["toward_wall_m_max"] == pytest.approx(0.7)


def test_collect_merges_one_fault_per_run(tmp_path):
    def rep(name: str) -> dict:
        d = {"episodes": 10, "crashes": 0, "crash_rate": 0.0, "crash_ci": [0.0, 0.28],
             "crash_upper_one_in": "1 in 3.6", "crash_reasons": {}, "contacts": 0,
             "contact_rate": 0.0, "contact_ci": [0.0, 0.28], "contact_upper_one_in": "1 in 3.6",
             "p95_m_mean": 0.2, "toward_wall_m_mean": 0.1, "toward_wall_m_max": 0.2}
        return {"seeds": [1], "num_envs": 10, "episodes_per_fault": 10,
                "controllers": [{"name": "v22", "policy": "x.onnx"}, {"name": "pd"}],
                "faults": [{"name": name, "fmea": "F1", "label": name,
                            "controllers": {"v22": d, "pd": d}}]}

    for name in ("reset", "drift"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.json").write_text(json.dumps(rep(name)))
    out = tmp_path / "t" / "fault_rates.md"
    assert fault_rates.main(["collect", str(tmp_path / "reset"), str(tmp_path / "drift"),
                             "--out", str(out)]) == 0
    md = out.read_text()
    assert "| F1 | reset | 0 / 10 | 1 in 3.6 |" in md and "| F1 | drift |" in md
    assert [f["name"] for f in json.loads(out.with_suffix(".json").read_text())["faults"]] == [
        "reset", "drift"]


def test_collect_refuses_mixed_sample_sizes(tmp_path):
    for name, n in (("a", 10), ("b", 20)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.json").write_text(json.dumps(
            {"episodes_per_fault": n, "controllers": [], "faults": []}))
    with pytest.raises(ValueError, match="episodes per fault"):
        fault_rates.collect([tmp_path / "a", tmp_path / "b"])


class Writer:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.rows: list[dict] = []

    def metric(self, step: int, **values) -> None:
        self.rows.append(values)


def test_run_flies_the_pd_through_a_reset_and_a_drift(tmp_path):
    cfg = {"params": {
        "controllers": [], "seeds": [7301], "num_envs": 2, "window_s": 0.5,
        "base_env": {"airframe": "gz_x500_sitl", "episode_length_s": 1.0, "hold_distance": 1.0,
                     "hold_height": 2.4, "contact_radius_m": 0.4, "start_offset_std_m": 0.0,
                     "start_yaw_std_rad": 0.0,
                     "estimate": {"jump_rate_hz": 0.0, "dropout_rate_hz": 0.0}},
        "faults": [
            {"name": "reset", "fmea": "F1", "label": "reset",
             "env": {"estimate": {"jump_decay_s": None}},
             "events": [{"t_s": 0.3, "jump": [1.0, 0.0, -2.4]}]},
            {"name": "drift", "fmea": "F7", "label": "drift",
             "env": {"estimate": {"bias_walk_std_m_per_sqrt_s": 0.0, "bias_max_m": 10.0}},
             "drift": {"start_s": 0.2, "duration_s": 0.5, "rate_mps": 0.03,
                       "direction": [-1.0, 0.0, 0.0]}},
            {"name": "skipped", "label": "not flown", "events": []},
        ],
        "only": ["reset", "drift"],
    }}
    w = Writer(tmp_path)
    out = fault_rates.run(cfg, "cpu", w)
    rep = json.loads((tmp_path / "report.json").read_text())
    assert [f["name"] for f in rep["faults"]] == ["reset", "drift"]
    assert rep["episodes_per_fault"] == 2
    assert set(rep["faults"][0]["controllers"]) == {"pd"}
    assert len(w.rows) == 4 and {r["fault"] for r in w.rows} == {"reset", "drift"}
    assert out["headline"]["reset_pd_crash"].endswith("/2")
    assert (tmp_path / "report.md").read_text().startswith("# Failure rates per fault")


def test_run_refuses_an_unknown_fault_in_only(tmp_path):
    cfg = {"params": {"controllers": [], "seeds": [7301], "num_envs": 2,
                      "faults": [{"name": "reset"}], "only": ["typo"]}}
    with pytest.raises(ValueError, match="unknown fault"):
        fault_rates.run(cfg, "cpu", Writer(tmp_path))


def test_shipped_config_names_every_fmea_row_once_per_fault():
    cfg = yaml.safe_load((REPO / "eval/configs/fault_rates_v22.yaml").read_text())
    names = [f["name"] for f in cfg["params"]["faults"]]
    assert len(names) == len(set(names)) == 9
    assert all(f.get("fmea", "").startswith("F") for f in cfg["params"]["faults"])
    assert all(s not in range(9000, 10000) for s in cfg["params"]["seeds"])
    assert cfg["params"]["num_envs"] * len(cfg["params"]["seeds"]) == 20480
    assert "smoke" in cfg and cfg["expected_hours"] <= 1.0
