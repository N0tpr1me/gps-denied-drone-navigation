"""The Gazebo-like check: its wobble measure, and a smoke run through launch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from eval import gazebo_check  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_band_rms_picks_out_the_wobble_band():
    dt = 0.012
    t = np.arange(0, 40.0, dt)[:, None]
    wobble = 0.3 * np.sin(2 * np.pi * 1.63 * t)  # Gazebo's limit cycle
    slow = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    fast = 0.3 * np.sin(2 * np.pi * 8.0 * t)
    x = np.hstack([wobble, slow, fast, wobble + slow])
    rms = gazebo_check.band_rms(x, dt)
    assert rms[0] == pytest.approx(0.3 / np.sqrt(2), rel=0.02)
    assert rms[1] < 0.01 and rms[2] < 0.01
    assert rms[3] == pytest.approx(0.3 / np.sqrt(2), rel=0.05)


def test_paired_ratio_and_describe():
    rows_a = [{"p95_m": 0.5, "crash": False, "lift_m": 0.1, "wobble": True, "wobble_rms": 0.2}] * 8
    rows_b = [{"p95_m": 1.0, "crash": False, "lift_m": 0.3, "wobble": False, "wobble_rms": 0.0}] * 8
    cmp = gazebo_check.paired_p95_ratio(rows_a, rows_b, np.random.default_rng(0), 100)
    assert cmp["ratio"] == pytest.approx(0.5) and cmp["ci"][1] < 1.0
    d = gazebo_check.describe(rows_a + [{"p95_m": 2.0, "crash": True}])
    assert d["crash_rate"] == pytest.approx(1 / 9) and d["wobble_rate"] == 1.0
    assert d["lift_m_mean"] == pytest.approx(0.1)


def test_check_seeds_stay_out_of_the_gate1_block():
    import yaml

    cfg = yaml.safe_load((REPO / "eval/configs/gazebo_check.yaml").read_text())
    assert all(not 9000 <= s < 10000 for s in cfg["params"]["seeds"])
    assert cfg["params"]["base_env"]["airframe"] == "gz_x500_sitl"
    assert cfg["smoke"]["base_env"]["airframe"] == "gz_x500_sitl"


def test_smoke_run_writes_report(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "training.launch", "--config", "eval/configs/gazebo_check.yaml",
         "--smoke", "--device", "cpu", "--runs-dir", str(tmp_path), "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    report = json.loads((run_dir / "report.json").read_text())
    assert [c["name"] for c in report["cells"]] == ["calm_clean", "calm", "wind4", "wind8",
                                                    "wind8_wall"]
    assert report["reference"] == "v9" and "beats" in report["verdicts"]["v6"]
    assert "| cell | controller |" in (run_dir / "report.md").read_text()
