from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "training" / "configs" / "structure_hold_rollout.yaml"


def test_rollout_entrypoint_writes_metrics_and_summary(tmp_path: Path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"]["env"]["episode_length_s"] = 1.0
    cfg["params"]["num_envs"] = 4
    cfg["params"]["score_envs"] = 2
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "envs.structure_hold.run", "--run-dir", str(run_dir),
         "--config", str(config), "--device", "cpu"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "ok"
    head = summary["headline"]
    assert head["episodes_scored"] == 2
    assert head["p95_m_mean"] < 1.0 and head["contact_rate"] == 0.0
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and {"p95_m", "worst_toward_surface_m", "contacts"} <= set(rows[0])


def test_rollout_entrypoint_reports_a_bad_config(tmp_path: Path):
    cfg = {"name": "x", "entrypoint": "envs.structure_hold.run", "seed": 0,
           "params": {"num_envs": 2, "controller": "teleport"}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "envs.structure_hold.run", "--run-dir", str(run_dir),
         "--config", str(config), "--device", "cpu"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "failed" and "teleport" in summary["error"]
