"""Route Follow scoring and its run entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from envs.route_follow import RouteFollowConfig, RouteFollowEnv  # noqa: E402
from eval import route_follow  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_score_controller_counts_episodes_and_crashes():
    cfg = RouteFollowConfig.from_dict({"episode_length_s": 8.0, "contact_radius_m": 0.4})
    env = RouteFollowEnv(cfg, 4, seed=1)
    crash = route_follow.score_controller(env, lambda obs: torch.tensor(
        [[1.0, 0.0, 0.0, 1.0]]).repeat(4, 1))
    assert crash["crash_rate"] == 1.0 and crash["completion_rate"] == 0.0
    assert crash["episodes"] >= 4
    env = RouteFollowEnv(cfg, 4, seed=1)
    pd = route_follow.YARDSTICKS["pd"](env)
    h = route_follow.score_controller(env, pd)
    assert h["episodes"] == 4 and h["crash_rate"] == 0.0
    assert 0.0 < h["cross_track_p95_m"] < 0.6 and h["min_clearance_m"] > 0.4
    assert set(route_follow.HEADLINE_KEYS) <= set(h)


def test_entrypoint_scores_the_yardsticks_on_every_set(tmp_path: Path):
    cfg = yaml.safe_load((REPO / "eval" / "configs" / "route_follow_baseline.yaml").read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"]["num_envs"] = 2
    cfg["params"]["sets"] = cfg["params"]["sets"][:2]
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    proc = subprocess.run([sys.executable, "-m", "eval.route_follow", "--run-dir", str(run),
                           "--config", str(config), "--device", "cpu", "--smoke"],
                          cwd=REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert "corridor/pd/cross_track_p95_m" in summary["headline"]
    assert "corridor_ranges/pd_centring/crash_rate" in summary["headline"]
    assert "pd_centring" in summary["vs_baseline"]


def test_compare_scores_several_policies_before_the_yardsticks():
    cfg = RouteFollowConfig.from_dict({"episode_length_s": 7.0, "contact_radius_m": 0.4,
                                       "route": {"branches": 1, "arch_prob": 1.0}})
    pd = route_follow.YARDSTICKS["pd"]
    res = route_follow.compare(cfg, {"a": pd, "b": pd}, num_envs=2, episodes=1, seed=3)
    assert list(res) == ["a", "b", "pd", "pd_centring"]
    assert res["a"] == res["pd"]  # same controller, identically seeded envs
