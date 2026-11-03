"""Route Follow training: the vec env adapter and the smoke run end to end."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("onnxruntime")

from envs.route_follow import NUM_OBS, RouteFollowConfig, RouteFollowEnv  # noqa: E402
from training.policy_io import load_controller  # noqa: E402
from training.route_ppo_env import PRIVILEGED_OBS, RouteFollowVecEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_vecenv_groups_and_privileged_truth():
    cfg = RouteFollowConfig.from_dict({"episode_length_s": 8.0, "contact_radius_m": 0.4})
    vec = RouteFollowVecEnv(RouteFollowEnv(cfg, 4, seed=0), history=3)
    td = vec.get_observations()
    assert td["policy"].shape == (4, 3 * NUM_OBS)
    assert td["privileged"].shape == (4, PRIVILEGED_OBS)
    td, rew, done, _ = vec.step(torch.zeros(4, 4))
    # The newest observation is the last NUM_OBS values of the stack.
    assert torch.equal(td["policy"][:, -NUM_OBS:], vec.env.obs_buf)
    assert (td["privileged"][:, -1] > 0).all()  # true clearance, inside the walls
    vec.episode_length_buf = torch.full((4,), 100)
    assert torch.allclose(vec.env.episode_time, torch.full((4,), 100 * cfg.dt))
    # The randomised phase puts the drone on the target as it stands then, flying with it,
    # with a flat history; left at the first waypoint it would be lost on the first step.
    e = vec.env
    assert torch.allclose(e.sim.state.pos, e.target_pos, atol=1e-5)
    assert torch.allclose(e.sim.state.vel, e.target_vel, atol=1e-5)
    td = vec.get_observations()
    assert torch.equal(td["policy"][:, -NUM_OBS:], td["policy"][:, :NUM_OBS])
    _, _, done, _ = vec.step(torch.zeros(4, 4))
    assert not done.any()


def test_smoke_training_exports_a_policy_scored_against_both_yardsticks(tmp_path: Path):
    cfg = yaml.safe_load((REPO / "training" / "configs" / "route_follow_ppo.yaml").read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"].update({"num_envs": 8, "iterations": 2, "num_steps_per_env": 8,
                          "eval": {"num_envs": 2, "episodes": 1, "env": {}}})
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    proc = subprocess.run([sys.executable, "-m", "training.train_route_ppo", "--run-dir",
                           str(run), "--config", str(config), "--device", "cpu", "--smoke"],
                          cwd=REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert {"policy_crash_rate", "pd_crash_rate", "pd_centring_crash_rate"} <= set(
        summary["headline"])
    meta = json.loads((run / "policy.json").read_text())
    assert meta["format"] == "route_follow_policy/1" and meta["input"]["num_obs"] == NUM_OBS
    env = RouteFollowEnv(RouteFollowConfig.from_dict(cfg["params"]["env"]), 3, seed=4)
    ctl = load_controller(run / "policy.onnx", 3)
    act = ctl(env.obs_buf)
    assert act.shape == (3, 4) and act.abs().max() <= 1.0
