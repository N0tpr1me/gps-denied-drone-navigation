"""PPO training loop: the VecEnv adapter, the export, the eval entry and the smoke run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("onnxruntime")

from envs.structure_hold.config import StructureHoldConfig  # noqa: E402
from envs.structure_hold.env import NUM_OBS, StructureHoldEnv  # noqa: E402
from training import train_ppo  # noqa: E402
from training.policy_io import export_policy, load_controller, policy_metadata  # noqa: E402
from training.ppo_env import (  # noqa: E402
    PRIVILEGED_OBS,
    DropoutCurriculum,
    ObsHistory,
    StructureHoldVecEnv,
)
from training.runlib import RunWriter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PPO_CONFIG = REPO / "training" / "configs" / "structure_hold_ppo.yaml"
EVAL_CONFIG = REPO / "training" / "configs" / "structure_hold_eval.yaml"

SMALL = {"num_envs": 8, "iterations": 2, "num_steps_per_env": 8, "save_interval": 1,
         "env": {"episode_length_s": 1.0}, "eval": {"num_envs": 4, "episodes": 1,
                                                    "score_envs": 2, "select_every": 1,
                                                    "select_envs": 4}}


def smoke_params(**over):
    return train_ppo.merge(train_ppo.DEFAULTS, {**SMALL, **over})


# ---------------------------------------------------------------- history and VecEnv


def test_history_is_oldest_first_and_refills_on_reset():
    h = ObsHistory(num_envs=2, num_obs=2, length=3)
    h.fill(torch.tensor([[1.0, 1.0], [10.0, 10.0]]))
    h.push(torch.tensor([[2.0, 2.0], [20.0, 20.0]]))
    stacked = h.push(torch.tensor([[3.0, 3.0], [30.0, 30.0]]),
                     reset_mask=torch.tensor([False, True]))
    assert stacked[0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]  # oldest ... newest
    assert stacked[1].tolist() == [30.0] * 6  # a reset env sees only its new episode
    assert h.size == 6


def test_vecenv_groups_shapes_and_privileged_truth():
    env = StructureHoldEnv(StructureHoldConfig(episode_length_s=1.0), num_envs=4, seed=0)
    vec = StructureHoldVecEnv(env, history=3)
    obs = vec.get_observations()
    assert obs["policy"].shape == (4, 3 * NUM_OBS)
    assert obs["privileged"].shape == (4, PRIVILEGED_OBS)
    assert torch.equal(obs["policy"][:, :NUM_OBS], obs["policy"][:, -NUM_OBS:])  # filled
    obs2, reward, dones, extras = vec.step(torch.zeros(4, 4))
    assert reward.shape == (4,) and dones.shape == (4,) and "time_outs" in extras
    assert torch.allclose(obs2["privileged"][:, :3], env.state.pos - env.hold_point)
    assert torch.equal(vec.episode_length_buf, env.episode_step)
    vec.episode_length_buf = torch.tensor([5, 6, 7, 8])
    assert env.episode_step.tolist() == [5, 6, 7, 8]
    assert torch.allclose(env.episode_time, torch.tensor([5, 6, 7, 8]) * env.dt)


def test_dropout_curriculum_grows_the_cap_with_the_iteration():
    c = DropoutCurriculum(start_s=0.5, end_s=3.0, iterations=10)
    assert c.value(0) == 0.5 and c.value(5) == 1.75 and c.value(10) == 3.0 and c.value(99) == 3.0
    with pytest.raises(ValueError):
        DropoutCurriculum(start_s=4.0, end_s=3.0, iterations=10)

    env = StructureHoldEnv(StructureHoldConfig(episode_length_s=1.0), num_envs=2, seed=0)
    vec = StructureHoldVecEnv(env, history=2, curriculum=c, steps_per_iteration=4)
    assert env.dropout_max_s == pytest.approx(0.5)  # applied before the first step
    for _ in range(4):
        vec.step(torch.zeros(2, 4))
    assert env.dropout_max_s == pytest.approx(0.75)  # iteration 1
    for _ in range(40):
        vec.step(torch.zeros(2, 4))
    assert env.dropout_max_s == pytest.approx(3.0)  # back at the config's value
    assert vec.curriculum_state() == {"dropout_max_s": pytest.approx(3.0)}

    p = smoke_params(curriculum={"dropout_max_s": 0.5, "fraction": 0.5}, iterations=20)
    assert train_ppo.make_curriculum(p).iterations == 10
    assert train_ppo.make_curriculum(smoke_params()) is None  # off by default


# ---------------------------------------------------------------- training and export


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    writer = RunWriter(run_dir)
    p = smoke_params()
    runner, vec, env_cfg, train_cfg = train_ppo.make_runner(p, 0, "cpu", run_dir / "tb", writer)
    runner.learn(2)
    meta = policy_metadata(env_cfg, p["history"], train_cfg)
    onnx_path = export_policy(runner, run_dir, meta)
    writer.close()
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return {"runner": runner, "run_dir": run_dir, "onnx": onnx_path, "env_cfg": env_cfg,
            "metrics": rows}


def test_actor_is_small_and_metrics_are_written(trained):
    n = sum(x.numel() for x in trained["runner"].alg.get_policy().parameters())
    assert 5_000 < n < 20_000, n  # the plan's "about 10k parameters"
    assert trained["onnx"].stat().st_size < 1_000_000
    assert (trained["run_dir"] / "policy.pt").stat().st_size < 5_000_000
    rows = trained["metrics"]
    assert [r["step"] for r in rows] == [0, 1]
    assert {"reward_mean", "loss_value", "action_std", "env_steps_per_s",
            "dropout_max_s"} <= set(rows[0])
    assert (trained["run_dir"] / "policy.json").exists()
    assert (trained["run_dir"] / "tb" / "model_1.pt").exists()  # rsl_rl's periodic save


def test_onnx_matches_torch_actor_and_takes_any_batch(trained):
    import onnxruntime as ort

    policy = trained["runner"].get_inference_policy("cpu")
    from tensordict import TensorDict

    x = torch.randn(16, 3 * NUM_OBS)
    with torch.no_grad():
        want = policy(TensorDict({"policy": x}, batch_size=[16])).numpy()
    sess = ort.InferenceSession(str(trained["onnx"]), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"obs": x.numpy()})
    assert got.shape == (16, 4)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_onnx_and_pt_controllers_agree_on_the_env(trained):
    env = StructureHoldEnv(trained["env_cfg"], num_envs=3, seed=1)
    onnx_ctl = load_controller(trained["onnx"], 3)
    pt_ctl = load_controller(trained["run_dir"] / "policy.pt", 3)
    obs, _ = env.get_observations()
    for _ in range(5):
        a, b = onnx_ctl(obs), pt_ctl(obs)
        assert a.shape == (3, 4) and a.abs().max() <= 1.0
        assert torch.allclose(a, b, atol=1e-4)
        obs, *_ = env.step(a)


def test_load_controller_rejects_a_mismatched_history(trained, tmp_path):
    import shutil

    bad = tmp_path / "policy.onnx"
    shutil.copy(trained["onnx"], bad)
    meta = json.loads((trained["run_dir"] / "policy.json").read_text())
    meta["input"]["history"] = 2
    (tmp_path / "policy.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="history"):
        load_controller(bad, 2)


# ---------------------------------------------------------------- entrypoints


def run_entry(module: str, cfg: dict, run_dir: Path) -> dict:
    config = run_dir.parent / f"{run_dir.name}.yaml"
    config.write_text(yaml.safe_dump(cfg))
    proc = subprocess.run(
        [sys.executable, "-m", module, "--run-dir", str(run_dir), "--config", str(config),
         "--device", "cpu", "--smoke"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "ok", summary
    return summary


def test_train_then_eval_entrypoints(tmp_path: Path):
    cfg = yaml.safe_load(PPO_CONFIG.read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"].update(SMALL)
    summary = run_entry("training.train_ppo", cfg, tmp_path / "train")
    head = summary["headline"]
    assert {"policy_p95_m_mean", "policy_crash_rate", "pd_p95_m_mean", "pd_crash_rate",
            "train_reward_mean", "selected_iteration"} <= set(head)
    assert head["selected_iteration"] in (0, 1)  # chosen among checkpoint 0 and the final
    meta = json.loads((tmp_path / "train" / "policy.json").read_text())
    assert meta["selected_iteration"] == head["selected_iteration"]
    assert "PD" in summary["vs_baseline"]
    assert head["pd_crash_rate"] == 0.0  # the yardstick holds a 1 s episode

    # Resume from that run: training continues after its last iteration.
    rcfg = yaml.safe_load(PPO_CONFIG.read_text())
    rcfg["params"].update(rcfg.pop("smoke"))
    rcfg["params"].update({**SMALL, "resume": str(tmp_path / "train" / "policy.pt")})
    rsummary = run_entry("training.train_ppo", rcfg, tmp_path / "resume")
    rmeta = json.loads((tmp_path / "resume" / "policy.json").read_text())
    assert rmeta["start_iteration"] == head["selected_iteration"] + 1
    assert rsummary["headline"]["selected_iteration"] in (rmeta["start_iteration"],
                                                          rmeta["start_iteration"] + 1)
    rlines = (tmp_path / "resume" / "metrics.jsonl").read_text().splitlines()
    rrows = [json.loads(x) for x in rlines]
    assert rrows[0]["step"] == rmeta["start_iteration"]

    ecfg = yaml.safe_load(EVAL_CONFIG.read_text())
    ecfg["params"].update(ecfg.pop("smoke"))
    ecfg["params"].update({"policy": str(tmp_path / "train" / "policy.onnx"), "num_envs": 4,
                           "score_envs": 2, "env": {"episode_length_s": 1.0}})
    esummary = run_entry("training.eval_policy", ecfg, tmp_path / "eval")
    assert esummary["headline"]["policy_p95_m_mean"] > 0
    rows = [json.loads(x) for x in (tmp_path / "eval" / "metrics.jsonl").read_text().splitlines()]
    assert {r["controller"] for r in rows if "controller" in r} == {"policy", "baseline"}
    assert {r["summary"] for r in rows if "summary" in r} == {"policy", "baseline"}
    assert esummary["headline"]["seeds"] == 2  # one summary row per seed and controller
    assert sum("summary" in r for r in rows) == 4


def test_eval_entrypoint_reports_a_missing_policy(tmp_path: Path):
    cfg = {"name": "x", "entrypoint": "training.eval_policy", "seed": 0,
           "params": {"policy": str(tmp_path / "nope.onnx"), "num_envs": 2}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "training.eval_policy", "--run-dir", str(run_dir),
         "--config", str(config), "--device", "cpu"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "failed" and "nope.json" in summary["error"]
