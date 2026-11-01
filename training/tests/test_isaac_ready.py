"""Ready for Isaac Lab: the released recipe runs end to end on the Isaac backend's own code.

Isaac Lab needs an RTX GPU, so here the backend (``envs/sims/isaac_quad.py``) runs over the
PhysX stand-in (``envs/sims/tests/fake_physx.py``), behind a factory that keeps Isaac Lab's
one-physics-scene-per-process rule: building a new Isaac sim closes the one before, so a
step on an env built earlier fails as it would on the GPU server. What these tests prove,
on every CI run, is that the day the server arrives only Isaac Lab itself is new:

- the v22 task (gz_x500_sitl, EKF2-like fusion, near-wall flow, airframe randomisation,
  wall yaw moment, re-engages and the engage-time input, crashes and resets) steps the same
  on the Isaac backend as on the mock;
- the v22 training recipe (``structure_hold_ppo_v22.yaml``, smoke) trains, selects a
  checkpoint, exports and scores with ``sim: isaac``, with every env built one after another;
- the eval entry scores a policy with ``sim: isaac``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("onnxruntime")

from envs.sims.isaac_quad import IsaacQuadSim  # noqa: E402
from envs.sims.tests.fake_physx import FakePhysxWorld  # noqa: E402
from envs.structure_hold import env as hold_env  # noqa: E402
from envs.structure_hold.config import StructureHoldConfig  # noqa: E402
from training import eval_policy, train_ppo  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
V22_CONFIG = REPO / "training" / "configs" / "structure_hold_ppo_v22.yaml"
EVAL_CONFIG = REPO / "training" / "configs" / "structure_hold_eval.yaml"
V22_POLICY = REPO / "training" / "policies" / "structure_hold_v22.onnx"

# The v22 training env plus the near-wall flow terms at their evaluation ranges, and
# re-engages often enough to happen inside a short test.
V22_ENV = {
    **yaml.safe_load(V22_CONFIG.read_text())["params"]["env"],
    "episode_length_s": 3.0,
    "reengage_rate_hz": 1.0,
    "reengage_min_s": 0.5,
}
V22_ENV["disturbance"] = {**V22_ENV["disturbance"], "mean_speed": 8.0,
                          "wall_deflect": [0.3, 0.8], "wall_recirc": [0.2, 0.5],
                          "wake_ti": [0.1, 0.3], "shed_amp": [0.1, 0.4]}


class OneScenePerProcess:
    """``make_sim`` for the env module: ``isaac`` builds the Isaac backend over the PhysX
    stand-in and closes the previous one, as Isaac Lab does; ``torch`` is the mock."""

    def __init__(self):
        self.real = hold_env.make_sim
        self.built = 0
        self.active: FakePhysxWorld | None = None

    def __call__(self, name, params, n, dt, device="cpu"):
        if name != "isaac":
            return self.real(name, params, n, dt, device=device)
        if self.active is not None:
            self.active.close()
        self.active = FakePhysxWorld(params, n, dt / params.substeps, device)
        self.built += 1
        return IsaacQuadSim(params, n, dt, device=device, world=self.active)


@pytest.fixture
def isaac(monkeypatch):
    make = OneScenePerProcess()
    monkeypatch.setattr(hold_env, "make_sim", make)
    return make


def test_one_scene_per_process_is_enforced(isaac):
    cfg = StructureHoldConfig.from_dict({"airframe": "gz_x500_sitl"})
    first = hold_env.StructureHoldEnv(cfg, 4, seed=0, sim="isaac")
    hold_env.StructureHoldEnv(cfg, 4, seed=0, sim="isaac")
    with pytest.raises(RuntimeError, match="one physics scene per process"):
        first.step(torch.zeros(4, 4))


def test_v22_task_steps_the_same_on_the_isaac_backend(isaac):
    """Crash rules on, so envs crash, time out and reset, and re-engage, on both backends:
    observations, rewards and episode ends agree step for step."""
    cfg = StructureHoldConfig.from_dict(V22_ENV)
    assert cfg.fusion.enabled and cfg.disturbance.near_wall_flow and cfg.airframe_rand.active
    n, steps = 64, 600  # 7.2 s at the 12 ms tick: two episodes and more
    mock = hold_env.StructureHoldEnv(cfg, n, seed=3, sim="torch")
    glue = hold_env.StructureHoldEnv(cfg, n, seed=3, sim="isaac")
    assert isinstance(glue.sim, IsaacQuadSim)
    g = torch.Generator().manual_seed(0)
    worst_obs = worst_rew = 0.0
    ends = resets = 0
    for _ in range(steps):
        # Random commands push envs into crashes as well as time-outs.
        act = (0.6 * torch.randn(n, 4, generator=g)).clamp(-1.0, 1.0)
        act[:, 3] = act[:, 3] * 0.3 + 0.1
        o1, r1, d1, _ = mock.step(act)
        o2, r2, d2, _ = glue.step(act)
        assert torch.equal(d1, d2)
        assert torch.equal(mock.reengaged, glue.reengaged)
        worst_obs = max(worst_obs, float((o1 - o2).abs().max()))
        worst_rew = max(worst_rew, float((r1 - r2).abs().max()))
        ends += int(d1.sum())
        resets += int(mock.reengaged.sum())
    assert ends > n and resets > 0, (ends, resets)  # the run did cross episodes
    assert mock.counts == glue.counts
    assert worst_obs < 1e-3 and worst_rew < 1e-3, (worst_obs, worst_rew)


def run(module, cfg: dict, run_dir: Path) -> dict:
    config = run_dir.parent / f"{run_dir.name}.yaml"
    config.write_text(yaml.safe_dump(cfg))
    code = module.main(["--run-dir", str(run_dir), "--config", str(config), "--device", "cpu"])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert code == 0, summary
    return summary


def test_v22_recipe_trains_exports_and_scores_on_the_isaac_backend(isaac, tmp_path):
    """``launch --config structure_hold_ppo_v22.yaml --sim isaac --smoke``, minus the GPU: the
    critic warm-up and training, checkpoint selection, export and the PD comparison each
    build their envs on the Isaac backend, one after another."""
    cfg = yaml.safe_load(V22_CONFIG.read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"].update({"sim": "isaac", "critic_warmup": 1, "iterations": 2,
                          "num_steps_per_env": 8, "num_envs": 8})
    cfg["params"]["env"]["episode_length_s"] = 1.0
    summary = run(train_ppo, cfg, tmp_path / "train")
    meta = json.loads((tmp_path / "train" / "policy.json").read_text())
    assert meta["input"]["num_obs"] == 21 and meta["input"]["history"] == 15
    assert summary["headline"]["policy_p95_m_mean"] > 0
    # Training, the selection of each candidate checkpoint, and the policy and PD scoring.
    assert isaac.built >= 5, isaac.built


def test_eval_entry_scores_v22_on_the_isaac_backend(isaac, tmp_path):
    cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    cfg["params"].update(cfg.pop("smoke"))
    cfg["params"].update({"policy": str(V22_POLICY), "sim": "isaac", "num_envs": 4,
                          "score_envs": 2, "seeds": [1],
                          "env": {"airframe": "gz_x500_sitl", "episode_length_s": 1.0}})
    summary = run(eval_policy, cfg, tmp_path / "eval")
    assert summary["headline"]["policy_p95_m_mean"] > 0
    assert isaac.built == 2  # the policy's envs, then the PD's
