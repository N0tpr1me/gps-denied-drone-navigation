"""Gate 1 protocol: held-out seeds, paired scoring, intervals, verdicts, smoke run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from envs.structure_hold import StructureHoldConfig, StructureHoldEnv  # noqa: E402
from envs.structure_hold.baseline import CascadedPd  # noqa: E402
from eval import gate1  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_held_out_seeds_are_refused():
    gate1.check_not_held_out(0, 1000, 2000, 1, 2, 3)
    for s in gate1.HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="held-out"):
            gate1.check_not_held_out(s)


def test_training_refuses_a_held_out_seed(tmp_path):
    from training import train_ppo
    from training.runlib import RunWriter

    with pytest.raises(ValueError, match="held-out"):
        train_ppo.train({"seed": 7001, "params": {}}, tmp_path, "cpu", RunWriter(tmp_path))


def test_wilson_interval():
    lo, hi = gate1.wilson(30, 1000)
    assert lo < 0.03 < hi and hi - lo == pytest.approx(0.021, abs=0.002)
    assert gate1.wilson(0, 100)[0] == 0.0


def test_fly_scores_each_first_episode_once_and_pd_is_deterministic():
    cfg = StructureHoldConfig.from_dict({"episode_length_s": 1.0})
    rows = [gate1.fly(env, CascadedPd(env)) for env in
            (StructureHoldEnv(cfg, 5, seed=9001) for _ in range(2))]
    assert [r["env"] for r in rows[0]] == list(range(5))
    assert [r["p95_m"] for r in rows[0]] == [r["p95_m"] for r in rows[1]]
    assert all(r["length_s"] == pytest.approx(1.0) and not r["crash"] for r in rows[0])


def test_paired_and_judge():
    def rows(p95, crash):
        return [{"p95_m": a, "crash": bool(c)} for a, c in zip(p95, crash, strict=True)]

    n = 400
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.0, n)
    pol, pd = rows(0.6 * base, np.zeros(n)), rows(base, rng.random(n) < 0.1)
    idx = rng.integers(0, n, size=(500, n))
    cmp = gate1.paired(pol, pd, idx)
    assert cmp["p95_change"] == pytest.approx(-0.4)
    assert cmp["p95_change_ci"][1] < 0 and cmp["crash_diff_ci"][1] < 0
    v = gate1.judge({"headline": True}, cmp, gate1.DEFAULT_RULES)
    assert v == {"hold": True, "safety": True, "headline": True, "pass": True}
    worse = gate1.paired(rows(base, rng.random(n) < 0.3), pd, idx)
    assert not gate1.judge({}, worse, gate1.DEFAULT_RULES)["safety"]


def test_a_policy_from_another_airframe_flies_through_the_module_thrust_mapping():
    """v3 (0.8 kg, 16 N) on the x500: the env must get the command the module would send."""
    meta = json.loads((REPO / "training/policies/structure_hold_v3.json").read_text())
    env = StructureHoldEnv(StructureHoldConfig.from_dict({"airframe": "gz_x500"}), 2, seed=0)
    k = gate1.module_thrust_scale(meta, env)
    assert k == pytest.approx((16.0 / 0.8) / (34.194 / 2.0643), rel=1e-3)
    seen = []

    def inner(obs):
        seen.append(obs[:, gate1.OBS["last_action"].start + 3].clone())
        a = torch.zeros(obs.shape[0], 4)
        a[:, 3] = torch.tensor([-0.5, 0.0])
        return a

    ctl = gate1.ModuleThrustScale(inner, k)
    obs, _ = env.reset()
    for _ in range(2):
        a = ctl(obs)
        obs, *_ = env.step(a)
    ctl(obs)
    # Module: u = thrust_n / (MASS_KG g) * hover, with v3's own newtons and mass.
    q, g = meta["env_cfg"]["quad"], 9.80665
    u_module = 0.5 * (torch.tensor([-0.5, 0.0]) + 1.0) * q["max_thrust"] / (q["mass"] * g)
    u_env = 0.5 * (a[:, 3] + 1.0) * env.cfg.quad.max_thrust * env.thrust_cmd_per_n
    u_hover = env.thrust_cmd_per_n * env.cfg.quad.mass * g
    torch.testing.assert_close(u_env, u_module * u_hover)
    torch.testing.assert_close(seen[-1], torch.tensor([-0.5, 0.0]))  # its own action


@pytest.mark.parametrize("config", ["gate1.yaml", "gate1_stress.yaml", "gate1_x500.yaml"])
def test_smoke_run_writes_report(tmp_path, config):
    run_dir = tmp_path / "run"
    out = subprocess.run(
        [sys.executable, "-m", "training.launch", "--config", f"eval/configs/{config}",
         "--smoke", "--device", "cpu", "--runs-dir", str(tmp_path), "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    report = json.loads((run_dir / "report.json").read_text())
    assert report["cells"] and "verdict" in report
    assert "| Cell |" in (run_dir / "report.md").read_text()
