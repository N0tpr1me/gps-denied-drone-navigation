"""The Gazebo-like check: its wobble measure, the re-engage sequence, and a smoke run
through launch."""

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


def _reengage_env(estimate: dict, n: int = 8):
    from envs.structure_hold.config import StructureHoldConfig
    from envs.structure_hold.env import StructureHoldEnv

    cfg = StructureHoldConfig.from_dict({
        "airframe": "gz_x500_sitl", "episode_length_s": 7.0, "hold_distance": 1.0,
        "hold_height": 2.4, "start_offset_std_m": 0.0, "start_yaw_std_rad": 0.0,
        "contact_radius_m": 0.4, "disturbance": {"mean_speed": 0.0, "num_gusts": 0,
                                                 "turb_sigma": 0.0},
        "estimate": {"delay_min_s": 0.0, "delay_max_s": 0.0, "pos_noise_std_m": 0.0,
                     "vel_noise_std_mps": 0.0, "bias_walk_std_m_per_sqrt_s": 0.0,
                     "bias_max_m": 0.0, "jump_rate_hz": 0.0, "dropout_rate_hz": 0.0,
                     **estimate}})
    return StructureHoldEnv(cfg, n, seed=7001, device="cpu")


def test_reengage_captures_the_estimate_and_follows_resets():
    from envs.structure_hold.baseline import CascadedPd

    seq = {"disengage_s": 3.0, "stock_s": 1.0, "after_s": 3.0, "sane_m": 0.6}
    env = _reengage_env({})
    rows = gazebo_check.fly_reengage(env, CascadedPd(env), seq, 7001)
    assert all(r["reengaged"] and r["sane_capture"] and not r["crash"] for r in rows)
    assert all(abs(r["capture_m"] - 1.0) < 0.05 and r["reset_lead_s"] is None for r in rows)
    assert gazebo_check.describe_reengage(rows)["reengage_fail"] == 0

    # A 1.4 m jump toward the wall 1.5 s before the switch: EKF2 rejects the fix for 1 s,
    # resets onto it, stock mode keeps the jumped setpoint, and the re-engage lands near the
    # wall (Gazebo seed 3).
    env = _reengage_env({"jump_decay_s": 5.0})
    rows = gazebo_check.fly_reengage(env, CascadedPd(env),
                                     {**seq, "jump_m": 1.4, "jump_lead_s": [1.5, 1.5]}, 7001)
    assert all(r["reset_lead_s"] is not None and 0.2 < r["reset_lead_s"] < 1.0 for r in rows)
    caps = [r["capture_m"] for r in rows if r["reengaged"]]
    assert caps and max(caps) < 0.6
    d = gazebo_check.describe_reengage(rows)
    assert d["wall_captures"] == len(caps) and d["reengage_fail"] == 0


def test_scripted_events_start_the_faults_they_name():
    from envs.structure_hold.baseline import CascadedPd

    env = _reengage_env({"jump_decay_s": 5.0}, n=4)
    pd = CascadedPd(env)
    obs, _ = env.get_observations()
    for _ in range(10):
        obs, *_ = env.step(pd(obs))
    gazebo_check.apply_event(env, {"t_s": 0.0, "jump": [0.0, 1.8, 0.0]})
    obs, *_ = env.step(pd(obs))
    off = env.estimator.jump_offset.cpu().numpy()
    assert np.allclose(off[:, 1], 1.8, atol=0.01) and np.allclose(off[:, [0, 2]], 0.0)
    gazebo_check.apply_event(env, {"t_s": 0.0, "dropout_s": 0.5, "accel_bias": [0.3, 0, 0]})
    left = int(env.estimator.dropout_left[0])
    assert round(0.5 / env.estimator.dt) - 1 <= left <= round(0.5 / env.estimator.dt)
    obs, *_ = env.step(pd(obs))
    assert int(env.estimator.dropout_left[0]) == left - 1


def test_a_jump_away_from_the_wall_draws_a_controller_toward_it():
    # The estimate says the drone moved 0.4 m off the wall; holding it moves the drone the
    # other way, and the row's worst toward the wall shows it.
    from envs.structure_hold.baseline import CascadedPd

    env = _reengage_env({"jump_decay_s": 5.0}, n=4)
    rows = gazebo_check.fly(env, CascadedPd(env), 2.0, 0.1,
                            [{"t_s": 2.0, "jump": [-0.4, 0.0, 0.0]}])
    assert all(r["toward_wall_m"] > 0.2 and not r["crash"] for r in rows)
    env = _reengage_env({}, n=4)
    rows = gazebo_check.fly(env, CascadedPd(env), 2.0, 0.1)
    assert all(r["toward_wall_m"] < 0.05 for r in rows)


def test_smoke_run_writes_report(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "training.launch", "--config", "eval/configs/gazebo_check.yaml",
         "--smoke", "--device", "cpu", "--runs-dir", str(tmp_path), "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    report = json.loads((run_dir / "report.json").read_text())
    assert [c["name"] for c in report["cells"]] == ["calm_clean", "calm", "wind4", "wind8",
                                                    "wind8_wall", "seed1_jump"]
    assert [c["name"] for c in report["reengage"]] == ["reengage", "reengage_jump"]
    assert report["reference"] == "v10" and "beats" in report["verdicts"]["v12"]
    checks = report["verdicts"]["v12"]["checks"]
    assert "ring_not_higher" in checks and "reengage_no_reengage_fail" in checks
    assert "seed1_jump_toward_wall_ok" in checks and "seed1_jump_p95_ok" not in checks
    assert not any(k.startswith("reengage_jump") for k in checks)  # not scored
    assert "| cell | controller |" in (run_dir / "report.md").read_text()
