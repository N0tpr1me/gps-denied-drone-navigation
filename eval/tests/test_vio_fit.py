"""The VIO error fit: alignment, gap and jump detection, and a round trip through our own
corruptor (corrupt a known trajectory with known numbers, fit, get the numbers back)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.estimate import EstimateCorruptionConfig, EstimateCorruptor
from eval import vio_fit


def _truth(duration_s: float, dt: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A slow wander around a point, like a drone holding in gusts: well under 0.3 m/s.
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt)
    phases = rng.uniform(0, 2 * np.pi, (3, 3))
    freqs = (0.03, 0.07, 0.11)
    pos = np.stack(
        [sum(0.3 * np.sin(2 * np.pi * f * t + phases[k, i]) for i, f in enumerate(freqs))
         for k in range(3)], axis=1
    )
    vel = np.gradient(pos, t, axis=0)
    return t, pos, vel


def _corrupt(cfg: EstimateCorruptionConfig, t, pos, vel, seed: int):
    dt = float(t[1] - t[0])
    c = EstimateCorruptor(cfg, num_envs=1, dt=dt, seed=seed)
    c.reset(pos[:1], vel[:1])
    out, stale = [], []
    for k in range(len(t)):
        s = c.step(pos[k : k + 1], vel[k : k + 1])
        out.append(s.pos[0])
        stale.append(bool(s.stale[0]))
    return np.array(out), np.array(stale)


def test_align_pos_yaw_undoes_a_yaw_and_shift():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(200, 3))
    yaw = 0.7
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    est = (gt - [1.0, 2.0, 3.0]) @ rot.T
    aligned, found, _ = vio_fit.align_pos_yaw(est, gt)
    assert np.allclose(aligned, gt, atol=1e-9)
    assert found == pytest.approx(-yaw)


def test_pair_indices_respects_gaps():
    t = np.array([0.0, 0.05, 0.10, 0.60, 0.65])
    i, j = vio_fit.pair_indices(t, 0.05, tol=0.025)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 1), (1, 2), (3, 4)]


def test_round_trip_recovers_noise_and_walk():
    dt = 0.05  # a 20 Hz VIO
    cfg = EstimateCorruptionConfig(
        delay_min_s=0.0, delay_max_s=0.0, pos_noise_std_m=0.01,
        bias_walk_std_m_per_sqrt_s=0.02, bias_max_m=100.0,
        jump_rate_hz=0.0, dropout_rate_hz=0.0,
    )
    flights = []
    for seed in range(6):
        t, pos, vel = _truth(300.0, dt, seed)
        est, _ = _corrupt(cfg, t, pos, vel, seed)
        flights.append(vio_fit.measure_flight(
            f"s{seed}", vio_fit.Trajectory(t, est), vio_fit.Trajectory(t, pos, vel),
            vio_fit.FitParams()))
    result = vio_fit.summarise(flights, vio_fit.FitParams(), EstimateCorruptionConfig())
    assert result["used_windows"] == "hover"
    prop = result["proposal"]
    assert prop["pos_noise_std_m"] == pytest.approx(0.01, rel=0.15)
    # Global alignment absorbs part of the walk, and 6 x 300 s is a small sample of it.
    assert prop["bias_walk_std_m_per_sqrt_s"] == pytest.approx(0.02, rel=0.3)
    assert result["jumps_m"] == [] and result["gaps_s"] == []
    # No event seen keeps the defaults rather than switching the fault off.
    assert prop["jump_rate_hz"] == EstimateCorruptionConfig().jump_rate_hz
    assert "dropout_rate_hz" in result["kept"] and "delay_min_s" in result["kept"]


def test_round_trip_finds_jumps_and_gaps():
    dt = 0.05
    cfg = EstimateCorruptionConfig(
        delay_min_s=0.0, delay_max_s=0.0, pos_noise_std_m=0.002,
        bias_walk_std_m_per_sqrt_s=0.002, jump_rate_hz=0.02, jump_min_m=0.5,
        jump_max_m=1.0, jump_decay_s=None, dropout_rate_hz=0.02, dropout_min_s=0.5,
        dropout_max_s=1.5,
    )
    t, pos, vel = _truth(2000.0, dt, 7)
    est, stale = _corrupt(cfg, t, pos, vel, 7)
    # A real VIO sends nothing while it has lost tracking: drop the stale samples.
    fresh = ~stale
    f = vio_fit.measure_flight("s", vio_fit.Trajectory(t[fresh], est[fresh]),
                               vio_fit.Trajectory(t, pos, vel), vio_fit.FitParams())
    rate_j = len(f.jumps_m) / f.duration_s
    rate_d = len(f.gaps_s) / f.duration_s
    assert rate_j == pytest.approx(0.02, rel=0.35)
    assert rate_d == pytest.approx(0.02, rel=0.35)
    assert min(f.jumps_m) >= 0.45 and max(f.jumps_m) <= 1.05
    # A gap is the dropout plus one sample period.
    assert min(f.gaps_s) >= 0.5 and max(f.gaps_s) <= 1.5 + 2 * dt
    # Jumps that never decay stay in the error.
    assert np.median(f.jump_remaining) == pytest.approx(1.0, abs=0.1)


def test_tum_loader_reads_velocity_columns(tmp_path):
    path = tmp_path / "est.txt"
    path.write_text("# t x y z qx qy qz qw vx vy vz\n"
                    "1.0 1 2 3 0 0 0 1 0.1 0.2 0.3\n2.0 1 2 3 0 0 0 1 0.1 0.2 0.3\n")
    traj = vio_fit.load_tum(path)
    assert traj.t.tolist() == [1.0, 2.0]
    assert traj.vel is not None and traj.vel[0].tolist() == [0.1, 0.2, 0.3]


def test_euroc_csv_loader_reads_ns_and_velocity(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z\n"
                    "1000000000,1,2,3,1,0,0,0,0.1,0.2,0.3,0,0,0,0,0,0\n"
                    "2000000000,1,2,3,1,0,0,0,0.1,0.2,0.3,0,0,0,0,0,0\n")
    traj = vio_fit.load_euroc_csv(path)
    assert traj.t.tolist() == [1.0, 2.0]
    assert traj.pos[0].tolist() == [1.0, 2.0, 3.0]
    assert traj.vel is not None and traj.vel[1].tolist() == [0.1, 0.2, 0.3]


def test_dir_source_resolves_inside_the_repo(tmp_path):
    with pytest.raises(FileNotFoundError):
        vio_fit.ensure_source({"name": "x", "dir": "runs/no_such_folder_for_test"})


def test_a_flight_with_almost_no_overlap_is_refused_in_plain_words():
    t, pos, vel = _truth(10.0, 0.05, 0)
    tiny = slice(0, 5)
    with pytest.raises(ValueError, match="fewer than 20"):
        vio_fit.measure_flight("s", vio_fit.Trajectory(t[tiny], pos[tiny]),
                               vio_fit.Trajectory(t, pos, vel), vio_fit.FitParams())
