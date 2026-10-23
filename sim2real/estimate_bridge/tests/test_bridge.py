import dataclasses
import math

import numpy as np
import pytest

from envs.estimate import EstimateCorruptionConfig
from envs.frames import enu_to_ned, flu_to_frd, ned_to_enu
from sim2real.estimate_bridge.bridge import (
    TRACE_COLUMNS,
    EstimateBridge,
    TraceWriter,
    odometry_ned,
    read_trace,
)
from sim2real.estimate_bridge.config import PROFILE_DIR, EstimateProfile, load_profile
from sim2real.estimate_bridge.params import read_params, to_environment
from sim2real.estimate_bridge.verify_trace import verify

DT = 0.02


def circle(n, dt=DT, r=1.0):
    t = dt * np.arange(n)
    return t, np.stack([r * np.cos(t), r * np.sin(t), 2.0 + 0 * t], 1)


def test_frames_round_trip_and_signs():
    v = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    np.testing.assert_array_equal(enu_to_ned(v), [[2.0, 1.0, -3.0], [0.5, -1.0, 0.0]])
    np.testing.assert_array_equal(ned_to_enu(enu_to_ned(v)), v)
    np.testing.assert_array_equal(flu_to_frd([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])


def test_clean_profile_passes_truth_through_with_finite_difference_velocity():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, seed=3)
    t, pos = circle(50)
    out = [b.update(ti, p) for ti, p in zip(t, pos, strict=True)]
    assert all(s is not None and s.publish and not s.stale for s in out)
    np.testing.assert_allclose(np.stack([s.est_pos for s in out]), pos)
    np.testing.assert_array_equal(out[0].truth_vel, 0.0)
    np.testing.assert_allclose(out[5].truth_vel, (pos[5] - pos[4]) / DT)
    np.testing.assert_allclose(out[5].est_vel, out[5].truth_vel)


def test_dropouts_are_not_published_and_stale_rows_are_flagged():
    # Stale rows are never published, so what the corruptor reports inside a dropout only
    # matters to the log; EKF2 dead-reckons on its own IMU meanwhile (which is what the
    # corruptor's default `dead_reckon` mode mimics for training). `hold` keeps the reported
    # value frozen, which is what this test checks.
    cfg = EstimateCorruptionConfig(dropout_rate_hz=5.0, dropout_min_s=0.1, dropout_max_s=0.3,
                                   dropout_mode="hold")
    b = EstimateBridge(cfg, DT, seed=0)
    t, pos = circle(400)
    out = [b.update(ti, p) for ti, p in zip(t, pos, strict=True)]
    stale = [s.stale for s in out]
    assert any(stale) and not all(stale)
    for s in out:
        assert s.publish == (not s.stale)
    # Inside a dropout the reported value is held, not refreshed.
    for a, c in zip(out, out[1:], strict=False):
        if a.stale and c.stale:
            np.testing.assert_array_equal(a.est_pos, c.est_pos)


def test_same_seed_same_stream_different_seed_differs():
    cfg = EstimateCorruptionConfig()
    t, pos = circle(200)
    runs = []
    for seed in (7, 7, 8):
        b = EstimateBridge(cfg, DT, seed)
        runs.append(np.stack([b.update(ti, p).est_pos for ti, p in zip(t, pos, strict=True)]))
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def test_divisor_thins_the_output_and_slows_the_corruptor():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, 0, divisor=2)
    assert math.isclose(b.output_rate_hz, 25.0)
    t, pos = circle(10)
    out = [b.update(ti, p) for ti, p in zip(t, pos, strict=True)]
    assert [s is None for s in out] == [False, True] * 5
    assert [s.step for s in out if s is not None] == list(range(5))
    assert b.corruptor.dt == pytest.approx(2 * DT)


def test_non_increasing_time_is_an_error():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, 0)
    b.update(0.0, [0, 0, 0])
    with pytest.raises(ValueError):
        b.update(0.0, [0, 0, 0])


def test_odometry_is_ned():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, 0)
    s = b.update(0.0, [1.0, 2.0, 3.0])
    p, v = odometry_ned(s)
    np.testing.assert_array_equal(p, [2.0, 1.0, -3.0])
    np.testing.assert_array_equal(v, 0.0)


def test_trace_round_trips_and_verifies(tmp_path):
    prof = EstimateProfile.from_dict({"name": "t", "seed": 5, "estimate": {
        "dropout_rate_hz": 2.0, "dropout_min_s": 0.1, "dropout_max_s": 0.2,
        "jump_rate_hz": 2.0}})
    b = EstimateBridge(prof.estimate, prof.dt, prof.seed, prof.divisor)
    w = TraceWriter(tmp_path / "trace.csv")
    t, pos = circle(300, dt=prof.dt)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    tr = read_trace(tmp_path / "trace.csv")
    assert set(tr) == set(TRACE_COLUMNS) and tr["t"].size == 300
    report = verify(prof, tmp_path / "trace.csv")
    assert report["ok"], report
    assert report["max_pos_error_m"] == 0.0 and report["dropouts"] >= 1
    assert report["published"] + report["stale"] == 300
    assert report["spacing_irregular"] == 0


def test_verify_catches_a_wrong_seed_and_a_published_dropout(tmp_path):
    prof = EstimateProfile.from_dict({"name": "t", "seed": 5, "estimate": {
        "dropout_rate_hz": 2.0, "dropout_min_s": 0.1, "dropout_max_s": 0.2}})
    b = EstimateBridge(prof.estimate, prof.dt, prof.seed)
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    t, pos = circle(300, dt=prof.dt)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    wrong = EstimateProfile.from_dict({"name": "t", "seed": 6, "estimate": {
        "dropout_rate_hz": 2.0, "dropout_min_s": 0.1, "dropout_max_s": 0.2}})
    assert not verify(wrong, path)["ok"]
    lines = path.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if i and line.split(",")[3] == "1")
    cols = lines[i].split(",")
    cols[2] = "1"  # published while stale
    lines[i] = ",".join(cols)
    path.write_text("\n".join(lines) + "\n")
    rep = verify(prof, path)
    assert not rep["ok"] and rep["published_while_stale"] == 1


@pytest.mark.parametrize("name", ["clean", "vio_default", "vio_after_takeoff"])
def test_shipped_profiles_load(name):
    p = load_profile(name)
    assert p.name == name and (PROFILE_DIR / f"{name}.yaml").exists()
    assert p.clean == (name == "clean")
    assert EstimateProfile.from_dict(p.to_dict()) == p


def test_vio_default_profile_is_the_training_default():
    assert load_profile("vio_default").estimate == EstimateCorruptionConfig()


def test_vio_after_takeoff_is_vio_default_with_faults_held_back():
    late = load_profile("vio_after_takeoff")
    assert late.estimate == dataclasses.replace(EstimateCorruptionConfig(), fault_start_s=24.0)
    assert late.seed == load_profile("vio_default").seed


def test_profile_rejects_unknown_keys():
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "x", "foo": 1})
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "x", "estimate": {"nope": 1}})


def test_ekf2_ev_params_disable_gps_and_enable_vision():
    p = read_params(PROFILE_DIR.parent / "ekf2_ev.params")
    assert p["EKF2_GPS_CTRL"] == 0 and p["SYS_HAS_GPS"] == 0
    assert p["EKF2_EV_CTRL"] & 0b111 == 0b111 and not p["EKF2_EV_CTRL"] & 0b1000
    assert p["EKF2_HGT_REF"] == 3
    est = load_profile("vio_default").estimate
    assert p["EKF2_EV_DELAY"] == pytest.approx(500.0 * (est.delay_min_s + est.delay_max_s))
    env = to_environment(p)
    assert env["PX4_PARAM_EKF2_EV_CTRL"] == "7" and env["PX4_PARAM_EKF2_EV_DELAY"] == "100.0"


def test_params_reader_rejects_bad_rows(tmp_path):
    f = tmp_path / "x.params"
    f.write_text("1\t1\tA\t1\n")
    with pytest.raises(ValueError):
        read_params(f)
    f.write_text("# c\n1\t1\tA\t2\t6\n1\t1\tB\t0.5\t9\n")
    assert read_params(f) == {"A": 2, "B": 0.5}
