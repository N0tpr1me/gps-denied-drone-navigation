import numpy as np

from sim2real.estimate_bridge import run_estimate
from sim2real.estimate_bridge.ulog_checks import (
    ESTIMATE,
    ESTIMATOR_STATUS,
    EV_POS,
    FLAGS,
    STATUS,
    TRUTH,
    VISION,
    aiding,
    estimate_error,
    hover_window,
    modes,
    vision_samples,
)

START = 1_000_000


def synthetic_topics(n=600, dt=0.02, bias=(0.1, 0.0, 0.0)):
    """A take-off to 2.5 m then a hover; the estimate equals truth plus a constant bias."""
    t = np.arange(n) * dt
    z_up = np.clip(t / 2.0, 0.0, 2.5)  # climb for 5 s, then hover
    truth = {"timestamp": (START + t * 1e6).astype(np.int64),
             "x": 0.0 * t, "y": 0.0 * t, "z": -z_up}  # NED: down = -up
    est = {"timestamp": (START + t * 1e6).astype(np.int64),
           "x": 0.0 * t + bias[1], "y": 0.0 * t + bias[0], "z": -z_up - bias[2],
           "xy_reset_counter": np.where(t > 8, 1, 0), "z_reset_counter": np.zeros(n, int),
           "xy_valid": np.ones(n, dtype=np.uint8)}
    flags = {"timestamp": truth["timestamp"], "cs_ev_pos": np.ones(n, np.uint8),
             "cs_gnss_pos": np.zeros(n, np.uint8), "cs_gps": np.zeros(n, np.uint8),
             "cs_inertial_dead_reckoning": np.zeros(n, np.uint8)}
    vt = t.copy()
    vt = vt[(vt < 6) | (vt > 6.5)]  # a 0.5 s hole
    vision = {"timestamp": (START + vt * 1e6).astype(np.int64),
              "timestamp_sample": (START + vt * 1e6 - 1000).astype(np.int64)}
    ev = {"timestamp": vision["timestamp"], "timestamp_sample": vision["timestamp_sample"],
          "fused": np.ones(vt.size, np.uint8), "innovation_rejected": np.zeros(vt.size, np.uint8),
          "innovation[0]": 0.03 + 0 * vt, "innovation[1]": -0.04 + 0 * vt,
          "test_ratio[0]": 0.1 + 0 * vt, "test_ratio[1]": 0.2 + 0 * vt}
    ev["fused"][:10] = 0
    status = {"timestamp": truth["timestamp"],
              "nav_state": np.where(t < 5, 17, np.where(t < 11, 4, 18)),
              "failsafe": np.zeros(n, np.uint8)}
    return {TRUTH: truth, ESTIMATE: est, FLAGS: flags, VISION: vision, EV_POS: ev,
            STATUS: status}


def test_hover_window_is_the_time_at_altitude_minus_settling():
    w = hover_window(synthetic_topics(), START, altitude=2.5)
    assert w is not None
    t0, t1 = w
    assert 6.0 <= t0 <= 6.6 and abs(t1 - 11.98) < 0.03


def test_estimate_error_is_the_bias_in_enu():
    topics = synthetic_topics(bias=(0.1, 0.0, 0.0))
    e = estimate_error(topics, START, (7.0, 11.0))
    assert abs(e["rms_m"] - 0.1) < 1e-9 and abs(e["max_m"] - 0.1) < 1e-9
    np.testing.assert_allclose(e["mean_bias_enu_m"], [0.1, 0.0, 0.0], atol=1e-12)
    assert e["xy_resets"] == 1 and e["z_resets"] == 0 and e["xy_invalid_fraction"] == 0.0
    assert estimate_error(topics, START, (9.0, 11.0))["xy_resets"] == 0


def test_aiding_reports_vision_and_no_gnss():
    a = aiding(synthetic_topics(), START)
    assert a["ev_pos_fraction"] == 1.0 and a["gnss_pos_fraction"] == 0.0
    assert a["gnss_ever_used"] is False


def test_aiding_falls_back_to_the_control_mode_bitmask():
    # Images built after PR #73 log no estimator_status_flags (PX4 logger format limit).
    topics = synthetic_topics()
    n = topics[FLAGS]["timestamp"].size
    del topics[FLAGS]
    bits = np.full(n, (1 << 12) | (1 << 24) | (1 << 9), dtype=np.uint64)  # ev_pos, ev_vel, baro
    bits[n // 2:] |= np.uint64(1 << 29)  # dead-reckoning for the second half
    topics[ESTIMATOR_STATUS] = {"timestamp": topics[TRUTH]["timestamp"], "control_mode_flags": bits}
    a = aiding(topics, START)
    assert a["source"] == "estimator_status.control_mode_flags"
    assert a["ev_pos_fraction"] == 1.0 and a["ev_vel_fraction"] == 1.0
    assert a["gnss_pos_fraction"] == 0.0 and a["gnss_ever_used"] is False
    assert a["inertial_dead_reckoning_fraction"] == 0.5
    bits[-1] |= np.uint64(1 << 2)  # one GNSS position sample
    assert aiding(topics, START)["gnss_ever_used"] is True


def test_vision_samples_find_the_gap_the_latency_and_the_fusion_stats():
    topics = synthetic_topics()
    v = vision_samples(topics, START, window=(1.0, 12.0))
    assert v["source"] == EV_POS
    assert v["gaps_over_0p1s"] == 1 and abs(v["longest_gap_s"] - 0.54) < 1e-6
    assert 47.0 < v["rate_hz"] < 50.0  # 50 Hz stream with one 0.5 s hole
    assert abs(v["stamp_minus_sample_mean_s"] - 0.001) < 1e-9
    assert v["fused_fraction"] == 1.0 and v["rejected_fraction"] == 0.0
    assert abs(v["innovation_rms_m"] - 0.05) < 1e-9 and v["test_ratio_max"] == 0.2
    assert vision_samples(topics, START)["fused_fraction"] < 1.0
    del topics[EV_POS]
    fallback = vision_samples(topics, START)
    assert fallback["source"] == VISION and "fused_fraction" not in fallback


def test_modes_sequence_is_takeoff_hold_land():
    m = modes(synthetic_topics(), START)
    assert [s["mode"] for s in m["sequence"]] == ["Takeoff", "Hold", "Land"]
    assert m["failsafe_fraction"] == 0.0


def test_missing_topics_are_reported_not_raised():
    assert "error" in estimate_error({}, START, None)
    assert "error" in aiding({}, START)
    assert "error" in vision_samples({}, START)
    assert "error" in modes({}, START)
    assert hover_window({}, START, 2.5) is None


def test_docker_command_mounts_repo_and_applies_params(tmp_path):
    cmd = run_estimate.docker_command("img", tmp_path, "-2,0,0,0,0,0", 2.5, 60.0, "c",
                                      {"PX4_PARAM_EKF2_GPS_CTRL": "0"})
    joined = " ".join(cmd)
    assert f"{run_estimate.REPO_ROOT}:/repo:ro" in joined
    assert "ESTIMATE_BRIDGE_PROFILE=/out/estimate_config.yaml" in cmd
    assert "PX4_PARAM_EKF2_GPS_CTRL=0" in cmd
    assert cmd[-6:] == ["hover-test", "--altitude", "2.5", "--hover-s", "60.0",
                        "--local-position-only"]


def test_docker_command_watch_publishes_the_browser_view(tmp_path):
    cmd = run_estimate.docker_command("img", tmp_path, "-2,0,0,0,0,0", 2.5, 60.0, "c", {},
                                      watch=True)
    assert "WATCH=1" in cmd and cmd.index("6080:6080") < cmd.index("img")
