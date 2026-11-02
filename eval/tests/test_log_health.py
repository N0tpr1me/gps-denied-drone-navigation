"""eval.log_health on synthetic logs with known frame mistakes and delays."""

import math

import numpy as np
import pytest

from eval import log_health as lh

START = 1_000_000
GOOD_PARAMS = {"EKF2_EV_CTRL": 7, "EKF2_EV_DELAY": 100.0, "EKF2_HGT_REF": 0,
               "EKF2_GPS_CTRL": 0, "EKF2_EV_QMIN": 0, "EKF2_MAG_TYPE": 0}


def trajectory(t):
    """NED position: a 2 m climb, then wandering side to side (several frequencies)."""
    north = 0.6 * np.sin(0.7 * t) + 0.3 * np.sin(1.9 * t + 1)
    east = 0.5 * np.sin(0.9 * t + 2) + 0.3 * np.sin(2.3 * t)
    climb = np.clip((t - 2) / 4, 0, 1)
    down = -2.0 * climb * climb * (3 - 2 * climb) - 0.2 * np.sin(1.3 * t) * climb
    return np.stack([north, east, down], 1)


def synthetic_log(delay_s=0.1, yaw_deg=0.0, mirror=False, flip_z=False, vision_hz=30.0,
                  seconds=60.0, heading_deg=30.0, fused=True, gps=False, imu_bias=0.05,
                  seed=0, vision_noise_m=0.005, jumps=()):
    rng = np.random.default_rng(seed)
    dt_imu = 0.004
    t = np.arange(0, seconds, dt_imu)
    p = trajectory(t)
    a = np.gradient(np.gradient(p, dt_imu, axis=0), dt_imu, axis=0)
    # body yaw = heading, level attitude: specific force in body = R^T (a - g)
    h = math.radians(heading_deg)
    rz = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
    f_world = a - np.array([0, 0, lh.GRAVITY])
    f_body = f_world @ rz + imu_bias + rng.normal(0, 0.05, f_world.shape)
    ts = (START + t * 1e6).astype(np.int64)
    imu = {"timestamp": ts, **{f"accelerometer_m_s2[{i}]": f_body[:, i] for i in range(3)}}
    q = np.array([math.cos(h / 2), 0, 0, math.sin(h / 2)])
    att = {"timestamp": ts[::10], **{f"q[{i}]": np.full(ts[::10].size, q[i]) for i in range(4)}}
    # vision: the position at (stamp - delay), in a frame turned by yaw_deg from north
    tv = np.arange(1.0, seconds, 1 / vision_hz)
    pv = trajectory(tv - delay_s) + rng.normal(0, vision_noise_m, (tv.size, 3))
    for t_jump, size_m in jumps:  # a VIO relocalisation: a step that decays over 5 s
        after = tv >= t_jump
        pv[after, :2] += size_m * np.exp(-(tv[after, None] - t_jump) / 5.0) * [0.8, 0.6]
    y = math.radians(yaw_deg)
    # a vision frame whose x axis sits yaw_deg from north: coordinates = R(-yaw) p
    r = np.array([[math.cos(y), math.sin(y)], [-math.sin(y), math.cos(y)]])
    pv[:, :2] = pv[:, :2] @ r.T
    if mirror:
        pv[:, [0, 1]] = pv[:, [1, 0]]
    if flip_z:
        pv[:, 2] = -pv[:, 2]
    tvs = (START + tv * 1e6).astype(np.int64)
    vision = {"timestamp": tvs, "timestamp_sample": tvs, "pose_frame": np.ones(tv.size, int),
              "quality": np.full(tv.size, 100), **{f"position[{i}]": pv[:, i] for i in range(3)},
              **{f"q[{i}]": np.full(tv.size, np.nan) for i in range(4)}}
    n = tv.size
    ev = {"timestamp": tvs, "timestamp_sample": tvs, "fused": np.full(n, int(fused)),
          "innovation_rejected": np.zeros(n, int), "innovation[0]": rng.normal(0, .03, n),
          "innovation[1]": rng.normal(0, .03, n), "test_ratio[0]": np.full(n, .1),
          "test_ratio[1]": np.full(n, .1)}
    tf = ts[::25]
    m = tf.size
    flags = {"timestamp": tf, "cs_ev_pos": np.full(m, fused), "cs_ev_hgt": np.full(m, fused),
             "cs_ev_vel": np.zeros(m, bool), "cs_ev_yaw": np.zeros(m, bool),
             "cs_mag_hdg": np.ones(m, bool), "cs_gnss_pos": np.full(m, gps),
             "cs_gnss_vel": np.full(m, gps)}
    land = {"timestamp": np.array([START, START + int(2.5e6), START + int((seconds - 1) * 1e6)]),
            "landed": np.array([1, 0, 1])}
    topics = {lh.IMU: imu, lh.ATTITUDE: att, lh.VISION: vision, lh.EV_POS: ev, lh.FLAGS: flags,
              lh.LAND: land}
    return topics


def by_name(report):
    return {c.name: c for c in report.checks}


def test_good_setup_passes_and_measures_the_delay():
    r = lh.check_topics(synthetic_log(delay_s=0.1), GOOD_PARAMS, START)
    c = by_name(r)
    assert r.worst() == "pass", r.text()
    assert c["Vision delay"].numbers["measured_delay_s"] == pytest.approx(0.1, abs=0.015)
    assert abs(c["Vision frame"].numbers["yaw_offset_deg"]) < 3
    assert c["Vision delay"].numbers["measured_on"] == "vertical"
    assert r.verdict.startswith("OK")


@pytest.mark.parametrize("delay_s", [0.0, 0.25, 0.4])
def test_wrong_delay_is_measured_and_a_value_suggested(delay_s):
    r = lh.check_topics(synthetic_log(delay_s=delay_s), GOOD_PARAMS, START)
    c = by_name(r)["Vision delay"]
    assert c.numbers["measured_delay_s"] == pytest.approx(delay_s, abs=0.015)
    assert c.status == "fail"
    assert "Set EKF2_EV_DELAY to about" in c.says


@pytest.mark.parametrize("yaw", [30.0, -90.0, 170.0])
def test_turned_vision_frame_is_caught_in_a_ned_frame(yaw):
    r = lh.check_topics(synthetic_log(yaw_deg=yaw), GOOD_PARAMS, START)
    c = by_name(r)["Vision frame"]
    assert c.status == "fail", c.says
    assert c.numbers["yaw_offset_deg"] == pytest.approx(yaw, abs=3)
    assert not c.numbers["mirrored"]
    # the delay is still measured right through the wrong frame
    assert by_name(r)["Vision delay"].numbers["measured_delay_s"] == pytest.approx(0.1, abs=.015)


def test_enu_sent_as_ned_is_called_mirrored_and_upside_down():
    r = lh.check_topics(synthetic_log(mirror=True, flip_z=True), GOOD_PARAMS, START)
    c = by_name(r)["Vision frame"]
    assert c.status == "fail"
    assert c.numbers["mirrored"] and c.numbers["vertical_correlation"] < 0
    assert "ENU" in c.says
    assert r.verdict.startswith("PROBLEM")


def test_noisy_jumpy_vio_still_gives_the_delay():
    log = synthetic_log(delay_s=0.15, vision_noise_m=0.02, jumps=[(10.0, 1.0), (30.0, -0.6),
                                                                   (45.0, 1.5)])
    r = lh.check_topics(log, GOOD_PARAMS, START)
    c = by_name(r)
    assert c["Vision delay"].numbers["measured_delay_s"] == pytest.approx(0.15, abs=0.02)
    assert c["Vision frame"].status == "pass", c["Vision frame"].says


def test_wrong_frame_leaves_the_delay_unjudged():
    r = lh.check_topics(synthetic_log(mirror=True, flip_z=True, delay_s=0.3), GOOD_PARAMS, START)
    c = by_name(r)["Vision delay"]
    assert c.status == "skip" and "Fix the frame" in c.says


def test_not_fused_and_gps_on_are_reported():
    r = lh.check_topics(synthetic_log(fused=False, gps=True), {**GOOD_PARAMS, "EKF2_EV_CTRL": 0},
                        START)
    c = by_name(r)
    assert c["Vision fused"].status == "fail"
    assert "does not ask for vision position" in c["Vision fused"].says
    assert c["GPS off"].status == "warn"


def test_no_vision_topic_fails_with_the_estimator_type_hint():
    topics = synthetic_log()
    del topics[lh.VISION]
    r = lh.check_topics(topics, GOOD_PARAMS, START)
    assert by_name(r)["Vision reaches PX4"].status == "fail"
    assert "estimator_type" in r.text()


def test_mocap_estimator_type_is_named():
    topics = synthetic_log()
    topics[lh.MOCAP] = topics.pop(lh.VISION)
    r = lh.check_topics(topics, GOOD_PARAMS, START)
    assert "estimator_type MOCAP" in by_name(r)["Vision reaches PX4"].says


def test_all_nan_positions_point_at_frame_id():
    topics = synthetic_log()
    for i in range(3):
        topics[lh.VISION][f"position[{i}]"][:] = np.nan
    r = lh.check_topics(topics, GOOD_PARAMS, START)
    c = by_name(r)["Vision reaches PX4"]
    assert c.status == "fail" and "frame_id" in c.says


def test_hover_without_motion_skips_frame_and_delay_instead_of_guessing():
    topics = synthetic_log()
    # freeze the vehicle: vision constant, IMU only gravity and noise
    rng = np.random.default_rng(1)
    v = topics[lh.VISION]
    for i in range(3):
        v[f"position[{i}]"] = np.full(v["timestamp"].size, [0.0, 0.0, -2.0][i])
    imu = topics[lh.IMU]
    n = imu["timestamp"].size
    imu["accelerometer_m_s2[0]"] = rng.normal(0, 0.02, n)
    imu["accelerometer_m_s2[1]"] = rng.normal(0, 0.02, n)
    imu["accelerometer_m_s2[2]"] = -lh.GRAVITY + rng.normal(0, 0.02, n)
    r = lh.check_topics(topics, GOOD_PARAMS, START)
    c = by_name(r)
    assert c["Vision delay"].status == "skip"
    assert c["Vision frame"].status == "skip"


def test_vision_height_reference_is_a_warning():
    r = lh.check_topics(synthetic_log(), {**GOOD_PARAMS, "EKF2_HGT_REF": 3}, START)
    assert by_name(r)["Height reference"].status == "warn"


def test_report_round_trips_to_json():
    import json
    r = lh.check_topics(synthetic_log(), GOOD_PARAMS, START)
    d = json.loads(json.dumps(r.to_dict(), default=lh._jsonable))
    assert d["worst"] == "pass" and len(d["checks"]) == 8


def with_compass(topics, heading_deg=30.0, ekf_turn_deg=0.0, takeoff_s=2.5):
    """Add a magnetometer that reads the true heading, and turn EKF2's yaw by
    ``ekf_turn_deg`` once airborne, as EKF2 does when it bends its heading to fit a vision
    frame that is turned from north."""
    att = topics[lh.ATTITUDE]
    t = (att["timestamp"] - START) * 1e-6
    yaw = np.radians(heading_deg + np.where(t >= takeoff_s, ekf_turn_deg, 0.0))
    att["q[0]"], att["q[3]"] = np.cos(yaw / 2), np.sin(yaw / 2)
    att["q[1]"] = att["q[2]"] = np.zeros(t.size)
    ts = topics[lh.IMU]["timestamp"][::5]
    h = math.radians(heading_deg)
    field_ned = np.array([0.2, 0.0, 0.4])
    rz = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
    body = field_ned @ rz  # R^T B as a row vector
    topics[lh.MAG] = {"timestamp": ts,
                      **{f"magnetometer_ga[{i}]": np.full(ts.size, body[i]) for i in range(3)}}
    return topics


def test_compass_catches_a_turned_frame_that_ekf2_bent_its_heading_to_fit():
    # vision turned +90 deg; EKF2 turned its heading -90 deg to match, so against EKF2's own
    # attitude the vision looks aligned: only the compass shows the mistake
    log = synthetic_log(yaw_deg=90.0)
    # the IMU is rotated by the true heading in synthetic_log; EKF2's yaw now differs
    log = with_compass(log, ekf_turn_deg=-90.0)
    r = lh.check_topics(log, GOOD_PARAMS, START)
    c = by_name(r)
    assert c["Vision frame"].status == "fail", c["Vision frame"].says
    assert c["Vision frame"].numbers["yaw_offset_deg"] == pytest.approx(90.0, abs=3)
    assert "compass" in c["Vision frame"].says
    assert c["Heading"].status == "fail"
    assert c["Heading"].numbers["ekf_vs_compass_drift_deg"] == pytest.approx(-90, abs=2)


def test_compass_agreeing_with_ekf2_passes():
    r = lh.check_topics(with_compass(synthetic_log()), GOOD_PARAMS, START)
    c = by_name(r)
    assert c["Vision frame"].status == "pass" and c["Heading"].status == "pass"
    assert abs(c["Heading"].numbers["ekf_vs_compass_drift_deg"]) < 1
