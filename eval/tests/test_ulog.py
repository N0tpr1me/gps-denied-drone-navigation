"""The ulog reader, on a small .ulg written byte by byte from the ULog spec.

The writer below is independent of pyulog, so the test checks pyulog's parse and our
NED/FRD to ENU/FLU conversion together.
"""

import math
import struct

import numpy as np
import pytest

from eval.metrics import Surface, compute_metrics, main
from eval.ulog import gazebo_reference, reference_numbers, trace_from_topics

START_US = 1_000_000


def msg(kind: str, payload: bytes) -> bytes:
    return struct.pack("<HB", len(payload), ord(kind)) + payload


def write_ulog(path, topics):
    """topics: name -> (field spec "float x;float y;...", struct fmt, rows of values)."""
    out = bytearray(b"ULog\x01\x12\x35\x01" + struct.pack("<Q", START_US))
    out += msg("B", bytes(16) + struct.pack("<3Q", 0, 0, 0))
    for name, (fields, _, _) in topics.items():
        out += msg("F", f"{name}:uint64_t timestamp;{fields}".encode())
    for msg_id, name in enumerate(topics):
        out += msg("A", struct.pack("<BH", 0, msg_id) + name.encode())
    for msg_id, (_, fmt, rows) in enumerate(topics.values()):
        for row in rows:
            out += msg("D", struct.pack("<H", msg_id) + struct.pack("<Q" + fmt, *row))
    path.write_bytes(bytes(out))


def sitl_like_log(path, with_groundtruth=True):
    """10 s at 50 Hz. Hold 2 m above the origin (NED z = -2). The true position drifts
    0.3 m north (NED +x) for t in [4, 6) s. The EKF estimate never moves."""
    ts = np.arange(0, 10, 0.02)
    gt = [(START_US + int(t * 1e6), 0.3 if 4 <= t < 6 else 0.0, 0.0, -2.0) for t in ts]
    est = [(START_US + int(t * 1e6), 0.0, 0.0, -2.0) for t in ts]
    sp = [(START_US + int(t * 1e6), 0.0, 0.0, -2.0) for t in ts[::5]]
    # Rates in FRD: roll 0.1, pitch 0.2 (nose up), yaw 0.3 (clockwise from above);
    # thrust_body z = -0.6 (upward).
    rates = [(START_US + int(t * 1e6), 0.1, 0.2, 0.3, 0.0, 0.0, -0.6) for t in ts]
    xyz = "float x;float y;float z;"
    topics = {
        "vehicle_local_position": (xyz, "fff", est),
        "vehicle_local_position_setpoint": (xyz, "fff", sp),
        "vehicle_rates_setpoint": (
            "float roll;float pitch;float yaw;float[3] thrust_body;", "ffffff", rates),
    }
    if with_groundtruth:
        topics["vehicle_local_position_groundtruth"] = (xyz, "fff", gt)
    write_ulog(path, topics)


def test_reads_groundtruth_and_converts_to_enu(tmp_path):
    pytest.importorskip("pyulog")
    from eval.ulog import read_ulog

    path = tmp_path / "flight.ulg"
    sitl_like_log(path)
    tr = read_ulog(path)
    assert tr.meta["position_source"] == "vehicle_local_position_groundtruth"
    assert tr.t[0] == pytest.approx(0.0)
    np.testing.assert_allclose(tr.pos_sp, np.tile([0.0, 0.0, 2.0], (len(tr.t), 1)))
    drift = (tr.t >= 4) & (tr.t < 6)
    np.testing.assert_allclose(tr.pos[drift], np.tile([0.0, 0.3, 2.0], (drift.sum(), 1)),
                               atol=1e-6)  # north is ENU +y, up is +z
    np.testing.assert_allclose(tr.rates_cmd[0], [0.1, -0.2, -0.3], atol=1e-6)
    assert tr.thrust_cmd[0] == pytest.approx(0.6)

    wall = Surface(point=[0.0, 1.0, 0.0], normal=[0.0, 1.0, 0.0])  # wall 1 m north
    m = compute_metrics(tr, wall, gusts=[(4.0, 6.0)])
    assert m["max_m"] == pytest.approx(0.3, abs=1e-6)
    assert m["worst_toward_surface_m"] == pytest.approx(0.3, abs=1e-6)
    assert m["min_clearance_m"] == pytest.approx(0.7, abs=1e-6)
    assert m["recovery_s"] == [pytest.approx(0.0, abs=0.021)]
    assert m["rate_roughness_rad_s2"] == pytest.approx(0.0)


def test_refuses_estimate_unless_allowed(tmp_path):
    pytest.importorskip("pyulog")
    from eval.ulog import read_ulog

    path = tmp_path / "real.ulg"
    sitl_like_log(path, with_groundtruth=False)
    with pytest.raises(ValueError, match="groundtruth"):
        read_ulog(path)
    tr = read_ulog(path, allow_estimate=True)
    assert tr.meta["position_source"] == "vehicle_local_position"
    assert compute_metrics(tr)["max_m"] == pytest.approx(0.0)  # the estimate hides the drift


def test_cli_prints_metrics(tmp_path, capsys):
    pytest.importorskip("pyulog")
    path = tmp_path / "flight.ulg"
    sitl_like_log(path)
    assert main([str(path), "--wall-point", "0,1,0", "--wall-normal", "0,1,0",
                 "--gust", "4,6", "--window", "1", "9"]) == 0
    out = capsys.readouterr().out
    assert '"worst_toward_surface_m": 0.3' in out
    assert "groundtruth" in out


def test_hold_point_and_nan_setpoints_without_pyulog():
    us = START_US + np.arange(5) * 100_000
    topics = {
        "vehicle_local_position_groundtruth": {
            "timestamp": us, "x": np.zeros(5), "y": np.ones(5), "z": np.full(5, -1.0)},
        # Setpoint starts at sample 1 and is NaN at sample 3 (a rate-only mode).
        "vehicle_local_position_setpoint": {
            "timestamp": us[[1, 3]], "x": np.array([0.0, math.nan]),
            "y": np.array([0.0, math.nan]), "z": np.array([-1.0, math.nan])},
    }
    tr = trace_from_topics(topics, START_US)
    np.testing.assert_allclose(tr.t, [0.1, 0.2])
    np.testing.assert_allclose(tr.pos[0], [1.0, 0.0, 1.0])  # NED (0, 1, -1) -> ENU (1, 0, 1)

    tr = trace_from_topics(topics, START_US, hold_point=[1.0, 0.0, 1.0])
    assert len(tr.t) == 5
    assert compute_metrics(tr)["max_m"] == pytest.approx(0.0)


# Mocap: the mocap frame is the EKF's local frame turned 90 degrees about up and moved.
MOCAP_YAW = math.pi / 2
MOCAP_OFFSET = np.array([5.0, -2.0, 0.1])  # ENU, m


def to_mocap_ned(enu):
    """EKF-frame ENU (N, 3) to the mocap frame, written as NED like PX4 logs it."""
    c, s = math.cos(MOCAP_YAW), math.sin(MOCAP_YAW)
    m = np.stack([c * enu[:, 0] - s * enu[:, 1], s * enu[:, 0] + c * enu[:, 1], enu[:, 2]],
                 axis=1) + MOCAP_OFFSET
    return np.stack([m[:, 1], m[:, 0], -m[:, 2]], axis=1)


def rig_like_log(path, pose_frame=1, link_delay_s=0.03):
    """12 s at 50 Hz. 0-4 s: climb to 1.5 m and fly 2 m east to the hold point (EKF frame
    ENU (2, 0, 1.5)); the estimate is right. From 5 s the hold: the estimate stays on the
    setpoint while the truth drifts 0.3 m north for t in [7, 9) s (an estimate error that
    only mocap sees). Mocap is published ``link_delay_s`` after it was sampled."""
    ts = np.arange(0, 12, 0.02)
    truth = np.zeros((len(ts), 3))
    truth[:, 0] = np.clip(ts / 4, 0, 1) * 2.0
    truth[:, 2] = np.clip(ts / 2, 0, 1) * 1.5
    truth[(ts >= 7) & (ts < 9), 1] = 0.3
    est = truth.copy()
    est[:, 1] = 0.0
    us = START_US + (ts * 1e6).astype(int)
    mocap = to_mocap_ned(truth)
    pub = us + int(link_delay_s * 1e6)
    xyz = "float x;float y;float z;"
    topics = {
        "vehicle_local_position": (
            xyz, "fff", [(u, e[1], e[0], -e[2]) for u, e in zip(us, est, strict=True)]),
        "vehicle_local_position_setpoint": (xyz, "fff", [(u, 0.0, 2.0, -1.5) for u in us[::5]]),
        "vehicle_mocap_odometry": (
            "uint64_t timestamp_sample;uint8_t pose_frame;float[3] position;", "QBfff",
            [(p, u, pose_frame, *m) for p, u, m in zip(pub, us, mocap, strict=True)]),
    }
    write_ulog(path, topics)


def test_mocap_truth_with_aligned_setpoint(tmp_path):
    pytest.importorskip("pyulog")
    from eval.ulog import read_ulog

    path = tmp_path / "rig.ulg"
    rig_like_log(path)
    with pytest.raises(ValueError, match="mocap frame"):
        read_ulog(path)
    tr = read_ulog(path, align_window=(0.0, 4.5))
    assert tr.meta["position_source"] == "vehicle_mocap_odometry"
    assert tr.meta["align_yaw_deg"] == pytest.approx(90.0, abs=0.01)
    np.testing.assert_allclose(tr.meta["align_offset_m"], MOCAP_OFFSET, atol=1e-4)
    assert tr.meta["align_rms_m"] < 1e-4
    assert tr.t[0] == pytest.approx(0.0)  # timestamp_sample, not the later publish time
    # The setpoint lands on the hold point's mocap position: EKF (2, 0, 1.5) -> (5, 0, 1.6).
    np.testing.assert_allclose(tr.pos_sp[-1], [5.0, 0.0, 1.6], atol=1e-4)
    # The drift the estimate hid shows against mocap; north in the EKF frame is mocap -x.
    m = compute_metrics(tr, window=(5.0, 11.9))
    assert m["max_m"] == pytest.approx(0.3, abs=1e-4)
    wall = Surface(point=[4.0, 0.0, 0.0], normal=[-1.0, 0.0, 0.0])  # 1 m off, in mocap frame
    m = compute_metrics(tr, wall, window=(5.0, 11.9))
    assert m["worst_toward_surface_m"] == pytest.approx(0.3, abs=1e-4)
    assert m["min_clearance_m"] == pytest.approx(0.7, abs=1e-4)


def test_mocap_truth_with_hold_point_in_mocap_frame(tmp_path):
    pytest.importorskip("pyulog")
    from eval.ulog import read_ulog

    path = tmp_path / "rig.ulg"
    rig_like_log(path, pose_frame=2)
    tr = read_ulog(path, hold_point=[5.0, 0.0, 1.6])
    assert "align_yaw_deg" not in tr.meta
    assert compute_metrics(tr, window=(5.0, 11.9))["max_m"] == pytest.approx(0.3, abs=1e-4)


def test_mocap_refuses_unknown_frame_and_a_window_without_motion(tmp_path):
    pytest.importorskip("pyulog")
    from eval.ulog import read_ulog

    path = tmp_path / "rig.ulg"
    rig_like_log(path, pose_frame=0)
    with pytest.raises(ValueError, match="pose_frame"):
        read_ulog(path, hold_point=[5.0, 0.0, 1.6])
    rig_like_log(path)
    with pytest.raises(ValueError, match="horizontally"):
        read_ulog(path, align_window=(5.0, 11.0))  # the hold: no motion to fix the yaw


def test_cli_scores_mocap_with_align(tmp_path, capsys):
    pytest.importorskip("pyulog")
    path = tmp_path / "rig.ulg"
    rig_like_log(path)
    assert main([str(path), "--align", "0", "4.5", "--window", "5", "11.9"]) == 0
    out = capsys.readouterr().out
    assert '"position_source": "vehicle_mocap_odometry"' in out
    assert '"align_yaw_deg"' in out


def test_fit_yaw_offset_recovers_a_known_transform():
    from eval.ulog import fit_yaw_offset, rotate_z

    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    dst = rotate_z(src, -2.0) + [1.0, 2.0, 3.0]
    psi, off = fit_yaw_offset(src, dst)
    assert psi == pytest.approx(-2.0)
    np.testing.assert_allclose(off, [1.0, 2.0, 3.0])


# Gazebo's reference (docs/decisions.md, 2026-11-10): the estimator's height on the ground
# comes out of the setpoint, and the hold is scored against the setpoint it began with.
def gazebo_topics(ground_offset=0.3, sink_after=None, jump_at=None, with_estimate=True):
    """60 s at 50 Hz, ENU numbers written as NED. On the ground for 10 s, a 5 s climb to
    2.5 m, then a hold. EKF2's height reads ``ground_offset`` above the truth throughout,
    and PX4 holds 2.5 m in EKF2's frame, so the setpoint sits that much above the truth.
    ``sink_after``: the truth sinks 0.2 m from that time (real hold error). ``jump_at``:
    the setpoint steps 1 m north then, and the truth goes with it within a second."""
    t = np.arange(0.0, 60.0, 0.02)
    us = START_US + np.round(t * 1e6).astype(np.int64)
    up = np.clip((t - 10.0) * 0.5, 0.0, 2.5)
    north = np.zeros_like(t)
    sp_north = np.zeros_like(t)
    if sink_after is not None:
        up = up - np.where(t >= sink_after, 0.2, 0.0)
    if jump_at is not None:
        sp_north = np.where(t >= jump_at, 1.0, 0.0)
        north = np.clip(t - jump_at, 0.0, 1.0)
    topics = {
        "vehicle_local_position_groundtruth": {
            "timestamp": us, "x": north, "y": np.zeros_like(t), "z": -up},
        "vehicle_local_position_setpoint": {
            "timestamp": us, "x": sp_north, "y": np.zeros_like(t),
            "z": np.full_like(t, -(2.5 + ground_offset))},
    }
    if with_estimate:
        topics["vehicle_local_position"] = {
            "timestamp": us, "x": north, "y": np.zeros_like(t), "z": -(up + ground_offset)}
    return topics


WINDOW = (20.0, 55.0)


def test_gazebo_reference_takes_out_the_height_the_estimator_read_on_the_ground():
    topics = gazebo_topics(ground_offset=0.3)
    logged = trace_from_topics(topics, START_US)
    scored = gazebo_reference(logged, topics, START_US, WINDOW)
    # As logged the setpoint is 0.3 m above a drone that is exactly where it was sent.
    assert compute_metrics(logged, window=WINDOW)["p95_m"] == pytest.approx(0.3, abs=1e-6)
    assert compute_metrics(scored, window=WINDOW)["p95_m"] == pytest.approx(0.0, abs=1e-6)
    numbers = reference_numbers(scored, logged, WINDOW)
    assert numbers["ground_offset_m"] == pytest.approx(0.3, abs=1e-6)
    assert numbers["p95_logged_setpoint_m"] == pytest.approx(0.3, abs=1e-6)
    assert numbers["setpoint_moved_m"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(logged.pos_sp[:, 2], 2.8)  # the logged trace is untouched


def test_gazebo_reference_keeps_height_error_made_after_lift_off():
    topics = gazebo_topics(ground_offset=0.3, sink_after=30.0)
    scored = gazebo_reference(trace_from_topics(topics, START_US), topics, START_US, WINDOW)
    # The estimate sank with the truth here, so PX4 saw it; the drone is still 0.2 m low.
    assert compute_metrics(scored, window=WINDOW)["p95_m"] == pytest.approx(0.2, abs=1e-6)


def test_gazebo_reference_scores_against_the_setpoint_the_hold_began_with():
    topics = gazebo_topics(ground_offset=0.0, jump_at=30.0)
    logged = trace_from_topics(topics, START_US)
    scored = gazebo_reference(logged, topics, START_US, WINDOW)
    # PX4 moved its setpoint 1 m and the drone followed: little error against the setpoint
    # as logged, a metre from where the hold began.
    assert compute_metrics(logged, window=WINDOW)["p95_m"] < 0.1
    assert compute_metrics(scored, window=WINDOW)["p95_m"] == pytest.approx(1.0, abs=1e-6)
    assert scored.meta["setpoint_moved_m"] == pytest.approx(1.0)
    assert scored.meta["setpoint"] == "fixed"


def test_a_moving_reference_keeps_moving_and_only_loses_the_ground_offset():
    topics = gazebo_topics(ground_offset=0.3, jump_at=30.0)
    logged = trace_from_topics(topics, START_US)
    scored = gazebo_reference(logged, topics, START_US, WINDOW, fixed=False)
    np.testing.assert_allclose(scored.pos_sp[:, :2], logged.pos_sp[:, :2])
    np.testing.assert_allclose(scored.pos_sp[:, 2], 2.5)
    assert scored.meta["setpoint"] == "moving"


def test_without_a_stretch_on_the_ground_nothing_is_taken_out():
    # No estimate in the log (or a log that starts in the air): the offset is unknown, so
    # the height is scored as logged and the row says so with None.
    topics = gazebo_topics(ground_offset=0.3, with_estimate=False)
    logged = trace_from_topics(topics, START_US)
    scored = gazebo_reference(logged, topics, START_US, WINDOW)
    assert scored.meta["ground_offset_m"] is None
    assert compute_metrics(scored, window=WINDOW)["p95_m"] == pytest.approx(0.3, abs=1e-6)
