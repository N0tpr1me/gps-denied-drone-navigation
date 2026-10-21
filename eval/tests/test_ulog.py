"""The ulog reader, on a small .ulg written byte by byte from the ULog spec.

The writer below is independent of pyulog, so the test checks pyulog's parse and our
NED/FRD to ENU/FLU conversion together.
"""

import math
import struct

import numpy as np
import pytest

from eval.metrics import Surface, compute_metrics, main
from eval.ulog import trace_from_topics

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
