"""export_traces on synthetic topics: tick window, last-sample lookup, missing columns."""

from __future__ import annotations

import csv
import json

import numpy as np

from sim2real.px4.tools import export_traces as et


def fake_topics() -> et.Topics:
    # status at 50 Hz: 10 standby ticks, 100 engaged, 10 handback
    n = 120
    ts = 1_000_000 + np.arange(n) * 20_000
    state = np.r_[np.ones(10), np.full(100, 2), np.full(10, 3)]
    status = {"timestamp": ts, "state": state, "stale_s": np.zeros(n),
              "action[0]": np.linspace(-1, 1, n)}
    for f in et.STATUS_FIELDS:
        status.setdefault(f, np.arange(n, dtype=float))
    # truth at 250 Hz, starting 1 ms after the status so ticks read the sample before
    tt = 1_001_000 + np.arange(600) * 4_000
    truth = {"timestamp": tt, "x": tt * 1e-6, "y": np.zeros(600), "z": -np.ones(600),
             "vx": np.zeros(600), "vy": np.zeros(600), "vz": np.zeros(600)}
    # odometry at 50 Hz with a reset counter, no quality field
    ot = 900_000 + np.arange(200) * 20_000
    odom = {"timestamp": ot, "timestamp_sample": ot - 100_000,
            "position[0]": np.ones(200), "position[1]": np.ones(200),
            "position[2]": np.ones(200), "velocity[0]": np.zeros(200),
            "velocity[1]": np.zeros(200), "velocity[2]": np.zeros(200),
            "q[0]": np.ones(200), "q[1]": np.zeros(200), "q[2]": np.zeros(200),
            "q[3]": np.zeros(200), "reset_counter": np.zeros(200, dtype=np.int64)}
    return {et.STATUS: status, "vehicle_local_position_groundtruth": truth, et.ODOMETRY: odom}


def test_window_and_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "load", lambda path: fake_topics())
    run = tmp_path / "20261023_sitl_hold_wall1m_calm_0ms_vio_default_abc"
    run.mkdir()
    (run / "flight.ulg").write_bytes(b"")
    (run / "hold_summary.json").write_text(json.dumps(
        {"scenario": "wall1m_calm_0ms", "profile": "vio_default", "seed": 2, "commit": "abc"}))
    out = tmp_path / "traces"
    info = et.export(run, "v6", out)

    assert info["ticks"] == 100 and abs(info["tick_hz"] - 50) < 1e-6
    assert info["odometry_rows"] > 0
    # attitude, gyro, hover thrust and the EKF estimate are not in the fake log
    assert "att_q[0]" in info["missing"] and "est_x" in info["missing"]
    assert "odometry_quality" in info["missing"]

    rows = list(csv.DictReader((out / "v6_wall1m_calm_0ms_vio_default_seed2_ticks.csv").open()))
    assert len(rows) == 100
    assert float(rows[0]["t"]) == 0.0 and int(rows[0]["sh_state"]) == 2
    # first engaged tick is status index 10 (t = 1.2 s since boot); truth sample before it
    # is at 1.197 s, so x reads 1.197 and the sample age is 3 ms
    assert abs(float(rows[0]["truth_x"]) - 1.197) < 1e-6
    assert abs(float(rows[0]["truth_sample_age_s"]) - 0.003) < 1e-9
    assert "est_x" not in rows[0]

    odom = list(csv.DictReader((out / "v6_wall1m_calm_0ms_vio_default_seed2_odometry.csv").open()))
    # odometry starts 0.3 s before the hold and is kept until 1 s after it (3.18 s): 165 rows
    assert len(odom) == 165
    ts = [float(r["t"]) for r in odom]
    assert ts[0] == -0.3 and all(-1.0 <= t <= 1.98 + 1.0 + 1e-9 for t in ts)
    assert "timestamp_sample" in odom[0]


def test_never_engaged():
    status = {"timestamp": np.arange(5), "state": np.ones(5)}
    try:
        et.first_engaged_window(status)
    except ValueError as exc:
        assert "never engaged" in str(exc)
    else:
        raise AssertionError("expected ValueError")
