"""The partner session scorer on synthetic topics (no .ulg needed)."""

import math

import numpy as np
import pytest
import yaml

from eval.partner_session import (
    NAV_AUTO_LOITER,
    NAV_EXTERNAL1,
    NAV_POSCTL,
    report,
    score_run,
    score_session,
)
from eval.ulog import rotate_z

START_US = 1_000_000
DT = 0.02
HOLD_ENU = np.array([2.0, 0.0, 2.0])  # EKF frame: 2 m east, 2 m up at the hold
PSI = math.radians(90.0)  # the mocap frame is the EKF frame turned 90 deg, then shifted
OFFSET = np.array([5.0, -1.0, 0.0])


def us(t):
    return (START_US + np.round(np.asarray(t) * 1e6)).astype(np.int64)


def enu_to_ned(p):
    return p[:, 1], p[:, 0], -p[:, 2]


def flight(controller="structure_hold", mode_end=82.0, truth="mocap", wobble=0.1):
    """Arm at 5 s, climb and move to the hold point by 18 s, switch mode at 20 s, hold
    until ``mode_end``, then Position mode. The true position wobbles ``wobble`` m along
    east around the hold point while holding. The EKF estimate is exact (in its frame)."""
    t = np.arange(0.0, 100.0, DT)
    frac = np.clip((t - 5.0) / 13.0, 0.0, 1.0)[:, None]
    p_ekf = frac * HOLD_ENU
    holding = (t >= 20.0) & (t < mode_end)
    p_ekf[holding, 0] += wobble * np.sin(2 * np.pi * 0.5 * t[holding])
    ex, ey, ez = enu_to_ned(p_ekf)
    topics = {"vehicle_local_position": {"timestamp": us(t), "x": ex, "y": ey, "z": ez}}
    if truth == "mocap":
        p_mc = rotate_z(p_ekf, PSI) + OFFSET
        mx, my, mz = enu_to_ned(p_mc)
        topics["vehicle_mocap_odometry"] = {
            "timestamp": us(t), "timestamp_sample": us(t), "pose_frame": np.ones(len(t)),
            "position[0]": mx, "position[1]": my, "position[2]": mz}
    else:
        topics["vehicle_local_position_groundtruth"] = {"timestamp": us(t), "x": ex, "y": ey,
                                                        "z": ez}
    ts = np.arange(0.0, 100.0, 0.1)
    nav_hold = NAV_EXTERNAL1 if controller == "structure_hold" else NAV_AUTO_LOITER
    nav = np.where((ts >= 20.0) & (ts < mode_end), nav_hold, NAV_POSCTL)
    topics["vehicle_status"] = {"timestamp": us(ts), "nav_state": nav,
                                "arming_state": np.where(ts >= 5.0, 2, 1),
                                "failsafe": np.zeros(len(ts), dtype=bool)}
    hx, hy, hz = enu_to_ned(HOLD_ENU[None])
    sp_ned = np.where(holding[:, None] | (t[:, None] < 20.0), [hx[0], hy[0], hz[0]], np.nan)
    if controller == "structure_hold":
        topics["structure_hold_status"] = {
            "timestamp": us(t), "state": np.where(holding, 2, 1),
            "trip_reason": np.where(t >= mode_end, 5, 0),
            "setpoint_ned[0]": sp_ned[:, 0], "setpoint_ned[1]": sp_ned[:, 1],
            "setpoint_ned[2]": sp_ned[:, 2]}
        # Rate-only hold: PX4's position setpoint is NaN while the module flies.
        nan = np.full(len(t), np.nan)
        topics["vehicle_local_position_setpoint"] = {"timestamp": us(t), "x": nan, "y": nan,
                                                     "z": nan}
    else:
        topics["vehicle_local_position_setpoint"] = {
            "timestamp": us(t), "x": np.full(len(t), hx[0]), "y": np.full(len(t), hy[0]),
            "z": np.full(len(t), hz[0])}
    return topics


SESSION = {"hold_s": 60.0, "settle_s": 2.0}
RUN = {"file": "a.ulg", "case": "calm", "run": 1}


@pytest.mark.parametrize("controller", ["structure_hold", "tuned_px4"])
def test_mocap_hold_point_is_carried_into_the_mocap_frame(controller):
    row = score_run(flight(controller), START_US, {**RUN, "controller": controller}, SESSION)
    expected = rotate_z(HOLD_ENU[None], PSI)[0] + OFFSET
    assert np.allclose(row["hold_point_enu"], expected, atol=1e-6)
    assert row["window_s"] == pytest.approx([22.0, 80.0])
    assert row["align_window_s"] == pytest.approx([5.0, 20.0])
    assert row["align_yaw_deg"] == pytest.approx(90.0, abs=1e-6)
    assert row["position_source"] == "vehicle_mocap_odometry"
    # A 0.1 m sine: p95 of |e| is about 0.1 m whatever the frame.
    assert row["p95_m"] == pytest.approx(0.1, abs=0.005)
    assert not row["ended_early"] and not row["crash"]


def test_groundtruth_needs_no_alignment():
    row = score_run(flight(truth="groundtruth"), START_US,
                    {**RUN, "controller": "structure_hold"}, SESSION)
    assert np.allclose(row["hold_point_enu"], HOLD_ENU)
    assert "align_window_s" not in row
    assert row["p95_m"] == pytest.approx(0.1, abs=0.005)


def test_a_short_hold_ended_early_and_reports_the_trip():
    row = score_run(flight(mode_end=40.0), START_US, {**RUN, "controller": "structure_hold"},
                    SESSION)
    assert row["ended_early"] and row["crash"]
    assert row["trip_reason"] == "position_error"
    assert row["next_nav_state"] == NAV_POSCTL


def test_wall_metrics_in_the_mocap_frame():
    # The wall 1 m from the hold point along mocap +x.
    hold_mc = rotate_z(HOLD_ENU[None], PSI)[0] + OFFSET
    session = {**SESSION, "wall": {"point": (hold_mc + [1.0, 0, 0]).tolist(),
                                   "normal": [1.0, 0.0, 0.0]}}
    row = score_run(flight("tuned_px4", wobble=0.0), START_US, {**RUN, "controller": "tuned_px4"},
                    session)
    assert row["min_clearance_m"] == pytest.approx(1.0, abs=1e-6)
    assert row["contacts"] == 0
    # A big drone whose props reach past the wall from the hold point: a contact, a crash.
    big = score_run(flight("tuned_px4", wobble=0.0), START_US,
                    {**RUN, "controller": "tuned_px4"}, {**session, "contact_radius_m": 1.1})
    assert big["contacts"] == 1 and big["crash"]


def test_session_tables_and_aborted_runs(tmp_path):
    runs = [{"file": f"{c}_{i}.ulg", "controller": c, "case": "calm", "run": i}
            for i, c in [(1, "tuned_px4"), (1, "structure_hold"), (2, "tuned_px4"),
                         (2, "structure_hold")]]
    runs.append({"file": "3.ulg", "controller": "structure_hold", "case": "calm", "run": 3,
                 "aborted": True, "reason": "VIO lost tracking on the climb"})
    sheet = tmp_path / "session.yaml"
    sheet.write_text(yaml.safe_dump({"drone": "test", "site": "lab", "date": "2027-05-01",
                                     **SESSION, "runs": runs}))
    wobble = {"tuned_px4": 0.2, "structure_hold": 0.1}

    def loader(path):
        run = next(r for r in runs if r["file"] == path.name)
        return flight(run["controller"], wobble=wobble[run["controller"]]), START_US

    result = score_session(sheet, loader)
    assert not result["errors"]
    agg = {c: b["aggregate"]["calm"] for c, b in result["batches"].items()}
    assert agg["tuned_px4"]["n"] == 2 and agg["structure_hold"]["n"] == 2
    assert agg["structure_hold"]["p95_m_mean"] < agg["tuned_px4"]["p95_m_mean"]
    text = report(result)
    assert "`tuned_px4` → `structure_hold`" in text and "-50 %" in text
    assert "aborted, not scored: 3.ulg" in text


def test_the_partner_template_parses(tmp_path):
    from pathlib import Path

    template = Path(__file__).resolve().parents[2] / "sim2real/partner/session_template.yaml"
    sheet = tmp_path / "session.yaml"
    sheet.write_text(template.read_text())

    def missing(path):
        raise OSError(f"{path.name} not sent")

    result = score_session(sheet, missing)
    assert [a["file"] for a in result["aborted"]] == ["03_tuned_calm_2.ulg"]
    assert {e["file"] for e in result["errors"]} == {"01_tuned_calm_1.ulg",
                                                    "02_hold_calm_1.ulg"}
    assert result["session"]["wall"]["normal"] == [1.0, 0.0, 0.0]


def test_a_refused_engage_before_the_hold_is_skipped():
    """The module refuses to engage on the ground: nav_state flicks to External 1 for one
    sample and back. The scored hold is the first real stretch."""
    topics = flight()
    vs = topics["vehicle_status"]
    ts = (vs["timestamp"] - START_US) * 1e-6
    vs["nav_state"] = np.where(np.isclose(ts, 3.0), NAV_EXTERNAL1, vs["nav_state"])
    row = score_run(topics, START_US, {**RUN, "controller": "structure_hold"}, SESSION)
    assert row["window_s"] == pytest.approx([22.0, 80.0]) and not row["ended_early"]


def test_an_earlier_refusal_is_not_the_holds_trip():
    """trip_reason is the module's LAST trip or refusal and clears on disarm only. A refused
    engage on the ground (7, "landed") before the flight must not be reported as the reason
    the hold ended; that is the reason on the first handback sample."""
    topics = flight(mode_end=40.0)
    st = topics["structure_hold_status"]
    t = (st["timestamp"] - START_US) * 1e-6
    st["trip_reason"] = np.where(t < 40.0, 7, st["trip_reason"])  # refused at 3 s, then held
    st["state"] = np.where((t >= 40.0) & (t < 41.0), 3, st["state"])  # handback for 1 s
    row = score_run(topics, START_US, {**RUN, "controller": "structure_hold"}, SESSION)
    assert row["ended_early"] and row["trip_reason"] == "position_error"


def test_hold_mode_on_the_ground_before_arming_is_not_the_hold():
    """PX4's SITL sits in Hold mode on the ground; the scored hold is the one flown."""
    topics = flight("tuned_px4")
    vs = topics["vehicle_status"]
    ts = (vs["timestamp"] - START_US) * 1e-6
    vs["nav_state"] = np.where(ts < 6.0, NAV_AUTO_LOITER, vs["nav_state"])  # armed at 5 s
    row = score_run(topics, START_US, {**RUN, "controller": "tuned_px4"}, SESSION)
    assert row["window_s"] == pytest.approx([22.0, 80.0])
