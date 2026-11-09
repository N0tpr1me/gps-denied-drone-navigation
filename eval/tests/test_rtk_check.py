"""The RTK-against-Gazebo check on synthetic topics: both truths in one log."""

import numpy as np
import pytest

from eval.rtk_check import check_topics, survey_points, table
from eval.rtk_truth import wall_from_survey
from eval.tests.test_partner_session import DT, HOLD_ENU, START_US, ekf_flags, flight, us
from sim2real.estimate_bridge.gps import WORLD_ORIGIN, enu_to_lat_lon_alt

WALL = ([3.0, 0.0, 0.0], [1.0, 0.0, 0.0])  # 1 m past the hold point along world east


def gazebo_flight(receiver_offset=(0.0, 0.0, 0.0), gnss=False):
    """A hold whose log has Gazebo's truth and, as the scripted receiver logs it, the same
    track in latitude and longitude about the world's origin (plus ``receiver_offset``)."""
    topics = flight("structure_hold", truth="groundtruth")
    gt = topics["vehicle_local_position_groundtruth"]
    enu = np.stack([gt["y"], gt["x"], -gt["z"]], axis=1)[::5] + np.asarray(receiver_offset)
    llh = np.array([enu_to_lat_lon_alt(p, *WORLD_ORIGIN) for p in enu])
    t = np.arange(0.0, 100.0, DT)[::5]
    topics["sensor_gps#1"] = {
        "timestamp": us(t), "fix_type": np.full(len(t), 6), "latitude_deg": llh[:, 0],
        "longitude_deg": llh[:, 1], "altitude_msl_m": llh[:, 2],
        "altitude_ellipsoid_m": llh[:, 2]}
    topics["estimator_status_flags"] = ekf_flags(gnss)
    return topics


def test_survey_points_lie_on_the_scenario_wall():
    point, normal, rms = wall_from_survey(survey_points(*WALL), WORLD_ORIGIN, HOLD_ENU)
    assert normal == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert (point - HOLD_ENU) @ normal == pytest.approx(1.0, abs=1e-3)
    assert rms < 1e-3


def test_a_perfect_receiver_scores_the_same_as_gazebo_truth():
    r = check_topics(gazebo_flight(), START_US, "structure_hold", (22.0, 80.0), *WALL)
    for other in ("gazebo", "gazebo_whole_hold", "benchmark"):
        assert r["rtk"]["p95_m"] == pytest.approx(r[other]["p95_m"], abs=2e-3)
        assert r["rtk"]["min_clearance_m"] == pytest.approx(r[other]["min_clearance_m"],
                                                             abs=2e-3)
    assert abs(r["difference"]["p95_m"]) < 2e-3
    assert r["receiver_error"]["horizontal_rms_m"] < 1e-3
    assert r["truth_coverage"] == pytest.approx(1.0, abs=1e-3) and r["truth_independent"]
    assert r["position_source"] == "sensor_gps#1"


def test_a_receiver_that_is_off_shows_in_the_receiver_error_and_the_wall_distance():
    # 5 cm toward the wall all flight: the fit to the estimate takes it out of the hold
    # error, but the drone really is 5 cm further from the wall than RTK says.
    r = check_topics(gazebo_flight(receiver_offset=(0.05, 0.0, 0.0)), START_US,
                     "structure_hold", (22.0, 80.0), *WALL)
    assert r["receiver_error"]["horizontal_rms_m"] == pytest.approx(0.05, abs=1e-3)
    assert abs(r["difference"]["p95_m"]) < 2e-3
    assert r["difference"]["min_clearance_m"] == pytest.approx(-0.05, abs=2e-3)


def test_what_the_lost_fix_hid_is_the_gap_to_the_whole_hold():
    # The drone sits 0.4 m off for 10 s of the hold, and the fix is float for exactly then:
    # RTK and Gazebo's truth agree on the samples RTK kept, and only the whole hold shows it.
    topics = gazebo_flight()
    gt = topics["vehicle_local_position_groundtruth"]
    t = np.arange(0.0, 100.0, DT)
    away = (t >= 40.0) & (t < 50.0)
    gt["y"] = gt["y"] + np.where(away, 0.4, 0.0)  # NED y is east
    gps = topics["sensor_gps#1"]
    gps["fix_type"] = np.where(away[::5], 5, 6)
    r = check_topics(topics, START_US, "structure_hold", (22.0, 80.0), *WALL)
    assert r["truth_coverage"] == pytest.approx(1.0 - 10.1 / 58.0, abs=0.01)
    assert r["rtk"]["p95_m"] == pytest.approx(r["gazebo"]["p95_m"], abs=2e-3)
    assert r["rtk"]["max_m"] < 0.12 and r["gazebo_whole_hold"]["max_m"] > 0.4
    assert r["hidden_by_lost_fix"]["p95_m"] > 0.25


def test_the_table_has_a_row_per_flight_and_flags_fused_gps():
    rows = [dict(check_topics(gazebo_flight(gnss=g), START_US, "structure_hold", (22.0, 80.0),
                              *WALL), run=name, profile="p", seed=1)
            for g, name in ((False, "a"), (True, "b"))]
    text = table(rows)
    assert text.count("\n") == 3 and "| YES |" in text.splitlines()[3]
    assert "| no |" in text.splitlines()[2]
