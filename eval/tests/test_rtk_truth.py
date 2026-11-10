"""RTK truth from ``sensor_gps`` on synthetic topics (no .ulg needed)."""

import math

import numpy as np
import pytest

from eval.rtk_truth import (
    covered,
    fit_delay,
    gps_fused,
    read_rtk,
    wall_from_survey,
)
from sim2real.estimate_bridge.gps import enu_to_lat_lon_alt

START_US = 1_000_000
ORIGIN = (47.397742, 8.545594, 488.0)
FIXED, FLOAT, FIX3D = 6, 5, 3


def us(t):
    return (START_US + np.round(np.asarray(t) * 1e6)).astype(np.int64)


def receiver(t, enu, fix=FIXED, layout="v1.17"):
    """A ``sensor_gps`` topic: the ENU track about ORIGIN as the receiver would log it."""
    llh = np.array([enu_to_lat_lon_alt(p, *ORIGIN) for p in enu])
    msg = {"timestamp": us(t), "fix_type": np.broadcast_to(fix, len(t)).copy()}
    if layout == "v1.17":
        msg.update({"latitude_deg": llh[:, 0], "longitude_deg": llh[:, 1],
                    "altitude_msl_m": llh[:, 2], "altitude_ellipsoid_m": llh[:, 2] + 47.0})
    else:  # PX4 up to v1.13: integers, 1e-7 deg and mm
        msg.update({"lat": np.round(llh[:, 0] * 1e7).astype(np.int32),
                    "lon": np.round(llh[:, 1] * 1e7).astype(np.int32),
                    "alt": np.round(llh[:, 2] * 1e3).astype(np.int32),
                    "alt_ellipsoid": np.round((llh[:, 2] + 47.0) * 1e3).astype(np.int32)})
    return msg


def track(t):
    return np.stack([0.5 * t, 1.0 + 0.1 * t, 2.0 + 0 * t], axis=1)


T = np.arange(0.0, 20.0, 0.1)


def test_only_rtk_fixed_samples_become_truth():
    fix = np.where((T >= 8.0) & (T < 12.0), FLOAT, FIXED)
    truth = read_rtk({"sensor_gps": receiver(T, track(T), fix)}, START_US, ORIGIN)
    assert truth.samples == 200 and truth.fixed == 160
    assert not np.any((truth.t >= 8.0) & (truth.t < 12.0))
    assert truth.pos == pytest.approx(track(truth.t), abs=1e-3)
    assert truth.source == "sensor_gps" and truth.period_s == pytest.approx(0.1)


def test_the_receiver_with_the_fixes_is_the_one_read():
    # Instance 0 is a second receiver without corrections; the RTK one is instance 1.
    topics = {"sensor_gps": receiver(T, track(T) + 1.5, FIX3D),
              "sensor_gps#1": receiver(T, track(T))}
    truth = read_rtk(topics, START_US, ORIGIN)
    assert truth.source == "sensor_gps#1"
    assert truth.pos == pytest.approx(track(T), abs=1e-3)


def test_the_selected_receiver_stands_in_when_sensor_gps_is_not_logged():
    truth = read_rtk({"vehicle_gps_position": receiver(T, track(T))}, START_US, ORIGIN)
    assert truth.source == "vehicle_gps_position" and truth.fixed == 200


def test_the_topic_logged_at_the_higher_rate_is_the_one_read():
    # PX4's default logger: sensor_gps at 1 Hz, vehicle_gps_position at 10 Hz.
    slow = np.arange(0.0, 20.0, 1.0)
    topics = {"sensor_gps": receiver(slow, track(slow)),
              "vehicle_gps_position": receiver(T, track(T))}
    truth = read_rtk(topics, START_US, ORIGIN)
    assert truth.source == "vehicle_gps_position" and truth.period_s == pytest.approx(0.1)
    # Logged alike, the receiver's own topic is read.
    same = {"sensor_gps": receiver(T, track(T)), "vehicle_gps_position": receiver(T, track(T))}
    assert read_rtk(same, START_US, ORIGIN).source == "sensor_gps"


def test_the_integer_layout_of_older_px4_reads_the_same():
    truth = read_rtk({"sensor_gps": receiver(T, track(T), layout="old")}, START_US, ORIGIN)
    assert truth.pos == pytest.approx(track(T), abs=0.012)  # 1e-7 deg is 1.1 cm


def test_altitude_is_the_kind_the_sheet_names():
    topics = {"sensor_gps": receiver(T, track(T))}
    msl = read_rtk(topics, START_US, ORIGIN, altitude="msl")
    ell = read_rtk(topics, START_US, ORIGIN, altitude="ellipsoid")
    assert ell.pos[:, 2] - msl.pos[:, 2] == pytest.approx(47.0, abs=1e-6)
    with pytest.raises(ValueError, match="msl or ellipsoid"):
        read_rtk(topics, START_US, ORIGIN, altitude="geoid")


def test_without_an_origin_the_first_fixed_sample_is_the_origin():
    fix = np.where(T < 2.0, FIX3D, FIXED)
    truth = read_rtk({"sensor_gps": receiver(T, track(T), fix)}, START_US)
    assert truth.pos[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert truth.pos[-1] == pytest.approx(track(T)[-1] - track(T)[20], abs=1e-3)


def test_a_log_without_a_fix_or_a_receiver_says_so():
    with pytest.raises(ValueError, match="no sensor_gps"):
        read_rtk({}, START_US, ORIGIN)
    with pytest.raises(ValueError, match="RTK-fixed"):
        read_rtk({"sensor_gps": receiver(T, track(T), FLOAT)}, START_US, ORIGIN)


def test_covered_is_the_share_of_time_with_fixed_samples():
    fix = np.where((T >= 8.0) & (T < 12.0), FLOAT, FIXED)
    truth = read_rtk({"sensor_gps": receiver(T, track(T), fix)}, START_US, ORIGIN)
    assert covered(truth, 0.0, 19.9) == pytest.approx(1.0 - 4.1 / 19.9, abs=0.01)
    assert covered(truth, 2.0, 6.0) == pytest.approx(1.0)
    assert covered(truth, 9.0, 11.0) == 0.0
    # One float sample is not a gap: the neighbours are 0.2 s apart.
    one = np.where(np.isclose(T, 5.0), FLOAT, FIXED)
    assert covered(read_rtk({"sensor_gps": receiver(T, track(T), one)}, START_US, ORIGIN),
                   0.0, 19.9) == pytest.approx(1.0, abs=1e-3)


# --- the wall from surveyed points -------------------------------------------------------

def llh(points):
    return [list(enu_to_lat_lon_alt(p, *ORIGIN)) for p in points]


def test_two_points_make_a_vertical_wall_facing_the_drone():
    # The wall runs north-south, 3 m east of the origin; the drone is west of it.
    point, normal, rms = wall_from_survey(llh([[3.0, -4.0, 0.2], [3.0, 5.0, 0.4]]), ORIGIN,
                                          drone_enu=[1.0, 0.0, 2.0])
    assert normal == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)
    assert (point - [1.0, 0.0, 2.0]) @ normal == pytest.approx(2.0, abs=1e-3)
    assert rms == pytest.approx(0.0, abs=1e-3)
    # From the other side the normal turns round: it always points from the drone to the wall.
    _, back, _ = wall_from_survey(llh([[3.0, -4.0, 0.2], [3.0, 5.0, 0.4]]), ORIGIN,
                                  drone_enu=[6.0, 0.0, 2.0])
    assert back == pytest.approx([-1.0, 0.0, 0.0], abs=1e-4)


def test_points_along_the_base_of_a_wall_are_still_a_vertical_wall():
    pts = [[3.0, -4.0, 0.0], [3.02, 0.0, 0.05], [2.98, 5.0, 0.0]]
    _, normal, rms = wall_from_survey(llh(pts), ORIGIN, drone_enu=[0.0, 0.0, 2.0])
    assert normal[2] == 0.0 and normal[0] == pytest.approx(1.0, abs=1e-3)
    assert 0.005 < rms < 0.03


def test_points_spread_over_a_surface_fit_its_plane():
    # A deck underside 6 m up, tilted 2 % along east; the drone flies under it.
    pts = [[x, y, 6.0 + 0.02 * x] for x, y in [(0, 0), (8, 0), (8, 5), (0, 5), (4, 2)]]
    point, normal, rms = wall_from_survey(llh(pts), ORIGIN, drone_enu=[4.0, 2.0, 4.5])
    assert normal == pytest.approx(np.array([-0.02, 0.0, 1.0]) / math.hypot(0.02, 1.0), abs=1e-3)
    assert (point - [4.0, 2.0, 4.5]) @ normal == pytest.approx(1.58 / math.hypot(0.02, 1.0),
                                                                abs=2e-3)
    assert rms < 2e-3


def test_a_survey_that_cannot_place_a_wall_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        wall_from_survey(llh([[3.0, 0.0, 0.0]]), ORIGIN, drone_enu=[0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="above each other"):
        wall_from_survey(llh([[3.0, 0.0, 0.0], [3.0, 0.0, 2.0]]), ORIGIN,
                         drone_enu=[0.0, 0.0, 2.0])


# --- is the truth independent of the estimate? -------------------------------------------

def flags(gnss_from=None):
    t = np.arange(0.0, 20.0, 0.5)
    on = np.zeros(len(t), dtype=bool) if gnss_from is None else t >= gnss_from
    return {"timestamp": us(t), "cs_gnss_pos": on, "cs_gnss_vel": on, "cs_ev_pos": ~on}


def test_gps_fused_reads_the_ekf_aiding_flags_inside_the_window():
    assert gps_fused({"estimator_status_flags": flags()}, START_US, (2.0, 18.0)) is False
    assert gps_fused({"estimator_status_flags": flags(10.0)}, START_US, (2.0, 18.0)) is True
    # Fused only after the scored stretch: the scored truth was still independent.
    assert gps_fused({"estimator_status_flags": flags(10.0)}, START_US, (2.0, 8.0)) is False


def test_gps_fused_falls_back_to_the_control_mode_bits():
    t = np.arange(0.0, 20.0, 0.5)
    bits = np.where(t >= 10.0, (1 << 2) | (1 << 12), 1 << 12).astype(np.uint64)
    topics = {"estimator_status": {"timestamp": us(t), "control_mode_flags": bits}}
    assert gps_fused(topics, START_US, (2.0, 18.0)) is True
    assert gps_fused(topics, START_US, (2.0, 8.0)) is False


def test_gps_fused_is_unknown_without_the_ekf_flags():
    assert gps_fused({}, START_US, (2.0, 18.0)) is None


# --- the receiver's latency ---------------------------------------------------------------

def test_fit_delay_finds_how_late_the_receiver_stamps():
    te = np.arange(0.0, 20.0, 0.02)
    path = np.stack([2.0 * np.sin(0.5 * te), 1.5 * np.cos(0.4 * te), 0.1 * te], axis=1)
    est = {"timestamp": us(te), "x": path[:, 1], "y": path[:, 0], "z": -path[:, 2]}
    late = 0.15  # the receiver saw each position 0.15 s before PX4 stamped it
    seen = np.stack([2.0 * np.sin(0.5 * (T - late)), 1.5 * np.cos(0.4 * (T - late)),
                     0.1 * (T - late)], axis=1)
    topics = {"sensor_gps": receiver(T, seen), "vehicle_local_position": est}
    truth = read_rtk(topics, START_US, ORIGIN)
    assert fit_delay(topics, START_US, truth, (1.0, 19.0)) == pytest.approx(late, abs=0.011)


def test_fit_delay_keeps_zero_when_a_lag_does_not_improve_the_fit():
    te = np.arange(0.0, 20.0, 0.02)
    path = np.stack([2.0 * np.sin(0.5 * te), 1.5 * np.cos(0.4 * te), 0.1 * te], axis=1)
    est = {"timestamp": us(te), "x": path[:, 1], "y": path[:, 0], "z": -path[:, 2]}
    topics = {"sensor_gps": receiver(T, path[::5]), "vehicle_local_position": est}
    truth = read_rtk(topics, START_US, ORIGIN)
    assert fit_delay(topics, START_US, truth, (1.0, 19.0)) == 0.0


def test_a_receiver_logged_at_one_hertz_is_not_a_truth():
    # PX4's default logging of sensor_gps: one sample a second, whatever the receiver's rate.
    slow = np.arange(0.0, 20.0, 1.0)
    with pytest.raises(ValueError, match="1.0 Hz.*SDLOG_PROFILE"):
        read_rtk({"sensor_gps": receiver(slow, track(slow))}, START_US, ORIGIN)
