"""Latitude, longitude and altitude into a local ENU frame (``envs.frames.geodetic_to_enu``)."""

import numpy as np
import pytest

from envs.frames import geodetic_to_enu
from sim2real.estimate_bridge.gps import enu_to_lat_lon_alt

ORIGIN = (47.397742, 8.545594, 488.0)  # PX4's default SITL home


def test_the_origin_is_zero():
    assert np.allclose(geodetic_to_enu(*ORIGIN, ORIGIN), 0.0, atol=1e-9)


def test_north_east_and_up_come_out_on_their_own_axes():
    # One arc second: about 30.9 m of northing (meridional radius 6.367e6 m at 47 deg) and
    # 31.0 m * cos(lat) of easting (prime vertical radius 6.390e6 m).
    north = geodetic_to_enu(ORIGIN[0] + 1 / 3600, ORIGIN[1], ORIGIN[2], ORIGIN)
    east = geodetic_to_enu(ORIGIN[0], ORIGIN[1] + 1 / 3600, ORIGIN[2], ORIGIN)
    up = geodetic_to_enu(ORIGIN[0], ORIGIN[1], ORIGIN[2] + 12.5, ORIGIN)
    assert north[1] == pytest.approx(30.88, abs=0.02) and abs(north[0]) < 1e-6
    assert east[0] == pytest.approx(30.978 * np.cos(np.radians(ORIGIN[0])), abs=0.02)
    assert abs(east[1]) < 1e-3  # the parallel curves away from the tangent plane's east axis
    assert up == pytest.approx([0.0, 0.0, 12.5], abs=1e-6)


def test_it_inverts_the_scripted_receiver_at_flight_distances():
    # The Gazebo receiver turns ENU truth into latitude and longitude (gps.py); reading them
    # back must give the same metres, or the RTK scorer and the simulator would disagree.
    # The receiver's radii leave out the site's altitude (0.5 km in 6400 km), so the two
    # differ by 8 parts in 100 000: 0.2 mm at the hold, 5 mm at 55 m.
    for enu, tol in (([1.0, -2.0, 2.5], 3e-4), ([-40.0, 55.0, 30.0], 6e-3)):
        back = geodetic_to_enu(*enu_to_lat_lon_alt(enu, *ORIGIN), ORIGIN)
        assert back == pytest.approx(enu, abs=tol)


def test_arrays_keep_their_shape():
    lat = ORIGIN[0] + np.array([0.0, 1e-5, 2e-5])
    out = geodetic_to_enu(lat, np.full(3, ORIGIN[1]), np.full(3, ORIGIN[2]), ORIGIN)
    assert out.shape == (3, 3)
    assert np.all(np.diff(out[:, 1]) > 1.0)
