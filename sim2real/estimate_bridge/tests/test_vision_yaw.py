import math
from pathlib import Path

import numpy as np
import pytest

from envs.frames import enu_flu_quat_to_ned_frd, quat_to_matrix
from sim2real.estimate_bridge.config import EstimateProfile, load_profile
from sim2real.estimate_bridge.run_estimate import DEFAULT_PARAMS, px4_params_for
from sim2real.estimate_bridge.vision_yaw import VisionYaw, ned_yaw_deg

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def yaw_enu(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def test_frames_quaternion_matches_the_axis_swaps():
    # ENU yaw 90 deg (facing north) is NED yaw 0; ENU yaw 0 (east) is NED yaw 90.
    assert ned_yaw_deg(enu_flu_quat_to_ned_frd(yaw_enu(90.0))) == pytest.approx(0.0, abs=1e-9)
    assert ned_yaw_deg(enu_flu_quat_to_ned_frd(yaw_enu(0.0))) == pytest.approx(90.0)
    # A tilted attitude: body x in NED equals body x in ENU with the axes swapped.
    q = np.array([0.9, 0.1, -0.2, 0.3])
    q /= np.linalg.norm(q)
    fwd_enu = quat_to_matrix(q) @ [1.0, 0.0, 0.0]
    fwd_ned = quat_to_matrix(enu_flu_quat_to_ned_frd(q)) @ [1.0, 0.0, 0.0]
    np.testing.assert_allclose(fwd_ned, [fwd_enu[1], fwd_enu[0], -fwd_enu[2]], atol=1e-12)
    left_enu = quat_to_matrix(q) @ [0.0, 1.0, 0.0]
    right_ned = quat_to_matrix(enu_flu_quat_to_ned_frd(q)) @ [0.0, 1.0, 0.0]
    np.testing.assert_allclose(right_ned, [-left_enu[1], -left_enu[0], left_enu[2]], atol=1e-12)


def test_error_offset_then_drift_up_to_its_limit():
    v = VisionYaw(offset_deg=10.0, drift_at_s=20.0, drift_rate_deg_s=-1.0, drift_max_deg=15.0)
    assert v.error_deg(None) == 10.0  # before the anchor: the offset only
    assert v.error_deg(20.0) == 10.0
    assert v.error_deg(25.0) == pytest.approx(5.0)
    assert v.error_deg(100.0) == pytest.approx(-5.0)


def test_attitude_is_the_truth_turned_about_the_vertical():
    v = VisionYaw(offset_deg=30.0)
    q = np.array([0.95, 0.05, 0.1, 0.2])
    q /= np.linalg.norm(q)
    true = quat_to_matrix(enu_flu_quat_to_ned_frd(q))
    sent = quat_to_matrix(v.attitude_ned(q, 0.0))
    assert ned_yaw_deg(v.attitude_ned(q, 0.0)) - ned_yaw_deg(enu_flu_quat_to_ned_frd(q)) == \
        pytest.approx(30.0)
    # Same tilt: body z (down) is unchanged in its angle to the vertical.
    assert (sent @ [0, 0, 1.0])[2] == pytest.approx((true @ [0, 0, 1.0])[2])


def test_bad_values_refused():
    with pytest.raises(ValueError):
        VisionYaw(offset_deg=170.0, drift_max_deg=20.0)
    with pytest.raises(ValueError):
        VisionYaw.from_dict({"offset": 3})
    with pytest.raises(ValueError):
        VisionYaw(drift_max_deg=-1.0)


def test_profiles_carry_it_and_the_runners_turn_on_vision_yaw():
    base = load_profile("vio_at_engage")
    assert base.vision_yaw is None
    assert px4_params_for(base, DEFAULT_PARAMS)["EKF2_EV_CTRL"] == 7
    for name in ("field_heading_none", "field_heading_offset", "field_heading_drift"):
        for anchor in ("engage", "hover"):
            p = load_profile(f"{name}_at_{anchor}")
            assert p.vision_yaw is not None
            assert px4_params_for(p, DEFAULT_PARAMS)["EKF2_EV_CTRL"] == 15
            assert EstimateProfile.from_dict(p.to_dict()) == p
            twin = load_profile(f"{name}_at_{'hover' if anchor == 'engage' else 'engage'}")
            assert twin.vision_yaw == p.vision_yaw and twin.estimate == p.estimate
