import numpy as np
import pytest

from sim2real.estimate_bridge.bridge import TraceWriter
from sim2real.estimate_bridge.config import EstimateProfile, load_profile
from sim2real.estimate_bridge.gps import FIXES, GpsFault, enu_to_lat_lon_alt, parse_gps_faults
from sim2real.estimate_bridge.verify_trace import verify

DT = 0.02
GPS = {"rate_hz": 10.0, "fix": "rtk_fixed"}


def climb_then_hover(n, dt=DT):
    t = dt * np.arange(n)
    z = np.minimum(t, 3.0)
    return t, np.stack([1.0 + 0 * t, 2.0 + 0.05 * t, z], 1)


def fly(gps_faults, n=800, gps=GPS, seed=1, anchor="start", engage_at=None):
    prof = EstimateProfile.from_dict({"name": "t", "seed": seed, "estimate": "clean",
                                      "fault_anchor": anchor, "gps": gps,
                                      "gps_faults": gps_faults})
    b = prof.bridge()
    t, pos = climb_then_hover(n)
    out = [b.update(ti, p, engaged=engage_at is not None and ti >= engage_at)
           for ti, p in zip(t, pos, strict=True)]
    return t, pos, out, prof


def sent(out):
    return [(i, s) for i, s in enumerate(out) if s.gps.sent]


def test_receiver_ticks_at_its_rate_with_seeded_noise_on_the_truth():
    t, pos, out, _ = fly([], n=500)
    ticks = sent(out)
    assert len(ticks) == 100  # 10 s at 10 Hz
    assert all(j - i == 5 for (i, _), (j, _) in zip(ticks, ticks[1:], strict=False))
    err = np.array([s.gps.pos - pos[i] for i, s in ticks])
    assert 0.0 < np.abs(err[:, :2]).max() < 0.06 and np.abs(err[:, 2]).max() < 0.12  # 1 cm, 2 cm
    assert all(s.gps.fix_type == 6 and s.gps.eph_m == pytest.approx(0.02) for _, s in ticks)
    assert all(not s.gps.sent for s in out if s not in [x for _, x in ticks])
    _, _, again, _ = fly([], n=500)
    np.testing.assert_array_equal([s.gps.pos for _, s in ticks],
                                  [s.gps.pos for _, s in sent(again)])


def test_float_fault_degrades_the_fix_for_its_window_and_offsets_the_solution():
    f = {"kind": "gps_float", "at_s": 4.0, "duration_s": 3.0, "offset_m": 0.8,
         "direction": [0.0, 2.0, 0.0]}
    t, pos, out, _ = fly([f])
    inside = [(i, s) for i, s in sent(out) if 4.0 <= t[i] < 7.0]
    outside = [(i, s) for i, s in sent(out) if not 4.0 <= t[i] < 7.0]
    assert inside and all(s.gps.fix_type == 5 and s.gps.eph_m == pytest.approx(0.5)
                          for _, s in inside)
    assert all(s.gps.fix_type == 6 for _, s in outside)
    err = np.array([s.gps.pos - pos[i] for i, s in inside])
    assert err[:, 1].mean() == pytest.approx(0.8, abs=0.15)  # 0.3 m noise over 30 samples
    assert np.abs(err[:, 0]).max() < 1.5 and np.abs(err[:, 0]).max() > 0.1  # float noise


def test_jump_holds_and_a_short_one_is_a_glitch():
    held = {"kind": "gps_jump", "at_s": 4.0, "offset_m": 1.5, "direction": [1.0, 0.0, 0.0]}
    t, pos, out, _ = fly([held])
    err = np.array([(t[i], (s.gps.pos - pos[i])[0]) for i, s in sent(out)])
    assert np.abs(err[err[:, 0] < 4.0, 1]).max() < 0.06
    assert np.all(np.abs(err[err[:, 0] >= 4.0, 1] - 1.5) < 0.06)
    glitch = dict(held, duration_s=0.5)
    t, pos, out, _ = fly([glitch])
    err = np.array([(t[i], (s.gps.pos - pos[i])[0]) for i, s in sent(out)])
    hit = (err[:, 0] >= 4.0) & (err[:, 0] < 4.5)
    assert hit.sum() == 5 and np.all(np.abs(err[hit, 1] - 1.5) < 0.06)
    assert np.abs(err[~hit, 1]).max() < 0.06


def test_loss_sends_nothing_then_comes_back_somewhere_else():
    f = {"kind": "gps_loss", "at_s": 2.0, "duration_s": 3.0, "offset_m": 1.0,
         "direction": [1.0, 0.0, 0.0]}
    t, pos, out, _ = fly([f])
    assert not any(s.gps.sent for s in out if 2.0 <= t[s.step] < 5.0)
    assert all(s.gps.fix_type == 0 for s in out if 2.0 <= t[s.step] < 5.0)
    after = [(i, s) for i, s in sent(out) if t[i] >= 5.0]
    assert after and all(abs((s.gps.pos - pos[i])[0] - 1.0) < 0.06 for i, s in after)


def test_gps_faults_are_timed_from_the_anchor_like_field_faults():
    f = {"kind": "gps_loss", "at_s": 1.0, "duration_s": 1.0}
    t, pos, out, _ = fly([f], anchor="engage", engage_at=6.0)
    quiet = [t[s.step] for s in out if not s.gps.sent and s.step % 5 == 0]
    assert quiet and min(quiet) == pytest.approx(7.0) and max(quiet) == pytest.approx(7.9)
    t, pos, out, _ = fly([f], anchor="engage")
    assert all(s.gps.sent for s in out if s.step % 5 == 0)  # never engaged: never lost


def test_trace_with_gps_verifies_with_its_profile_and_not_with_another(tmp_path):
    faults = [{"kind": "gps_float", "at_s": 3.0, "duration_s": 2.0, "offset_m": 0.5},
              {"kind": "gps_loss", "at_s": 6.0, "duration_s": 1.0, "offset_m": 0.3}]
    prof = EstimateProfile.from_dict({"name": "t", "seed": 4, "gps": GPS, "gps_faults": faults})
    b = prof.bridge()
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    t, pos = climb_then_hover(600)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    report = verify(prof, path)
    assert report["ok"] and report["gps_samples"] == 110 and report["gps_mismatches"] == 0
    no_loss = EstimateProfile.from_dict({"name": "t", "seed": 4, "gps": GPS,
                                         "gps_faults": faults[:1]})
    assert not verify(no_loss, path)["ok"]
    other_seed = EstimateProfile.from_dict({"name": "t", "seed": 5, "gps": GPS,
                                            "gps_faults": faults})
    assert not verify(other_seed, path)["ok"]
    assert verify(EstimateProfile.from_dict({"name": "t", "seed": 4}), path)["gps_mismatches"] > 0


def test_a_trace_without_gps_still_verifies(tmp_path):
    prof = EstimateProfile.from_dict({"name": "t", "seed": 2})
    b = prof.bridge()
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    t, pos = climb_then_hover(200)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    assert verify(prof, path)["ok"]


def test_profile_round_trips_gps():
    prof = EstimateProfile.from_dict({"name": "t", "gps": {"rate_hz": 5, "fix": "fix3d"},
                                      "gps_faults": [{"kind": "gps_jump", "at_s": 1,
                                                      "offset_m": 2, "direction": [0, 1, 0]}]})
    again = EstimateProfile.from_dict(prof.to_dict())
    assert again.gps == prof.gps and again.gps_faults == prof.gps_faults
    assert again.gps_faults[0].direction == (0.0, 1.0, 0.0)
    assert EstimateProfile.from_dict({"name": "t"}).gps is None


@pytest.mark.parametrize("bad", [
    {"kind": "teleport", "at_s": 1},
    {"kind": "gps_loss", "at_s": 1},  # no duration
    {"kind": "gps_jump", "at_s": -1},
    {"kind": "gps_jump", "at_s": 1, "offset_m": -1},
    {"kind": "gps_jump", "at_s": 1, "direction": [0, 0, 0]},
    {"kind": "gps_jump", "at_s": 1, "colour": "red"},
])
def test_bad_gps_faults_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_gps_faults([bad])


def test_gps_faults_need_a_receiver_and_a_known_fix():
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "t", "gps_faults": [{"kind": "gps_jump", "at_s": 1}]})
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "t", "gps": {"fix": "rtk_maybe"}})
    assert set(FIXES) == {"fix3d", "rtk_float", "rtk_fixed"}
    assert isinstance(GpsFault("gps_jump", 1.0), GpsFault)


def test_enu_to_lat_lon_round_trips_through_the_tangent_plane():
    lat0, lon0 = 47.397971057728974, 8.546163739800146
    lat, lon, alt = enu_to_lat_lon_alt([100.0, -50.0, 2.5], lat0, lon0, 0.0)
    assert alt == pytest.approx(2.5)
    # Back with the same radii: 100 m east, 50 m south, within a millimetre.
    a, e2 = 6378137.0, 6.69437999014e-3
    w = np.sqrt(1 - e2 * np.sin(np.radians(lat0)) ** 2)
    east = np.radians(lon - lon0) * (a / w) * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * a * (1 - e2) / w ** 3
    assert east == pytest.approx(100.0, abs=1e-3) and north == pytest.approx(-50.0, abs=1e-3)
    assert lat < lat0 and lon > lon0


@pytest.mark.parametrize("name", ["field_rtk_float", "field_gps_jump", "field_gps_glitch",
                                  "field_gps_loss", "gps_on"])
@pytest.mark.parametrize("runner", ["engage", "hover"])
def test_shipped_gps_profiles_pair_up(name, runner):
    p = load_profile(f"{name}_at_{runner}")
    other = load_profile(f"{name}_at_{'hover' if runner == 'engage' else 'engage'}")
    assert p.fault_anchor == ("engage" if runner == "engage" else "altitude")
    assert p.gps is not None and p.gps == other.gps and p.gps.fix == "rtk_fixed"
    assert p.gps_faults == other.gps_faults
    assert p.estimate == other.estimate and p.bias_from_anchor and other.bias_from_anchor
    if name.startswith("field_"):
        assert p.gps_faults and p.estimate.jump_rate_hz == 0 and p.estimate.dropout_rate_hz == 0
    else:
        assert not p.gps_faults and p.estimate == load_profile(f"vio_at_{runner}").estimate


# --- a receiver that is logged but not fused: RTK as truth (eval/rtk_truth.py) -------------

def test_a_receiver_can_be_logged_without_being_fused():
    from sim2real.estimate_bridge.run_estimate import (
        DEFAULT_PARAMS,
        gps_as_expected,
        px4_params_for,
    )

    fused = EstimateProfile.from_dict({"name": "t", "gps": GPS})
    logged = EstimateProfile.from_dict({"name": "t", "gps": {**GPS, "fused": False}})
    assert fused.gps.fused and not logged.gps.fused
    assert EstimateProfile.from_dict(logged.to_dict()).gps == logged.gps
    on, off = px4_params_for(fused, DEFAULT_PARAMS), px4_params_for(logged, DEFAULT_PARAMS)
    # Both have PX4 take the bridge's HIL_GPS as its receiver; only one lets EKF2 use it.
    assert on["MAV_USEHILGPS"] == off["MAV_USEHILGPS"] == 1
    assert on["SIM_GPS_USED"] == off["SIM_GPS_USED"] == 0
    assert on["EKF2_GPS_CTRL"] == 7 and off["EKF2_GPS_CTRL"] == 0
    # PX4's logger keeps sensor_gps at 1 Hz unless the high-rate sensors profile (bit 11) is
    # on, and with SYS_HAS_GPS 0 there is no vehicle_gps_position: 1 Hz is no truth.
    assert off["SDLOG_PROFILE"] & (1 << 11) and off["SDLOG_PROFILE"] & 1
    # A logged-only receiver must leave the flight GPS-free, or the truth is not independent.
    assert gps_as_expected(logged, {"gps_free": True})
    assert not gps_as_expected(logged, {"gps_free": False})
    assert gps_as_expected(fused, {"gps_free": False})


@pytest.mark.parametrize("name", ["rtk_truth", "rtk_truth_float"])
@pytest.mark.parametrize("runner", ["engage", "hover"])
def test_shipped_rtk_truth_profiles_fly_the_benchmark_estimate_with_gps_logged(name, runner):
    p = load_profile(f"{name}_at_{runner}")
    assert p.gps is not None and p.gps.fix == "rtk_fixed" and not p.gps.fused
    assert p.estimate == load_profile(f"vio_at_{runner}").estimate
    assert p.fault_anchor == load_profile(f"vio_at_{runner}").fault_anchor
    assert bool(p.gps_faults) == (name == "rtk_truth_float")
    assert all(f.kind == "gps_float" for f in p.gps_faults)
