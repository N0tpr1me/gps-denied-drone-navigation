import numpy as np
import pytest

from sim2real.estimate_bridge.bridge import TraceWriter
from sim2real.estimate_bridge.config import EstimateProfile, load_profile
from sim2real.estimate_bridge.field_faults import FieldFault, parse_field_faults
from sim2real.estimate_bridge.verify_trace import verify

DT = 0.02


def climb_then_hover(n, dt=DT):
    """Take off from (1, 2, 0) to 3 m up and drift slowly along y, so the truth moves."""
    t = dt * np.arange(n)
    z = np.minimum(t, 3.0)
    return t, np.stack([1.0 + 0 * t, 2.0 + 0.05 * t, z], 1)


def fly(faults, n=600, anchor="start", engage_at=None, estimate="clean"):
    prof = EstimateProfile.from_dict({"name": "t", "estimate": estimate, "fault_anchor": anchor,
                                      "field_faults": faults})
    b = prof.bridge()
    t, pos = climb_then_hover(n)
    out = [b.update(ti, p, engaged=engage_at is not None and ti >= engage_at)
           for ti, p in zip(t, pos, strict=True)]
    return t, pos, out


def test_reset_origin_snaps_to_where_the_vio_started_and_moves_from_there():
    t, pos, out = fly([{"kind": "reset_origin", "at_s": 5.0}])
    k = int(round(5.0 / DT))
    before, at, later = out[k - 1], out[k], out[k + 100]
    np.testing.assert_allclose(before.est_pos, pos[k - 1])  # clean: truth until the reset
    np.testing.assert_allclose(at.est_pos, pos[0], atol=1e-12)  # back to the takeoff point
    np.testing.assert_allclose(later.est_pos - at.est_pos, pos[k + 100] - pos[k], atol=1e-12)
    assert (before.resets, at.resets, later.resets) == (0, 1, 1)
    assert all(s.publish for s in out)


def test_unsignalled_reset_keeps_the_counter_and_axes_limit_the_snap():
    t, pos, out = fly([{"kind": "reset_origin", "at_s": 5.0, "axes": "xy", "signal": False}])
    k = int(round(5.0 / DT))
    np.testing.assert_allclose(out[k].est_pos[:2], pos[0][:2], atol=1e-12)
    assert out[k].est_pos[2] == pytest.approx(pos[k][2])  # height untouched
    assert out[-1].resets == 0


def test_restart_withholds_the_gap_then_snaps():
    t, pos, out = fly([{"kind": "restart", "at_s": 4.0, "duration_s": 2.0}])
    a, b = int(round(4.0 / DT)), int(round(6.0 / DT))
    assert all(not s.publish for s in out[a:b])
    assert out[a - 1].publish and out[b].publish
    np.testing.assert_allclose(out[b].est_pos, pos[0], atol=1e-12)
    assert out[b].resets == 1


def test_silence_withholds_only_its_window_and_moves_nothing():
    t, pos, out = fly([{"kind": "silence", "at_s": 2.0, "duration_s": 6.0}])
    a, b = int(round(2.0 / DT)), int(round(8.0 / DT))
    assert [s.publish for s in out].count(False) == b - a
    np.testing.assert_allclose(out[b].est_pos, pos[b])


def test_freeze_keeps_arriving_fresh_but_stops_moving():
    t, pos, out = fly([{"kind": "freeze", "at_s": 1.0, "duration_s": 3.0}])
    a, b = int(round(1.0 / DT)), int(round(4.0 / DT))
    frozen = out[a:b]
    assert all(s.publish and not s.stale for s in frozen)
    assert all(np.array_equal(s.est_pos, frozen[0].est_pos) for s in frozen)
    assert all(not s.est_vel.any() for s in frozen)
    assert not np.allclose(pos[a], pos[b - 1])  # truth climbed meanwhile
    np.testing.assert_allclose(out[b].est_pos, pos[b])  # tracks again afterwards


def test_drift_ramps_then_holds():
    f = {"kind": "drift", "at_s": 2.0, "duration_s": 4.0, "rate_mps": 0.05,
         "direction": [2.0, 0.0, 0.0]}
    t, pos, out = fly([f])
    err = np.array([s.est_pos - p for s, p in zip(out, pos, strict=True)])
    k2, k4, k6 = (int(round(x / DT)) for x in (2.0, 4.0, 6.0))
    assert np.abs(err[:k2]).max() == pytest.approx(0.0)
    assert err[k4] == pytest.approx([0.1, 0.0, 0.0])  # 2 s at 0.05 m/s, direction normalised
    assert err[-1] == pytest.approx([0.2, 0.0, 0.0])  # held after 4 s


def test_engage_anchor_times_faults_from_the_engage_only():
    t, pos, out = fly([{"kind": "silence", "at_s": 1.0, "duration_s": 1.0}],
                      anchor="engage", engage_at=6.0)
    unpublished = [ti for ti, s in zip(t, out, strict=True) if not s.publish]
    assert unpublished and min(unpublished) == pytest.approx(7.0)
    assert max(unpublished) == pytest.approx(8.0 - DT)
    t, pos, out = fly([{"kind": "silence", "at_s": 1.0, "duration_s": 1.0}], anchor="engage")
    assert all(s.publish for s in out)  # never engaged: never anchored, never faulted


def test_faulted_trace_verifies_with_its_profile_and_not_without(tmp_path):
    faults = [{"kind": "restart", "at_s": 3.0, "duration_s": 1.0},
              {"kind": "drift", "at_s": 5.0, "duration_s": 2.0, "rate_mps": 0.1},
              {"kind": "freeze", "at_s": 8.0, "duration_s": 1.0}]
    prof = EstimateProfile.from_dict({"name": "t", "seed": 4, "field_faults": faults})
    b = prof.bridge()
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    t, pos = climb_then_hover(600)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    assert verify(prof, path)["ok"]
    assert not verify(EstimateProfile.from_dict({"name": "t", "seed": 4}), path)["ok"]


def test_profile_round_trips_field_faults():
    prof = EstimateProfile.from_dict({"name": "t", "field_faults": [
        {"kind": "drift", "at_s": 1, "duration_s": 2, "rate_mps": 0.1, "direction": [0, 1, 0]}]})
    again = EstimateProfile.from_dict(prof.to_dict())
    assert again.field_faults == prof.field_faults
    assert again.field_faults[0].direction == (0.0, 1.0, 0.0)


@pytest.mark.parametrize("bad", [
    {"kind": "teleport", "at_s": 1},
    {"kind": "silence", "at_s": 1},  # no duration
    {"kind": "drift", "at_s": 1, "duration_s": 2},  # no rate
    {"kind": "reset_origin", "at_s": -1},
    {"kind": "reset_origin", "at_s": 1, "axes": "xw"},
    {"kind": "drift", "at_s": 1, "duration_s": 2, "rate_mps": 0.1, "direction": [0, 0, 0]},
    {"kind": "reset_origin", "at_s": 1, "colour": "red"},
])
def test_bad_field_faults_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_field_faults([bad])


@pytest.mark.parametrize("name", ["field_reset", "field_restart", "field_silence",
                                  "field_freeze", "field_drift"])
@pytest.mark.parametrize("runner", ["engage", "hover"])
def test_shipped_field_profiles_pair_up(name, runner):
    """Each field fault ships as a hold-runner and a stock-runner profile that differ only in
    their anchor, so both meet the same failure at the same time into the scored window."""
    p = load_profile(f"{name}_at_{runner}")
    other = load_profile(f"{name}_at_{'hover' if runner == 'engage' else 'engage'}")
    assert p.fault_anchor == ("engage" if runner == "engage" else "altitude")
    assert p.field_faults and p.field_faults == other.field_faults
    assert p.estimate == other.estimate and p.bias_from_anchor and other.bias_from_anchor
    assert p.estimate.jump_rate_hz == 0 and p.estimate.dropout_rate_hz == 0
    assert all(isinstance(f, FieldFault) for f in p.field_faults)
