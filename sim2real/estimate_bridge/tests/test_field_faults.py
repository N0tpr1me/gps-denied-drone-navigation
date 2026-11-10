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


def test_runaway_position_and_velocity_agree_with_each_other_then_hold():
    f = {"kind": "runaway", "at_s": 2.0, "duration_s": 4.0, "accel_mps2": 0.5,
         "direction": [0.0, 3.0, 0.0]}
    t, pos, out = fly([f])
    _, _, clean = fly([])
    perr = np.array([s.est_pos - c.est_pos for s, c in zip(out, clean, strict=True)])
    verr = np.array([s.est_vel - c.est_vel for s, c in zip(out, clean, strict=True)])
    k2, k4, k6 = (int(round(x / DT)) for x in (2.0, 4.0, 6.0))
    assert np.abs(perr[:k2]).max() == pytest.approx(0.0)
    assert perr[k4] == pytest.approx([0.0, 1.0, 0.0])  # 0.5 * 0.5 * 2^2, direction normalised
    assert verr[k4] == pytest.approx([0.0, 1.0, 0.0])  # the velocity it reports runs away too
    assert perr[-1] == pytest.approx([0.0, 4.0, 0.0])  # held after the 4 s window
    assert np.abs(verr[k6 + 1:]).max() == pytest.approx(0.0)


def test_runaway_with_a_speed_cap_runs_on_at_it():
    f = {"kind": "runaway", "at_s": 2.0, "duration_s": 4.0, "accel_mps2": 0.5,
         "max_speed_mps": 0.5, "direction": [0.0, 1.0, 0.0]}
    t, pos, out = fly([f])
    _, _, clean = fly([])
    perr = np.array([s.est_pos - c.est_pos for s, c in zip(out, clean, strict=True)])
    verr = np.array([s.est_vel - c.est_vel for s, c in zip(out, clean, strict=True)])
    k25, k4, k6 = (int(round(x / DT)) for x in (2.5, 4.0, 6.0))
    assert verr[k25] == pytest.approx([0.0, 0.25, 0.0])  # still accelerating
    assert verr[k4] == pytest.approx([0.0, 0.5, 0.0])    # at the cap from 3 s
    assert perr[k4] == pytest.approx([0.0, 0.75, 0.0])   # 0.25 m to the cap, then 1 s at 0.5 m/s
    assert perr[-1] == pytest.approx([0.0, 1.75, 0.0])   # held after the window
    assert np.abs(verr[k6 + 1:]).max() == pytest.approx(0.0)
    # The cap is carried in the profile only when set, so older profiles read as before.
    assert "max_speed_mps" in parse_field_faults([f])[0].to_dict()
    del f["max_speed_mps"]
    assert "max_speed_mps" not in parse_field_faults([f])[0].to_dict()


def test_shipped_prearm_runaway_profile():
    """The VIO runs off on the floor from the bridge's first seconds, before any arming."""
    p = load_profile("field_runaway_prearm")
    assert p.fault_anchor == "start"
    (f,) = p.field_faults
    assert f.kind == "runaway" and f.at_s <= 5.0 and f.max_speed_mps == pytest.approx(2.8)
    assert p.estimate.jump_rate_hz == 0 and p.estimate.dropout_rate_hz == 0


DEGRADE = {"kind": "degrade", "at_s": 2.0, "duration_s": 6.0, "ramp_s": 2.0, "noise_m": 0.05,
           "vel_noise_mps": 0.2, "walk_mps": 0.1, "loss_hz": 0.5, "loss_s": 0.4,
           "range_noise_m": 0.05, "range_loss_frac": 0.2, "range_short_frac": 0.1}


def test_degrade_adds_noise_losses_and_a_walk_that_stays():
    t, pos, out = fly([DEGRADE], n=800)
    a, b = int(round(2.0 / DT)), int(round(8.0 / DT))
    err = np.array([s.est_pos - p for s, p in zip(out, pos, strict=True)])
    assert np.all(err[:a] == 0)  # clean until the fault
    early, late = err[a:a + 25], err[b - 50:b]
    assert np.abs(early).std() < np.abs(late).std()  # the ramp: worse as it goes on
    assert np.abs(late).max() > 0.05 and np.abs(late).max() < 1.0
    vel_err = np.array([np.abs(s.est_vel[2]) for s in out[b - 50:b]])  # truth hovers
    assert vel_err.max() > 0.1
    losses = [i for i in range(a, b) if not out[i].publish]
    assert losses and all(out[i].publish for i in range(0, a))
    assert all(s.publish for s in out[b:])  # tracking is back after the window
    after = err[b:]
    assert np.all(after == after[0]) and np.linalg.norm(after[0]) > 0  # the walk is kept


def test_degrade_is_reproducible_from_the_seed_and_differs_across_seeds():
    def run(seed):
        prof = EstimateProfile.from_dict({"name": "t", "seed": seed, "estimate": "clean",
                                          "field_faults": [DEGRADE]})
        b = prof.bridge()
        t, pos = climb_then_hover(500)
        return np.array([b.update(ti, p).est_pos for ti, p in zip(t, pos, strict=True)])
    np.testing.assert_array_equal(run(3), run(3))
    assert np.abs(run(3) - run(4)).max() > 0


def test_degrade_spoils_the_rangefinder_without_touching_the_estimate_replay(tmp_path):
    prof = EstimateProfile.from_dict({"name": "t", "seed": 2, "field_faults": [DEGRADE]})
    b = prof.bridge()
    t, pos = climb_then_hover(500)
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    readings = []
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
        readings.append(b.field_faults.range_reading(1.0, 0.1, b.t_rel(ti)))
    w.close()
    assert verify(prof, path)["ok"]  # the range draws are a separate stream
    a = int(round(2.0 / DT))
    assert all(r == 1.0 for r in readings[:a])
    window = readings[a + 150:]
    assert any(r is None for r in window)  # lost returns
    assert any(r is not None and r < 0.7 for r in window)  # short returns
    assert any(r is not None and 0.9 < r < 1.3 and r != 1.0 for r in window)  # noise
    assert all(r is None or r >= 0.1 for r in window)
    assert b.field_faults.range_reading(None, 0.1, 5.0) is None


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
    {"kind": "degrade", "at_s": 1, "duration_s": 2},  # no effect
    {"kind": "degrade", "at_s": 1, "duration_s": 2, "loss_hz": 1},  # loss needs a length
    {"kind": "degrade", "at_s": 1, "duration_s": 2, "noise_m": -0.1},
    {"kind": "degrade", "at_s": 1, "duration_s": 2, "range_loss_frac": 0.6,
     "range_short_frac": 0.6},
    {"kind": "silence", "at_s": 1, "duration_s": 2, "noise_m": 0.1},  # degrade only
    {"kind": "runaway", "at_s": 1, "duration_s": 2},  # no acceleration
    {"kind": "drift", "at_s": 1, "duration_s": 2, "rate_mps": 0.1, "accel_mps2": 1},
    {"kind": "drift", "at_s": 1, "duration_s": 2, "rate_mps": 0.1, "max_speed_mps": 1},
    {"kind": "runaway", "at_s": 1, "duration_s": 2, "accel_mps2": 0.5, "max_speed_mps": -1},
])
def test_bad_field_faults_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_field_faults([bad])


@pytest.mark.parametrize("name", ["field_reset", "field_restart", "field_silence",
                                  "field_freeze", "field_drift", "field_degrade",
                                  "field_runaway"])
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
