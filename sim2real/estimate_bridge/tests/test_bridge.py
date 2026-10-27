import dataclasses
import math

import numpy as np
import pytest

from envs.estimate import EstimateCorruptionConfig
from envs.frames import enu_to_ned, flu_to_frd, ned_to_enu
from sim2real.estimate_bridge.bridge import (
    TRACE_COLUMNS,
    EstimateBridge,
    Rangefinder,
    TraceWriter,
    odometry_ned,
    read_trace,
)
from sim2real.estimate_bridge.config import PROFILE_DIR, EstimateProfile, load_profile
from sim2real.estimate_bridge.params import read_params, to_environment
from sim2real.estimate_bridge.verify_trace import verify

DT = 0.02


def circle(n, dt=DT, r=1.0):
    t = dt * np.arange(n)
    return t, np.stack([r * np.cos(t), r * np.sin(t), 2.0 + 0 * t], 1)


def test_frames_round_trip_and_signs():
    v = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    np.testing.assert_array_equal(enu_to_ned(v), [[2.0, 1.0, -3.0], [0.5, -1.0, 0.0]])
    np.testing.assert_array_equal(ned_to_enu(enu_to_ned(v)), v)
    np.testing.assert_array_equal(flu_to_frd([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])


def test_clean_profile_passes_truth_through_with_finite_difference_velocity():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, seed=3)
    t, pos = circle(50)
    out = [b.update(ti, p) for ti, p in zip(t, pos, strict=True)]
    assert all(s is not None and s.publish and not s.stale for s in out)
    np.testing.assert_allclose(np.stack([s.est_pos for s in out]), pos)
    np.testing.assert_array_equal(out[0].truth_vel, 0.0)
    np.testing.assert_allclose(out[5].truth_vel, (pos[5] - pos[4]) / DT)
    np.testing.assert_allclose(out[5].est_vel, out[5].truth_vel)


def test_dropouts_are_not_published_and_stale_rows_are_flagged():
    # Stale rows are never published, so what the corruptor reports inside a dropout only
    # matters to the log; EKF2 dead-reckons on its own IMU meanwhile (which is what the
    # corruptor's default `dead_reckon` mode mimics for training). `hold` keeps the reported
    # value frozen, which is what this test checks.
    cfg = EstimateCorruptionConfig(dropout_rate_hz=5.0, dropout_min_s=0.1, dropout_max_s=0.3,
                                   dropout_mode="hold")
    b = EstimateBridge(cfg, DT, seed=0)
    t, pos = circle(400)
    out = [b.update(ti, p) for ti, p in zip(t, pos, strict=True)]
    stale = [s.stale for s in out]
    assert any(stale) and not all(stale)
    for s in out:
        assert s.publish == (not s.stale)
    # Inside a dropout the reported value is held, not refreshed.
    for a, c in zip(out, out[1:], strict=False):
        if a.stale and c.stale:
            np.testing.assert_array_equal(a.est_pos, c.est_pos)


def test_same_seed_same_stream_different_seed_differs():
    cfg = EstimateCorruptionConfig()
    t, pos = circle(200)
    runs = []
    for seed in (7, 7, 8):
        b = EstimateBridge(cfg, DT, seed)
        runs.append(np.stack([b.update(ti, p).est_pos for ti, p in zip(t, pos, strict=True)]))
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def test_divisor_thins_the_output_and_slows_the_corruptor():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, 0, divisor=2)
    assert math.isclose(b.output_rate_hz, 25.0)
    t, pos = circle(10)
    out = [b.update(ti, p) for ti, p in zip(t, pos, strict=True)]
    assert [s is None for s in out] == [False, True] * 5
    assert [s.step for s in out if s is not None] == list(range(5))
    assert b.corruptor.dt == pytest.approx(2 * DT)


def test_non_increasing_time_is_an_error():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, 0)
    b.update(0.0, [0, 0, 0])
    with pytest.raises(ValueError):
        b.update(0.0, [0, 0, 0])


def test_odometry_is_ned():
    b = EstimateBridge(EstimateCorruptionConfig.clean(), DT, 0)
    s = b.update(0.0, [1.0, 2.0, 3.0])
    p, v = odometry_ned(s)
    np.testing.assert_array_equal(p, [2.0, 1.0, -3.0])
    np.testing.assert_array_equal(v, 0.0)


def test_trace_round_trips_and_verifies(tmp_path):
    prof = EstimateProfile.from_dict({"name": "t", "seed": 5, "estimate": {
        "dropout_rate_hz": 2.0, "dropout_min_s": 0.1, "dropout_max_s": 0.2,
        "jump_rate_hz": 2.0}})
    b = EstimateBridge(prof.estimate, prof.dt, prof.seed, prof.divisor)
    w = TraceWriter(tmp_path / "trace.csv")
    t, pos = circle(300, dt=prof.dt)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    tr = read_trace(tmp_path / "trace.csv")
    assert set(tr) == set(TRACE_COLUMNS) and tr["t"].size == 300
    report = verify(prof, tmp_path / "trace.csv")
    assert report["ok"], report
    assert report["max_pos_error_m"] == 0.0 and report["dropouts"] >= 1
    assert report["published"] + report["stale"] == 300
    assert report["spacing_irregular"] == 0


def test_verify_catches_a_wrong_seed_and_a_published_dropout(tmp_path):
    prof = EstimateProfile.from_dict({"name": "t", "seed": 5, "estimate": {
        "dropout_rate_hz": 2.0, "dropout_min_s": 0.1, "dropout_max_s": 0.2}})
    b = EstimateBridge(prof.estimate, prof.dt, prof.seed)
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    t, pos = circle(300, dt=prof.dt)
    for ti, p in zip(t, pos, strict=True):
        w.write(b.update(ti, p))
    w.close()
    wrong = EstimateProfile.from_dict({"name": "t", "seed": 6, "estimate": {
        "dropout_rate_hz": 2.0, "dropout_min_s": 0.1, "dropout_max_s": 0.2}})
    assert not verify(wrong, path)["ok"]
    lines = path.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if i and line.split(",")[3] == "1")
    cols = lines[i].split(",")
    cols[2] = "1"  # published while stale
    lines[i] = ",".join(cols)
    path.write_text("\n".join(lines) + "\n")
    rep = verify(prof, path)
    assert not rep["ok"] and rep["published_while_stale"] == 1


@pytest.mark.parametrize("name", ["clean", "vio_default", "vio_after_takeoff", "vio_at_engage",
                                  "vio_at_hover"])
def test_shipped_profiles_load(name):
    p = load_profile(name)
    assert p.name == name and (PROFILE_DIR / f"{name}.yaml").exists()
    assert p.clean == (name == "clean")
    assert EstimateProfile.from_dict(p.to_dict()) == p


def test_vio_default_profile_is_the_training_default():
    assert load_profile("vio_default").estimate == EstimateCorruptionConfig()


def test_vio_after_takeoff_is_vio_default_with_faults_held_back():
    late = load_profile("vio_after_takeoff")
    assert late.estimate == dataclasses.replace(EstimateCorruptionConfig(), fault_start_s=24.0)
    assert late.seed == load_profile("vio_default").seed


def test_seed_override_keeps_the_fault_anchor():
    # run_gazebo / run_estimate --seed (always passed by run_baseline) once rebuilt the
    # profile without fault_anchor, so vio_at_engage flew keyed to the start.
    p = load_profile("vio_at_engage")
    q = p.with_seed(3)
    assert q.seed == 3 and q.fault_anchor == "engage"
    assert q == dataclasses.replace(p, seed=3)


def yaw_quat(yaw: float, pitch: float = 0.0) -> tuple[float, float, float, float]:
    """(w, x, y, z) for yaw about z then pitch about the new y (FLU: positive pitch is nose
    down)."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def test_rangefinder_reads_the_wall_along_body_x():
    # wall_slab: wall face at x = 0, normal toward it +x; the x500 spawns 1 m short, facing +x.
    rf = Rangefinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert math.isclose(rf.range_m((-1.0, 0.3, 2.5), yaw_quat(0.0)), 1.0, rel_tol=1e-9)
    # Yawed 30 deg: the beam is longer than the perpendicular distance.
    a = math.radians(30)
    assert math.isclose(rf.range_m((-1.0, 0.0, 2.5), yaw_quat(a)), 1 / math.cos(a), rel_tol=1e-9)
    # Pitched 10 deg nose down: also 1 / cos.
    b = math.radians(10)
    assert math.isclose(rf.range_m((-0.5, 0.0, 2.5), yaw_quat(0.0, b)), 0.5 / math.cos(b),
                        rel_tol=1e-9)
    # Facing away, nearly parallel, out of range, or past the wall: no reading.
    assert rf.range_m((-1.0, 0.0, 2.5), yaw_quat(math.pi)) is None
    assert rf.range_m((-1.0, 0.0, 2.5), yaw_quat(math.radians(85))) is None
    assert rf.range_m((-0.05, 0.0, 2.5), yaw_quat(0.0)) is None
    assert rf.range_m((-13.0, 0.0, 2.5), yaw_quat(0.0)) is None
    assert rf.range_m((0.2, 0.0, 2.5), yaw_quat(0.0)) is None
    with pytest.raises(ValueError):
        Rangefinder((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_profile_carries_the_rangefinder_through_its_yaml():
    p = dataclasses.replace(load_profile("vio_at_engage"),
                            rangefinder=Rangefinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    d = p.to_dict()
    assert d["rangefinder"]["wall_point"] == [0.0, 0.0, 0.0]
    assert EstimateProfile.from_dict(d) == p
    # The corruption, and so verify_trace, does not depend on it.
    assert p.bridge().update(0.0, (0.0, 0.0, 0.0)).publish
    assert load_profile("vio_at_engage").rangefinder is None


def test_profile_rejects_unknown_keys():
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "x", "foo": 1})
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "x", "estimate": {"nope": 1}})


def test_ekf2_ev_params_disable_gps_and_enable_vision():
    p = read_params(PROFILE_DIR.parent / "ekf2_ev.params")
    assert p["EKF2_GPS_CTRL"] == 0 and p["SYS_HAS_GPS"] == 0
    assert p["EKF2_EV_CTRL"] & 0b111 == 0b111 and not p["EKF2_EV_CTRL"] & 0b1000
    assert p["EKF2_HGT_REF"] == 3
    est = load_profile("vio_default").estimate
    assert p["EKF2_EV_DELAY"] == pytest.approx(500.0 * (est.delay_min_s + est.delay_max_s))
    env = to_environment(p)
    assert env["PX4_PARAM_EKF2_EV_CTRL"] == "7" and env["PX4_PARAM_EKF2_EV_DELAY"] == "100.0"


def test_params_reader_rejects_bad_rows(tmp_path):
    f = tmp_path / "x.params"
    f.write_text("1\t1\tA\t1\n")
    with pytest.raises(ValueError):
        read_params(f)
    f.write_text("# c\n1\t1\tA\t2\t6\n1\t1\tB\t0.5\t9\n")
    assert read_params(f) == {"A": 2, "B": 0.5}


# ---------------------------------------------------------------------------------------
# Fault schedule keyed to the engage (bridge.py, "Fault anchor")


def hovering_path(n, dt=DT):
    t = dt * np.arange(n)
    pos = np.stack([0.3 * np.sin(0.2 * t), 0.2 * np.cos(0.3 * t), 2.0 + 0.05 * np.sin(t)], 1)
    return t, pos


def run_bridge(bridge, t, pos, engage_at=None):
    out = []
    for i, (ti, p) in enumerate(zip(t, pos, strict=True)):
        engaged = engage_at is not None and i >= engage_at
        out.append(bridge.update(ti, p, engaged=engaged))
    return out


def test_engage_anchor_repeats_vio_default_faults_from_the_anchor_for_any_engage_time():
    # Busy faults so a 60 s run holds several jumps and dropouts.
    cfg = dataclasses.replace(EstimateCorruptionConfig(), jump_rate_hz=0.2, dropout_rate_hz=0.2,
                              dropout_max_s=1.0)
    n, anchor_s = 3000, 25.0
    t, pos = hovering_path(n)
    ref = run_bridge(EstimateBridge(cfg, DT, seed=5), t, pos)  # vio_default's schedule
    ref_faults = [(s.jumped, s.dropout_started) for s in ref]
    anchor_step = round(anchor_s / DT)
    early = ref_faults[:anchor_step]
    assert any(f[0] for f in early) and any(f[1] for f in early)
    for engage_at in (900, 1250, 1400):
        b = EstimateBridge(cfg, DT, seed=5, fault_anchor="engage", fault_anchor_s=anchor_s)
        out = run_bridge(b, t, pos, engage_at=engage_at)
        assert b.anchor_step == engage_at and b.anchor_t == pytest.approx(t[engage_at])
        # No jump or dropout before the engage, whatever vio_default did then.
        assert not any(s.jumped or s.dropout_started or s.stale for s in out[:engage_at])
        # From the engage on, the faults are vio_default's from the anchor on, step for step.
        m = min(n - engage_at, n - anchor_step)  # the stretch both runs cover
        got = [(s.jumped, s.dropout_started) for s in out[engage_at:engage_at + m]]
        want = ref_faults[anchor_step:anchor_step + m]
        assert got == want
        assert sum(f[0] for f in got) >= 2 and sum(f[1] for f in got) >= 2
        # And the dropout flags (what is withheld from PX4) follow suit.
        ref_stale = [s.stale for s in ref[anchor_step:anchor_step + m]]
        assert [s.stale for s in out[engage_at:engage_at + m]] == ref_stale
        assert all(s.engaged == (i >= engage_at) for i, s in enumerate(out))


def test_engage_anchor_keeps_the_stream_continuous_at_the_switch():
    cfg = EstimateCorruptionConfig()  # default rates: no fault lands in a few seconds
    n, engage_at = 500, 300
    t, pos = hovering_path(n)
    b = EstimateBridge(cfg, DT, seed=11, fault_anchor="engage", fault_anchor_s=25.0)
    out = run_bridge(b, t, pos, engage_at=engage_at)
    err = np.stack([s.est_pos - s.truth_pos for s in out])
    # The estimate error across the switch moves by no more than the white noise would
    # anyway (bias and delayed history carried over), nowhere near a jump.
    before, after = err[engage_at - 1], err[engage_at]
    assert np.abs(after - before).max() < 6 * cfg.pos_noise_std_m
    plain = EstimateBridge(cfg, DT, seed=11)
    plain.update(t[0], pos[0])
    assert int(b.corruptor.delay_steps[0]) == int(plain.corruptor.delay_steps[0]) > 0


def test_bias_from_anchor_gives_every_anchor_time_the_same_bias_and_stays_continuous():
    cfg = EstimateCorruptionConfig()
    n = 2500
    t, pos = hovering_path(n)

    def bias_after_anchor(engage_at, bias_from_anchor):
        b = EstimateBridge(cfg, DT, seed=4, fault_anchor="engage", fault_anchor_s=25.0,
                           bias_from_anchor=bias_from_anchor)
        trail = []
        for i, (ti, p) in enumerate(zip(t, pos, strict=True)):
            b.update(ti, p, engaged=i >= engage_at)
            trail.append(b.corruptor.bias[0].copy())
        trail = np.stack(trail)
        return trail[:engage_at], trail[engage_at:engage_at + 1000]

    before_a, a = bias_after_anchor(900, True)
    before_b, b_ = bias_after_anchor(1400, True)
    assert not before_a.any() and not before_b.any()  # no walk before the anchor
    assert np.allclose(a, b_)  # the same bias series from the anchor, whenever it fires
    assert np.abs(a).max() > 0.02  # and it does walk from there
    _, c = bias_after_anchor(900, False)
    _, d = bias_after_anchor(1400, False)
    assert not np.allclose(c, d)  # the carried-over walk made them differ before the fix
    # Continuity at the switch holds as before.
    b = EstimateBridge(cfg, DT, seed=4, fault_anchor="engage", fault_anchor_s=25.0,
                       bias_from_anchor=True)
    out = run_bridge(b, t[:1000], pos[:1000], engage_at=900)
    err = np.stack([s.est_pos - s.truth_pos for s in out])
    assert np.abs(err[900] - err[899]).max() < 6 * cfg.pos_noise_std_m
    # The fault schedule is untouched: same jumps and dropouts as without the option.
    busy = dataclasses.replace(cfg, jump_rate_hz=0.2, dropout_rate_hz=0.2, dropout_max_s=1.0)
    with_ = run_bridge(EstimateBridge(busy, DT, seed=5, fault_anchor="engage",
                                      bias_from_anchor=True), t, pos, engage_at=1250)
    without = run_bridge(EstimateBridge(busy, DT, seed=5, fault_anchor="engage"), t, pos,
                         engage_at=1250)
    assert [(s.jumped, s.dropout_started, s.stale) for s in with_] == \
        [(s.jumped, s.dropout_started, s.stale) for s in without]
    # "start" anchors ignore the option: there is nothing to key the walk to.
    assert EstimateBridge(cfg, DT, seed=4, bias_from_anchor=True).bias_from_anchor is False


def test_shipped_anchored_profiles_start_the_bias_at_the_anchor_and_old_ones_do_not():
    for name in ("vio_at_engage", "vio_at_hover"):
        prof = load_profile(name)
        assert prof.bias_from_anchor is True
        assert EstimateProfile.from_dict(prof.to_dict()).bias_from_anchor is True
    old = {"name": "old", "fault_anchor": "engage"}  # a run folder's profile from before
    assert EstimateProfile.from_dict(old).bias_from_anchor is False
    assert load_profile("vio_default").bias_from_anchor is False


def test_engage_anchor_without_engage_never_faults_and_verifies(tmp_path):
    cfg = dataclasses.replace(EstimateCorruptionConfig(), jump_rate_hz=0.5, dropout_rate_hz=0.5,
                              dropout_max_s=0.5)
    prof = EstimateProfile.from_dict({"name": "t", "seed": 2, "fault_anchor": "engage",
                                      "fault_anchor_s": 3.0, "estimate": cfg.__dict__})
    t, pos = hovering_path(600)
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    b = prof.bridge()
    for s in run_bridge(b, t, pos, engage_at=200):
        w.write(s)
    w.close()
    assert not b.anchored or b.anchor_step == 200
    tr = read_trace(path)
    assert "engaged" in tr and tr["engaged"][:200].sum() == 0 and tr["engaged"][200:].all()
    rep = verify(prof, path)
    assert rep["ok"] and rep["anchor_step"] == 200 and rep["jumps"] > 0
    # The replay must honour the engaged column: without it the faults land elsewhere.
    plain = dataclasses.replace(prof, fault_anchor="start")
    assert not verify(plain, path)["ok"]
    # A trace without the column (older flights) still reads, with engaged = 0.
    lines = path.read_text().splitlines()
    cols = lines[0].split(",")
    i = cols.index("engaged")
    stripped = "\n".join(",".join(x for j, x in enumerate(line.split(",")) if j != i)
                         for line in lines) + "\n"
    (tmp_path / "old.csv").write_text(stripped)
    old = read_trace(tmp_path / "old.csv")
    assert old["engaged"].sum() == 0 and old["t"].size == 600


def test_vio_at_engage_profile_is_vio_default_keyed_to_the_engage():
    p = load_profile("vio_at_engage")
    assert p.estimate == EstimateCorruptionConfig() and p.fault_anchor == "engage"
    assert p.fault_anchor_s == 25.0 and p.seed == load_profile("vio_default").seed
    assert load_profile("vio_default").fault_anchor == "start"
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "x", "fault_anchor": "takeoff"})
    with pytest.raises(ValueError):
        EstimateBridge(EstimateCorruptionConfig(), DT, 1, fault_anchor="engage", fault_anchor_s=-1)


def takeoff_path(n, dt=DT, climb_steps=250, altitude=2.0):
    """Ground until step 50, then a straight climb to ``altitude`` over ``climb_steps``,
    then the hovering wander, from a spawn height of 0.25 m."""
    t = dt * np.arange(n)
    z = 0.25 + altitude * np.clip((np.arange(n) - 50) / climb_steps, 0.0, 1.0)
    return t, np.stack([0.3 * np.sin(0.2 * t), 0.2 * np.cos(0.3 * t), z], 1)


def test_altitude_anchor_fires_at_the_height_and_repeats_vio_default_from_there():
    cfg = dataclasses.replace(EstimateCorruptionConfig(), jump_rate_hz=0.2, dropout_rate_hz=0.2,
                              dropout_max_s=1.0)
    n, anchor_s, alt = 3000, 25.0, 1.7
    t, pos = takeoff_path(n)
    ref = run_bridge(EstimateBridge(cfg, DT, seed=5), t, pos)  # vio_default's schedule
    ref_faults = [(s.jumped, s.dropout_started) for s in ref]
    anchor_step = round(anchor_s / DT)
    b = EstimateBridge(cfg, DT, seed=5, fault_anchor="altitude", fault_anchor_s=anchor_s,
                       fault_anchor_alt_m=alt)
    out = run_bridge(b, t, pos)  # never "engaged": the stock baseline flies no mode
    expect = int(np.flatnonzero(pos[:, 2] - pos[0, 2] >= alt)[0])
    assert 50 < expect < 300 and b.anchor_step == expect
    assert b.anchor_t == pytest.approx(t[expect])
    assert not any(s.jumped or s.dropout_started or s.stale for s in out[:expect])
    m = min(n - expect, n - anchor_step)
    got = [(s.jumped, s.dropout_started) for s in out[expect:expect + m]]
    assert got == ref_faults[anchor_step:anchor_step + m]
    assert sum(f[0] for f in got) >= 2 and sum(f[1] for f in got) >= 2
    assert not any(s.engaged for s in out)


def test_vio_at_hover_profile_needs_the_runner_to_fill_the_height(tmp_path):
    p = load_profile("vio_at_hover")
    assert p.fault_anchor == "altitude" and p.fault_anchor_alt_m is None
    assert p.estimate == EstimateCorruptionConfig() and p.fault_anchor_s == 25.0
    with pytest.raises(ValueError):
        p.bridge()  # no height yet
    filled = p.with_anchor_altitude(2.0)
    assert filled.fault_anchor_alt_m == pytest.approx(1.7)  # hover_window's 0.3 m margin
    assert filled.with_anchor_altitude(5.0) is filled  # a set height stays
    assert load_profile("vio_at_engage").with_anchor_altitude(2.0).fault_anchor_alt_m is None
    assert EstimateProfile.from_dict(filled.to_dict()) == filled
    with pytest.raises(ValueError):
        EstimateProfile.from_dict({"name": "x", "fault_anchor": "altitude",
                                   "fault_anchor_alt_m": 0.0})
    # A flight on the filled profile verifies from its trace alone (the truth carries the
    # height), and the unfilled hover profile cannot replay it.
    cfg = dataclasses.replace(EstimateCorruptionConfig(), jump_rate_hz=0.5, dropout_rate_hz=0.5,
                              dropout_max_s=0.5)
    prof = dataclasses.replace(filled, seed=3, fault_anchor_s=3.0, estimate=cfg)
    t, pos = takeoff_path(900)
    path = tmp_path / "trace.csv"
    w = TraceWriter(path)
    b = prof.bridge()
    for s in run_bridge(b, t, pos):
        w.write(s)
    w.close()
    rep = verify(prof, path)
    assert rep["ok"] and rep["fault_anchor"] == "altitude" and rep["anchor_step"] == b.anchor_step
    assert rep["anchor_t"] == pytest.approx(b.anchor_t) and rep["jumps"] > 0
    assert rep["engaged_samples"] == 0
    assert not verify(dataclasses.replace(prof, fault_anchor="start"), path)["ok"]


def test_heartbeat_decoding_matches_sitl_test():
    from sim2real.estimate_bridge.heartbeat import structure_hold_engaged

    def custom(main, sub):
        return (main << 16) | (sub << 24)

    assert structure_hold_engaged(custom(4, 11))  # auto / external 1
    assert structure_hold_engaged(custom(4, 12))  # a later external mode
    assert not structure_hold_engaged(custom(4, 3))  # Hold
    assert not structure_hold_engaged(custom(3, 0))  # Position
