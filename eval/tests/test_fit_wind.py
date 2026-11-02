"""eval.fit_wind: the wind and near-wall fit, on model-made logs where the answer is known."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import yaml

from envs.disturbance import DisturbanceConfig, sample_params
from envs.disturbance.core import drag_force
from eval import fit_wind as fw

MASS = 2.064
WALL = {"point": [0.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0]}  # ENU, into the free air
TRUE = dict(mean_speed=8.0, mean_heading_deg=10.0, num_gusts=0, turb_sigma=0.8, turb_tau=1.5,
            k_quad=0.032, wall_decay=1.4, wall_floor=0.12)


def hold(name: str, kind: str, d: float | None, cfg: dict, seed: int = 3,
         p: fw.FitParams | None = None, **kw) -> fw.Segment:
    """One model-made hold, ``d`` m from the wall (None: 30 m out, free air)."""
    point = np.array([-(30.0 if d is None else d), 0.0, 2.5])
    topics = fw.synthetic_topics(DisturbanceConfig.from_dict(cfg), seed, point, MASS,
                                 duration_s=60.0, rate_hz=100.0, wall_point=WALL["point"],
                                 wall_normal=WALL["normal"], **kw)
    spec = {"name": name, "kind": kind, "wall": WALL, "window_s": [10.0, 70.0]}
    return fw.segment_from_topics(topics, spec, p or fw.FitParams())


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4)) * [3.0, 0.3, 0.3, 0.3]
    q[:, 0] = np.abs(q[:, 0])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    r = fw.quat_to_rot(q)
    assert np.allclose(np.einsum("nij,nkj->nik", r, r), np.eye(3), atol=1e-12)
    assert np.allclose(fw.rot_to_quat(r), q, atol=1e-9)


def test_invert_drag_undoes_the_drag():
    rng = np.random.default_rng(1)
    v = rng.normal(scale=4.0, size=(100, 3))
    v[:, 2] = 0.0
    for k_lin, k_quad in [(0.0, 0.03), (0.1, 0.02), (0.2, 0.0)]:
        f = drag_force(v, np.full(100, k_lin), np.full(100, k_quad))
        assert np.allclose(fw.invert_drag(f, k_lin, k_quad), v, atol=1e-9)


def test_world_force_matches_the_model_force_with_offset_and_sway():
    """Tilt, sway and an accelerometer offset: the calm hold's offset taken out, the force
    from body x and y equals the model's horizontal force."""
    cfg = dict(TRUE)
    offset = (0.05, -0.03, 0.0)
    seg = hold("w1", "wall", 1.0, cfg, accel_offset_frd=offset, sway_m=0.2)
    calm = hold("calm", "calm", None, dict(cfg, mean_speed=0.0, turb_sigma=0.0),
                accel_offset_frd=offset, sway_m=0.2)
    off = fw.calm_offset([seg, calm])
    assert np.allclose(off, offset[:2], atol=2e-3)  # the calm sway is drag-only: tiny force
    p = sample_params(DisturbanceConfig.from_dict(cfg), 1, 3, wall_point=WALL["point"],
                      wall_normal=WALL["normal"])
    from envs.disturbance import disturbance_force
    w = 2.0 * math.pi / 7.0
    t = seg.t
    pos = np.array([-1.0, 0.0, 2.5]) + 0.2 * np.stack([np.cos(w * t), np.sin(w * t), 0 * t], 1)
    vel = 0.2 * w * np.stack([-np.sin(w * t), np.cos(w * t), 0 * t], 1)
    true = disturbance_force(p, t - 10.0, pos, vel)
    got = fw.world_force(seg, MASS, np.asarray(offset[:2]))
    # The grid smooths the force over 0.1 s, so compare to the same smoothing of the truth.
    true_s = fw._moving_average(true[:, :2], 5)
    assert np.sqrt(np.mean((got[:, :2] - true_s) ** 2)) < 0.01


def test_fit_shelter_recovers_the_curve():
    d = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    g = fw.shelter_curve(d, 1.3, 0.15)
    length, floor = fw.fit_shelter(d, g)
    assert length == pytest.approx(1.3, rel=1e-4)
    assert floor == pytest.approx(0.15, abs=1e-4)
    # One distance: the floor has to be fixed, and L then follows exactly.
    length1, floor1 = fw.fit_shelter(d[2:3], fw.shelter_curve(d[2:3], 1.3, 0.0), floor=0.0)
    assert (length1, floor1) == (pytest.approx(1.3, rel=1e-4), 0.0)


def test_full_fit_recovers_the_model():
    """Free air with a reference speed, five wall distances and a calm hold, in the same
    wind: the drag, the heading and the shelter come back."""
    p = fw.FitParams()
    segs = [hold("calm", "calm", None, dict(TRUE, mean_speed=0.0, turb_sigma=0.0), p=p)]
    free = hold("free", "free", None, TRUE, p=p)
    # The reference an anemometer at the hold point would read over the same samples.
    air = fw.air_velocity(free, MASS, np.zeros(2), 0.0, TRUE["k_quad"])
    free.ref_speed = float(np.linalg.norm(air[free.valid, :2].mean(axis=0)))
    segs.append(free)
    for d in (0.75, 1.0, 1.5, 2.0, 3.0):
        segs.append(hold(f"w{d}", "wall", d, TRUE, p=p))
    res = fw.fit(segs, MASS, p)
    assert res["drag"]["method"] == "one speed, k_lin fixed"
    assert res["drag"]["k_quad"] == pytest.approx(TRUE["k_quad"], rel=0.03)
    case = res["cases"]["default"]
    assert case["heading_deg"] == pytest.approx(TRUE["mean_heading_deg"], abs=1.0)
    assert res["shelter"]["wall_decay_m"] == pytest.approx(TRUE["wall_decay"], rel=0.03)
    assert res["shelter"]["wall_floor"] == pytest.approx(TRUE["wall_floor"], abs=0.02)
    for h in case["holds"]:
        assert h["crossflow_frac"] < 0.01  # shelter alone keeps the direction
        assert h["sigma_over_sheltered"] == pytest.approx(1.0, abs=0.05)
    turb = res["turbulence"]
    # A 60 s hold sees part of the model's spectrum (periods up to 300 s): within 15 %.
    assert turb["sigma_m_s"] == pytest.approx(TRUE["turb_sigma"], rel=0.15)
    assert 0.8 < turb["tau_s"] < 2.0
    prop = fw.proposal(res)
    assert prop["covered"]["wall_decay"] and prop["covered"]["wall_floor"]
    lo, hi = prop["block"]["wall_decay"]
    assert lo < TRUE["wall_decay"] < hi


def test_two_reference_speeds_fit_both_drag_terms():
    cfg = dict(TRUE, k_lin=0.08, k_quad=0.025, turb_sigma=0.3)
    segs = []
    for speed in (4.0, 8.0):
        s = hold(f"free{speed}", "free", None, dict(cfg, mean_speed=speed))
        air = fw.air_velocity(s, MASS, np.zeros(2), cfg["k_lin"], cfg["k_quad"])
        s.ref_speed = float(np.linalg.norm(air[s.valid, :2].mean(axis=0)))
        s.case = f"{speed}"
        segs.append(s)
    res = fw.fit(segs, MASS, fw.FitParams())
    assert res["drag"]["method"] == "two speeds"
    assert res["drag"]["k_lin"] == pytest.approx(0.08, abs=0.01)
    assert res["drag"]["k_quad"] == pytest.approx(0.025, rel=0.04)


def test_shelter_needs_no_drag_calibration():
    """Without a reference speed the configured k_quad is used; it is wrong by 50 % here,
    and the shelter still comes back, since it is a ratio of speeds."""
    p = fw.FitParams(k_quad=0.048)
    segs = [hold("free", "free", None, TRUE, p=p)]
    segs += [hold(f"w{d}", "wall", d, TRUE, p=p) for d in (0.75, 1.5, 3.0)]
    res = fw.fit(segs, MASS, p)
    assert res["drag"]["method"] == "configured"
    assert res["shelter"]["wall_decay_m"] == pytest.approx(TRUE["wall_decay"], rel=0.03)
    assert res["cases"]["default"]["mean_speed_m_s"] < 7.0  # speeds scale with 1/sqrt(k)


def test_exclude_window_keeps_a_gust_out_of_the_turbulence():
    cfg = dict(TRUE, num_gusts=1, gust_start=(30.0, 30.0), gust_duration=(3.0, 3.0),
               gust_speed=(4.0, 4.0))
    p = fw.FitParams(k_quad=TRUE["k_quad"])
    plain = fw.segment_stats(hold("g", "free", None, cfg, p=p), MASS, np.zeros(2), 0.0,
                             TRUE["k_quad"], p)
    topics = fw.synthetic_topics(DisturbanceConfig.from_dict(cfg), 3, np.array([-30., 0, 2.5]),
                                 MASS, wall_point=WALL["point"], wall_normal=WALL["normal"])
    masked = fw.segment_from_topics(topics, {"name": "g", "kind": "free", "window_s": [10, 70],
                                             "exclude_s": [[39.5, 43.5]]}, p)
    excl = fw.segment_stats(masked, MASS, np.zeros(2), 0.0, TRUE["k_quad"], p)
    calm = fw.segment_stats(hold("n", "free", None, TRUE, p=p), MASS, np.zeros(2), 0.0,
                            TRUE["k_quad"], p)
    assert excl.sigma_m_s == pytest.approx(calm.sigma_m_s, rel=0.08)
    assert plain.sigma_m_s >= excl.sigma_m_s


def test_auto_window_finds_the_hold():
    t = np.arange(0.0, 100.0, 0.1)
    z = np.clip((t - 10.0) * 0.5, 0.0, 2.5)
    z[t > 80] = np.clip(2.5 - (t[t > 80] - 80.0) * 0.5, 0.0, 2.5)
    t0, t1 = fw.auto_window(t, z, settle_s=5.0)
    assert t0 == pytest.approx(14.7 + 5.0, abs=0.2)
    assert t1 == pytest.approx(80.3 - 1.0, abs=0.2)


def test_compare_truth_rows():
    res = {"cases": {"a": {"mean_speed_m_s": 8.2, "heading_deg": 359.0}},
           "turbulence": {"sigma_m_s": 0.7, "tau_s": None}, "drag": {"k_quad": 0.033},
           "shelter": {"wall_decay_m": 1.5, "wall_floor": 0.1}}
    rows = fw.compare_truth(res, {"heading_deg": 1.0, "wall_decay": 1.4, "turb_tau": 1.5})
    assert rows["heading_deg[a]"]["error"] == pytest.approx(-2.0)
    assert rows["wall_decay"]["error_pct"] == pytest.approx(100 * 0.1 / 1.4)
    assert "turb_tau" not in rows  # no fitted value: nothing to compare
    with pytest.raises(ValueError):
        fw.compare_truth(res, {"nonsense": 1.0})


def test_wall_hold_without_distance_is_refused():
    topics = fw.synthetic_topics(DisturbanceConfig.from_dict(TRUE), 3, np.array([-1., 0, 2.5]),
                                 MASS, wall_point=WALL["point"], wall_normal=WALL["normal"])
    topics[fw.ESTIMATE] = topics.pop(fw.GROUNDTRUTH)  # a real log: no truth
    with pytest.raises(ValueError, match="distance_m"):
        fw.segment_from_topics(topics, {"name": "w", "kind": "wall", "wall": WALL,
                                        "window_s": [10, 70]}, fw.FitParams())
    seg = fw.segment_from_topics(topics, {"name": "w", "kind": "wall", "distance_m": 1.0,
                                          "window_s": [10, 70]}, fw.FitParams())
    assert np.all(seg.dist == 1.0)


def test_entrypoint_smoke(tmp_path):
    """The shipped config's smoke segments (model-made logs) through main()."""
    from training.runlib import read_summary
    cfg = yaml.safe_load(open(fw.REPO_ROOT / "eval/configs/wind_fit_gazebo.yaml"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert fw.main(["--run-dir", str(tmp_path), "--config", str(path), "--smoke"]) == 0
    result = json.loads((tmp_path / "fit.json").read_text())
    assert result["shelter"]["wall_decay_m"] == pytest.approx(1.4, rel=0.05)
    assert read_summary(tmp_path)["status"] == "ok"
