"""eval.fit_airframe: a customer's airframe from one flight, on flights with known answers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from eval import fit_airframe as fa

TRUE = dict(mass=1.6, rotor_k=7.0, esc_min=0.12, motor_tau=0.025,
            effectiveness=(180.0, 160.0, 12.0), yaw_to_roll=0.05)


@pytest.fixture(scope="module")
def fitted():
    f, prm = fa.synthetic_flight(**TRUE)
    return fa.fit_airframe(f, TRUE["mass"], prm, fa.FitParams()), prm


def test_thrust_curve_and_motor_lag(fitted):
    res, _ = fitted
    th = res["thrust"]
    assert th["rotors"] == 4
    assert th["max_thrust_n"] == pytest.approx(4 * TRUE["rotor_k"], rel=0.01)
    assert th["esc_min_frac"] == pytest.approx(TRUE["esc_min"], abs=0.02)
    assert th["motor_tau_s"] == pytest.approx(TRUE["motor_tau"], abs=0.004)
    # Hover: the curve's thrust there carries the weight.
    u = th["hover_command"]
    e = th["esc_min_frac"]
    assert 4 * th["rotor_k_n"] * (e + (1 - e) * u) ** 2 == pytest.approx(TRUE["mass"] * fa.GRAVITY)


def test_torque_effectiveness_and_coupling(fitted):
    res, _ = fitted
    axes = res["torque"]["axes"]
    for axis, b in zip(("roll", "pitch", "yaw"), TRUE["effectiveness"], strict=True):
        assert axes[axis]["effectiveness_rad_s2"] == pytest.approx(b, rel=0.03)
        assert axes[axis]["replay_r2"] > 0.99
    assert res["torque"]["coupling"]["yaw_to_roll"] == pytest.approx(TRUE["yaw_to_roll"],
                                                                     abs=0.01)
    assert abs(res["torque"]["coupling"]["roll_to_pitch"]) < 0.01
    # Only b reaches the dynamics: torque per unit over the inertia is b, whatever inertia.
    af = res["airframe"]
    for k, b in enumerate(TRUE["effectiveness"]):
        assert af["actuation"]["torque_per_unit"][k] / af["quad"]["inertia"][k] == \
            pytest.approx(b, rel=0.03)


def test_given_inertia_is_used(fitted):
    f, prm = fa.synthetic_flight(**TRUE)
    res = fa.fit_airframe(f, TRUE["mass"], prm, fa.FitParams(), inertia=(0.01, 0.02, 0.03))
    assert res["airframe"]["quad"]["inertia"] == [0.01, 0.02, 0.03]
    assert res["airframe"]["actuation"]["torque_per_unit"][1] == pytest.approx(0.02 * 160.0,
                                                                               rel=0.03)


def test_rate_pid_from_params_scales_by_k():
    pid = fa.rate_pid_from_params({"MC_PITCHRATE_P": 0.1, "MC_PITCHRATE_I": 0.2,
                                   "MC_PITCHRATE_K": 1.5, "MC_YR_INT_LIM": 0.25,
                                   "IMU_GYRO_CUTOFF": 30.0})
    assert pid["p"][1] == pytest.approx(0.15)
    assert pid["i"][1] == pytest.approx(0.3)
    assert pid["int_lim"][2] == 0.25
    assert pid["gyro_cutoff_hz"] == 30.0
    assert pid["p"][0] == 0.15  # PX4's default where the file is silent


def test_fitted_airframe_loads_into_the_env_config(fitted, tmp_path):
    pytest.importorskip("torch")
    from envs.structure_hold import StructureHoldConfig

    res, _ = fitted
    path = tmp_path / "airframe.yaml"
    path.write_text(yaml.safe_dump(res["airframe"], sort_keys=False))
    cfg = StructureHoldConfig.from_dict({"airframe": str(path)})
    assert cfg.quad.mass == pytest.approx(TRUE["mass"])
    assert cfg.quad.max_thrust == pytest.approx(res["thrust"]["max_thrust_n"])
    assert cfg.actuation.esc_min_frac == pytest.approx(res["thrust"]["esc_min_frac"])
    assert cfg.rate_pid.p == tuple(res["airframe"]["rate_pid"]["p"])


def test_flight_from_topics_uses_the_airborne_stretch():
    f, _ = fa.synthetic_flight(seconds=30.0, **TRUE)
    us = np.round((100.0 + f.t) * 1e6).astype(np.int64)
    motors = np.full((len(f.t), 12), np.nan)
    motors[:, :4] = f.motors
    topics = {
        fa.MOTORS: {"timestamp": us, **{f"control[{i}]": motors[:, i] for i in range(12)}},
        fa.TORQUE: {"timestamp": us, **{f"xyz[{i}]": f.torque_sp[:, i] for i in range(3)}},
        fa.IMU: {"timestamp": us, **{f"gyro_rad[{i}]": f.gyro[:, i] for i in range(3)},
                 **{f"accelerometer_m_s2[{i}]": np.zeros(len(f.t)) for i in range(2)},
                 "accelerometer_m_s2[2]": f.accel_z},
        fa.LANDED: {"timestamp": np.array([us[0], us[0] + 2_000_000, us[-1] - 2_000_000]),
                    "landed": np.array([1, 0, 1])},
    }
    got = fa.flight_from_topics(topics, fa.FitParams())
    assert got.motors.shape[1] == 4  # the NaN outputs are dropped
    assert got.t[0] == pytest.approx(102.0 + 3.0, abs=0.01)
    assert got.t[-1] == pytest.approx(130.0 - 2.0 - 3.0, abs=0.01)
    del topics[fa.MOTORS]
    with pytest.raises(ValueError, match="SDLOG_PROFILE"):
        fa.flight_from_topics(topics, fa.FitParams())


def test_compare_to_the_x500_itself_is_exact():
    from envs.structure_hold.airframes import gz_x500

    ref = gz_x500()
    b = [t / i for t, i in zip(ref["actuation"]["torque_per_unit"], ref["quad"]["inertia"],
                               strict=True)]
    res = {"airframe": {"quad": {**ref["quad"]}, "actuation": {**ref["actuation"]}},
           "torque": {"axes": {a: {"effectiveness_rad_s2": v}
                               for a, v in zip(("roll", "pitch", "yaw"), b, strict=True)}}}
    rows = fa.compare_to(res, "gz_x500")
    assert all(abs(r["error_pct"]) < 1e-9 for r in rows.values())


def test_entrypoint_smoke(tmp_path):
    from training.runlib import read_summary

    cfg = yaml.safe_load(open(fa.REPO_ROOT / "eval/configs/airframe_fit_gz_x500.yaml"))
    cfg["params"].update(cfg["smoke"])  # what training.launch --smoke merges
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert fa.main(["--run-dir", str(tmp_path), "--config", str(path), "--smoke"]) == 0
    res = json.loads((tmp_path / "fit.json").read_text())
    assert res["thrust"]["rotors"] == 4
    assert yaml.safe_load((tmp_path / "airframe.yaml").read_text())["quad"]["mass"] > 0
    assert read_summary(tmp_path)["status"] == "ok"
