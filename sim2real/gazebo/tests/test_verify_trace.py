import numpy as np
import pytest

from envs.disturbance import disturbance_force, drag_force, shelter_factor, wind_velocity
from sim2real.gazebo.params_io import write_params
from sim2real.gazebo.scenario import load_scenario, sample_scenario
from sim2real.gazebo.verify_trace import main, read_trace, verify


def synthetic_trace(path, p, n=200, dt=0.004, corrupt=0.0):
    """A trace the plugin would write for params p along a made-up trajectory."""
    rng = np.random.default_rng(0)
    t = np.arange(n) * dt
    pos = np.array([-2.0, 0.0, 2.5]) + 0.3 * np.sin(np.outer(t, [1.0, 2.0, 0.5]))
    vel = rng.normal(0.0, 0.5, (n, 3))
    tiled = type(p)(**{k: np.repeat(np.asarray(getattr(p, k)), n, axis=0)
                       for k in p.__dataclass_fields__})
    wind = wind_velocity(tiled, t)
    shelter = shelter_factor(tiled, pos)
    force = disturbance_force(tiled, t, pos, vel) - drag_force(-vel, tiled.k_lin, tiled.k_quad)
    force = force + corrupt
    rows = np.column_stack([np.arange(n), t, pos, vel, wind, shelter, force])
    header = "step,t,x,y,z,vx,vy,vz,wx,wy,wz,shelter,fx,fy,fz"
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.17g")
    return force


@pytest.fixture
def run_dir(tmp_path):
    p = sample_scenario(load_scenario("wind_8ms"))
    write_params(tmp_path / "wind_params.txt", p)
    return tmp_path, p


def test_matching_trace_passes(run_dir):
    tmp, p = run_dir
    force = synthetic_trace(tmp / "trace.csv", p)
    report = verify(tmp / "wind_params.txt", tmp / "trace.csv")
    assert report["ok"]
    assert report["rows"] == 200 and report["applied_lag_steps"] == 0
    assert report["max_force_error_n"] < 1e-9
    assert report["dt_median_s"] == pytest.approx(0.004)
    assert report["applied_force_max_n"] == pytest.approx(np.linalg.norm(force, axis=1).max())
    assert 0.0 < report["shelter_min"] < 1.0


def test_wrong_force_fails(run_dir):
    tmp, p = run_dir
    synthetic_trace(tmp / "trace.csv", p, corrupt=1e-4)
    report = verify(tmp / "wind_params.txt", tmp / "trace.csv")
    assert not report["ok"] and report["max_force_error_n"] == pytest.approx(1e-4)
    assert report["max_wind_error_m_s"] == 0.0


def test_cli_writes_report_and_exit_code(run_dir, capsys):
    tmp, p = run_dir
    synthetic_trace(tmp / "trace.csv", p)
    code = main([str(tmp / "wind_params.txt"), str(tmp / "trace.csv"), "--out",
                 str(tmp / "r.json")])
    assert code == 0 and (tmp / "r.json").exists()
    synthetic_trace(tmp / "trace.csv", p, corrupt=1.0)
    assert main([str(tmp / "wind_params.txt"), str(tmp / "trace.csv")]) == 1


def test_read_trace_rejects_missing_columns(tmp_path):
    (tmp_path / "t.csv").write_text("step,t\n0,0\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_trace(tmp_path / "t.csv")
