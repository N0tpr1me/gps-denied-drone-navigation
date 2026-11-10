import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from eval.vio_runaway import RunawayFix, describe, poisson_upper

REPO = Path(__file__).resolve().parents[2]


def stream(seconds=20.0, runaway_at=None, accel=0.0, noise=0.2, bias=0.05, seed=0,
           reset_at=None, gap=None):
    """IMU at 100 Hz and fixes at 50 Hz from a vehicle swinging in gusts, as (kind, t_us,
    xyz, counter) tuples."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(int(seconds * 100)):
        t = 0.01 * k
        t_us = 1_000_000 + 10_000 * k
        w = 2.0
        acc = np.array([0.8 * np.cos(w * t), -0.5 * np.sin(1.3 * w * t), 0.1])
        vel = np.array([0.4 * np.sin(w * t), 0.5 / (1.3 * w) * np.cos(1.3 * w * t), 0.1 * t])
        out.append(("i", t_us, acc + bias + 0.05 * rng.standard_normal(3), 0))
        if k % 2 == 0 and not (gap and gap[0] <= t < gap[1]):
            err = np.zeros(3)
            if runaway_at is not None and t > runaway_at:
                err[1] = accel * (t - runaway_at)
            counter = 1 if reset_at is not None and t >= reset_at else 0
            out.append(("f", t_us, vel + err + noise * rng.standard_normal(3), counter))
    return out


def replay_python(samples, limit):
    f = RunawayFix(1, limit)
    res = []
    for kind, t, xyz, counter in samples:
        if kind == "i":
            f.inertial(t, np.asarray(xyz, float)[None])
        else:
            f.fix(t, np.asarray(xyz, float)[None], counter)
            res.append((float(f.disagreement[0]), bool(f.runaway[0])))
    return res


@pytest.fixture(scope="module")
def replay_cpp(tmp_path_factory):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not installed")
    exe = tmp_path_factory.mktemp("rr") / "runaway_replay"
    subprocess.run([gxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    f"-I{REPO / 'sim2real/px4/structure_hold'}",
                    str(REPO / "sim2real/px4/tests/runaway_replay.cpp"), "-o", str(exe)],
                   check=True)

    def run(samples, limit):
        lines = [f"{limit}"]
        for kind, t, xyz, counter in samples:
            x, y, z = xyz
            lines.append(f"{kind} {t} {x:.9g} {y:.9g} {z:.9g}" + (f" {counter}" if kind == "f"
                                                                  else ""))
        out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True,
                             text=True, check=True).stdout.split("\n")
        return [(float(a), b == "1") for a, b in (ln.split() for ln in out if ln)]

    return run


@pytest.mark.parametrize("case", [
    {},  # healthy
    {"runaway_at": 6.0, "accel": 0.5},
    {"runaway_at": 6.0, "accel": 3.0, "reset_at": 12.0},  # caught, then the VIO restarts
    {"runaway_at": 4.0, "accel": 1.0, "gap": (8.0, 9.0)},  # a dropout restarts the window
])
def test_python_check_matches_the_module(replay_cpp, case):
    samples = stream(**case)
    py, cc = replay_python(samples, 1.0), replay_cpp(samples, 1.0)
    assert len(py) == len(cc)
    dv_py, dv_cc = np.array([d for d, _ in py]), np.array([d for d, _ in cc])
    assert np.abs(dv_py - dv_cc).max() < 2e-3  # float32 in the module
    assert [r for _, r in py] == [r for _, r in cc]


def test_healthy_vio_never_trips_and_a_runaway_does():
    assert not any(r for _, r in replay_python(stream(seconds=60), 0.8))
    res = replay_python(stream(runaway_at=6.0, accel=0.5), 0.8)
    first = next(i for i, (_, r) in enumerate(res) if r)
    t_trip = first * 0.02
    assert 6.0 + 1.6 < t_trip < 6.0 + 3.0  # 0.8 m/s over the 3 s window, plus two 0.25 s slots
    assert res[-1][1]  # latched


def test_restart_counter_and_limit_zero():
    res = replay_python(stream(runaway_at=4.0, accel=3.0, reset_at=10.0, seconds=11), 1.0)
    assert any(r for _, r in res[:300]) and not res[-1][1]
    assert not any(r for _, r in replay_python(stream(runaway_at=4.0, accel=3.0), 0.0))


def test_describe_counts_false_trips_and_catches():
    rows = [{"length_s": 3600.0, "reason": "timeout", "trip_s": None, "trip_move_m": None,
             "max_dv_mps": 0.3, "runaway_at_s": None}]
    d = describe(rows)
    assert d["trips"] == 0 and d["false_trips_per_h_95"] == pytest.approx(3.0, rel=0.01)
    rows = [{"length_s": 30.0, "reason": "box", "trip_s": 23.0, "trip_move_m": 0.4,
             "max_dv_mps": 2.0, "runaway_at_s": 20.0},
            {"length_s": 60.0, "reason": "timeout", "trip_s": 25.0, "trip_move_m": 0.8,
             "max_dv_mps": 2.0, "runaway_at_s": 20.0}]
    d = describe(rows)
    assert d["caught"] == 2 and d["time_to_trip_s"]["median"] == pytest.approx(4.0)
    assert d["ended_early"] == 1 and d["end_reasons"] == {"box": 1}
    assert poisson_upper(5) == pytest.approx(10.51, abs=0.05)
