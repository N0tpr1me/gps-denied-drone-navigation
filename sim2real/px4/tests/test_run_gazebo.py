"""The Gazebo hold runner's pure parts: the hold window and the module statistics from the
status topic, the docker command, and the pass line."""

from __future__ import annotations

import numpy as np

from sim2real.px4 import run_gazebo
from sim2real.px4.sitl_test import STATE_ENGAGED, STATE_HANDBACK, STATE_STANDBY

START = 5_000_000


def synthetic_status(n: int = 1200, dt: float = 0.01) -> dict[str, np.ndarray]:
    """Standby 2 s, engaged 6 s (fallback for 1 s in the middle), standby 1 s, engaged 2 s,
    handback 1 s."""
    t = np.arange(n) * dt
    state = np.full(n, STATE_STANDBY, dtype=np.uint8)
    state[(t >= 2) & (t < 8)] = STATE_ENGAGED
    state[(t >= 9) & (t < 11)] = STATE_ENGAGED
    state[t >= 11] = STATE_HANDBACK
    fallback = ((t >= 5) & (t < 6)).astype(np.uint8)
    stale = np.where(fallback == 1, t - 4.4, 0.02)  # 0.6 .. 1.59 s while in the fallback
    status = {
        "timestamp": (START + t * 1e6).astype(np.int64),
        "state": state,
        "fallback_active": fallback,
        "yaw_guarded": np.zeros(n, np.uint8),
        "stale_s": stale.astype(np.float32),
        "controller": np.ones(n, np.uint8),
        "loop_time_us": np.full(n, 40, np.uint32),
    }
    for k in range(3):
        status[f"setpoint_ned[{k}]"] = np.full(n, [1.0, 2.0, -2.5][k], np.float32)
        status[f"position_error_ned[{k}]"] = np.full(n, [0.3, -0.4, 0.0][k], np.float32)
    for k in range(4):
        status[f"action[{k}]"] = np.full(n, [1.0, 0.2, -0.1, 0.0][k], np.float32)
    return status


def test_hold_window_first_engaged_stretch_trimmed():
    status = synthetic_status()
    w = run_gazebo.hold_window(status, START, settle_s=2.0)
    assert w is not None
    assert abs(w[0] - 4.0) < 1e-6 and abs(w[1] - 7.99) < 1e-6
    w2 = run_gazebo.hold_window(status, START, settle_s=0.5, which=1)
    assert w2 is not None and abs(w2[0] - 9.5) < 1e-6 and abs(w2[1] - 10.99) < 1e-6
    assert run_gazebo.hold_window(status, START, which=2) is None
    # A stretch shorter than the settle time is no window.
    assert run_gazebo.hold_window(status, START, settle_s=3.0, which=1) is None


def test_hold_point_is_the_captured_setpoint_in_enu():
    status = synthetic_status()
    w = run_gazebo.hold_window(status, START, settle_s=2.0)
    p = run_gazebo.hold_point_enu(status, START, w)
    assert np.allclose(p, [2.0, 1.0, 2.5])  # NED (1, 2, -2.5) -> ENU (2, 1, 2.5)


def test_hold_stats_counts_fallback_and_actions():
    status = synthetic_status()
    w = run_gazebo.hold_window(status, START, settle_s=2.0)  # 4 .. 8 s: 400 ticks
    s = run_gazebo.hold_stats(status, START, w)
    assert s["ticks"] == 400
    assert abs(s["fallback_share"] - 0.25) < 1e-6 and s["fallback_episodes"] == 1
    assert s["yaw_guard_share"] == 0.0
    assert abs(s["stale_max_s"] - 1.59) < 1e-3
    assert abs(s["stale_share_over_0p5s"] - 0.25) < 1e-6
    assert abs(s["estimate_error_xy_max_m"] - 0.5) < 1e-6
    assert abs(s["action_saturated_share"] - 1 / 3) < 1e-9
    assert s["controller"] == 1 and s["loop_time_max_us"] == 40
    empty = run_gazebo.hold_stats(status, START, (0.0, 1.0))
    assert empty == {"ticks": 0}


def test_docker_command_forwards_module_params(tmp_path):
    cmd = run_gazebo.docker_command("img", tmp_path, "0,0,0,0,0,0", 2.5, 60.0, 5.0, 1.0,
                                    ["SH_FB_EN=1", "SH_CTRL=0"], "name",
                                    {"PX4_PARAM_EKF2_EV_CTRL": "7"})
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "hold-test" in cmd and cmd[cmd.index("img") + 1] == "hold-test"
    assert "-e" in cmd and "PX4_PARAM_EKF2_EV_CTRL=7" in cmd
    assert cmd.count("--param") == 2 and "SH_CTRL=0" in cmd
    assert "ESTIMATE_BRIDGE_PROFILE=/out/estimate_config.yaml" in cmd
    i = cmd.index("--hold-s")
    assert cmd[i + 1] == "60.0"


def test_pass_line_needs_every_check():
    good = {"flight_ok": True, "estimate_ok": True, "wind_ok": True, "gps_free": True,
            "vision_fused": True}
    assert run_gazebo.assemble(dict(good))["ok"]
    for key in good:
        bad = dict(good)
        bad[key] = False
        assert not run_gazebo.assemble(bad)["ok"]
    assert not run_gazebo.assemble({})["ok"]
