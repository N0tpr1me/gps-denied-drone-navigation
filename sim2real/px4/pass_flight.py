"""Stock PX4 flying a surface pass on trajectory setpoints: the baseline for the module's pass.

The module flies a pass itself (``SH_PASS_*``, PassPath.hpp). Stock PX4 has no such mode, so
its side of the comparison is what an integrator would write without us: a companion script
that streams the same path to PX4's own position controller in Offboard mode. Each setpoint
(MAVLink ``SET_POSITION_TARGET_LOCAL_NED``, 50 Hz) carries the position, the velocity and
the acceleration of the same trapezoid, so the tuned controller gets every feed-forward it
can use. Runs inside the SITL image next to PX4 (entrypoint.sh ``with-sitl``), or against
any PX4 SITL on udp 14540:

    python3 pass_flight.py --out /out/summary.json --pass-len 5 --hold-s 30

Sequence: take off to ``--altitude`` and settle in stock Hold; wait until PX4's time since
boot reaches ``--engage-at-boot-s`` (the moment the module's flight engages, so both meet
the same wind and the same estimate faults at the same point of the pass); latch the
position and heading there and switch to Offboard; hold ``--pass-t0`` seconds, fly
``--pass-len`` metres to the right of the heading (left for a negative length, up or down
the wall with ``--pass-up``) at ``--pass-speed``, hold at the end; after ``--hold-s`` seconds
in Offboard, land.

The path runs on PX4's own clock (``time_boot_ms`` of its position messages), which in
Gazebo is the simulation's, so a slow simulation does not stretch the pass. The checks here
are on PX4's estimate; the score is made on the true position from the ulog
(``sim2real.estimate_bridge.run_estimate``, ``eval/pass_score.py``).

Unlike the module, this path does not pause when the position estimate goes stale, has no
wall guard and no keep-out: stock PX4 flies what it is sent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pymavlink.dialects.v20 import common as mavlink  # noqa: E402
from sitl_test import (  # noqa: E402
    MAIN_AUTO,
    MAVLINK_URL,
    SUB_AUTO_LAND,
    RtfSampler,
    Vehicle,
    mode_name,
    pass_direction,
    pass_duration,
    pass_profile,
)

MAIN_OFFBOARD = 6  # px4_custom_mode.h
SETPOINT_HZ = 50.0
# PX4 takes Offboard only once setpoints already arrive (its proof of life).
PRESTREAM_S = 1.0
# Position, velocity, acceleration and yaw are all used; only the yaw rate is ignored.
TYPE_MASK = mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="summary JSON path")
    p.add_argument("--altitude", type=float, default=2.5, help="takeoff altitude, m")
    p.add_argument("--settle-s", type=float, default=5.0, help="stock Hold before the pass, s")
    p.add_argument("--hold-s", type=float, default=30.0,
                   help="time in Offboard: the settle, the pass and the hold at its end, s")
    p.add_argument("--pass-len", type=float, default=5.0,
                   help="pass length, m (positive right or up, negative left or down)")
    p.add_argument("--pass-up", action="store_true", help="the pass goes up the wall")
    p.add_argument("--pass-speed", type=float, default=0.5, help="cruise speed, m/s")
    p.add_argument("--pass-t0", type=float, default=3.0, help="hold before the pass moves, s")
    p.add_argument("--engage-at-boot-s", type=float, default=0.0,
                   help="switch to Offboard when PX4's time since boot reaches this, s")
    p.add_argument("--drift-tol", type=float, default=1.0,
                   help="max distance from the moving point on PX4's estimate, m")
    p.add_argument("--ready-timeout-s", type=float, default=300.0)
    return p.parse_args(argv)


def setpoint(t_pass: float, p0: tuple[float, float, float], d: tuple[float, float, float],
             length: float, speed: float) -> tuple[list[float], list[float], list[float]]:
    """Position, velocity and acceleration (NED) ``t_pass`` seconds after the pass starts."""
    s, v = pass_profile(t_pass, length, speed)
    # The acceleration of the trapezoid, from its speed a moment either side.
    eps = 0.01
    a = (pass_profile(t_pass + eps, length, speed)[1]
         - pass_profile(t_pass - eps, length, speed)[1]) / (2.0 * eps)
    return ([p0[i] + s * d[i] for i in range(3)], [v * d[i] for i in range(3)],
            [a * d[i] for i in range(3)])


def send(v: Vehicle, pos: list[float], vel: list[float], acc: list[float], yaw: float) -> None:
    v.m.mav.set_position_target_local_ned_send(
        int(v.tel.boot_s * 1000) & 0xFFFFFFFF, v.m.target_system, v.m.target_component,
        mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK, pos[0], pos[1], pos[2],
        vel[0], vel[1], vel[2], acc[0], acc[1], acc[2], yaw, 0.0)


def in_offboard(v: Vehicle) -> bool:
    return v.tel.main_mode == MAIN_OFFBOARD


def fly(args: argparse.Namespace) -> dict:
    result: dict = {"px4_tag": os.environ.get("PX4_TAG", "unknown"), "checks": {}}
    checks = result["checks"]
    # Centred sticks as in the module's flight, so both fly with the same links alive.
    v = Vehicle(MAVLINK_URL, joystick=True)
    start = time.monotonic()
    v.m.wait_heartbeat(timeout=args.ready_timeout_s)
    v.wait_for(lambda: not math.isnan(v.tel.down), 60, "local position")

    def try_arm() -> bool:
        return v.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1) == mavlink.MAV_RESULT_ACCEPTED

    v.set_param("MIS_TAKEOFF_ALT", args.altitude)
    v.wait_for(try_arm, args.ready_timeout_s, "arming")
    result["ready_s"] = round(time.monotonic() - start, 1)
    ground_down = v.tel.down
    nan = float("nan")
    assert v.command(mavlink.MAV_CMD_NAV_TAKEOFF, nan, 0, 0, nan, nan, nan, nan) \
        == mavlink.MAV_RESULT_ACCEPTED, "takeoff rejected"
    result["climb_s"] = round(v.wait_for(
        lambda: ground_down - v.tel.down >= args.altitude - 0.3, 60, "takeoff altitude"), 1)
    v.pump(args.settle_s)

    # Setpoints at the place the vehicle is, until the engage time: PX4 needs the stream
    # before it takes Offboard, and until then the setpoints change nothing.
    period = 1.0 / SETPOINT_HZ
    result["engage_late_s"] = round(max(
        v.tel.boot_s + PRESTREAM_S - args.engage_at_boot_s, 0.0), 2) \
        if args.engage_at_boot_s > 0 else 0.0
    t_pre = v.tel.boot_s
    while v.tel.boot_s < max(args.engage_at_boot_s, t_pre + PRESTREAM_S):
        here = [v.tel.north, v.tel.east, v.tel.down]
        send(v, here, [0.0] * 3, [0.0] * 3, v.tel.yaw)
        v.pump(period)

    # The engage: latch where and which way, as the module does, and switch.
    p0 = (v.tel.north, v.tel.east, v.tel.down)
    yaw0 = v.tel.yaw
    boot0 = v.tel.boot_s
    result["engage_boot_s"] = round(boot0, 2)
    length = abs(args.pass_len)
    d = pass_direction(yaw0, args.pass_up, args.pass_len)
    n = (math.cos(yaw0), math.sin(yaw0))
    send(v, list(p0), [0.0] * 3, [0.0] * 3, yaw0)
    ack = v.set_mode(MAIN_OFFBOARD)
    checks["offboard_accepted"] = ack == mavlink.MAV_RESULT_ACCEPTED

    lag: list[float] = []
    standoff: list[float] = []
    off: list[float] = []
    left_offboard_at = None
    seen_offboard = False
    while True:
        # PX4's clock between its position messages, so the path does not step with them.
        now = v.boot_at(time.monotonic())
        if now - boot0 >= args.hold_s:
            break
        pos, vel, acc = setpoint(now - boot0 - args.pass_t0, p0, d, length, args.pass_speed)
        send(v, pos, vel, acc, yaw0)
        v.pump(period)
        seen_offboard = seen_offboard or in_offboard(v)
        if seen_offboard and not in_offboard(v) and left_offboard_at is None:
            left_offboard_at = now - boot0
        rel = (v.tel.north - pos[0], v.tel.east - pos[1], v.tel.down - pos[2])
        lag.append(rel[0] * d[0] + rel[1] * d[1] + rel[2] * d[2])
        standoff.append(rel[0] * n[0] + rel[1] * n[1])
        off.append(math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2))
    end = (v.tel.north - p0[0], v.tel.east - p0[1], v.tel.down - p0[2])
    result["pass"] = {
        "length_m": args.pass_len, "up": bool(args.pass_up), "speed_mps": args.pass_speed,
        "t0_s": args.pass_t0, "expected_s": round(pass_duration(length, args.pass_speed), 2),
        "hold_s": args.hold_s, "setpoints_sent": len(off),
        "end_along_m": round(end[0] * d[0] + end[1] * d[1] + end[2] * d[2], 3),
        "max_lag_m": round(max(abs(x) for x in lag), 3) if lag else None,
        "max_standoff_change_m": round(max(abs(x) for x in standoff), 3) if standoff else None,
        "max_off_point_m": round(max(off), 3) if off else None,
        "left_offboard_after_s": None if left_offboard_at is None else round(left_offboard_at, 2),
    }
    checks["offboard_engaged"] = seen_offboard
    checks["still_offboard_after_pass"] = seen_offboard and left_offboard_at is None
    # On PX4's estimate, as sitl_test.py's hold checks; the runner re-judges on truth.
    checks["pass_tracked"] = bool(off) and max(off) <= args.drift_tol
    checks["pass_reached_end"] = abs(result["pass"]["end_along_m"] - length) <= args.drift_tol

    result["land_ack"] = v.command(mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, nan, nan, nan, nan)
    result["land_mode"] = "Land"
    v.wait_for(lambda: v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode == SUB_AUTO_LAND
               or v.tel.landed, 5, "Land in heartbeat")
    result["land_s"] = round(v.wait_for(lambda: v.tel.landed, 90, "touchdown"), 1)
    v.wait_for(lambda: not v.tel.armed, 30, "auto-disarm")
    checks["landed_and_disarmed"] = True
    result["last_mode"] = mode_name(v)
    result["wall_s"] = round(time.monotonic() - start, 1)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rtf = RtfSampler()
    try:
        with rtf:
            summary = fly(args)
    except Exception as exc:  # report every failure in the summary, then fail the run
        summary = {"checks": {}, "error": f"{type(exc).__name__}: {exc}"}
    if rtf.summary() is not None:
        summary["real_time_factor"] = rtf.summary()
    summary["passed"] = bool(summary["checks"]) and all(summary["checks"].values()) \
        and "error" not in summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
