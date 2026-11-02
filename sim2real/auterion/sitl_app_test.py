"""Acceptance flight for the Structure Hold app (external-mode variant) in PX4 SITL.

Stock PX4 v1.17.0 with the SIH quadrotor (no Structure Hold module compiled in), the app
running in its Docker image (docker/Dockerfile.sitl) and talking to PX4 over ROS 2 through
the Micro XRCE-DDS Agent. sitl_app_test.sh starts both; this script flies:

0. on the ground, armed: a switch into Structure Hold is refused (the app's arming check:
   on the ground);
1. take off in stock PX4 and hover;
2. engage Structure Hold (DO_SET_MODE auto/external1, the first external mode to register)
   and hold ``--hold-s``: drift from the engage point under ``--drift-tol``;
3. disengage to Position mode: accepted;
4. re-engage, trip the watchdog with the app's ``test_fault`` parameter: PX4 is in a stock
   mode within ``--handback-tol-s`` and still airborne, and a further engage is refused
   (trip latched until disarm);
5. land, disarm, take off again, engage, then stop the app's container: commander's own
   external-mode timeout must hand the vehicle to a stock mode with it still airborne (the
   case a crashed app or a lost companion link takes);
6. land.

Hold quality is not judged here beyond a loose drift bound: SIH's velocity estimate is
unusable on v1.17.0 (sim2real/px4/README.md), so the app flies with ``vel_src`` 1 as the
module's SIH test does. Gazebo judges hold quality.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "px4"))

from pymavlink.dialects.v20 import common as mavlink  # noqa: E402
from sitl_test import (  # noqa: E402
    MAIN_AUTO,
    MAIN_POSCTL,
    MAVLINK_URL,
    SUB_AUTO_LAND,
    SUB_EXTERNAL1,
    Vehicle,
    mode_name,
    record_hold,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="summary JSON path")
    p.add_argument("--container", default="structure-hold-app", help="the app's container name")
    p.add_argument("--image", default="structure-hold/app-sitl")
    p.add_argument("--app-arg", action="append", default=["-p", "vel_src:=1"],
                   help="extra ROS arguments for the app")
    p.add_argument("--altitude", type=float, default=2.5)
    p.add_argument("--hold-s", type=float, default=20.0)
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument("--drift-tol", type=float, default=0.5)
    p.add_argument("--handback-tol-s", type=float, default=1.5)
    p.add_argument("--app-loss-tol-s", type=float, default=5.0,
                   help="time for commander to leave the mode after the app stops")
    p.add_argument("--ready-timeout-s", type=float, default=120.0)
    return p.parse_args()


def in_hold(v: Vehicle) -> bool:
    return v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode == SUB_EXTERNAL1


def start_app(args: argparse.Namespace) -> None:
    subprocess.run(["docker", "rm", "-f", args.container], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--network", "host", "--name", args.container,
                    args.image, "app", *args.app_arg], check=True, capture_output=True)


def app_param(v: Vehicle, args: argparse.Namespace, name: str, value: str) -> None:
    """``ros2 param set`` on the app; MAVLink keeps pumping meanwhile (it takes seconds, and
    without the joystick stream PX4 would run its manual-control-loss failsafe)."""
    proc = subprocess.Popen(["docker", "exec", args.container, "bash", "-c",
                             ". /opt/ros/humble/setup.sh && . /ws/install/setup.sh && "
                             f"ros2 param set /structure_hold {name} {value}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic() + 30
    while proc.poll() is None and time.monotonic() < end:
        v.pump(0.1)
    if proc.poll() is None:
        proc.kill()
        raise TimeoutError(f"ros2 param set {name} timed out")


def app_logs(args: argparse.Namespace) -> str:
    r = subprocess.run(["docker", "logs", args.container], capture_output=True, text=True)
    return r.stdout + r.stderr


def handback_delay(args: argparse.Namespace) -> float:
    """Seconds from the app's trip line to its "stock PX4 in charge" line (ROS log stamps)."""
    t_trip = t_back = None
    for line in app_logs(args).splitlines():
        m = re.search(r"\[(\d+\.\d+)\]", line)
        if m is None:
            continue
        if "watchdog trip" in line:
            t_trip, t_back = float(m.group(1)), None
        elif "stock PX4 in charge" in line and t_trip is not None and t_back is None:
            t_back = float(m.group(1))
    return t_back - t_trip if t_trip is not None and t_back is not None else math.inf


def takeoff(v: Vehicle, args: argparse.Namespace) -> None:
    ground = v.tel.down
    nan = float("nan")
    assert v.command(mavlink.MAV_CMD_NAV_TAKEOFF, nan, 0, 0, nan, nan, nan, nan) \
        == mavlink.MAV_RESULT_ACCEPTED, "takeoff rejected"
    v.wait_for(lambda: ground - v.tel.down >= args.altitude - 0.3, 60, "takeoff altitude")
    v.pump(args.settle_s)


def land(v: Vehicle) -> None:
    v.set_mode(MAIN_AUTO, SUB_AUTO_LAND)
    v.wait_for(lambda: not v.tel.armed, 60, "landed and disarmed")


def engage(v: Vehicle, retry_s: float = 10.0) -> bool:
    """Ask for Structure Hold until commander takes it (the mode may still be registering)."""
    end = time.monotonic() + retry_s
    while time.monotonic() < end:
        if v.set_mode(MAIN_AUTO, SUB_EXTERNAL1) == mavlink.MAV_RESULT_ACCEPTED:
            try:
                v.wait_for(lambda: in_hold(v), 3, "Structure Hold in heartbeat")
                return True
            except TimeoutError:
                pass
        v.pump(0.5)
    return False


def fly(args: argparse.Namespace) -> dict:
    result: dict = {"checks": {}}
    checks = result["checks"]
    v = Vehicle(MAVLINK_URL, joystick=True)
    v.m.wait_heartbeat(timeout=args.ready_timeout_s)
    v.wait_for(lambda: not math.isnan(v.tel.down), 60, "local position")
    v.set_param("MIS_TAKEOFF_ALT", args.altitude)

    def try_arm() -> bool:
        return v.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1) == mavlink.MAV_RESULT_ACCEPTED

    # The app logs once commander has registered the mode.
    t0 = time.monotonic()
    while "registered with PX4" not in app_logs(args):
        if time.monotonic() - t0 > args.ready_timeout_s:
            raise TimeoutError("the app did not register Structure Hold")
        v.pump(1.0)
    result["register_s"] = round(time.monotonic() - t0, 1)
    v.wait_for(try_arm, args.ready_timeout_s, "arming")

    # 0. Armed on the ground: refused by the app's arming check.
    # Commander may ack the request before the mode's arming check answers; what counts is
    # that the mode is never entered.
    ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    entered = False
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        v.pump(0.1)
        entered = entered or in_hold(v)
    checks["ground_engage_refused"] = not entered
    result["ground_engage_ack"] = ack

    # 1-2. Take off, engage, hold.
    takeoff(v, args)
    checks["engage_accepted"] = engage(v)
    ref = (v.tel.north, v.tel.east, v.tel.down)
    drift: list[float] = []
    alt_err: list[float] = []
    record_hold(v, args.hold_s, ref, drift, alt_err)
    checks["still_engaged_after_hold"] = in_hold(v)
    result["hold"] = {"duration_s": args.hold_s, "xy_max_drift_m": round(max(drift), 3),
                      "alt_max_err_m": round(max(alt_err), 3), "samples": len(drift)}
    checks["position_held"] = max(drift) <= args.drift_tol
    checks["altitude_held"] = max(alt_err) <= args.drift_tol

    # 3. Disengage to Position mode.
    ack = v.set_mode(MAIN_POSCTL, 0)
    checks["disengage_accepted"] = ack == mavlink.MAV_RESULT_ACCEPTED
    v.wait_for(lambda: v.tel.main_mode == MAIN_POSCTL, 5, "Position mode")
    v.pump(args.settle_s)

    # 4. Re-engage and trip the watchdog.
    checks["reengage_accepted"] = engage(v)
    v.pump(3.0)
    t_trip = time.monotonic()
    app_param(v, args, "test_fault", "1")
    # The handback is timed from the trip the app logs, not from the slow parameter call.
    dt = math.inf
    if in_hold(v):
        try:
            v.wait_for(lambda: not in_hold(v), args.handback_tol_s + 5, "leave Structure Hold")
        except TimeoutError:
            pass
    if not in_hold(v):
        dt = handback_delay(args)
    result["handback_s"] = round(dt, 2)
    result["mode_after_trip"] = mode_name(v)
    checks["handback_in_time"] = dt <= args.handback_tol_s
    checks["stock_mode_after_trip"] = v.tel.main_mode == MAIN_POSCTL
    v.pump(2.0)
    checks["airborne_after_trip"] = not v.tel.landed and v.tel.armed
    checks["trip_logged"] = "watchdog trip: test_fault" in app_logs(args)
    v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    entered = False
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        v.pump(0.1)
        entered = entered or in_hold(v)
    checks["latched_after_trip"] = not entered
    result["trip_to_check_s"] = round(time.monotonic() - t_trip, 1)

    # 5. Land (clears the latch on disarm), fly again, engage, stop the app.
    land(v)
    v.wait_for(try_arm, 30, "re-arm")
    takeoff(v, args)
    checks["engage_after_disarm"] = engage(v)
    v.pump(3.0)
    subprocess.run(["docker", "kill", args.container], capture_output=True)
    t_kill = time.monotonic()
    try:
        dt = v.wait_for(lambda: not in_hold(v), args.app_loss_tol_s + 10, "leave the mode")
    except TimeoutError:
        dt = math.inf
    result["app_loss_handback_s"] = round(dt, 2)
    result["mode_after_app_loss"] = mode_name(v)
    checks["app_loss_handback"] = dt <= args.app_loss_tol_s
    v.pump(3.0)
    checks["airborne_after_app_loss"] = not v.tel.landed and v.tel.armed
    result["app_loss_check_s"] = round(time.monotonic() - t_kill, 1)

    # 6. Land.
    land(v)
    result["passed"] = all(checks.values())
    return result


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        start_app(args)
        result = fly(args)
    except Exception as e:  # noqa: BLE001 - every failure goes into the summary
        result = {"error": f"{type(e).__name__}: {e}", "passed": False}
    finally:
        logs = app_logs(args)
        (args.out.parent / "app.log").write_text(logs)
        subprocess.run(["docker", "rm", "-f", args.container], capture_output=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
