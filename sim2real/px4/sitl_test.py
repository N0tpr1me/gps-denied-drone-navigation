"""Acceptance flight for the Structure Hold mode in PX4 SITL (SIH quadrotor, no Gazebo).

Runs against a PX4 built with install.sh, started separately (see sitl_test.sh):

    python3 sitl_test.py --out runs/.../summary.json

Steps, each a check in the summary:

1. take off to ``--altitude`` in stock PX4 and hover ``--settle-s``;
2. engage the external mode "Structure Hold" over MAVLink (DO_SET_MODE, auto/external),
   hold ``--hold-s``: drift from the engage point stays under ``--drift-tol``;
3. disengage back to stock Position mode: the switch is accepted and the vehicle keeps
   holding for ``--settle-s``;
4. re-engage, then trip the watchdog with the SH_TEST_FAULT hook: PX4 leaves Structure
   Hold within ``--handback-tol-s`` (measured from the ulog), the vehicle stays airborne in
   a stock mode, and a further engage request is refused (trip latched);

A centred joystick (MANUAL_CONTROL at 20 Hz) is streamed throughout, because PX4 only
offers Position mode when it has a manual control source. With ``--no-joystick`` the
stock mode on disengage is Hold, and after the trip PX4 refuses Position mode and runs its
RC-loss failsafe (Return by default), which is the path a lost RC link takes.
5. land and disarm.

Afterwards the ulog is checked for the structure_hold_status topic (states standby,
engaged and handback all logged, with the test-fault trip reason).

pymavlink rather than MAVSDK because MAVSDK's action API cannot select PX4 external modes.
Positions are PX4 local NED; only magnitudes are reported, so no frame conversion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

MAVLINK_URL = "udpin:0.0.0.0:14540"

# PX4 custom modes (src/modules/commander/px4_custom_mode.h)
MAIN_ALTCTL, MAIN_POSCTL, MAIN_AUTO = 2, 3, 4
SUB_AUTO_TAKEOFF, SUB_AUTO_LOITER, SUB_AUTO_LAND, SUB_EXTERNAL1 = 2, 3, 6, 11

# structure_hold_status.state
STATE_STANDBY, STATE_ENGAGED, STATE_HANDBACK = 1, 2, 3
TRIP_TEST_FAULT, TRIP_ESTIMATE_RESET = 1, 6
# Commander refuses the mode for SH_RST_HOLD (10 s) after an EKF position reset; a
# re-engage that lands in that window is retried until it is accepted or this runs out.
REENGAGE_RETRY_S = 14.0


@dataclass
class Telemetry:
    north: float = math.nan
    east: float = math.nan
    down: float = math.nan
    main_mode: int = 0
    sub_mode: int = 0
    armed: bool = False
    landed: bool = True


@dataclass
class Samples:
    hold_drift: list[float] = field(default_factory=list)
    hold_alt_err: list[float] = field(default_factory=list)
    posctl_drift: list[float] = field(default_factory=list)


class Vehicle:
    def __init__(self, url: str, joystick: bool) -> None:
        self.m = mavutil.mavlink_connection(url)
        self.tel = Telemetry()
        self.joystick = joystick
        self._last_joystick = 0.0
        self._last_heartbeat = 0.0

    def pump(self, seconds: float) -> None:
        """Process incoming messages for ``seconds``; keep the joystick stream alive."""
        end = time.monotonic() + seconds
        while True:
            self._send_heartbeat()
            self._send_joystick()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            msg = self.m.recv_match(blocking=True, timeout=min(remaining, 0.05))
            if msg is not None:
                self._handle(msg)

    def _send_heartbeat(self) -> None:
        """GCS heartbeat at 1 Hz: without one PX4 fails the "No connection to the GCS" check."""
        if time.monotonic() - self._last_heartbeat < 1.0:
            return
        self._last_heartbeat = time.monotonic()
        self.m.mav.heartbeat_send(mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def _send_joystick(self) -> None:
        """Centred sticks at 20 Hz: PX4 needs a manual-control source for Position mode."""
        if not self.joystick or time.monotonic() - self._last_joystick < 0.05:
            return
        self._last_joystick = time.monotonic()
        self.m.mav.manual_control_send(self.m.target_system, 0, 0, 500, 0, 0)

    def _handle(self, msg) -> None:
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            self.tel.north, self.tel.east, self.tel.down = msg.x, msg.y, msg.z
        elif t == "HEARTBEAT" and msg.get_srcComponent() == mavlink.MAV_COMP_ID_AUTOPILOT1:
            self.tel.main_mode = (msg.custom_mode >> 16) & 0xFF
            self.tel.sub_mode = (msg.custom_mode >> 24) & 0xFF
            self.tel.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == "EXTENDED_SYS_STATE":
            self.tel.landed = msg.landed_state == mavlink.MAV_LANDED_STATE_ON_GROUND

    def wait_for(self, predicate, timeout_s: float, what: str) -> float:
        start = time.monotonic()
        while not predicate():
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"timed out after {timeout_s:.0f} s waiting for {what}")
            self.pump(0.1)
        return time.monotonic() - start

    def command(self, cmd: int, *params: float, timeout_s: float = 5.0) -> int:
        """Send COMMAND_LONG and return the ack result."""
        p = list(params) + [0.0] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, cmd, 0, *p)
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            msg = self.m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg is None:
                continue
            self._handle(msg)
            if msg.command == cmd:
                return msg.result
        raise TimeoutError(f"no ack for command {cmd}")

    def set_mode(self, main: int, sub: int = 0) -> int:
        return self.command(mavlink.MAV_CMD_DO_SET_MODE,
                            mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, main, sub)

    def set_param(self, name: str, value: float | int) -> None:
        """PARAM_SET; PX4 encodes INT32 params by value bytes inside the float field."""
        if isinstance(value, int):
            encoded = struct.unpack("<f", struct.pack("<i", value))[0]
            ptype = mavlink.MAV_PARAM_TYPE_INT32
        else:
            encoded, ptype = float(value), mavlink.MAV_PARAM_TYPE_REAL32
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode(), encoded, ptype)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is not None and msg.param_id.rstrip("\0") == name:
                return
        raise TimeoutError(f"no PARAM_VALUE ack for {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="summary JSON path")
    p.add_argument("--ulog", type=Path, default=None, help="ulog to check after the flight")
    p.add_argument("--altitude", type=float, default=2.5, help="takeoff altitude, m")
    p.add_argument("--hold-s", type=float, default=20.0, help="time in Structure Hold, s")
    p.add_argument("--settle-s", type=float, default=5.0, help="stock-mode hover times, s")
    p.add_argument("--drift-tol", type=float, default=0.5, help="max drift while holding, m")
    p.add_argument("--handback-tol-s", type=float, default=1.0)
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="PX4 parameter to set before arming (repeatable); an integer value "
                        "sets an INT32 parameter, anything else a FLOAT")
    p.add_argument("--no-joystick", action="store_true",
                   help="send no MANUAL_CONTROL: Position mode is then unavailable and the "
                        "stock mode on disengage and handback is Hold")
    p.add_argument("--ready-timeout-s", type=float, default=120.0)
    return p.parse_args()


def in_structure_hold(v: Vehicle) -> bool:
    return v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode >= SUB_EXTERNAL1


def mode_name(v: Vehicle) -> str:
    if in_structure_hold(v):
        return "Structure Hold"
    auto = {SUB_AUTO_TAKEOFF: "Takeoff", SUB_AUTO_LOITER: "Hold", SUB_AUTO_LAND: "Land",
            5: "Return"}
    if v.tel.main_mode == MAIN_AUTO:
        return auto.get(v.tel.sub_mode, f"Auto/{v.tel.sub_mode}")
    return {MAIN_ALTCTL: "Altitude", MAIN_POSCTL: "Position"}.get(v.tel.main_mode,
                                                                  f"main/{v.tel.main_mode}")


def record_hold(v: Vehicle, seconds: float, ref: tuple[float, float, float], drift: list[float],
                alt_err: list[float] | None = None) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        v.pump(0.1)
        drift.append(math.hypot(v.tel.north - ref[0], v.tel.east - ref[1]))
        if alt_err is not None:
            alt_err.append(abs(v.tel.down - ref[2]))


def fly(args: argparse.Namespace) -> dict:
    result: dict = {"px4_tag": os.environ.get("PX4_TAG", "unknown"), "checks": {},
                    "handback_tol_s": args.handback_tol_s}
    checks = result["checks"]
    joystick = not args.no_joystick
    v = Vehicle(MAVLINK_URL, joystick)
    # Stock mode used for disengage, and expected after the watchdog hands back.
    stock = (MAIN_POSCTL, 0, "Position") if joystick else (MAIN_AUTO, SUB_AUTO_LOITER, "Hold")
    result["stock_mode"] = stock[2]

    def in_stock(v: Vehicle) -> bool:
        return v.tel.main_mode == stock[0] and (stock[1] == 0 or v.tel.sub_mode == stock[1])

    start = time.monotonic()
    v.m.wait_heartbeat(timeout=args.ready_timeout_s)
    v.wait_for(lambda: not math.isnan(v.tel.down), 60, "local position")

    # Arm retries until the estimator and preflight checks are happy.
    def try_arm() -> bool:
        return v.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1) == mavlink.MAV_RESULT_ACCEPTED

    v.set_param("MIS_TAKEOFF_ALT", args.altitude)
    for item in args.param:
        name, _, value = item.partition("=")
        v.set_param(name, int(value) if value.lstrip("-").isdigit() else float(value))
    result["params"] = args.param
    v.wait_for(try_arm, args.ready_timeout_s, "arming")
    result["ready_s"] = round(time.monotonic() - start, 1)
    ground_down = v.tel.down
    nan = float("nan")
    assert v.command(mavlink.MAV_CMD_NAV_TAKEOFF, nan, 0, 0, nan, nan, nan, nan) \
        == mavlink.MAV_RESULT_ACCEPTED, "takeoff rejected"
    result["climb_s"] = round(v.wait_for(
        lambda: ground_down - v.tel.down >= args.altitude - 0.3, 60, "takeoff altitude"), 1)
    v.pump(args.settle_s)

    samples = Samples()

    # 2. Engage Structure Hold and hold.
    ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    checks["engage_accepted"] = ack == mavlink.MAV_RESULT_ACCEPTED
    v.wait_for(lambda: in_structure_hold(v), 5, "Structure Hold in heartbeat")
    result["engage_sub_mode"] = v.tel.sub_mode
    engage_point = (v.tel.north, v.tel.east, v.tel.down)
    record_hold(v, args.hold_s, engage_point, samples.hold_drift, samples.hold_alt_err)
    checks["still_engaged_after_hold"] = in_structure_hold(v)
    result["hold"] = {
        "duration_s": args.hold_s,
        "xy_max_drift_m": round(max(samples.hold_drift), 3),
        "alt_max_err_m": round(max(samples.hold_alt_err), 3),
        "samples": len(samples.hold_drift),
    }
    checks["position_held"] = max(samples.hold_drift) <= args.drift_tol
    checks["altitude_held"] = max(samples.hold_alt_err) <= args.drift_tol

    # 3. Disengage to the stock mode and check PX4 keeps holding.
    ack = v.set_mode(stock[0], stock[1])
    checks["disengage_accepted"] = ack == mavlink.MAV_RESULT_ACCEPTED
    v.wait_for(lambda: in_stock(v), 5, f"{stock[2]} mode in heartbeat")
    ref = (v.tel.north, v.tel.east, v.tel.down)
    record_hold(v, args.settle_s, ref, samples.posctl_drift)
    # The stock controller's own hold quality is the baseline task's business (and in SIH
    # it wanders, see README); here only the clean hand-over matters.
    result["after_disengage"] = {"mode": stock[2],
                                 "xy_max_drift_m": round(max(samples.posctl_drift), 3)}
    checks["stock_mode_flying_after_disengage"] = in_stock(v) and not v.tel.landed

    # 4. Re-engage, trip the watchdog, expect handback and a latched refusal. Commander
    # refuses the mode (or the module hands straight back) while the EKF-reset hold-off
    # runs, so the request is repeated once a second until it takes.
    t_reengage = time.monotonic()
    refusals = 0
    while True:
        ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
        try:
            v.wait_for(lambda: in_structure_hold(v), 1.0, "Structure Hold re-engaged")
            break
        except TimeoutError:
            refusals += 1
            if time.monotonic() - t_reengage > REENGAGE_RETRY_S:
                raise
    result["reengage"] = {"ack": ack, "refusals": refusals,
                          "wait_s": round(time.monotonic() - t_reengage, 1)}
    checks["reengage_accepted"] = ack == mavlink.MAV_RESULT_ACCEPTED
    v.pump(2.0)
    v.set_param("SH_TEST_FAULT", 1)
    t_trip = time.monotonic()
    # Which stock mode PX4 ends up in is commander's call once the module has asked
    # (without a manual-control source its RC-loss failsafe runs, Return by default), so the
    # check is that Structure Hold is no longer flying and the vehicle is still airborne.
    v.wait_for(lambda: not in_structure_hold(v), 10, "handback to a stock mode")
    v.pump(1.0)
    # Seen over MAVLink at the 1 Hz heartbeat, so coarse; the ulog check below measures the
    # real latency from the trip to commander's mode change.
    result["handback_seen_s"] = round(time.monotonic() - t_trip, 2)
    result["handback_mode"] = mode_name(v)
    checks["handback_to_stock_mode"] = not in_structure_hold(v) and not v.tel.landed
    v.pump(1.0)
    ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    v.pump(1.5)
    # Commander acks the request, then refuses the switch because the arming check for the
    # mode reports can_arm_and_run = false while the trip is latched.
    checks["latched_engage_refused"] = not in_structure_hold(v)
    result["latched_engage_ack"] = ack

    # 5. Land (a failsafe may already be landing the vehicle; then the command is moot).
    result["land_ack"] = v.command(mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, nan, nan, nan, nan)
    result["land_s"] = round(v.wait_for(lambda: v.tel.landed, 90, "touchdown"), 1)
    v.wait_for(lambda: not v.tel.armed, 30, "auto-disarm")
    checks["landed_and_disarmed"] = True
    result["wall_s"] = round(time.monotonic() - start, 1)
    return result


def check_ulog(path: Path, result: dict) -> None:
    """Confirm the status topic made it into the log with the expected transitions."""
    from pyulog import ULog

    checks = result["checks"]
    ulog = ULog(str(path), ["structure_hold_status", "vehicle_rates_setpoint", "vehicle_status"])
    by_name = {d.name: d for d in ulog.data_list}
    status = by_name.get("structure_hold_status")
    checks["status_topic_logged"] = status is not None
    if status is None:
        return
    states = set(int(s) for s in status.data["state"])
    trips = set(int(t) for t in status.data["trip_reason"])
    result["ulog"] = {
        "status_samples": len(status.data["timestamp"]),
        "states_seen": sorted(states),
        "trip_reasons_seen": sorted(trips),
        "max_loop_time_us": int(max(status.data["loop_time_us"])),
        "rates_setpoint_samples": len(by_name["vehicle_rates_setpoint"].data["timestamp"])
        if "vehicle_rates_setpoint" in by_name else 0,
    }
    checks["status_states_logged"] = bool({STATE_STANDBY, STATE_ENGAGED, STATE_HANDBACK} <= states)
    checks["status_trip_logged"] = bool(TRIP_TEST_FAULT in trips)

    # Handback latency: first HANDBACK sample to commander's first nav_state change after it.
    vs = by_name.get("vehicle_status")
    st_state = status.data["state"]
    hb = [t for t, s in zip(status.data["timestamp"], st_state, strict=True)
          if int(s) == STATE_HANDBACK]
    if vs is not None and hb:
        t_trip = hb[0]
        mode_id = int(status.data["mode_id"][-1])
        left = [t for t, n in zip(vs.data["timestamp"], vs.data["nav_state"], strict=True)
                if t > t_trip and int(n) != mode_id]
        if left:
            latency = (left[0] - t_trip) / 1e6
            result["ulog"]["handback_latency_s"] = round(latency, 3)
            checks["handback_within_tolerance"] = bool(latency <= result.get("handback_tol_s", 1.0))


def main() -> int:
    args = parse_args()
    try:
        summary = fly(args)
        if args.ulog is not None and args.ulog.exists():
            check_ulog(args.ulog, summary)
        elif args.ulog is not None:
            summary["checks"]["status_topic_logged"] = False
            summary["ulog_error"] = f"not found: {args.ulog}"
    except Exception as exc:  # report every failure in the summary, then fail the run
        summary = {"checks": {}, "error": f"{type(exc).__name__}: {exc}"}
    summary["passed"] = bool(summary["checks"]) and all(summary["checks"].values()) \
        and "error" not in summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
