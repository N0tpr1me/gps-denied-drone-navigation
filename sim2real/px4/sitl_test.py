"""Acceptance flight for the Structure Hold mode in PX4 SITL (SIH quadrotor, no Gazebo).

Runs against a PX4 built with install.sh, started separately (see sitl_test.sh):

    python3 sitl_test.py --out runs/.../summary.json

Steps, each a check in the summary:

0. on the ground (skip with ``--no-ground-checks``): Structure Hold cannot be armed into,
   and once armed a switch into it is refused (land detector) and handed back without
   lifting off;
   with ``--prearm`` first the pre-arm VIO check (SH_WD_GND_V): the script sends a VIO
   (MAVLink ODOMETRY) that sits still, then runs off at 0.5 m/s^2 while the vehicle sits
   disarmed. Arming must be allowed on the still VIO, refused in the stock mode once the
   VIO runs (with the critical message on the ground station), still refused 6 s after it
   stops, allowed again after 10 s still, and allowed 5 s after a running VIO goes silent;
1. take off to ``--altitude`` in stock PX4 and hover ``--settle-s``;
2. engage the external mode "Structure Hold" over MAVLink (DO_SET_MODE, auto/external),
   hold ``--hold-s``: drift from the engage point stays under ``--drift-tol``;
3. disengage back to stock Position mode: the switch is accepted and the vehicle keeps
   holding for ``--settle-s``; with ``--moving-engage`` the sticks then push the vehicle
   sideways and Structure Hold is engaged while it still moves (the re-engage mid-gust
   case): it must stop and hold without a trip, then hand back to Position mode;
4. re-engage, then trip the watchdog with the SH_TEST_FAULT hook: PX4 leaves Structure
   Hold within ``--handback-tol-s`` (measured from the ulog), the vehicle stays airborne in
   a stock mode (Structure Hover with ``--wall``), the ground station receives a critical
   STATUSTEXT naming the trip, and a further engage request is refused (trip latched);

The pilot's live view (docs/pilot-status.md) is checked along the way: an "engaged" line at
each engage, the periodic status line during the hold, and the live number array
(DEBUG_FLOAT_ARRAY "SH_STATUS") showing the hold while engaged and the trip after it.
With ``--pilot-warning`` the hold-error limit SH_WD_POS_ERR is lowered for a moment before
the trip so the live hold error sits at about 60 % of it: the ground station must get the
near-limit warning, and the limit goes back before it can trip.

A centred joystick (MANUAL_CONTROL at 20 Hz) is streamed throughout, because PX4 only
offers Position mode when it has a manual control source. With ``--no-joystick`` the
stock mode on disengage is Hold, and after the trip PX4 refuses Position mode and runs its
RC-loss failsafe (Return by default), which is the path a lost RC link takes.
5. land and disarm, through the module's guarded landing mode "Structure Land" (ext2) by
   default, or stock Land with ``--land stock``.

With ``--wall`` a forward rangefinder is simulated over MAVLink (DISTANCE_SENSOR at 20 Hz,
forward facing) against a virtual wall placed ``--wall-m`` ahead of the first engage point
along the engage heading. Mid-hold the wall jumps to ``--wall-close-m`` ahead of the hold
point, which puts the hold point inside SH_WALL_MIN: the wall guard must move it back and
the vehicle must end up at least SH_WALL_MIN (less ``--wall-tol``) from the wall
(WallGuard.hpp), and the ground station must get a "wall guard" warning. The ulog check
then expects a nonzero standoff_shift_m.

With ``--range-faults`` (needs ``--wall``) the rangefinder sends ``--range-faults-s`` of
broken readings before the wall jump: 0 cm, 5 cm (below the sensor's minimum), beyond its
maximum, 65535 cm, and a short 50 cm reading flagged invalid. Any of them taken at face
value would move the hold point back by 0.2 to 0.65 m; the vehicle must not back off.

With ``--wall`` the landing is checked against the wall too: at the land command the wall
is placed ``--land-wall-m`` ahead and then creeps ``--land-creep-m`` closer at ``--land-
creep-mps``. That is the Gazebo seed 3 case seen from the estimate's side: after a handback
on a jumped estimate the estimate recovers during the landing and the vehicle, holding it,
walks toward the wall (docs/baseline.md). The clearance until touchdown is reported for
Structure Land and stock Land, not checked: in SIH PX4's own xy loop lags too much in the
descent to judge it (Gazebo does, ``run_gazebo.py``). The same creep runs for ``--hover-s``
right after the trip, when the handback has gone to Structure Hover (the guarded hover,
SH_HB_GUARD), which must keep the same distance.

With ``--keep-out`` a 360 deg lidar is simulated over MAVLink (OBSTACLE_DISTANCE at 10 Hz,
72 bins of 5 deg, body frame, as a LightWare SF45 or a companion computer sends it) that
sees one thin pole. The keep-out is set to ``--keep-out-m`` through PX4's own CP_DIST
(SH_KO_DIST stays at its default -1, which follows it), or through SH_KO_DIST with
``--keep-out-param sh``. The pole first stands ``--obstacle-far-m`` to the right of the
hold point, beyond the keep-out: nothing may move. Then it jumps to ``--obstacle-m``, inside
it: the hold point must move away (KeepOut.hpp), the vehicle must end at least the
keep-out (less ``--wall-tol``) from the pole and still be in Structure Hold, and the ground
station must get a "keep-out" warning. The ulog check then expects a nonzero
keep_out_shift_m.

With ``--pass-len`` the first hold is a surface pass (PassPath.hpp, docs/surface-passes.md):
SH_PASS_LEN, SH_PASS_DIR (``--pass-up``), SH_PASS_SPD and SH_PASS_T0 are set before the
engage, and the hold point must wait the settle, slide that far along the wall (to the right
of the engage heading for a positive length) or up it, and hold where it stopped. Checked on
PX4's estimate: the ground station gets the "pass" and "pass done" lines, the vehicle ends
the length away along the pass direction, it follows the moving point (the mock's trapezoid
from the "pass" line on) within ``--drift-tol``, and its distance along the wall normal, the
standoff, never changes by more than that. The pass is switched off again for the later
engages. With ``--wall`` and ``--pass-wall-step-m`` the virtual wall has a step that much
closer over the second half of the pass (a buttress): the wall guard must move the moving
hold point back and the pass must still finish. With ``--keep-out`` and ``--pass-pole-m`` a
pole stands on the pass line that far from the engage point: the pass must stop short of it
("pass stopped") and the vehicle must stay the keep-out away. The ulog check then expects
the pass state and its offset in the status topic.

``--engage-at-boot-s`` holds the first engage until PX4's own clock (time since boot, the
simulation's time in Gazebo) reaches that value, so two controllers flown one after the other
meet the same wind and the same estimate faults at the same moment of their pass
(``pass_flight.py`` is the stock-PX4 side).

Afterwards the ulog is checked for the structure_hold_status topic (states standby,
engaged and handback all logged, with the test-fault trip reason).

pymavlink rather than MAVSDK because MAVSDK's action API cannot select PX4 external modes.
Positions are PX4 local NED; only magnitudes are reported, so no frame conversion.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

MAVLINK_URL = "udpin:0.0.0.0:14540"

# PX4 custom modes (src/modules/commander/px4_custom_mode.h)
MAIN_ALTCTL, MAIN_POSCTL, MAIN_AUTO = 2, 3, 4
SUB_AUTO_TAKEOFF, SUB_AUTO_LOITER, SUB_AUTO_LAND = 2, 3, 6
SUB_EXTERNAL1, SUB_EXTERNAL2, SUB_EXTERNAL3 = 11, 12, 13  # Structure Hold, Land, Hover

# structure_hold_status.state
STATE_STANDBY, STATE_ENGAGED, STATE_HANDBACK, STATE_LANDING, STATE_HOVER = 1, 2, 3, 4, 5
TRIP_TEST_FAULT, TRIP_ESTIMATE_RESET = 1, 6
# With SH_RST_HOLD > 0 commander refuses the mode for that long after an EKF position
# reset; a re-engage that lands in the window is retried until accepted or this runs out.
REENGAGE_RETRY_S = 14.0
# The landing's wall clearance is scored above this height (the ground stops the vehicle,
# not the wall).
LAND_AIRBORNE_M = 0.3
TRIP_LANDED = 7
# The live array (PilotStatus.hpp, LiveIndex): name, and the indices read here.
LIVE_NAME = "SH_STATUS"
LIVE_STATE, LIVE_HOLD_ERROR, LIVE_WD_SECONDS, LIVE_WD_USED = 0, 1, 6, 7
LIVE_TRIP, LIVE_TRIP_COUNT = 9, 10
# Broken forward-range readings for --range-faults: (centimetres, MAVLink signal_quality,
# where 0 = unknown and 1 = invalid). Min 10 cm, max 1200 cm as in the wall's own readings.
RANGE_FAULTS = [(0, 0), (5, 0), (2000, 0), (65535, 0), (50, 1)]
# The simulated lidar for --keep-out: 72 bins of 5 deg (MAVLink OBSTACLE_DISTANCE), 0.2 to
# 10 m; an empty bin reads max + 1 ("nothing there").
LIDAR_BINS = 72
LIDAR_MIN_CM = 20
LIDAR_MAX_CM = 1000
MAV_FRAME_BODY_FRD = 12
MAV_FRAME_LOCAL_FRD = 20
# --prearm: the scripted VIO's runaway on the ground, as Basalt's in drifting dust
# (docs/vio-render.md: up to 2.8 m/s before take-off).
VIO_RUN_ACCEL = 0.5
VIO_RUN_SPEED = 2.8
# The surface pass (PassPath.hpp): the module's acceleration (the mock's
# SurfacePass.accel_mps2) and structure_hold_status.pass_state.
PASS_ACCEL_MPS2 = 0.5
PASS_DONE, PASS_STOPPED = 3, 4


def pass_duration(length_m: float, speed_mps: float, accel: float = PASS_ACCEL_MPS2) -> float:
    """Seconds a pass takes from its start to its stop (envs/structure_hold/passes.py)."""
    d = abs(length_m)
    if d <= 0:
        return 0.0
    peak = min(speed_mps, math.sqrt(d * accel))
    return 2.0 * peak / accel + (d - peak * peak / accel) / peak


def pass_profile(t: float, length_m: float, speed_mps: float,
                 accel: float = PASS_ACCEL_MPS2) -> tuple[float, float]:
    """(distance flown, speed) ``t`` seconds after a pass starts: the mock's trapezoid, which
    the module's stepped path follows to within a centimetre (tests/test_pass_path.cpp)."""
    d = abs(length_m)
    if d <= 0 or t <= 0:
        return 0.0, 0.0
    peak = min(speed_mps, math.sqrt(d * accel))
    ta = peak / accel
    tc = (d - peak * ta) / peak
    dur = 2.0 * ta + tc
    if t >= dur:
        return d, 0.0
    if t < ta:
        return 0.5 * accel * t * t, accel * t
    if t < ta + tc:
        return 0.5 * accel * ta * ta + peak * (t - ta), peak
    return d - 0.5 * accel * (dur - t) ** 2, accel * (dur - t)


def pass_direction(yaw: float, up: bool, length_m: float) -> tuple[float, float, float]:
    """Unit direction of a pass in NED, as PassPath::engage: to the right of the heading
    ``yaw`` for a positive length, or up (NED z falls) with ``up``."""
    sign = -1.0 if length_m < 0 else 1.0
    if up:
        return (0.0, 0.0, -sign)
    return (-sign * math.sin(yaw), sign * math.cos(yaw), 0.0)


@dataclass
class Telemetry:
    north: float = math.nan
    east: float = math.nan
    down: float = math.nan
    main_mode: int = 0
    sub_mode: int = 0
    armed: bool = False
    landed: bool = True
    yaw: float = math.nan
    pitch: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    boot_s: float = 0.0  # PX4's time since boot, from LOCAL_POSITION_NED


@dataclass
class VirtualWall:
    """A flat wall ahead of ``origin`` along ``yaw`` (NED), seen by a forward rangefinder."""

    origin: tuple[float, float]
    yaw: float
    ahead_m: float
    max_m: float = 12.0
    creep_mps: float = 0.0       # the wall walks toward the origin from creep_t0 ...
    creep_max_m: float = 0.0     # ... by at most this much
    creep_t0: float = 0.0
    # A step in the wall (a buttress): from step_at_m along step_dir (a level unit vector
    # along the wall, from the origin) on, the face is step_m closer.
    step_dir: tuple[float, float] = (0.0, 0.0)
    step_at_m: float = 0.0
    step_m: float = 0.0

    def gap_m(self, tel: Telemetry) -> float:
        """Perpendicular distance vehicle -> wall (negative once through it)."""
        n = (math.cos(self.yaw), math.sin(self.yaw))
        rel = (tel.north - self.origin[0], tel.east - self.origin[1])
        moved = rel[0] * n[0] + rel[1] * n[1]
        crept = min(self.creep_mps * max(time.monotonic() - self.creep_t0, 0.0), self.creep_max_m)
        along = rel[0] * self.step_dir[0] + rel[1] * self.step_dir[1]
        step = self.step_m if self.step_m > 0 and along >= self.step_at_m else 0.0
        return self.ahead_m - crept - step - moved

    def range_m(self, tel: Telemetry) -> float | None:
        gap = self.gap_m(tel)
        n = (math.cos(self.yaw), math.sin(self.yaw))
        cos_p = math.cos(tel.pitch)
        bx_n = cos_p * math.cos(tel.yaw) * n[0] + cos_p * math.sin(tel.yaw) * n[1]
        if bx_n < 0.2 or gap <= 0:
            return None
        return min(gap / bx_n, self.max_m)


@dataclass
class VirtualPole:
    """A thin pole at a fixed NED point, seen by a 360 deg lidar."""

    north: float
    east: float

    def distance_m(self, tel: Telemetry) -> float:
        return math.hypot(self.north - tel.north, self.east - tel.east)

    def scan(self, tel: Telemetry) -> list[int]:
        """72 body-frame bins (centimetres), the pole in the bin of its bearing."""
        bins = [LIDAR_MAX_CM + 1] * LIDAR_BINS
        bearing = math.atan2(self.east - tel.east, self.north - tel.north) - tel.yaw
        i = round(math.degrees(bearing) / 5.0) % LIDAR_BINS
        cm = int(round(self.distance_m(tel) * 100))
        if LIDAR_MIN_CM < cm < LIDAR_MAX_CM:
            bins[i] = cm
        return bins


@dataclass
class Samples:
    hold_drift: list[float] = field(default_factory=list)
    hold_alt_err: list[float] = field(default_factory=list)
    hold_pass: list[tuple[float, float, float, float]] = field(default_factory=list)
    posctl_drift: list[float] = field(default_factory=list)


class Vehicle:
    def __init__(self, url: str, joystick: bool) -> None:
        self.m = mavutil.mavlink_connection(url)
        self.tel = Telemetry()
        self.joystick = joystick
        self._last_joystick = 0.0
        self._last_heartbeat = 0.0
        self._last_range = 0.0
        self._last_scan = 0.0
        self.pole: VirtualPole | None = None
        self.scans = 0
        self.wall: VirtualWall | None = None
        self.ranges: list[float] = []  # every range sent, for the wall checks
        self.stick_y = 0                # MANUAL_CONTROL y (roll), -1000..1000
        self.range_faults = False       # send RANGE_FAULTS instead of the wall's range
        # --prearm: a scripted VIO over MAVLink ODOMETRY at 30 Hz. None: none sent; else its
        # acceleration along its own x axis (0 with vio_speed 0: a still VIO).
        self.vio_accel: float | None = None
        self.vio_speed = 0.0
        self.vio_x = 0.0
        self._last_vio = 0.0
        self._fault_i = 0
        # Autopilot STATUSTEXT as the ground station sees it: (time, severity, text), long
        # texts reassembled from their chunks (MAVLink 2 ``id``).
        self.statustext: list[tuple[float, int, str]] = []
        self._text_index: dict[int, int] = {}
        # The module's live array as the ground station sees it: (time, values).
        self.live: list[tuple[float, list[float]]] = []
        # (our clock, PX4's time since boot) per position message: a pass is timed on PX4's
        # clock, which in Gazebo runs at the simulation's speed, not ours.
        self.clock: list[tuple[float, float]] = []

    def pump(self, seconds: float) -> None:
        """Process incoming messages for ``seconds``; keep the joystick stream alive."""
        end = time.monotonic() + seconds
        while True:
            self._send_heartbeat()
            self._send_joystick()
            self._send_range()
            self._send_scan()
            self._send_vio()
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
        self.m.mav.manual_control_send(self.m.target_system, 0, self.stick_y, 500, 0, 0)

    def _send_range(self) -> None:
        """Forward rangefinder at 20 Hz against the virtual wall, when there is one."""
        if self.wall is None or math.isnan(self.tel.yaw):
            return
        if time.monotonic() - self._last_range < 0.05:
            return
        self._last_range = time.monotonic()
        boot_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
        if self.range_faults:
            cm, quality = RANGE_FAULTS[self._fault_i % len(RANGE_FAULTS)]
            self._fault_i += 1
            self.m.mav.distance_sensor_send(boot_ms, 10, int(self.wall.max_m * 100), cm,
                                            mavlink.MAV_DISTANCE_SENSOR_LASER, 0,
                                            mavlink.MAV_SENSOR_ROTATION_NONE, 0,
                                            signal_quality=quality)
            return
        r = self.wall.range_m(self.tel)
        if r is None:
            return
        self.ranges.append(r)
        self.m.mav.distance_sensor_send(boot_ms, 10, int(self.wall.max_m * 100),
                                        int(round(r * 100)), mavlink.MAV_DISTANCE_SENSOR_LASER, 0,
                                        mavlink.MAV_SENSOR_ROTATION_NONE, 0)

    def _send_vio(self) -> None:
        """The scripted VIO: position and body-frame velocity, noisy as a real one at rest."""
        now = time.monotonic()
        if self.vio_accel is None:
            self._last_vio = 0.0
            return
        dt = now - self._last_vio
        if dt < 1.0 / 30.0:
            return
        if self._last_vio > 0.0:
            self.vio_speed = min(self.vio_speed + self.vio_accel * dt, VIO_RUN_SPEED)
            self.vio_x += self.vio_speed * dt
        self._last_vio = now
        nan21 = [float("nan")] * 21
        jitter = 0.02 * math.sin(40.0 * now)
        self.m.mav.odometry_send(int(now * 1e6), MAV_FRAME_LOCAL_FRD, MAV_FRAME_BODY_FRD,
                                 self.vio_x, 0.0, 0.0, [1.0, 0.0, 0.0, 0.0],
                                 self.vio_speed + jitter, -jitter, 0.0, 0.0, 0.0, 0.0,
                                 nan21, nan21, 0, mavlink.MAV_ESTIMATOR_TYPE_VIO, 100)

    def stop_vio(self, still: bool) -> None:
        """The VIO stops running: it reports still again, or goes silent."""
        self.vio_speed = 0.0
        self.vio_accel = 0.0 if still else None

    def _send_scan(self) -> None:
        """360 deg lidar at 10 Hz against the virtual pole, when there is one."""
        if self.pole is None or math.isnan(self.tel.yaw):
            return
        if time.monotonic() - self._last_scan < 0.1:
            return
        self._last_scan = time.monotonic()
        self.scans += 1
        self.m.mav.obstacle_distance_send(int(time.monotonic() * 1e6),
                                          mavlink.MAV_DISTANCE_SENSOR_LASER,
                                          self.pole.scan(self.tel), 0, LIDAR_MIN_CM,
                                          LIDAR_MAX_CM, 5.0, 0.0, MAV_FRAME_BODY_FRD)

    def _handle(self, msg) -> None:
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            self.tel.north, self.tel.east, self.tel.down = msg.x, msg.y, msg.z
            self.tel.vn, self.tel.ve = msg.vx, msg.vy
            self.tel.boot_s = msg.time_boot_ms * 1e-3
            self.clock.append((time.monotonic(), self.tel.boot_s))
        elif t == "HEARTBEAT" and msg.get_srcComponent() == mavlink.MAV_COMP_ID_AUTOPILOT1:
            self.tel.main_mode = (msg.custom_mode >> 16) & 0xFF
            self.tel.sub_mode = (msg.custom_mode >> 24) & 0xFF
            self.tel.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == "ATTITUDE":
            self.tel.yaw, self.tel.pitch = msg.yaw, msg.pitch
        elif t == "EXTENDED_SYS_STATE":
            self.tel.landed = msg.landed_state == mavlink.MAV_LANDED_STATE_ON_GROUND
        elif t == "STATUSTEXT" and msg.get_srcComponent() == mavlink.MAV_COMP_ID_AUTOPILOT1:
            chunk_id = getattr(msg, "id", 0)
            if chunk_id and chunk_id in self._text_index:
                i = self._text_index[chunk_id]
                when, severity, text = self.statustext[i]
                self.statustext[i] = (when, severity, text + msg.text)
            else:
                if chunk_id:
                    self._text_index[chunk_id] = len(self.statustext)
                self.statustext.append((time.monotonic(), msg.severity, msg.text))
        elif t == "DEBUG_FLOAT_ARRAY" and msg.name == LIVE_NAME:
            self.live.append((time.monotonic(), list(msg.data)))

    def boot_at(self, when: float) -> float:
        """PX4's time since boot at ``when`` on our clock (between position messages)."""
        if not self.clock:
            return math.nan
        i = bisect.bisect_left(self.clock, (when, -math.inf))
        if i <= 0:
            return self.clock[0][1] + (when - self.clock[0][0])
        if i >= len(self.clock):
            return self.clock[-1][1] + (when - self.clock[-1][0])
        (t0, b0), (t1, b1) = self.clock[i - 1], self.clock[i]
        return b0 + (b1 - b0) * (when - t0) / max(t1 - t0, 1e-9)

    def live_since(self, t0: float) -> list[list[float]]:
        """Live arrays (DEBUG_FLOAT_ARRAY "SH_STATUS") received since ``t0``."""
        return [d for when, d in self.live if when >= t0]

    def texts_since(self, t0: float, max_severity: int = mavlink.MAV_SEVERITY_DEBUG) -> list[str]:
        """STATUSTEXT received since ``t0`` at ``max_severity`` or more urgent."""
        return [text for when, sev, text in self.statustext if when >= t0 and sev <= max_severity]

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
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            self._handle(msg)  # texts and the live array keep flowing while we wait
            if msg.get_type() == "COMMAND_ACK" and msg.command == cmd:
                return msg.result
        raise TimeoutError(f"no ack for command {cmd}")

    def set_mode(self, main: int, sub: int = 0) -> int:
        return self.command(mavlink.MAV_CMD_DO_SET_MODE,
                            mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, main, sub)

    def param_type(self, name: str, timeout_s: float = 2.0) -> int | None:
        """The vehicle's MAV_PARAM_TYPE for ``name`` (PARAM_REQUEST_READ), None if no answer."""
        self.m.mav.param_request_read_send(self.m.target_system, self.m.target_component,
                                           name.encode(), -1)
        msg = self._wait_param_value(name, timeout_s)
        return None if msg is None else int(msg.param_type)

    def _wait_param_value(self, name: str, timeout_s: float):
        """The next PARAM_VALUE for ``name``, or None; every other message is handled."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            self._handle(msg)
            if msg.get_type() == "PARAM_VALUE" and msg.param_id.rstrip("\0") == name:
                return msg
        return None

    def get_param(self, name: str, timeout_s: float = 2.0) -> float | int | None:
        """The vehicle's value of ``name`` (an int for INT32 parameters), None if no answer."""
        self.m.mav.param_request_read_send(self.m.target_system, self.m.target_component,
                                           name.encode(), -1)
        msg = self._wait_param_value(name, timeout_s)
        if msg is None:
            return None
        if int(msg.param_type) == mavlink.MAV_PARAM_TYPE_INT32:
            return struct.unpack("<i", struct.pack("<f", msg.param_value))[0]
        return float(msg.param_value)

    def set_param(self, name: str, value: float | int) -> None:
        """PARAM_SET; PX4 encodes INT32 params by value bytes inside the float field. The
        type is the vehicle's own when it answers, so ``SH_WALL_MIN=0`` is sent as a float
        (PX4 rejects a set whose type does not match)."""
        ptype = self.param_type(name)
        if ptype == mavlink.MAV_PARAM_TYPE_REAL32:
            value = float(value)
        elif ptype == mavlink.MAV_PARAM_TYPE_INT32:
            value = int(round(value))
        if isinstance(value, int):
            encoded = struct.unpack("<f", struct.pack("<i", value))[0]
            ptype = mavlink.MAV_PARAM_TYPE_INT32
        else:
            encoded, ptype = float(value), mavlink.MAV_PARAM_TYPE_REAL32
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode(), encoded, ptype)
        if self._wait_param_value(name, 5.0) is None:
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
    p.add_argument("--land-trip", action="store_true",
                   help="trip the watchdog during Structure Land; expect a handback to stock Land")
    p.add_argument("--land-trip-s", type=float, default=1.5,
                   help="time in Structure Land before the --land-trip fault")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="PX4 parameter to set before arming (repeatable); an integer value "
                        "sets an INT32 parameter, anything else a FLOAT")
    p.add_argument("--no-joystick", action="store_true",
                   help="send no MANUAL_CONTROL: Position mode is then unavailable and the "
                        "stock mode on disengage and handback is Hold")
    p.add_argument("--ready-timeout-s", type=float, default=120.0)
    p.add_argument("--wall", action="store_true",
                   help="simulate a forward rangefinder against a virtual wall (wall guard check)")
    p.add_argument("--wall-m", type=float, default=1.0, help="wall distance at engage, m")
    p.add_argument("--wall-close-m", type=float, default=0.3,
                   help="wall distance from the hold point after the jump, m")
    p.add_argument("--wall-min", type=float, default=0.7, help="SH_WALL_MIN flown, m")
    p.add_argument("--wall-tol", type=float, default=0.15, help="allowed shortfall, m")
    p.add_argument("--wall-s", type=float, default=15.0, help="time after the wall jump, s")
    p.add_argument("--land", choices=("structure", "stock"), default="structure",
                   help="land through Structure Land (ext2) or stock Land")
    p.add_argument("--land-wall-m", type=float, default=0.8,
                   help="with --wall: wall distance at the land command, m")
    p.add_argument("--land-creep-m", type=float, default=0.6,
                   help="with --wall: how far the wall creeps in during the landing, m")
    p.add_argument("--land-creep-mps", type=float, default=0.1,
                   help="with --wall: creep speed, m/s (Gazebo seed 3 crept 0.05-0.07 m/s)")
    p.add_argument("--prearm", action="store_true",
                   help="before take-off, check the pre-arm VIO check (SH_WD_GND_V) with a "
                        "scripted VIO that runs off on the ground")
    p.add_argument("--no-ground-checks", action="store_true",
                   help="skip step 0 (arm-into and ground-engage refusals) before take-off")
    p.add_argument("--range-faults", action="store_true",
                   help="with --wall: broken range readings before the wall jump; no back-off")
    p.add_argument("--range-faults-s", type=float, default=6.0)
    p.add_argument("--range-fault-tol", type=float, default=0.3,
                   help="max back-off from the wall during the faults, m")
    p.add_argument("--pilot-warning", action="store_true",
                   help="lower SH_WD_POS_ERR before the trip and expect the near-limit warning")
    p.add_argument("--moving-engage", action="store_true",
                   help="engage Structure Hold while the vehicle still moves sideways")
    p.add_argument("--moving-stick", type=int, default=700,
                   help="MANUAL_CONTROL roll stick before the moving engage (-1000..1000)")
    p.add_argument("--moving-push-s", type=float, default=2.0)
    p.add_argument("--keep-out", action="store_true",
                   help="simulate a 360 deg lidar seeing a pole beside the hold (keep-out check)")
    p.add_argument("--keep-out-m", type=float, default=1.0, help="keep-out distance flown, m")
    p.add_argument("--keep-out-param", choices=("cp", "sh"), default="cp",
                   help="set it through PX4's CP_DIST (default) or SH_KO_DIST")
    p.add_argument("--obstacle-far-m", type=float, default=1.6,
                   help="pole distance right of the hold point first (beyond the keep-out), m")
    p.add_argument("--obstacle-m", type=float, default=0.5,
                   help="pole distance right of the hold point after the jump, m")
    p.add_argument("--keep-out-s", type=float, default=12.0, help="time after the jump, s")
    p.add_argument("--pass-len", type=float, default=0.0,
                   help="fly the first hold as a surface pass of this length, m (SH_PASS_LEN; "
                        "positive right or up, negative left or down)")
    p.add_argument("--pass-up", action="store_true",
                   help="the pass goes up the wall (SH_PASS_DIR 1)")
    p.add_argument("--pass-speed", type=float, default=0.5, help="SH_PASS_SPD, m/s")
    p.add_argument("--pass-t0", type=float, default=3.0, help="SH_PASS_T0, s")
    p.add_argument("--pass-wall-step-m", type=float, default=0.0,
                   help="with --wall: the wall is this much closer over the second half of "
                        "the pass")
    p.add_argument("--pass-pole-m", type=float, default=0.0,
                   help="with --keep-out: a pole on the pass line this far from the engage point")
    p.add_argument("--pass-slack-s", type=float, default=2.0,
                   help="how much later than its trapezoid the pass may finish, s (a stale "
                        "estimate holds it back: the Gazebo runner allows more)")
    p.add_argument("--engage-at-boot-s", type=float, default=0.0,
                   help="hold the first engage until PX4's time since boot reaches this, s")
    p.add_argument("--hover-s", type=float, default=8.0,
                   help="with --wall: time after the trip with the wall creeping in, s")
    return p.parse_args()


def in_structure_hold(v: Vehicle) -> bool:
    return v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode == SUB_EXTERNAL1


def in_structure_land(v: Vehicle) -> bool:
    return v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode == SUB_EXTERNAL2


def in_structure_hover(v: Vehicle) -> bool:
    return v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode == SUB_EXTERNAL3


def mode_name(v: Vehicle) -> str:
    if in_structure_hold(v):
        return "Structure Hold"
    if in_structure_land(v):
        return "Structure Land"
    if in_structure_hover(v):
        return "Structure Hover"
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
    if args.wall:
        v.set_param("SH_WALL_MIN", args.wall_min)
    if args.keep_out:
        v.set_param("CP_DIST" if args.keep_out_param == "cp" else "SH_KO_DIST", args.keep_out_m)
    if args.pass_len:
        v.set_param("SH_PASS_LEN", args.pass_len)
        v.set_param("SH_PASS_DIR", 1 if args.pass_up else 0)
        v.set_param("SH_PASS_SPD", args.pass_speed)
        v.set_param("SH_PASS_T0", args.pass_t0)
        # Long enough for the settle, the pass and a hold at its end.
        args.hold_s = max(args.hold_s,
                          args.pass_t0 + pass_duration(args.pass_len, args.pass_speed) + 6.0)
    result["params"] = args.param

    # 0. On the ground. First wait until the vehicle can arm at all (estimator, preflight),
    # then disarm again: from here a refused arm can only be the mode's doing.
    v.wait_for(try_arm, args.ready_timeout_s, "arming")
    result["ready_s"] = round(time.monotonic() - start, 1)
    if args.prearm:
        prearm_checks(v, result, try_arm)
    if not args.no_ground_checks:
        ground_checks(v, args, result, stock, try_arm)
    ground_down = v.tel.down
    nan = float("nan")
    assert v.command(mavlink.MAV_CMD_NAV_TAKEOFF, nan, 0, 0, nan, nan, nan, nan) \
        == mavlink.MAV_RESULT_ACCEPTED, "takeoff rejected"
    result["climb_s"] = round(v.wait_for(
        lambda: ground_down - v.tel.down >= args.altitude - 0.3, 60, "takeoff altitude"), 1)
    v.pump(args.settle_s)

    samples = Samples()

    # 2. Engage Structure Hold and hold.
    if args.engage_at_boot_s > 0:
        result["engage_late_s"] = round(max(v.tel.boot_s - args.engage_at_boot_s, 0.0), 2)
        v.wait_for(lambda: v.tel.boot_s >= args.engage_at_boot_s, 120, "the engage time")
    result["engage_boot_s"] = round(v.tel.boot_s, 2)
    t_engage = time.monotonic()
    # Where and which way the module latches its hold: read before the command, because the
    # heartbeat that confirms the mode comes up to a second later and the heading has moved
    # a few degrees by then (over a 5 m pass that is most of a metre of standoff).
    latched_point = (v.tel.north, v.tel.east, v.tel.down)
    latched_yaw = v.tel.yaw
    ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    checks["engage_accepted"] = ack == mavlink.MAV_RESULT_ACCEPTED
    v.wait_for(lambda: in_structure_hold(v), 5, "Structure Hold in heartbeat")
    result["engage_sub_mode"] = v.tel.sub_mode
    engage_point = (v.tel.north, v.tel.east, v.tel.down)
    if args.wall:
        v.wall = VirtualWall((engage_point[0], engage_point[1]), v.tel.yaw, args.wall_m)
    if args.pass_len:
        engage_point = latched_point
        pass_step(v, args, engage_point, latched_yaw, t_engage, result, samples)
    else:
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
    pilot_view_hold(v, args, t_engage, result)
    # A pass that already met the buttress or the pole has used the once-per-engage guard
    # messages these two steps look for.
    if args.wall and not (args.pass_len and args.pass_wall_step_m > 0):
        wall_guard_step(v, args, engage_point, result)
    if args.keep_out and not (args.pass_len and args.pass_pole_m > 0):
        keep_out_step(v, args, result)

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
    if args.moving_engage:
        moving_engage_step(v, args, result, stock)
        if not in_stock(v):
            v.set_mode(stock[0], stock[1])
        v.wait_for(lambda: in_stock(v), 5, f"{stock[2]} mode after the moving engage")
        v.pump(args.settle_s)

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
    if args.pilot_warning:
        pilot_warning_step(v, result)
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
    # The pilot is told why (docs/market-research.md, gap 2): a critical STATUSTEXT naming
    # the trip, which QGroundControl shows as a warning.
    trip_texts = [t for t in v.texts_since(t_trip, mavlink.MAV_SEVERITY_CRITICAL)
                  if "watchdog trip" in t]
    result["trip_warning"] = trip_texts[0] if trip_texts else None
    checks["trip_warning_sent"] = bool(trip_texts)
    # The live array shows the trip: its code, and the module no longer flying the hold.
    after = v.live_since(t_trip + 0.5)
    checks["live_shows_trip"] = bool(after) and int(after[-1][LIVE_TRIP]) == TRIP_TEST_FAULT \
        and int(after[-1][LIVE_STATE]) != STATE_ENGAGED and after[-1][LIVE_TRIP_COUNT] >= 1
    if args.wall:
        # Next to a measured wall the trip hands back to Structure Hover. The wall creeps in
        # as the estimate's recovery would walk stock Position mode toward it.
        checks["handback_to_structure_hover"] = in_structure_hover(v)
        hover_wall = VirtualWall((v.tel.north, v.tel.east), v.wall.yaw, args.land_wall_m,
                                 creep_mps=args.land_creep_mps, creep_max_m=args.land_creep_m,
                                 creep_t0=time.monotonic())
        v.wall = hover_wall
        gaps_hover: list[float] = []
        end = time.monotonic() + args.hover_s
        while time.monotonic() < end:
            v.pump(0.1)
            gaps_hover.append(hover_wall.gap_m(v.tel))
        result["hover_wall"] = {"start_m": round(gaps_hover[0], 3),
                                "min_m": round(min(gaps_hover), 3),
                                "end_m": round(gaps_hover[-1], 3), "creep_m": args.land_creep_m}
        checks["hover_kept_off_wall"] = min(gaps_hover) >= args.wall_min - args.wall_tol
        checks["still_in_structure_hover"] = in_structure_hover(v)
    v.pump(1.0)
    ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    v.pump(1.5)
    # Commander acks the request, then refuses the switch because the arming check for the
    # mode reports can_arm_and_run = false while the trip is latched.
    checks["latched_engage_refused"] = not in_structure_hold(v)
    result["latched_engage_ack"] = ack

    # 5. Land (a failsafe may already be landing the vehicle; then the command is moot).
    if args.wall:
        assert v.wall is not None
        here = (v.tel.north, v.tel.east)
        v.wall = VirtualWall(here, v.wall.yaw, args.land_wall_m, creep_mps=args.land_creep_mps,
                             creep_max_m=args.land_creep_m, creep_t0=time.monotonic())
    gaps: list[float] = []
    land_mode = "Land"
    if args.land == "structure":
        result["land_ack"] = v.set_mode(MAIN_AUTO, SUB_EXTERNAL2)
        try:
            v.wait_for(lambda: in_structure_land(v), 3, "Structure Land in heartbeat")
            land_mode = "Structure Land"
        except TimeoutError:
            result["land_ack"] = v.command(mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, nan, nan, nan, nan)
        checks["structure_land_engaged"] = land_mode == "Structure Land"
        if args.land_trip and land_mode == "Structure Land":
            # A trip in the landing hands back to stock Land, which finishes it.
            v.pump(args.land_trip_s)
            v.set_param("SH_TEST_FAULT", 1)
            try:
                result["land_trip_handback_s"] = round(v.wait_for(
                    lambda: v.tel.main_mode == MAIN_AUTO and v.tel.sub_mode == SUB_AUTO_LAND,
                    args.handback_tol_s + 1.0, "stock Land after the trip"), 2)
                checks["land_trip_to_stock_land"] = True
            except TimeoutError:
                checks["land_trip_to_stock_land"] = False
            result["land_trip_mode"] = mode_name(v)
    else:
        result["land_ack"] = v.command(mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, nan, nan, nan, nan)
    result["land_mode"] = land_mode

    def landed() -> bool:
        # Only while airborne: on the ground the vehicle cannot move, and the virtual wall
        # (a stand-in for the estimate recovering) would walk into it.
        if v.wall is not None and ground_down - v.tel.down > LAND_AIRBORNE_M:
            gaps.append(v.wall.gap_m(v.tel))
        return v.tel.landed

    result["land_s"] = round(v.wait_for(landed, 90, "touchdown"), 1)
    v.wait_for(lambda: not v.tel.armed, 30, "auto-disarm")
    checks["landed_and_disarmed"] = True
    if v.wall is not None and gaps:
        result["land_wall"] = {"start_m": round(gaps[0], 3), "min_m": round(min(gaps), 3),
                               "end_m": round(gaps[-1], 3), "creep_m": args.land_creep_m}
        # Reported, not checked: SIH's EKF velocity is unusable on v1.17.0 (README), and
        # PX4's own xy loop, which flies Structure Land, lags 0.3-0.6 m in the descent
        # here (minimum 0.28-0.67 m over runs). Gazebo judges the landing (run_gazebo).
    result["wall_s"] = round(time.monotonic() - start, 1)
    return result


def prearm_checks(v: Vehicle, result: dict, try_arm) -> None:
    """SH_WD_GND_V: no arming, in the stock mode either, on a VIO that runs on the ground."""
    checks = result["checks"]
    critical = mavlink.MAV_SEVERITY_CRITICAL
    arm_cmd = mavlink.MAV_CMD_COMPONENT_ARM_DISARM

    def disarm() -> None:
        v.pump(1.2)  # the armed flag comes with the 1 Hz heartbeat
        if v.tel.armed:
            v.command(arm_cmd, 0)
            v.wait_for(lambda: not v.tel.armed, 5, "disarm")

    def refused() -> bool:
        ack = v.command(arm_cmd, 1)
        v.pump(1.2)
        was_refused = ack != mavlink.MAV_RESULT_ACCEPTED and not v.tel.armed
        disarm()
        return was_refused

    def run_until_blocked() -> float:
        t0 = time.monotonic()
        v.vio_accel = VIO_RUN_ACCEL
        return v.wait_for(lambda: any("arming blocked" in t for t in v.texts_since(t0, critical)),
                          15, "the arming-blocked message")

    def wait_allowed(t0: float, timeout_s: float) -> float:
        v.wait_for(lambda: any("arming allowed" in t for t in v.texts_since(t0)),
                   timeout_s, "the arming-allowed message")
        return time.monotonic() - t0

    disarm()
    out: dict = {}
    # A still VIO: the land detector needs its second at rest, the check its slots.
    v.vio_accel = 0.0
    v.pump(5.0)
    checks["prearm_still_vio_arms"] = try_arm()
    disarm()
    v.pump(2.0)
    # It runs off: blocked within a few seconds, said to the pilot, arming refused.
    out["blocked_after_s"] = round(run_until_blocked(), 1)
    v.pump(0.5)  # commander's next arming-check poll (300 ms)
    checks["prearm_running_vio_refused"] = refused()
    # Still again: not trusted at once, trusted after ten seconds.
    t_still = time.monotonic()
    v.stop_vio(still=True)
    v.pump(6.0)
    checks["prearm_refused_6s_after_still"] = refused()
    out["allowed_after_still_s"] = round(wait_allowed(t_still, 20), 1)
    v.pump(0.5)
    checks["prearm_arms_after_still"] = try_arm() and 9.0 <= out["allowed_after_still_s"] <= 16.0
    disarm()
    v.pump(2.0)
    # It runs off again and then goes silent: EKF2 no longer fuses it, arming allowed.
    out["blocked_again_after_s"] = round(run_until_blocked(), 1)
    v.pump(0.5)
    refused_again = refused()
    t_silent = time.monotonic()
    v.stop_vio(still=False)
    out["allowed_after_silent_s"] = round(wait_allowed(t_silent, 15), 1)
    v.pump(0.5)
    checks["prearm_arms_after_silent"] = refused_again and try_arm() \
        and 4.0 <= out["allowed_after_silent_s"] <= 9.0
    checks["prearm_blocked_within_5s"] = max(out["blocked_after_s"],
                                             out["blocked_again_after_s"]) <= 5.0
    result["prearm"] = out
    # The flight that follows counts its fixes from the local position again, as it does
    # with no VIO: the scripted one is gone.
    v.set_param("SH_FIX_SRC", 2)


def ground_checks(v: Vehicle, args: argparse.Namespace, result: dict,
                  stock: tuple[int, int, str], try_arm) -> None:
    """Step 0: the mode cannot be armed into, and is refused on the ground once armed."""
    checks = result["checks"]

    def in_stock(v: Vehicle) -> bool:
        return v.tel.main_mode == stock[0] and (stock[1] == 0 or v.tel.sub_mode == stock[1])

    v.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
    v.wait_for(lambda: not v.tel.armed, 5, "disarm before the ground checks")
    # Disarmed, the mode reports itself unavailable: commander lets it be selected, but not
    # armed into (a hold point on the ground; the policy never learned a take-off).
    v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    v.pump(1.0)
    arm_ack = v.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
    v.pump(1.0)
    result["ground_arm_in_mode"] = {"selected": in_structure_hold(v), "arm_ack": arm_ack}
    checks["arm_into_mode_refused"] = arm_ack != mavlink.MAV_RESULT_ACCEPTED and not v.tel.armed
    if v.tel.armed:
        v.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
        v.wait_for(lambda: not v.tel.armed, 5, "disarm")
    v.set_mode(stock[0], stock[1])
    v.wait_for(try_arm, 10, "arming in the stock mode")
    # Armed on the ground, after commander's next arming-check poll (300 ms): commander takes
    # the switch, the module refuses it (land detector) and hands back at idle thrust; the
    # vehicle must stay on the ground (land detector, not the estimate's height, which
    # moves 0.1 to 0.2 m on the ground in SIH).
    v.pump(1.0)
    v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    stayed_landed = True
    end = time.monotonic() + 2.5
    while time.monotonic() < end:
        v.pump(0.1)
        stayed_landed = stayed_landed and v.tel.landed
    result["ground_engage"] = {"mode_after": mode_name(v), "stayed_landed": stayed_landed}
    checks["ground_engage_refused"] = not in_structure_hold(v) and stayed_landed and v.tel.armed
    if not in_stock(v):
        v.set_mode(stock[0], stock[1])
        v.pump(0.5)


def pilot_view_hold(v: Vehicle, args: argparse.Namespace, t_engage: float, result: dict) -> None:
    """The pilot's view of the first hold: engage line, status line, live array."""
    checks = result["checks"]
    texts = v.texts_since(t_engage, mavlink.MAV_SEVERITY_INFO)
    engaged = [t for t in texts if t.startswith("Structure Hold: engaged")]
    status = [t for t in texts if "off the hold point" in t and "watchdog" not in t]
    live = [d for d in v.live_since(t_engage) if int(d[LIVE_STATE]) == STATE_ENGAGED]
    result["pilot_view"] = {
        "engaged_line": engaged[0] if engaged else None,
        "status_lines": status,
        "live_arrays_engaged": len(live),
        "live_hold_error_max_m": round(max(d[LIVE_HOLD_ERROR] for d in live), 3) if live else None,
        "live_wd_used_max_pct": round(max(d[LIVE_WD_USED] for d in live), 1) if live else None,
        "live_wd_seconds_min": round(min(d[LIVE_WD_SECONDS] for d in live), 2) if live else None,
    }
    checks["engaged_line_sent"] = bool(engaged)
    if args.hold_s >= 12.0:  # SH_GCS_PER is 10 s by default
        checks["status_line_sent"] = bool(status)
    # The numbers are sane: a hold error inside the watchdog's, a margin inside 100 %, and
    # time left before the no-fix trip. The rate depends on the link (10 Hz onboard).
    checks["live_array_while_engaged"] = len(live) >= 5 and all(
        0.0 <= d[LIVE_HOLD_ERROR] < 2.0 and 0.0 <= d[LIVE_WD_USED] < 100.0
        and d[LIVE_WD_SECONDS] > 0.0 for d in live)


def pilot_warning_step(v: Vehicle, result: dict) -> None:
    """Lower SH_WD_POS_ERR so the hold error sits near 60 % of it; expect the warning."""
    checks = result["checks"]
    live = [d for d in v.live_since(time.monotonic() - 1.0) if int(d[LIVE_STATE]) == STATE_ENGAGED]
    err = max((d[LIVE_HOLD_ERROR] for d in live), default=math.nan)
    limit = max(0.2, err / 0.6) if math.isfinite(err) else 0.2  # 0.2 m is SH_WD_POS_ERR's minimum
    was = v.get_param("SH_WD_POS_ERR")
    t0 = time.monotonic()
    v.set_param("SH_WD_POS_ERR", limit)
    warned: list[str] = []
    end = t0 + 4.0  # the module reads parameters once a second
    while time.monotonic() < end and not warned:
        v.pump(0.1)
        warned = [t for t in v.texts_since(t0, mavlink.MAV_SEVERITY_WARNING)
                  if "watchdog trips at" in t]
    v.set_param("SH_WD_POS_ERR", 2.0 if was is None else was)  # as the flight had it
    v.pump(1.5)
    result["pilot_warning"] = {"hold_error_m": round(err, 3), "limit_m": round(limit, 3),
                               "text": warned[0] if warned else None}
    # Only meaningful when the hold error was large enough to reach the warning level.
    if err >= 0.1:
        checks["near_limit_warning_sent"] = bool(warned)


def moving_engage_step(v: Vehicle, args: argparse.Namespace, result: dict,
                       stock: tuple[int, int, str]) -> None:
    """Push sideways in the stock mode, engage while moving, check the hold catches it."""
    checks = result["checks"]
    v.stick_y = args.moving_stick
    v.pump(args.moving_push_s)
    speed = math.hypot(v.tel.vn, v.tel.ve)
    ack = v.set_mode(MAIN_AUTO, SUB_EXTERNAL1)
    v.stick_y = 0  # centred again: only the module is flying from here
    v.wait_for(lambda: in_structure_hold(v), 5, "Structure Hold engaged while moving")
    # The heartbeat (and so the engage point seen here) lags the module's own engage by up
    # to a second; the overshoot is measured from the first position seen in the mode.
    p0 = (v.tel.north, v.tel.east, v.tel.down)
    dist: list[float] = []
    end = time.monotonic() + args.hold_s
    while time.monotonic() < end:
        v.pump(0.1)
        dist.append(math.dist((v.tel.north, v.tel.east, v.tel.down), p0))
    tail = dist[-int(5.0 / 0.1):]
    result["moving_engage"] = {"ack": ack, "speed_at_request_mps": round(speed, 2),
                               "max_from_engage_m": round(max(dist), 3),
                               "last5s_max_m": round(max(tail), 3)}
    checks["moving_engage_no_trip"] = in_structure_hold(v)
    # Stops and holds: no wandering off at the end (the overshoot itself is reported).
    checks["moving_engage_holds"] = max(tail) - min(tail) <= args.drift_tol
    v.set_mode(stock[0], stock[1])


def pass_step(v: Vehicle, args: argparse.Namespace, engage: tuple[float, float, float],
              yaw: float, t_engage: float, result: dict, samples: Samples) -> None:
    """The first hold flown as a surface pass: record it against the moving hold point."""
    checks = result["checks"]
    length = abs(args.pass_len)
    d = pass_direction(yaw, args.pass_up, args.pass_len)
    n = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    duration = pass_duration(length, args.pass_speed)
    buttress = args.wall and args.pass_wall_step_m > 0 and not args.pass_up
    pole = args.keep_out and args.pass_pole_m > 0 and not args.pass_up
    if buttress:
        v.wall = VirtualWall((engage[0], engage[1]), yaw, args.wall_m, step_dir=(d[0], d[1]),
                             step_at_m=0.5 * length, step_m=args.pass_wall_step_m)
    if pole:
        v.pole = VirtualPole(engage[0] + args.pass_pole_m * d[0],
                             engage[1] + args.pass_pole_m * d[1])
    # (time, along the pass, along the wall normal, across both), from the engage point
    rows = samples.hold_pass
    gaps: list[float] = []
    pole_d: list[float] = []
    end = time.monotonic() + args.hold_s
    while time.monotonic() < end:
        v.pump(0.1)
        rel = (v.tel.north - engage[0], v.tel.east - engage[1], v.tel.down - engage[2])
        along = rel[0] * d[0] + rel[1] * d[1] + rel[2] * d[2]
        normal = rel[0] * n[0] + rel[1] * n[1]
        across = rel[2] if not args.pass_up else rel[0] * right[0] + rel[1] * right[1]
        rows.append((v.tel.boot_s, along, normal, across))
        if buttress:
            gaps.append(v.wall.gap_m(v.tel))
        if pole:
            pole_d.append(v.pole.distance_m(v.tel))

    def first(pattern: str) -> tuple[float, str] | None:
        for when, _severity, text in v.statustext:
            if when >= t_engage and re.search(pattern, text):
                return when, text
        return None

    started = first(r"Structure Hold: pass -?[\d.]+ m at")
    done = first(r"Structure Hold: pass done")
    stopped = first(r"Structure Hold: pass stopped")
    # On PX4's clock from here: the module's path runs on it.
    t_start = v.boot_at(started[0]) if started else math.inf
    boot_engage = v.boot_at(t_engage)
    # Where the moving point is: the trapezoid from the "pass" line on, cut where the
    # module said it stopped.
    stop_m = float(re.search(r"after (-?[\d.]+) m", stopped[1]).group(1)) if stopped else length
    track: list[float] = []
    for t, along, normal, across in rows:
        ref = min(pass_profile(t - t_start, length, args.pass_speed)[0], stop_m)
        lag = along - ref
        track.append(lag)
        if args.pass_up:
            samples.hold_drift.append(math.hypot(normal, across))
            samples.hold_alt_err.append(abs(lag))
        else:
            samples.hold_drift.append(math.hypot(lag, normal))
            samples.hold_alt_err.append(abs(across))
    tail = [r[1] for r in rows[-30:]] or [math.nan]
    result["pass"] = {
        "length_m": args.pass_len, "up": bool(args.pass_up), "speed_mps": args.pass_speed,
        "t0_s": args.pass_t0, "expected_s": round(duration, 2),
        "started_line": started[1] if started else None,
        "start_after_engage_s": round(t_start - boot_engage, 2) if started else None,
        "done_after_start_s": round(v.boot_at(done[0]) - t_start, 2) if done and started
        else None,
        "stopped_line": stopped[1] if stopped else None,
        "end_along_m": round(sum(tail) / len(tail), 3),
        "max_lag_m": round(max(abs(x) for x in track), 3) if track else None,
        "max_standoff_change_m": round(max(abs(r[2]) for r in rows), 3) if rows else None,
        "samples": len(rows),
    }
    checks["pass_started"] = started is not None
    checks["pass_standoff_held"] = bool(rows) and max(abs(r[2]) for r in rows) <= args.drift_tol
    if pole:
        v.pole = None
        want = args.pass_pole_m - args.keep_out_m
        last = pole_d[-30:] or [math.nan]
        result["pass"].update(pole_m=args.pass_pole_m, keep_out_m=args.keep_out_m,
                              stopped_at_m=stop_m, pole_min_m=round(min(pole_d), 3),
                              pole_last3s_mean_m=round(sum(last) / len(last), 3))
        checks["pass_stopped_by_keep_out"] = stopped is not None and "keep-out" in stopped[1]
        # The hold point stops when it comes inside the keep-out of the pole, and the
        # keep-out's own push leaves it a little short of that.
        checks["pass_stopped_short_of_pole"] = want - 0.5 <= stop_m <= want + 0.1
        checks["pass_kept_off_pole"] = sum(last) / len(last) >= args.keep_out_m - args.wall_tol
    else:
        checks["pass_done"] = done is not None and started is not None \
            and v.boot_at(done[0]) - t_start <= duration + args.pass_slack_s
        checks["pass_reached_end"] = abs(sum(tail) / len(tail) - length) <= args.drift_tol
    if buttress:
        last = gaps[-30:] or [math.nan]
        result["pass"].update(wall_m=args.wall_m, wall_step_m=args.pass_wall_step_m,
                              wall_min_gap_m=round(min(gaps), 3),
                              wall_last3s_mean_m=round(sum(last) / len(last), 3))
        checks["pass_wall_guard_acted"] = any(
            "wall guard moved" in t for t in v.texts_since(t_engage, mavlink.MAV_SEVERITY_WARNING))
        checks["pass_kept_off_wall"] = sum(last) / len(last) >= args.wall_min - args.wall_tol
    # The later engages of the sequence are plain holds again (read at the next engage).
    v.set_param("SH_PASS_LEN", 0.0)


def wall_guard_step(v: Vehicle, args: argparse.Namespace, hold: tuple[float, float, float],
                    result: dict) -> None:
    """Jump the virtual wall to inside SH_WALL_MIN of the hold point; expect a back-off."""
    checks = result["checks"]
    assert v.wall is not None
    before = list(v.ranges)
    checks["wall_range_streamed"] = len(before) > 0
    # At --wall-m (above SH_WALL_MIN) the guard leaves the hold point alone: the vehicle
    # sits about --wall-m from the wall.
    settled = before[len(before) // 2:] or [math.nan]
    if args.range_faults:
        # Back-off = movement away from the wall along its normal, from where the faults
        # start (the policy's own wander in SIH is a few cm to 0.2 m).
        n = (math.cos(v.wall.yaw), math.sin(v.wall.yaw))
        p0 = (v.tel.north, v.tel.east)
        backoff = 0.0
        v.range_faults = True
        end = time.monotonic() + args.range_faults_s
        while time.monotonic() < end:
            v.pump(0.1)
            moved = (v.tel.north - p0[0]) * n[0] + (v.tel.east - p0[1]) * n[1]
            backoff = max(backoff, -moved)
        v.range_faults = False
        result["range_faults"] = {"duration_s": args.range_faults_s, "sent": v._fault_i,
                                  "max_backoff_m": round(backoff, 3)}
        checks["range_faults_ignored"] = backoff <= args.range_fault_tol and in_structure_hold(v)
    v.wall = VirtualWall((hold[0], hold[1]), v.wall.yaw, args.wall_close_m)
    n0 = len(v.ranges)
    t_close = time.monotonic()
    v.pump(args.wall_s)
    after = v.ranges[n0:]
    tail = after[-int(3.0 / 0.05):] or [math.nan]
    result["wall"] = {
        "wall_min_m": args.wall_min,
        "before_mean_m": round(sum(settled) / len(settled), 3),
        "close_m": args.wall_close_m,
        "after_min_m": round(min(after), 3) if after else None,
        "after_last3s_mean_m": round(sum(tail) / len(tail), 3),
    }
    checks["wall_backed_off"] = sum(tail) / len(tail) >= args.wall_min - args.wall_tol
    checks["still_engaged_after_wall"] = in_structure_hold(v)
    checks["wall_guard_warning_sent"] = any(
        "wall guard" in t for t in v.texts_since(t_close, mavlink.MAV_SEVERITY_WARNING))


def keep_out_step(v: Vehicle, args: argparse.Namespace, result: dict) -> None:
    """A pole beside the hold point, first beyond the keep-out, then inside it."""
    checks = result["checks"]
    # The hold point is where the vehicle sits now; the pole goes to its right (FRD +y).
    here = (v.tel.north, v.tel.east)
    right = v.tel.yaw + math.pi / 2

    def pole_at(d: float) -> VirtualPole:
        return VirtualPole(here[0] + d * math.cos(right), here[1] + d * math.sin(right))

    v.pole = pole_at(args.obstacle_far_m)
    t_far = time.monotonic()
    p0 = (v.tel.north, v.tel.east)
    far_moved = 0.0
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        v.pump(0.1)
        far_moved = max(far_moved, math.hypot(v.tel.north - p0[0], v.tel.east - p0[1]))
    far_warned = any("keep-out" in t for t in v.texts_since(t_far, mavlink.MAV_SEVERITY_WARNING))

    v.pole = pole_at(args.obstacle_m)
    t_close = time.monotonic()
    dists: list[float] = []
    end = time.monotonic() + args.keep_out_s
    while time.monotonic() < end:
        v.pump(0.05)
        dists.append(v.pole.distance_m(v.tel))
    tail = dists[-int(3.0 / 0.05):] or [math.nan]
    result["keep_out"] = {
        "keep_out_m": args.keep_out_m,
        "set_through": "CP_DIST" if args.keep_out_param == "cp" else "SH_KO_DIST",
        "far_m": args.obstacle_far_m,
        "far_max_move_m": round(far_moved, 3),
        "close_m": args.obstacle_m,
        "after_min_m": round(min(dists), 3) if dists else None,
        "after_last3s_mean_m": round(sum(tail) / len(tail), 3),
        "scans_sent": v.scans,
    }
    # The lidar stays on for the rest of the flight, seeing nothing (the pole far out of its
    # range), as a real one would. Switched off, PX4's own collision prevention (CP_DIST)
    # finds its obstacle data stale and commands Hold 5 s into the next Position stretch,
    # the same 5 s the check after the disengage waits: it then passed or failed by 0.1 s
    # on every PX4 version (docs/px4-versions.md).
    v.pole = pole_at(1000.0)
    # Beyond the keep-out the pole changes nothing (the policy's own wander in SIH is a few
    # cm to 0.2 m, as in the range-fault check).
    checks["keep_out_ignores_far_obstacle"] = not far_warned and far_moved <= args.range_fault_tol
    checks["keep_out_backed_off"] = sum(tail) / len(tail) >= args.keep_out_m - args.wall_tol
    checks["still_engaged_after_keep_out"] = in_structure_hold(v)
    checks["keep_out_warning_sent"] = any(
        "keep-out" in t for t in v.texts_since(t_close, mavlink.MAV_SEVERITY_WARNING))


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
    expected = {STATE_STANDBY, STATE_ENGAGED, STATE_HANDBACK}
    if result.get("land_mode") == "Structure Land":
        expected.add(STATE_LANDING)
    if result.get("checks", {}).get("handback_to_structure_hover"):
        expected.add(STATE_HOVER)
    checks["status_states_logged"] = bool(expected <= states)
    checks["status_trip_logged"] = bool(TRIP_TEST_FAULT in trips)
    if "ground_engage" in result:
        # Commander takes the switch on the ground; the module's refusal must be logged.
        checks["ground_refusal_logged"] = bool(TRIP_LANDED in trips)
    if "wall" in result and "standoff_shift_m" in status.data:
        shift = float(max(status.data["standoff_shift_m"]))
        result["ulog"]["max_standoff_shift_m"] = round(shift, 3)
        checks["wall_guard_shift_logged"] = shift > 0.0
    if "keep_out" in result and "keep_out_shift_m" in status.data:
        shift = float(max(status.data["keep_out_shift_m"]))
        result["ulog"]["max_keep_out_shift_m"] = round(shift, 3)
        result["ulog"]["keep_out_m_logged"] = round(float(max(status.data["keep_out_m"])), 3)
        checks["keep_out_shift_logged"] = shift > 0.0
        checks["keep_out_distance_logged"] = abs(
            float(max(status.data["keep_out_m"])) - result["keep_out"]["keep_out_m"]) < 1e-3

    if "pass" in result and "pass_state" in status.data:
        pass_states = set(int(x) for x in status.data["pass_state"])
        offset = max(math.sqrt(float(a) ** 2 + float(b) ** 2 + float(c) ** 2) for a, b, c in zip(
            status.data["pass_offset_ned[0]"], status.data["pass_offset_ned[1]"],
            status.data["pass_offset_ned[2]"], strict=True))
        result["ulog"]["pass_states_seen"] = sorted(pass_states)
        result["ulog"]["max_pass_offset_m"] = round(offset, 3)
        if result["pass"].get("stopped_line"):
            checks["pass_stop_logged"] = PASS_STOPPED in pass_states \
                and offset < abs(result["pass"]["length_m"]) - 0.5
        else:
            checks["pass_logged"] = PASS_DONE in pass_states \
                and abs(offset - abs(result["pass"]["length_m"])) < 0.01

    # Handback latency: the first HANDBACK sample of the sequence's own trip (SH_TEST_FAULT)
    # to commander's first nav_state change after it. The ground checks' refused engage goes
    # through a handback too (trip 7, "landed"), earlier in the log: not this one.
    vs = by_name.get("vehicle_status")
    st_state = status.data["state"]
    hb = [t for t, s, r in zip(status.data["timestamp"], st_state, status.data["trip_reason"],
                               strict=True)
          if int(s) == STATE_HANDBACK and int(r) == TRIP_TEST_FAULT]
    if vs is not None and hb:
        t_trip = hb[0]
        mode_id = int(status.data["mode_id"][-1])
        left = [t for t, n in zip(vs.data["timestamp"], vs.data["nav_state"], strict=True)
                if t > t_trip and int(n) != mode_id]
        if left:
            latency = (left[0] - t_trip) / 1e6
            result["ulog"]["handback_latency_s"] = round(latency, 3)
            checks["handback_within_tolerance"] = bool(latency <= result.get("handback_tol_s", 1.0))


class RtfSampler:
    """Gazebo's real-time factor every few seconds through the flight, as hover_test.py
    records it: a flight that ran slow met its wind and its estimate faults late on our
    clock, and a benchmark batch is only read next to this number. Nothing is counted in the
    first ``settle_s`` nor before the first sample at 0.9 or above: Gazebo reports a factor
    near zero while it is still starting (a minute on a cold machine), long before the
    takeoff. Does nothing in SIH (no ``gz``, no ``PX4_GZ_WORLD``)."""

    PATTERN = re.compile(r"real_time_factor:\s*([0-9.eE+-]+)")

    def __init__(self, period_s: float = 3.0, settle_s: float = 20.0) -> None:
        world = os.environ.get("PX4_GZ_WORLD")
        self.topic = f"/world/{world}/stats" if world and shutil.which("gz") else None
        self.period_s = period_s
        self.settle_s = settle_s
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        began = time.monotonic()
        up = False  # Gazebo has reached real time once
        while not self._stop.is_set():
            try:
                out = subprocess.run(["gz", "topic", "-e", "-t", self.topic, "-n", "1"],
                                     capture_output=True, text=True, timeout=5.0).stdout
                found = self.PATTERN.search(out)
                value = float(found.group(1)) if found else None
                if value is not None and time.monotonic() - began >= self.settle_s:
                    up = up or value >= 0.9
                if value is not None and up:
                    self.values.append(value)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                pass
            self._stop.wait(self.period_s)

    def __enter__(self) -> RtfSampler:
        if self.topic is not None:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=6.0)

    def summary(self) -> dict | None:
        """None outside Gazebo; else mean, min and the number of samples."""
        if self.topic is None:
            return None
        v = self.values
        return {"mean": round(sum(v) / len(v), 3) if v else None,
                "min": round(min(v), 3) if v else None, "samples": len(v)}


def main() -> int:
    args = parse_args()
    rtf = RtfSampler()
    try:
        with rtf:
            summary = fly(args)
        if args.ulog is not None and args.ulog.exists():
            check_ulog(args.ulog, summary)
        elif args.ulog is not None:
            summary["checks"]["status_topic_logged"] = False
            summary["ulog_error"] = f"not found: {args.ulog}"
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
