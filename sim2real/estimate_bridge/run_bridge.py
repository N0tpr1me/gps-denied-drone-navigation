"""The bridge process: runs inside the SITL image next to PX4 and Gazebo.

    PYTHONPATH=/repo python3 -m sim2real.estimate_bridge.run_bridge \
        --profile /out/estimate_config.yaml --trace /out/estimate_trace.csv

Subscribes to Gazebo's ``/world/<world>/pose/info`` (50 Hz, the stream PX4's own gz bridge
turns into ``vehicle_local_position_groundtruth``), takes the vehicle model's pose, runs
it through :class:`bridge.EstimateBridge` and sends each published sample to PX4 as a
MAVLink ODOMETRY message (local NED position and velocity, no attitude, no covariance,
estimator type VIO) on PX4's second onboard link (local port 14030), leaving 14540 to the
flight script. PX4's mavlink receiver turns ODOMETRY into ``vehicle_visual_odometry``,
which EKF2 fuses with ``ekf2_ev.params``.

Global origin: with no GPS, PX4 has no global position, and its Hold mode (where the
commander sits when nothing else is selected, and where it goes after an auto takeoff)
requires one, so arming is refused and takeoff would fail over. As a VIO companion does,
the bridge sends SET_GPS_GLOBAL_ORIGIN once, which only tells EKF2 which latitude and
longitude its local origin corresponds to; every position still comes from the vision
stream (``--no-origin`` skips it, to see the refusal).

Timing: PX4 stamps each sample on arrival (no TIMESYNC reply is sent, so its timesync
offset stays zero and ``sync_stamp`` falls back to the receive time). The corruptor's
delay therefore reaches the EKF as real latency, compensated only by the constant
``EKF2_EV_DELAY``, as on a VIO without clock sync. Dropouts send nothing at all.

The trace CSV (``bridge.TRACE_COLUMNS``) records every corruptor step; ``verify_trace``
replays it on the host.

Engage: the bridge reads PX4's HEARTBEAT on its own link (asked for at 20 Hz with
SET_MESSAGE_INTERVAL, so an engage is seen within about 50 ms) and marks each sample with
whether Structure Hold (external mode 1 under the auto main mode, as sitl_test.py decodes
it) is flying. A profile with ``fault_anchor: engage`` keys its fault schedule to the first
such sample (``bridge.py``).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

from sim2real.estimate_bridge.bridge import TraceWriter, odometry_ned
from sim2real.estimate_bridge.config import load_profile
from sim2real.estimate_bridge.heartbeat import structure_hold_engaged

NAN = float("nan")
NAN21 = [NAN] * 21
VIO_COMPONENT_ID = 197  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
HEARTBEAT_INTERVAL_US = 50_000


class Bridge:
    def __init__(self, args: argparse.Namespace):
        self.profile = load_profile(args.profile)
        self.bridge = self.profile.bridge()
        self.engaged = False
        self.model = args.model
        self.trace = TraceWriter(args.trace)
        self.lock = threading.Lock()
        self.sent = self.dropped = self.received = 0
        self.last_log = time.monotonic()
        self.mav = mavutil.mavlink_connection(args.mavlink, source_system=args.sysid,
                                              source_component=VIO_COMPONENT_ID)
        print(f"bridge: profile {self.profile.name} seed {self.profile.seed} "
              f"{self.bridge.output_rate_hz:.0f} Hz, faults keyed to the "
              f"{self.profile.fault_anchor}; waiting for PX4 on {args.mavlink}", flush=True)
        self.mav.wait_heartbeat(timeout=args.heartbeat_timeout_s)
        print(f"bridge: PX4 heartbeat from system {self.mav.target_system}", flush=True)
        # Heartbeats at 20 Hz on this link, so the engage is seen within about 50 ms.
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavlink.MAVLINK_MSG_ID_HEARTBEAT, HEARTBEAT_INTERVAL_US, 0, 0, 0, 0, 0)
        if not args.no_origin:
            self.mav.mav.set_gps_global_origin_send(
                self.mav.target_system, int(round(args.origin_lat * 1e7)),
                int(round(args.origin_lon * 1e7)), int(round(args.origin_alt * 1e3)))
            print(f"bridge: global origin set to {args.origin_lat}, {args.origin_lon}, "
                  f"{args.origin_alt} m", flush=True)
        self.node = Node()
        topic = f"/world/{args.world}/pose/info"
        if not self.node.subscribe(Pose_V, topic, self.on_pose):
            raise RuntimeError(f"could not subscribe to {topic}")
        print(f"bridge: subscribed to {topic}, model {self.model}", flush=True)

    def on_pose(self, msg: Pose_V) -> None:
        for pose in msg.pose:
            if pose.name == self.model:
                break
        else:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        with self.lock:
            self.received += 1
            was_anchored = self.bridge.anchored
            try:
                sample = self.bridge.update(
                    t, (pose.position.x, pose.position.y, pose.position.z), self.engaged)
            except ValueError as exc:  # a repeated stamp: skip, do not kill the stream
                print(f"bridge: skipped sample: {exc}", flush=True)
                return
            if sample is None:
                return
            if self.bridge.anchored and not was_anchored:
                print(f"bridge: Structure Hold engaged at t={sample.t:.2f} s, fault clock "
                      f"anchored at {self.profile.fault_anchor_s:.1f} s (step {sample.step})",
                      flush=True)
            self.trace.write(sample)
            if sample.publish:
                self.send(sample)
                self.sent += 1
            else:
                self.dropped += 1
            now = time.monotonic()
            if now - self.last_log > 10:
                self.last_log = now
                print(f"bridge: t={sample.t:.1f} s received {self.received} sent {self.sent} "
                      f"withheld {self.dropped}", flush=True)

    def on_mavlink(self, msg) -> None:
        """PX4's traffic on the bridge link; only the autopilot heartbeat matters."""
        if msg.get_type() != "HEARTBEAT":
            return
        if msg.get_srcComponent() != mavlink.MAV_COMP_ID_AUTOPILOT1:
            return
        engaged = structure_hold_engaged(msg.custom_mode)
        with self.lock:
            if engaged != self.engaged:
                print(f"bridge: Structure Hold {'engaged' if engaged else 'not flying'}",
                      flush=True)
            self.engaged = engaged

    def send(self, sample) -> None:
        p, v = odometry_ned(sample)
        self.mav.mav.odometry_send(
            int(round(sample.t * 1e6)),
            mavlink.MAV_FRAME_LOCAL_NED, mavlink.MAV_FRAME_LOCAL_NED,
            float(p[0]), float(p[1]), float(p[2]), [NAN, NAN, NAN, NAN],
            float(v[0]), float(v[1]), float(v[2]), NAN, NAN, NAN,
            NAN21, NAN21, 0, mavlink.MAV_ESTIMATOR_TYPE_VIO, 100,
        )

    def close(self) -> None:
        with self.lock:
            self.trace.close()
        print(f"bridge: done, received {self.received} sent {self.sent} withheld "
              f"{self.dropped}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--world", default=os.environ.get("PX4_GZ_WORLD", "default"))
    ap.add_argument("--model", default="x500_0")
    ap.add_argument("--mavlink", default="udpin:0.0.0.0:14030")
    ap.add_argument("--sysid", type=int, default=1)
    ap.add_argument("--heartbeat-timeout-s", type=float, default=120.0)
    # The world's spherical coordinates (wall_slab.sdf copies PX4's default world), so
    # PX4's ground-truth topic, which it derives from them, shares the vision origin.
    ap.add_argument("--origin-lat", type=float, default=47.397971057728974)
    ap.add_argument("--origin-lon", type=float, default=8.546163739800146)
    ap.add_argument("--origin-alt", type=float, default=0.0)
    ap.add_argument("--no-origin", action="store_true", help="send no global origin")
    args = ap.parse_args(argv)

    bridge = Bridge(args)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        # Drain PX4's traffic so pymavlink's buffers never fill; only the heartbeat is read.
        msg = bridge.mav.recv_match(blocking=True, timeout=0.5)
        if msg is not None:
            bridge.on_mavlink(msg)
    bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
