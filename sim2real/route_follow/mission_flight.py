"""Stock PX4 Mission mode flying a route, under MAVSDK: the Route Follow baseline flight.

Runs inside the SITL image next to PX4, Gazebo and the estimate bridge (entrypoint
``with-sitl``; ``run_mission.py`` starts it):

    python3 /repo/sim2real/route_follow/mission_flight.py --mission /out/mission.json \\
        --out /out/summary.json

``mission.json`` (written by the host runner) holds the route's waypoints as latitude,
longitude and height above home, the mission speed, the acceptance radius, the take-off
height and the end hold. The flight: wait until armable on the local (vision) position,
upload the mission, arm, take off to the first waypoint's height, start the mission, wait
for its last item, hold ``end_hold_s`` in PX4's own end-of-mission loiter, land, disarm.
Every corner is a fly-through waypoint, as QGroundControl plans one by default, so PX4
turns through it on its own trajectory; the last waypoint stops.

The flight's own numbers here are PX4's view (times, mission progress); the scoring on the
TRUE position is the host's (``run_mission.py`` reads the ulog). Standard library and MAVSDK
only, like ``sim2real/docker/hover_test.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.mission import MissionItem, MissionPlan

MAVLINK_URL = "udpin://0.0.0.0:14540"


def mission_item(lat: float, lon: float, rel_alt: float, speed: float, fly_through: bool,
                 acceptance_m: float) -> MissionItem:
    """One waypoint; MAVSDK's MissionItem takes every field positionally."""
    nan = float("nan")
    return MissionItem(lat, lon, rel_alt, speed, fly_through, nan, nan,
                       MissionItem.CameraAction.NONE, nan, nan, acceptance_m, nan, nan,
                       MissionItem.VehicleAction.NONE)


async def wait_for(predicate, timeout_s: float, what: str, poll_s: float = 0.1) -> float:
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"timed out after {timeout_s:.0f} s waiting for {what}")
        await asyncio.sleep(poll_s)
    return time.monotonic() - start


class Telemetry:
    def __init__(self) -> None:
        self.down = math.nan
        self.in_air = False
        self.armed = False
        self.current = -1
        self.total = 0
        self.flight_mode = ""


async def follow(drone: System, tel: Telemetry) -> None:
    async def position() -> None:
        async for pv in drone.telemetry.position_velocity_ned():
            tel.down = pv.position.down_m

    async def in_air() -> None:
        async for v in drone.telemetry.in_air():
            tel.in_air = v

    async def armed() -> None:
        async for v in drone.telemetry.armed():
            tel.armed = v

    async def progress() -> None:
        async for p in drone.mission.mission_progress():
            tel.current, tel.total = p.current, p.total

    async def mode() -> None:
        async for m in drone.telemetry.flight_mode():
            tel.flight_mode = str(m)

    await asyncio.gather(position(), in_air(), armed(), progress(), mode())


async def wait_until_armable(drone: System, timeout_s: float) -> float:
    start = time.monotonic()
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        # GPS-free: MAVSDK's local-position flag does not see external vision, so only the
        # preflight checks (hover_test.py, wait_until_ready).
        if health.is_armable:
            return time.monotonic() - start
        if time.monotonic() - start > timeout_s:
            raise TimeoutError("vehicle never became armable")
    raise RuntimeError("health stream ended")


async def arm_when_allowed(drone: System, timeout_s: float = 30.0) -> int:
    """Arm, trying again while PX4 denies it; returns the number of tries. The health stream
    can say armable a moment before EKF2 calls the heading stable ("Preflight Fail: heading
    estimate not stable" in px4.log), and a single denied ``arm()`` left one flight in 32 on
    the floor. A drone that never arms is a flight not flown, not a crash."""
    start = time.monotonic()
    tries = 0
    while True:
        tries += 1
        try:
            await drone.action.arm()
            return tries
        except ActionError as exc:
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"vehicle never became armable: arming denied for "
                                   f"{timeout_s:.0f} s ({exc})") from exc
            await asyncio.sleep(1.0)


async def fly(args: argparse.Namespace) -> dict:
    m = json.loads(Path(args.mission).read_text())
    drone = System()
    await drone.connect(system_address=MAVLINK_URL)
    result: dict = {"px4_tag": os.environ.get("PX4_TAG", "unknown"), "passed": False,
                    "route": m.get("route"), "events": {}}
    t0 = time.monotonic()
    ev = result["events"]

    def mark(name: str) -> None:
        ev[name] = round(time.monotonic() - t0, 2)

    ready = await asyncio.wait_for(wait_until_armable(drone, args.ready_timeout_s),
                                   args.ready_timeout_s + 10)
    result["ready_s"] = round(ready, 1)
    mark("armable")
    tel = Telemetry()
    streams = asyncio.create_task(follow(drone, tel))
    try:
        wps = m["waypoints"]
        items = [mission_item(w["lat"], w["lon"], w["rel_alt_m"], m["speed_mps"],
                              i < len(wps) - 1, m["acceptance_m"]) for i, w in enumerate(wps)]
        await drone.mission.set_return_to_launch_after_mission(False)
        await drone.mission.upload_mission(MissionPlan(items))
        mark("uploaded")
        await wait_for(lambda: not math.isnan(tel.down), 10, "local position")
        ground_down = tel.down
        await drone.action.set_takeoff_altitude(m["takeoff_alt_m"])
        result["arm_tries"] = await arm_when_allowed(drone)
        mark("armed")
        await drone.action.takeoff()
        await wait_for(lambda: ground_down - tel.down >= m["takeoff_alt_m"] - 0.2, 60,
                       "take-off height")
        # A short settle so the mission starts from a hover, as the mock's target starts from
        # rest on the first waypoint.
        await asyncio.sleep(args.settle_s)
        mark("mission_start")
        # The progress stream can still show the last mission's end (current == total)
        # until PX4 starts this one, so first see it running, then see it reach the end.
        tel.current = -1
        await drone.mission.start_mission()
        await wait_for(lambda: 0 <= tel.current < tel.total, 30, "the mission to start")
        await wait_for(lambda: tel.total > 0 and tel.current >= tel.total, args.mission_timeout_s
                       or 3.0 * m["planned_s"] + 60.0, "the mission's last waypoint")
        mark("mission_done")
        result["mission_finished"] = bool(await drone.mission.is_mission_finished())
        await asyncio.sleep(m["end_hold_s"])
        mark("land")
        await drone.action.land()
        await wait_for(lambda: not tel.in_air, 120, "touchdown")
        await wait_for(lambda: not tel.armed, 30, "auto-disarm")
        mark("disarmed")
        result["passed"] = bool(result["mission_finished"])
    finally:
        streams.cancel()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ready-timeout-s", type=float, default=300.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--mission-timeout-s", type=float, default=None,
                    help="default: three times the target's planned time plus 60 s")
    args = ap.parse_args()
    try:
        summary = asyncio.run(fly(args))
    except Exception as exc:  # report every failure in the summary, then fail the run
        summary = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    # os._exit: MAVSDK's gRPC threads otherwise keep the interpreter alive (hover_test.py).
    os._exit(main())
