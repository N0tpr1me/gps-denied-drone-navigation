"""Acceptance flight for the SITL image: take off, hover, land, all under MAVSDK.

Runs inside the container next to PX4 (see entrypoint.sh ``hover-test``):

    python3 hover_test.py --out /out/summary.json

Pass means: armed and climbed to the target altitude, held it for the hover time
(altitude within ``--alt-tol`` and horizontal drift within ``--drift-tol``), landed and
disarmed. Gazebo's real-time factor is sampled during the hover and recorded.

GPS stays on here (stock SITL); disabling it is the EKF2 glue task's job.
Positions are PX4's local NED; only magnitudes are reported, so no frame conversion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from mavsdk import System

MAVLINK_URL = "udpin://0.0.0.0:14540"
GZ_STATS_TOPIC = "/world/default/stats"
RTF_PATTERN = re.compile(r"real_time_factor:\s*([0-9.eE+-]+)")


@dataclass
class Telemetry:
    """Latest values from the MAVSDK streams, updated by background tasks."""

    north: float = math.nan
    east: float = math.nan
    down: float = math.nan
    in_air: bool = False
    armed: bool = False


@dataclass
class HoverSamples:
    altitude: list[float] = field(default_factory=list)
    drift: list[float] = field(default_factory=list)
    rtf: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="summary JSON path")
    p.add_argument("--altitude", type=float, default=2.5, help="takeoff altitude, m")
    p.add_argument("--hover-s", type=float, default=30.0, help="hover duration, s")
    p.add_argument("--alt-tol", type=float, default=0.5, help="max altitude error, m")
    p.add_argument("--drift-tol", type=float, default=1.0, help="max horizontal drift, m")
    p.add_argument("--ready-timeout-s", type=float, default=180.0)
    return p.parse_args()


async def follow_streams(drone: System, tel: Telemetry) -> None:
    async def position() -> None:
        async for pv in drone.telemetry.position_velocity_ned():
            pos = pv.position
            tel.north, tel.east, tel.down = pos.north_m, pos.east_m, pos.down_m

    async def in_air() -> None:
        async for value in drone.telemetry.in_air():
            tel.in_air = value

    async def armed() -> None:
        async for value in drone.telemetry.armed():
            tel.armed = value

    await asyncio.gather(position(), in_air(), armed())


async def wait_for(predicate, timeout_s: float, what: str, poll_s: float = 0.2) -> float:
    """Wait until ``predicate()`` is true; return the seconds it took."""
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"timed out after {timeout_s:.0f} s waiting for {what}")
        await asyncio.sleep(poll_s)
    return time.monotonic() - start


async def read_rtf() -> float | None:
    """One sample of Gazebo's real-time factor, or None if the topic gave nothing."""
    proc = await asyncio.create_subprocess_exec(
        "gz", "topic", "-e", "-t", GZ_STATS_TOPIC, "-n", "1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        return None
    match = RTF_PATTERN.search(out.decode(errors="replace"))
    return float(match.group(1)) if match else None


async def sample_rtf(samples: HoverSamples, stop: asyncio.Event, period_s: float = 3.0) -> None:
    while not stop.is_set():
        value = await read_rtf()
        if value is not None:
            samples.rtf.append(value)
        try:
            await asyncio.wait_for(stop.wait(), timeout=period_s)
        except asyncio.TimeoutError:
            pass


async def wait_until_ready(drone: System, timeout_s: float) -> float:
    start = time.monotonic()
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
            return time.monotonic() - start
        if time.monotonic() - start > timeout_s:
            raise TimeoutError("vehicle never became armable")
    raise RuntimeError("health stream ended")


async def hover(tel: Telemetry, args: argparse.Namespace, samples: HoverSamples) -> None:
    n0, e0 = tel.north, tel.east
    end = time.monotonic() + args.hover_s
    while time.monotonic() < end:
        samples.altitude.append(-tel.down)
        samples.drift.append(math.hypot(tel.north - n0, tel.east - e0))
        await asyncio.sleep(0.1)


async def fly(args: argparse.Namespace) -> dict:
    drone = System()
    await drone.connect(system_address=MAVLINK_URL)
    result: dict = {"px4_tag": os.environ.get("PX4_TAG", "unknown"), "passed": False}
    ready_s = await asyncio.wait_for(
        wait_until_ready(drone, args.ready_timeout_s), args.ready_timeout_s + 10
    )
    result["ready_s"] = round(ready_s, 1)

    tel = Telemetry()
    streams = asyncio.create_task(follow_streams(drone, tel))
    samples = HoverSamples()
    stop_rtf = asyncio.Event()
    rtf_task = asyncio.create_task(sample_rtf(samples, stop_rtf))
    wall_start = time.monotonic()
    try:
        await drone.action.set_takeoff_altitude(args.altitude)
        await drone.action.arm()
        await drone.action.takeoff()
        climb_s = await wait_for(
            lambda: -tel.down >= args.altitude - 0.2, 60, "takeoff altitude"
        )
        result["climb_s"] = round(climb_s, 1)

        await hover(tel, args, samples)

        await drone.action.land()
        result["land_s"] = round(await wait_for(lambda: not tel.in_air, 90, "touchdown"), 1)
        await wait_for(lambda: not tel.armed, 30, "auto-disarm")
    finally:
        stop_rtf.set()
        await rtf_task
        streams.cancel()

    result["wall_s"] = round(time.monotonic() - wall_start, 1)
    return summarise(result, samples, args)


def summarise(result: dict, samples: HoverSamples, args: argparse.Namespace) -> dict:
    alt_err = [abs(a - args.altitude) for a in samples.altitude]
    hover = {
        "duration_s": args.hover_s,
        "target_alt_m": args.altitude,
        "alt_mean_m": round(statistics.fmean(samples.altitude), 3),
        "alt_std_m": round(statistics.pstdev(samples.altitude), 3),
        "alt_max_err_m": round(max(alt_err), 3),
        "xy_max_drift_m": round(max(samples.drift), 3),
        "samples": len(samples.altitude),
    }
    rtf = {
        "mean": round(statistics.fmean(samples.rtf), 3) if samples.rtf else None,
        "min": round(min(samples.rtf), 3) if samples.rtf else None,
        "samples": len(samples.rtf),
    }
    checks = {
        "altitude_held": hover["alt_max_err_m"] <= args.alt_tol,
        "position_held": hover["xy_max_drift_m"] <= args.drift_tol,
        "rtf_recorded": rtf["samples"] > 0,
    }
    return {**result, "hover": hover, "real_time_factor": rtf, "checks": checks,
            "passed": all(checks.values())}


def main() -> int:
    args = parse_args()
    try:
        summary = asyncio.run(fly(args))
    except Exception as exc:  # report every failure in the summary, then fail the run
        summary = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    # os._exit, not sys.exit: MAVSDK's gRPC threads and mavsdk_server child otherwise keep
    # the interpreter alive after the flight is over.
    os._exit(main())
