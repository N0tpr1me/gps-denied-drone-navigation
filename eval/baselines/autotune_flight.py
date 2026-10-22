"""In-container flight for PX4's attitude autotune: take off, hold, run it, land, read gains.

Runs beside SITL through the image's ``with-sitl`` entrypoint (``eval/baselines/autotune.py``
starts it), with ``MC_AT_EN 1`` and ``MC_AT_APPLY 1`` set at boot:

    python3 /repo/eval/baselines/autotune_flight.py --out /out/autotune.json

Sequence: arm and climb to ``--altitude`` in Takeoff, settle in Hold, set ``MC_AT_START 1``
(PX4's own trigger, as QGroundControl's Autotune button does), wait until the module resets
it (roll, pitch and yaw identified, gains verified, or the 20 s per-state timeout failed it),
land, wait for disarm (``MC_AT_APPLY 1`` writes the gains on disarm), then read the gain
parameters. Success means the module changed the gains; the ulog's
``autotune_attitude_control_status`` has the per-state detail.

Needs MAVSDK, so it only runs inside the SITL image; the helpers come from hover_test.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/sitl")
from hover_test import (  # noqa: E402  (the image's flight helpers)
    MAVLINK_URL,
    Telemetry,
    follow_streams,
    wait_for,
    wait_until_ready,
)
from mavsdk import System  # noqa: E402

GAINS = [f"MC_{axis}RATE_{g}" for axis in ("ROLL", "PITCH", "YAW") for g in ("P", "I", "D", "K")]
GAINS += ["MC_ROLL_P", "MC_PITCH_P", "MC_YAW_P"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--altitude", type=float, default=2.5)
    p.add_argument("--settle-s", type=float, default=8.0, help="Hold before the autotune starts")
    p.add_argument("--timeout-s", type=float, default=150.0, help="whole autotune sequence")
    p.add_argument("--ready-timeout-s", type=float, default=180.0)
    return p.parse_args()


async def read_gains(drone: System) -> dict[str, float]:
    return {name: round(await drone.param.get_param_float(name), 6) for name in GAINS}


async def fly(args: argparse.Namespace) -> dict:
    drone = System()
    await drone.connect(system_address=MAVLINK_URL)
    await asyncio.wait_for(wait_until_ready(drone, args.ready_timeout_s, True),
                           args.ready_timeout_s + 10)
    result: dict = {"px4_tag": os.environ.get("PX4_TAG", "unknown"), "passed": False}
    result["gains_before"] = await read_gains(drone)

    tel = Telemetry()
    streams = asyncio.create_task(follow_streams(drone, tel))
    try:
        await drone.action.set_takeoff_altitude(args.altitude)
        await wait_for(lambda: not math.isnan(tel.down), 10, "local position")
        ground_down = tel.down
        await drone.action.arm()
        await drone.action.takeoff()
        await wait_for(lambda: ground_down - tel.down >= args.altitude - 0.2, 60,
                       "takeoff altitude")
        await asyncio.sleep(args.settle_s)

        await drone.param.set_param_int("MC_AT_START", 1)
        start = time.monotonic()
        n0, e0 = tel.north, tel.east
        max_drift = 0.0
        while await drone.param.get_param_int("MC_AT_START") == 1:
            max_drift = max(max_drift, math.hypot(tel.north - n0, tel.east - e0))
            if time.monotonic() - start > args.timeout_s:
                raise TimeoutError("autotune never finished")
            await asyncio.sleep(0.5)
        result["autotune_s"] = round(time.monotonic() - start, 1)
        result["autotune_max_drift_m"] = round(max_drift, 3)

        await drone.action.land()
        await wait_for(lambda: not tel.in_air, 90, "touchdown")
        await wait_for(lambda: not tel.armed, 30, "auto-disarm")
        await asyncio.sleep(3.0)  # MC_AT_APPLY 1 saves the gains on disarm
    finally:
        streams.cancel()

    result["gains_after"] = await read_gains(drone)
    changed = {k: v for k, v in result["gains_after"].items()
               if not math.isclose(v, result["gains_before"][k], rel_tol=1e-4)}
    result["changed"] = changed
    result["passed"] = bool(changed)
    return result


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
    os._exit(main())  # as hover_test.py: MAVSDK threads otherwise keep the process alive
