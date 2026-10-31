"""Props-off bench test of the Structure Hold module on a flight controller (docs/bench-test.md).

Talks to PX4's shell over MAVLink (SERIAL_CONTROL, as PX4's Tools/mavlink_shell.py does),
so the same script checks a real flight controller on USB and a SITL instance:

    python3 sim2real/px4/bench_test.py --device /dev/ttyACM0 --out runs/bench/summary.json
    python3 sim2real/px4/bench_test.py --device udpin:0.0.0.0:14540 --fake-range 1.5 ...

Checks, each a boolean in the summary:

1. the firmware answers and reports its version (v1.17.0 expected, the pinned release);
2. the module is running and registered as a mode (``structure_hold status``: standby),
   flying the expected policy (``--policy``, default the released one);
3. policy inference fits the tick (``structure_hold bench``: worst step under 1 ms of the
   10 ms control interval);
4. every SH_ parameter is readable, and the ones given with ``--expect NAME=VALUE`` match;
5. a forward-facing rangefinder is publishing (``listener distance_sensor``) and the module
   sees it as fresh, when ``--require-range`` is given (the wall guard needs it).

``--fake-range D`` streams a forward DISTANCE_SENSOR at D metres over MAVLink, for a dry run
against SITL, which has no forward rangefinder. Nothing here arms the vehicle; the armed
props-off steps are manual and listed in docs/bench-test.md.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

RELEASED_POLICY = "structure_hold_v22"
PROMPT = re.compile(r"(nsh|pxh)> ?$")


class Shell:
    """PX4's MAVLink shell: write a command, read until the prompt comes back."""

    def __init__(self, m) -> None:
        self.m = m

    def _send(self, text: bytes) -> None:
        for i in range(0, len(text), 70):
            chunk = text[i:i + 70]
            self.m.mav.serial_control_send(
                mavlink.SERIAL_CONTROL_DEV_SHELL,
                mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE | mavlink.SERIAL_CONTROL_FLAG_RESPOND,
                0, 0, len(chunk), list(chunk) + [0] * (70 - len(chunk)))

    def run(self, command: str, timeout_s: float = 10.0) -> str:
        self._send(command.encode() + b"\n")
        out = ""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            msg = self.m.recv_match(type="SERIAL_CONTROL", blocking=True, timeout=0.2)
            if msg is None:
                # An empty poll keeps the reply coming (the shell answers on RESPOND).
                self._send(b"")
                continue
            if msg.count:
                out += bytes(msg.data[:msg.count]).decode(errors="replace")
                lines = out.rstrip("\r\n ").splitlines()
                if len(lines) > 1 and PROMPT.search(out.rstrip()):
                    break
        # Drop the echoed command and the trailing prompt; strip ANSI colours.
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)
        lines = [ln.rstrip("\r") for ln in out.splitlines()]
        if lines and command in lines[0]:
            lines = lines[1:]
        if lines and PROMPT.search(lines[-1]):
            lines = lines[:-1]
        return "\n".join(lines)


def fake_range(m, distance_m: float, stop: threading.Event) -> None:
    """Forward rangefinder at 20 Hz for SITL dry runs (MAV_SENSOR_ROTATION_NONE = forward)."""
    while not stop.is_set():
        boot_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
        m.mav.distance_sensor_send(boot_ms, 10, 1200, int(round(distance_m * 100)),
                                   mavlink.MAV_DISTANCE_SENSOR_LASER, 0,
                                   mavlink.MAV_SENSOR_ROTATION_NONE, 0)
        stop.wait(0.05)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="udpin:0.0.0.0:14540",
                   help="serial port (e.g. /dev/ttyACM0) or a pymavlink URL")
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--out", type=Path, required=True, help="summary JSON path")
    p.add_argument("--policy", default=RELEASED_POLICY, help="policy NAME expected in the build")
    p.add_argument("--bench-steps", type=int, default=2000)
    p.add_argument("--expect", action="append", default=[], metavar="NAME=VALUE",
                   help="parameter value the vehicle must have (repeatable)")
    p.add_argument("--require-range", action="store_true",
                   help="fail unless a forward rangefinder is publishing")
    p.add_argument("--fake-range", type=float, default=None, metavar="M",
                   help="stream a forward DISTANCE_SENSOR at M metres (SITL dry run)")
    p.add_argument("--timeout-s", type=float, default=60.0)
    return p.parse_args()


def bench(args: argparse.Namespace) -> dict:
    result: dict = {"device": args.device, "checks": {}, "shell": {}}
    checks, shell_out = result["checks"], result["shell"]
    m = mavutil.mavlink_connection(args.device, baud=args.baud, source_system=255)
    m.wait_heartbeat(timeout=args.timeout_s)
    stop = threading.Event()
    if args.fake_range is not None:
        threading.Thread(target=fake_range, args=(m, args.fake_range, stop), daemon=True).start()
    try:
        sh = Shell(m)
        sh.run("")  # wake the shell, swallow the banner

        ver = shell_out["ver"] = sh.run("ver all")
        tag = re.search(r"PX4 version: Release (\S+)", ver)
        result["px4_version"] = tag.group(1) if tag else None
        result["hw_arch"] = (re.search(r"HW arch: (\S+)", ver) or [None, None])[1]
        sitl = result["hw_arch"] == "PX4_SITL"
        checks["firmware_answers"] = result["hw_arch"] is not None
        checks["px4_v1_17_0"] = result["px4_version"] == "1.17.0"

        status = shell_out["status"] = sh.run("structure_hold status")
        checks["module_registered"] = "state: standby" in status
        checks["policy_expected"] = f"policy {args.policy}," in status

        b = shell_out["bench"] = sh.run(f"structure_hold bench {args.bench_steps}", timeout_s=60)
        mt = re.search(r"mean ([\d.]+) us, worst (\d+) us", b)
        if mt:
            result["inference_mean_us"] = float(mt.group(1))
            result["inference_worst_us"] = int(mt.group(2))
        # SITL's lockstep clock moves in simulator steps (4 ms in SIH), so its timings say
        # nothing; the number only counts on a real flight controller.
        result["inference_check"] = "informational (SITL clock)" if sitl else "scored"
        if not sitl:
            checks["inference_within_budget"] = "bench ok" in b

        params = shell_out["params"] = sh.run("param show SH_*")
        values = dict(re.findall(r"(SH_[A-Z0-9_]+) \[[^\]]*\]\s*:\s*(\S+)", params))
        result["params"] = values
        checks["params_readable"] = len(values) >= 20
        for item in args.expect:
            name, _, want = item.partition("=")
            got = values.get(name)
            checks[f"param_{name}"] = got is not None and abs(float(got) - float(want)) < 1e-4

        if args.require_range or args.fake_range is not None:
            time.sleep(0.5)
            listener = shell_out["distance_sensor"] = sh.run("listener distance_sensor 1")
            checks["range_publishing"] = "orientation: 0" in listener
            status = shell_out["status_with_range"] = sh.run("structure_hold status")
            checks["module_sees_range"] = "forward range fresh" in status
            mr = re.search(r"forward range fresh ([\d.]+) m", status)
            result["range_m"] = float(mr.group(1)) if mr else None
    finally:
        stop.set()
    return result


def main() -> int:
    args = parse_args()
    try:
        summary = bench(args)
    except Exception as exc:  # report every failure in the summary, then fail
        summary = {"checks": {}, "error": f"{type(exc).__name__}: {exc}"}
    summary["passed"] = bool(summary["checks"]) and all(summary["checks"].values()) \
        and "error" not in summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "shell"}, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
