"""The evaluation kit: fly the field faults with Structure Hold and with tuned PX4,
then print one comparison table (``docs/eval-kit.md``).

    ./evaluate.sh                                  # the one command (checks Docker, builds)
    python -m eval.eval_kit                        # the same, once the image and deps exist
    python -m eval.eval_kit --faults reset drift   # a subset
    python -m eval.eval_kit --dry-run              # the plan and the time it should take
    python -m eval.eval_kit --rescore runs/<kit folder>

A drone maker's engineer wants to see what the module does when the position estimate
fails before they give it a call. Every piece already exists (the SITL image, the field
fault profiles, ``eval.field_faults``); this flies them in one go, in parallel where the
machine has the cores, and puts the two controllers side by side on each fault:

- Structure Hold (released policy in the module, forward rangefinder on, as the partner
  pack requires), engaged 1 m from the wall, fault 20 s into the hold
  (``field_<fault>_at_engage``);
- tuned stock PX4 (``eval/baselines/px4_tuned_v2.params``) holding the same spot in Hold
  mode, the same fault 20 s into its hover (``field_<fault>_at_hover``).

Both on the same scenario, wind seed and estimate seed, scored on the TRUE position over
the 25 s after the fault (``eval.field_faults``). Writes ``runs/<time>_eval_kit_<commit>/``:
``config.yaml``, ``flights.json`` (one run folder per flight), ``comparison.json`` and
``comparison.md`` (the table printed at the end).

The verdict per fault is a fixed rule, so nobody has to trust a judgement call: fewer wall
contacts wins; then more clearance to the wall by more than ``MARGIN_M``; then a smaller
largest move by more than ``MARGIN_M``; otherwise "about the same". Landing is not judged:
stock PX4's failsafe landing on a lost estimate is its designed answer, and counts against
it only when it touches the wall on the way down, which the first rule catches.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.baselines.run_baseline import REPO_ROOT, cpu_slots, fly, git_commit
from eval.field_faults import CONTACT_M, score_run

FAULTS = ("reset", "restart", "silence", "freeze", "drift", "degrade")
FAULT_TEXT = {
    "reset": "VIO resets to its origin in the air",
    "restart": "VIO process restarts (3 s gap, then reset)",
    "silence": "position link lost for 8 s",
    "freeze": "estimate freezes for 5 s",
    "drift": "estimate drifts 0.6 m away from the wall",
    "degrade": "dust, fog or low light for 20 s (VIO and rangefinder degrade)",
}
SCENARIO = "wall1m_wind_4ms"  # 1 m from the wall, 4 m/s mean wind into it, gusts
SEED = 1
PARAMS = Path("eval/baselines/px4_tuned_v2.params")
DEFAULT_IMAGE = "structure-hold/px4-sitl:v1.17.0"
MARGIN_M = 0.05  # differences under 5 cm are "about the same"
# Wall-clock minutes one flight takes, measured on the cloud container (4 cores, one at a
# time); only used for the --dry-run estimate.
MINUTES_PER_FLIGHT = {"hold": 2.2, "estimate": 1.8}
CONTROLLERS = {"hold": "Structure Hold", "estimate": "tuned PX4"}
TRIP_TEXT = {
    "none": "no trip",
    "position_error": "tripped (position jumped) and handed back",
    "estimate_reset": "tripped (estimate reset) and handed back",
    "estimate_invalid": "tripped (estimate lost) and handed back",
    "estimate_stale": "tripped (estimate late) and handed back",
    "estimate_frozen": "tripped (estimate frozen) and handed back",
    "range_lost": "tripped (rangefinder lost) and handed back",
    "tilt": "tripped (tilt limit) and handed back",
}


@dataclass
class Flight:
    fault: str
    runner: str  # "hold" (Structure Hold) or "estimate" (tuned stock PX4)
    run_args: list[str] = field(default_factory=list)
    run: str | None = None  # the run folder, once flown


def plan(faults: list[str], image: str | None = None) -> list[Flight]:
    """Two flights per fault, Structure Hold first, in the order given."""
    img = ["--image", image] if image else []
    out = []
    for f in faults:
        if f not in FAULTS:
            raise ValueError(f"unknown fault {f!r}; choose from {FAULTS}")
        out.append(Flight(f, "hold", ["--profile", f"field_{f}_at_engage", "--rangefinder",
                                      *img]))
        out.append(Flight(f, "estimate", ["--profile", f"field_{f}_at_hover", *img]))
    return out


def estimate_minutes(flights: list[Flight], slots: int) -> float:
    """Rough wall-clock time: flights go ``slots`` at a time, each as long as the slowest
    runner in its round."""
    per = sorted((MINUTES_PER_FLIGHT[f.runner] for f in flights), reverse=True)
    return sum(per[i] for i in range(0, len(per), max(slots, 1)))


def image_present(image: str) -> bool:
    try:
        out = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    except FileNotFoundError:
        return False
    return out.returncode == 0


def judge(hold: dict[str, Any], stock: dict[str, Any]) -> tuple[str, str]:
    """(winner, why): winner is "Structure Hold", "tuned PX4" or "about the same"."""
    if "error" in hold or "error" in stock:
        return "not scored", "a flight could not be scored (see the rows)"
    if hold["contact"] != stock["contact"]:
        return ("tuned PX4", "Structure Hold touched the wall") if hold["contact"] else \
            ("Structure Hold", "tuned PX4 touched the wall")
    d_clear = hold["min_clearance_m"] - stock["min_clearance_m"]
    if abs(d_clear) > MARGIN_M:
        who = "Structure Hold" if d_clear > 0 else "tuned PX4"
        return who, f"stayed {abs(d_clear):.2f} m further from the wall"
    d_move = stock["max_horizontal_m"] - hold["max_horizontal_m"]
    if abs(d_move) > MARGIN_M:
        who = "Structure Hold" if d_move > 0 else "tuned PX4"
        return who, f"moved {abs(d_move):.2f} m less"
    return "about the same", f"within {MARGIN_M * 100:.0f} cm on wall distance and movement"


def what_happened(row: dict[str, Any]) -> str:
    """One plain sentence on what the vehicle did after the fault."""
    if "error" in row:
        return f"not scored: {row['error']}"
    parts = []
    if row["runner"] == "hold":
        trip = row.get("trip") or "none"
        parts.append(TRIP_TEXT.get(trip, f"tripped ({trip}) and handed back"))
    if row["contact"]:
        parts.append("touched the wall")
    if row["max_climb_m"] > 0.5:
        parts.append(f"climbed {row['max_climb_m']:.1f} m")
    if row["max_drop_m"] > 0.5:
        parts.append(f"dropped {row['max_drop_m']:.1f} m")
    parts.append("still flying at the end" if row["airborne_end"] else "on the ground at the end")
    return ", ".join(parts)


def pair_rows(flights: list[Flight], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per fault: both controllers' rows and the verdict."""
    by = {(f.fault, f.runner): r for f, r in zip(flights, rows, strict=True)}
    out = []
    for fault in dict.fromkeys(f.fault for f in flights):
        hold, stock = by.get((fault, "hold")), by.get((fault, "estimate"))
        if hold is None or stock is None:
            continue
        winner, why = judge(hold, stock)
        out.append({"fault": fault, "what": FAULT_TEXT[fault], "structure_hold": hold,
                    "tuned_px4": stock, "better": winner, "why": why})
    return out


def _m(row: dict[str, Any], key: str, nd: int = 2) -> str:
    v = row.get(key)
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def comparison_table(pairs: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        f"# Structure Hold vs tuned PX4 on {len(pairs)} position-estimate failures",
        "",
        f"Gazebo SITL, PX4 v1.17.0, x500, scenario `{meta['scenario']}` (1 m from a wall, "
        f"4 m/s mean wind into it, gusts), seed {meta['seed']}, commit {meta['commit']}. "
        "Each fault hits 20 s into the hold; scored on the true position over the 25 s "
        f"after it. Props touch the wall at {CONTACT_M:.2f} m.",
        "",
        "| Fault | Controller | Closest to wall, m | Largest move, m | Lowest height, m "
        "| What happened | Better |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in pairs:
        for i, key in enumerate(("structure_hold", "tuned_px4")):
            r = p[key]
            name = "Structure Hold" if key == "structure_hold" else "tuned PX4"
            first = p["what"] if i == 0 else ""
            verdict = f"**{p['better']}**: {p['why']}" if i == 0 else ""
            clear = _m(r, "min_clearance_m") + (" (contact)" if r.get("contact") else "")
            lines.append(f"| {first} | {name} | {clear} | {_m(r, 'max_horizontal_m')} "
                         f"| {_m(r, 'min_height_m')} | {what_happened(r)} | {verdict} |")
    tally = {w: sum(p["better"] == w for p in pairs)
             for w in ("Structure Hold", "tuned PX4", "about the same", "not scored")}
    contacts = {k: sum(bool(p[k].get("contact")) for p in pairs)
                for k in ("structure_hold", "tuned_px4")}
    lines += [
        "",
        f"Structure Hold better on {tally['Structure Hold']} of {len(pairs)} faults, tuned "
        f"PX4 on {tally['tuned PX4']}, about the same on {tally['about the same']}"
        + (f", {tally['not scored']} not scored" if tally["not scored"] else "") + ". "
        f"Wall contacts: Structure Hold {contacts['structure_hold']}, tuned PX4 "
        f"{contacts['tuned_px4']}.",
        "",
        "One flight per cell on one seed: a first look, not a benchmark. The verdict rule: "
        f"fewer wall contacts, then more wall clearance by more than {MARGIN_M * 100:.0f} cm, "
        f"then a smaller largest move by more than {MARGIN_M * 100:.0f} cm.",
    ]
    return "\n".join(lines) + "\n"


def fly_all(flights: list[Flight], params: Path, slots: list[str | None], stamp: str
            ) -> None:
    """Fly every flight, ``len(slots)`` at a time, each pinned to its own CPUs; fills in
    ``Flight.run``."""
    free: queue.Queue[tuple[int, str | None]] = queue.Queue()
    for i, cpuset in enumerate(slots):
        free.put((i, cpuset))

    def one(f: Flight) -> None:
        slot, cpuset = free.get()
        try:
            name = f"px4-sitl-kit-{stamp}-{slot}" if len(slots) > 1 else None
            folder = fly(SCENARIO, SEED, params, f.run_args, f.runner, cpuset, name)
        finally:
            free.put((slot, cpuset))
        f.run = str(folder) if folder else None
        print(f"{f.fault:8s} {CONTROLLERS[f.runner]:15s} "
              f"{'flown: ' + str(folder) if folder else 'FAILED (no run folder)'}", flush=True)

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for fut in [pool.submit(one, f) for f in flights]:
            fut.result()


def score_all(flights: list[Flight]) -> list[dict[str, Any]]:
    rows = []
    for f in flights:
        base = {"run": f.run, "runner": "hold" if f.runner == "hold" else "stock",
                "profile": f.run_args[1], "seed": SEED}
        if not f.run:
            rows.append({**base, "error": "the flight did not produce a run folder"})
            continue
        try:
            rows.append(score_run(Path(f.run)))
        except (OSError, ValueError, KeyError) as e:
            rows.append({**base, "error": str(e)})
    return rows


def write_outputs(out: Path, flights: list[Flight], meta: dict[str, Any]) -> str:
    rows = score_all(flights)
    pairs = pair_rows(flights, rows)
    text = comparison_table(pairs, meta)
    (out / "comparison.json").write_text(json.dumps({"meta": meta, "pairs": pairs}, indent=1))
    (out / "comparison.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--faults", nargs="+", default=list(FAULTS), choices=FAULTS)
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="the SITL image to fly")
    ap.add_argument("--jobs", default="auto",
                    help="flights at once (each pinned to its own CPUs), or 'auto' (default)")
    ap.add_argument("--cpus-per-job", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--rescore", type=Path, metavar="KIT_FOLDER",
                    help="score an earlier kit folder again without flying")
    args = ap.parse_args(argv)

    if args.rescore:
        meta = yaml.safe_load((args.rescore / "config.yaml").read_text())
        flights = [Flight(**f) for f in json.loads((args.rescore / "flights.json").read_text())]
        print(write_outputs(args.rescore, flights, meta))
        return 0

    jobs = args.jobs if args.jobs == "auto" else int(args.jobs)
    try:
        slots = cpu_slots(jobs, args.cpus_per_job)
    except ValueError as e:
        ap.error(str(e))
    image = None if args.image == DEFAULT_IMAGE else args.image
    flights = plan(args.faults, image)
    minutes = estimate_minutes(flights, len(slots))
    print(f"{len(flights)} flights ({len(args.faults)} faults x Structure Hold and tuned PX4), "
          f"{len(slots)} at a time: about {minutes:.0f} min", flush=True)
    if args.dry_run:
        for f in flights:
            print(f"  {f.fault:8s} {CONTROLLERS[f.runner]:15s} {' '.join(f.run_args)}")
        return 0
    if not image_present(args.image):
        print(f"SITL image {args.image} not found: run ./evaluate.sh (it builds it) or "
              "sim2real/docker/run.sh build", file=sys.stderr)
        return 2

    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / "runs" / f"{stamp}_eval_kit_{commit}"
    out.mkdir(parents=True)
    meta = {"scenario": SCENARIO, "seed": SEED, "params": str(PARAMS), "image": args.image,
            "faults": list(args.faults), "jobs": len(slots), "commit": commit,
            "started": stamp}
    (out / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    fly_all(flights, REPO_ROOT / PARAMS, slots, stamp)
    (out / "flights.json").write_text(json.dumps([asdict(f) for f in flights], indent=1))
    text = write_outputs(out, flights, meta)
    print()
    print(text)
    print("kit folder:", out)
    return 0 if all(f.run for f in flights) else 1


if __name__ == "__main__":
    sys.exit(main())
