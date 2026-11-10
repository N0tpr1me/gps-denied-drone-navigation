"""The hold datasheet: where Structure Hold breaks, in one table, from the sweep runs.

    python -m eval.hold_datasheet runs/<into> runs/<along> runs/<lee> --out runs/hold_datasheet.md

The sweep is three ``eval.gate1`` runs (``eval/configs/hold_datasheet_{into,along,lee}.yaml``,
one per wind direction): every cell is one wind speed, one distance from the wall and one
estimate quality, flown by the released policy and the cascaded PD yardstick on the same
episodes. This script reads their ``report.json`` files and writes the table a buyer asks for
("holds within X m at Y m/s, Z m from the wall, on this kind of estimate"), plus the envelope
each controller actually has: the highest wind, per direction, distance and estimate, at
which it still holds inside ``--p95-limit`` with a crash rate whose 95 % upper bound stays
under ``--crash-limit``. Nothing here is a pass or fail; the numbers are the result, and the
Gazebo edge cells (``docs/hold-datasheet.md``) say how far to trust them.

Cell names carry the sweep's coordinates: ``<direction>_w<wind>_d<distance>_<estimate>``
(``into_w10_d1_vio``). ``datasheet.json`` beside the markdown holds every number.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

CELL = re.compile(r"^(?P<direction>[a-z]+)_w(?P<wind>\d+)_d(?P<distance>\d+)_(?P<estimate>\w+)$")

DIRECTION_LABEL = {"into": "wind into the wall", "along": "wind along the wall",
                   "lee": "wind away from the wall (the drone in its lee)"}
ESTIMATE_LABEL = {"clean": "clean estimate", "vio": "vio_default", "degraded": "degraded VIO",
                  "dropout3": "2 to 3 s dropouts"}
ESTIMATE_ORDER = ("clean", "vio", "degraded", "dropout3")


def parse_cell(name: str) -> dict[str, Any] | None:
    m = CELL.match(name)
    if not m:
        return None
    return {"direction": m["direction"], "wind": int(m["wind"]), "distance": int(m["distance"]),
            "estimate": m["estimate"]}


def load_reports(run_dirs: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for d in run_dirs:
        path = d / "report.json" if d.is_dir() else d
        reports.append(json.loads(path.read_text()))
    return reports


def build(reports: list[dict[str, Any]], p95_limit_m: float = 0.5,
          crash_limit: float = 0.01) -> dict[str, Any]:
    """Every cell's numbers per controller, and each controller's envelope."""
    cells: list[dict[str, Any]] = []
    controllers: list[str] = []
    for rep in reports:
        for c in rep["cells"]:
            coords = parse_cell(c["name"])
            if coords is None:
                raise ValueError(f"cell {c['name']!r} is not a sweep cell "
                                 "(<direction>_w<wind>_d<distance>_<estimate>)")
            row = dict(coords, name=c["name"], label=c.get("label", c["name"]), controllers={})
            for name, d in c["controllers"].items():
                if name not in controllers:
                    controllers.append(name)
                row["controllers"][name] = {
                    "episodes": d["episodes"], "p95_m": d["p95_m_mean"],
                    "p95_ci": list(d["p95_ci"]), "crash_rate": d["crash_rate"],
                    "crash_ci": list(d["crash_ci"]),
                    "worst_toward_surface_m": d["worst_toward_surface_m"],
                }
            cells.append(row)
    if not cells:
        raise ValueError("no cells in the reports")
    cells.sort(key=lambda r: (r["direction"], r["distance"],
                              ESTIMATE_ORDER.index(r["estimate"])
                              if r["estimate"] in ESTIMATE_ORDER else 99, r["wind"]))

    def holds(d: dict[str, Any]) -> bool:
        return (math.isfinite(d["p95_m"]) and d["p95_m"] <= p95_limit_m
                and d["crash_ci"][1] <= crash_limit)

    # The envelope: per (direction, distance, estimate), the highest wind in an unbroken run
    # of winds (from the lowest flown) at which the controller holds, and the first wind at
    # which it does not. The 0 m/s cells, flown once (into), count for every direction.
    envelope: list[dict[str, Any]] = []
    calm = {(r["distance"], r["estimate"]): r for r in cells if r["wind"] == 0}
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for r in cells:
        if r["wind"] == 0:
            continue
        groups.setdefault((r["direction"], r["distance"], r["estimate"]), []).append(r)
    for (direction, distance, estimate), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["wind"])
        c0 = calm.get((distance, estimate))
        if c0 is not None:
            rows = [c0] + rows
        entry: dict[str, Any] = {"direction": direction, "distance": distance,
                                 "estimate": estimate, "controllers": {}}
        for name in controllers:
            top: int | None = None
            breaks: int | None = None
            for r in rows:
                d = r["controllers"].get(name)
                if d is None:
                    continue
                if holds(d):
                    top = r["wind"]
                else:
                    breaks = r["wind"]
                    break
            entry["controllers"][name] = {"holds_to_mps": top, "breaks_at_mps": breaks,
                                          "winds_flown": [r["wind"] for r in rows]}
        envelope.append(entry)
    return {"p95_limit_m": p95_limit_m, "crash_limit": crash_limit, "controllers": controllers,
            "cells": cells, "envelope": envelope}


def _fmt_cell(d: dict[str, Any] | None) -> str:
    if d is None:
        return "-"
    p95 = f"{d['p95_m']:.2f}" if math.isfinite(d["p95_m"]) else "-"
    return f"{p95} m, {d['crash_rate']:.1%}"


def markdown(ds: dict[str, Any]) -> str:
    names = ds["controllers"]
    lines = ["# Hold datasheet (mock, true position, 60 s holds)", ""]
    n = ds["cells"][0]["controllers"][names[0]]["episodes"]
    lines.append(f"Each cell: mean per-episode p95 hold error and crash rate over {n} episodes, "
                 f"for {' / '.join(names)}. A cell holds when p95 is at most "
                 f"{ds['p95_limit_m']:.2f} m and the crash rate's 95 % upper bound is under "
                 f"{ds['crash_limit']:.1%}. Mock numbers; the Gazebo edge cells say how far to "
                 "trust them (docs/hold-datasheet.md).")
    lines.append("")
    lines.append("## Envelope: the highest wind each controller holds at")
    lines.append("")
    head = "| Wind direction | Distance | Estimate | " + " | ".join(
        f"{nm} holds to | {nm} breaks at" for nm in names) + " |"
    lines += [head, "|" + " --- |" * (3 + 2 * len(names))]
    for e in ds["envelope"]:
        parts = []
        for nm in names:
            c = e["controllers"].get(nm) or {}
            top = c.get("holds_to_mps")
            brk = c.get("breaks_at_mps")
            parts.append("never" if top is None else f"{top} m/s")
            parts.append(f"{brk} m/s" if brk is not None else
                         f"not within {max(c.get('winds_flown') or [0])} m/s")
        lines.append(f"| {DIRECTION_LABEL.get(e['direction'], e['direction'])} | "
                     f"{e['distance']} m | {ESTIMATE_LABEL.get(e['estimate'], e['estimate'])} | "
                     + " | ".join(parts) + " |")
    lines.append("")
    lines.append("## Every cell")
    lines.append("")
    lines.append(f"Each entry is {' / '.join(names)}: p95 in metres, then the crash rate.")
    lines.append("")
    directions = sorted({c["direction"] for c in ds["cells"]},
                        key=lambda d: list(DIRECTION_LABEL).index(d) if d in DIRECTION_LABEL else 9)
    for direction in directions:
        for distance in sorted({c["distance"] for c in ds["cells"]}):
            rows = [c for c in ds["cells"] if c["distance"] == distance
                    and (c["direction"] == direction or c["wind"] == 0)]
            if not any(r["direction"] == direction for r in rows):
                continue
            ests = [e for e in ESTIMATE_ORDER if any(r["estimate"] == e for r in rows)]
            ests += sorted({r["estimate"] for r in rows} - set(ests))
            lines.append(f"### {DIRECTION_LABEL.get(direction, direction).capitalize()}, "
                         f"{distance} m from the wall")
            lines.append("")
            lines.append("| Wind | " + " | ".join(ESTIMATE_LABEL.get(e, e) for e in ests) + " |")
            lines.append("|" + " --- |" * (1 + len(ests)))
            for wind in sorted({r["wind"] for r in rows}):
                cells_w = {r["estimate"]: r for r in rows if r["wind"] == wind}
                parts = []
                for e in ests:
                    r = cells_w.get(e)
                    parts.append(" / ".join(_fmt_cell(r["controllers"].get(nm) if r else None)
                                            for nm in names))
                lines.append(f"| {wind} m/s | " + " | ".join(parts) + " |")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="sweep run folders (or report.json files)")
    ap.add_argument("--out", type=Path,
                    help="markdown file (default: datasheet.md in the first run)")
    ap.add_argument("--p95-limit", type=float, default=0.5, help="hold error that still counts, m")
    ap.add_argument("--crash-limit", type=float, default=0.01,
                    help="95 %% upper bound on the crash rate that still counts")
    args = ap.parse_args(argv)
    ds = build(load_reports(args.runs), args.p95_limit, args.crash_limit)
    first = args.runs[0] if args.runs[0].is_dir() else args.runs[0].parent
    out = args.out or first / "datasheet.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(ds))
    out.with_suffix(".json").write_text(json.dumps(ds, indent=2, default=float) + "\n")
    print(f"hold_datasheet: {len(ds['cells'])} cells, {len(ds['envelope'])} envelope rows -> {out}")
    for e in ds["envelope"]:
        parts = ", ".join(f"{nm} to {c['holds_to_mps']} m/s"
                          for nm, c in e["controllers"].items())
        print(f"  {e['direction']} {e['distance']} m {e['estimate']}: {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
