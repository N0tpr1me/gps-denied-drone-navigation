"""Every environmental factor the mock models, one at a time: what each costs the released
policy and the cascaded PD yardstick, against the same hold without it.

    python -m training.launch --config eval/configs/environment_factors.yaml --smoke --device cpu
    python -m training.launch --config eval/configs/environment_factors.yaml --device cpu \\
        --set "params.only=[nominal,hot_high,cold_pack,low_battery,rain]"   # part of the list
    python -m eval.environment_factors collect runs/<a> runs/<b> ... --out runs/factors.md

The flying, the scoring and the statistics are ``eval.gate1``'s (one episode per env, every
controller on the same episodes, the PD always added). What this adds is the read-out for
``docs/environment-factors.md``: one row per factor with each controller's mean p95 error,
how far that is from its own number in the reference cell (``params.reference``, default
``nominal``), and its crash rate. A factor that moves a controller's error by less than the
seed spread is one the controller does not feel; one that moves both controllers the same
way sits in the estimate or the physics, not in the controller.

Not Gate 1: the seeds are outside its held-out block and no verdict here is a gate. The
numbers are mock numbers against the PD yardstick, not Gazebo against tuned PX4.

Inputs (``params``): as ``eval.gate1`` (``controllers``, ``seeds``, ``num_envs``,
``base_env``, ``cells`` of ``{name, label, group, env}``), plus ``reference`` and ``only``
(cell names to fly; the reference is always flown). Outputs: ``eval.gate1``'s
(``metrics.jsonl``, ``report.json``, ``report.md``) and ``factors.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from eval import gate1


def factor_rows(report: dict[str, Any], reference: str) -> list[dict[str, Any]]:
    """One row per cell: per controller its p95, the change from its own reference-cell p95
    (NaN when the reference was not flown) and its crash rate."""
    ref = next((c for c in report["cells"] if c["name"] == reference), None)
    rows = []
    for c in report["cells"]:
        row: dict[str, Any] = {"name": c["name"], "label": c["label"],
                               "group": (c.get("cell") or {}).get("group", ""),
                               "controllers": {}}
        for name, d in c["controllers"].items():
            base = ref["controllers"][name]["p95_m_mean"] if ref else math.nan
            row["controllers"][name] = {
                "p95_m": d["p95_m_mean"],
                "change": d["p95_m_mean"] / base - 1.0 if base and base == base else math.nan,
                "crash_rate": d["crash_rate"], "episodes": d["episodes"],
                "worst_toward_surface_m": d["worst_toward_surface_m"],
            }
        rows.append(row)
    return rows


def markdown(rows: list[dict[str, Any]], names: list[str], reference: str) -> str:
    head = "| Factor | " + " | ".join(f"{n} p95 (m) | {n} crash" for n in names) + " |"
    lines = [f"Mean per-episode p95 error and the change from each controller's own "
             f"`{reference}` row; crash rate.", "", head,
             "| --- |" + " --- | --- |" * len(names)]
    for r in rows:
        cells = []
        for n in names:
            d = r["controllers"][n]
            change = ("" if r["name"] == reference or d["change"] != d["change"]
                      else f" ({d['change']:+.0%})")
            cells.append(f"{d['p95_m']:.3f}{change} | {d['crash_rate']:.1%}")
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(cfg: dict[str, Any], device: str, writer: Any) -> dict[str, Any]:
    p = cfg["params"]
    reference = str(p.get("reference", "nominal"))
    cells = list(p["cells"])
    if reference not in [c["name"] for c in cells]:
        raise ValueError(f"reference cell {reference!r} is not in the cells")
    only = p.get("only")
    if only:
        unknown = set(only) - {c["name"] for c in cells}
        if unknown:
            raise ValueError(f"params.only names unknown cells: {sorted(unknown)}")
        cells = [c for c in cells if c["name"] in only or c["name"] == reference]
    # No cell here is a gate: nothing is judged beyond envelope, nothing is a headline.
    inner = {**cfg, "params": {**p, "cells": [{**c, "headline": False} for c in cells],
                               "rules": {"max_pooled_crash_rate": 1.0}}}
    gate1.run(inner, device, writer)
    run_dir = Path(writer.run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    for c, spec in zip(report["cells"], cells, strict=True):
        c["cell"] = {"group": spec.get("group", "")}
    names = [s["name"] for s in report["controllers"]]
    rows = factor_rows(report, reference)
    (run_dir / "factors.json").write_text(json.dumps(
        {"reference": reference, "controllers": names, "seeds": report["seeds"],
         "num_envs": report["num_envs"], "rows": rows}, indent=2, default=float) + "\n")
    (run_dir / "factors.md").write_text(markdown(rows, names, reference))
    gate = report["gate_controller"]
    worst = max((r for r in rows if r["name"] != reference),
                key=lambda r: r["controllers"][gate]["change"], default=None)
    headline = {"cells": len(rows), "reference": reference,
                f"{gate}_reference_p95_m": next(
                    r["controllers"][gate]["p95_m"] for r in rows if r["name"] == reference)}
    if worst is not None:
        headline["hardest_factor"] = worst["name"]
        headline[f"{gate}_hardest_change"] = worst["controllers"][gate]["change"]
        headline[f"{gate}_max_crash_rate"] = max(r["controllers"][gate]["crash_rate"]
                                                 for r in rows)
    pg, pd = report["pooled"][gate], report["pooled"]["pd"]
    wording = (f"mock, {gate} vs cascaded PD on the same episodes, {len(rows)} conditions x "
               f"{report['num_envs'] * len(report['seeds'])}: pooled p95 {pg['p95_m_mean']:.2f} "
               f"vs {pd['p95_m_mean']:.2f} m, crash {pg['crash_rate']:.1%} vs "
               f"{pd['crash_rate']:.1%}; not Gazebo, not against tuned PX4")
    print(f"environment_factors: {wording}", flush=True)
    return {"headline": headline, "vs_baseline": wording}


def collect(run_dirs: list[str], out: str | None) -> str:
    """Join the ``factors.json`` of several part runs (``params.only``) into one table; the
    reference row comes from the first run that has it, and each part's changes stay
    against its own reference."""
    rows, names, reference, seen = [], None, None, set()
    for d in run_dirs:
        data = json.loads((Path(d) / "factors.json").read_text())
        names, reference = names or data["controllers"], reference or data["reference"]
        if data["controllers"] != names:
            raise ValueError(f"{d}: controllers {data['controllers']} differ from {names}")
        for r in data["rows"]:
            if r["name"] not in seen:
                seen.add(r["name"])
                rows.append(r)
    text = markdown(rows, names or [], reference or "nominal")
    if out:
        Path(out).write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "collect":
        ap = argparse.ArgumentParser(prog="eval.environment_factors collect")
        ap.add_argument("runs", nargs="+")
        ap.add_argument("--out")
        args = ap.parse_args(argv[1:])
        print(collect(args.runs, args.out))
        return 0
    from training.runlib import RunWriter, entry_args, load_yaml

    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        result = run(cfg, args.device, writer)
        writer.summary("ok", headline=result["headline"], vs_baseline=result["vs_baseline"])
        return 0
    except Exception as e:  # every failure lands in summary.json, then exit non-zero
        traceback.print_exc()
        writer.summary("failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
