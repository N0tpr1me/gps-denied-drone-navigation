"""Fly a PX4 parameter set over the baseline scenarios and seeds, then tabulate it.

    python -m eval.baselines.run_baseline --label stock \\
        --params sim2real/estimate_bridge/ekf2_ev.params
    python -m eval.baselines.run_baseline --label tuned_v1 \\
        --params eval/baselines/px4_tuned_v1.params
    python -m eval.baselines.run_baseline --compare runs/<stock>/batch.json runs/<tuned>/batch.json
    python -m eval.baselines.run_baseline --runner hold --label structure_hold_v3 \\
        --params eval/baselines/px4_tuned_v1.params -- --param SH_FB_EN=1

Each flight is one ``sim2real.estimate_bridge.run_estimate`` run (Gazebo wall_slab, wind
plugin, GPS off, EKF2 on the corrupted vision estimate), scored on the TRUE position with
``eval.metrics``; with ``--runner hold`` it is one ``sim2real.px4.run_gazebo`` run instead
(the Structure Hold mode holding over the same window, scored the same way), so the two
tables compare like for like. Seed ``s`` sets both the estimate profile's seed and the
scenario's wind seed, so seed 2 is a different estimate corruption and a different
gust/turbulence draw.

Writes ``runs/<time>_px4_baseline_<label>_<commit>/``:

- ``batch.json``: one row per flight (run folder, metrics, crash, failsafe) and per-scenario
  aggregates (mean and worst over seeds)
- ``table.md``: the same as a Markdown table
- ``log_lines.md``: one experiments/LOG.md line per flight, to paste after review

Crash: the vehicle came within ``eval.metrics.CONTACT_RADIUS_M`` of the wall (a contact)
or the flight never completed (hover test error, no log). Failsafe: PX4 spent any part of
the flight in a failsafe (``ekf.json`` ``modes.failsafe_fraction`` > 0); ``failsafe_in_hover``
when that happened inside the scored hover window.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# runner -> (module, summary file, the mode the scored window is flown in)
RUNNERS = {"estimate": ("sim2real.estimate_bridge.run_estimate", "estimate_summary.json", "Hold"),
           "hold": ("sim2real.px4.run_gazebo", "hold_summary.json", "External 1")}
DEFAULT_SCENARIOS = ("wall1m_calm_0ms", "wall1m_wind_4ms", "wall1m_wind_8ms")
DEFAULT_SEEDS = (1, 2, 3)
METRICS = ("p95_m", "rms_m", "max_m", "worst_toward_surface_m", "min_clearance_m",
           "recovery_max_s")


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def failsafe_in_window(modes: dict[str, Any] | None, window: list[float] | None,
                       expected: str = "Hold") -> bool:
    """True when PX4 left the expected mode for anything but the commanded Land inside
    the window."""
    if not modes or not window:
        return False
    t0, t1 = window
    return any(t0 <= m["t"] <= t1 and m["mode"] != expected
               for m in modes.get("sequence", []))


def row_from_summary(summary: dict[str, Any], ekf: dict[str, Any] | None,
                     expected_mode: str = "Hold") -> dict[str, Any]:
    """One table row from a run's estimate_summary.json (or hold_summary.json) and
    ekf.json."""
    metrics = summary.get("metrics") or {}
    hover = summary.get("hover") or summary.get("flight") or {}
    # A flight that ran to the end but could not be scored is not a crash, just unscored.
    completed = "error" not in hover and summary.get("ulog_error") != "no flight.ulg" and (
        bool(metrics) or "ulog_error" in summary)
    modes = (ekf or {}).get("modes")
    fs_fraction = float(modes["failsafe_fraction"]) if modes else math.nan
    window = metrics.get("window_s")
    row: dict[str, Any] = {
        "scenario": summary["scenario"], "seed": summary["seed"], "run": summary["run"],
        "ok": bool(summary.get("ok")), "completed": completed,
        "contacts": int(metrics.get("contacts", 0)),
        "failsafe_fraction": fs_fraction,
        "failsafe": bool(fs_fraction > 0) if modes else False,
        "failsafe_in_hover": failsafe_in_window(modes, window, expected_mode),
        "window_s": window,
    }
    # Mode changes inside the window are cut at the commanded Land at the end of the hover
    # (stock runner only: the hold runner's window ends at the commanded disengage, and the
    # rest of its sequence is commanded mode changes, not failsafes).
    if window and modes and expected_mode == "Hold":
        land = [m["t"] for m in modes.get("sequence", []) if m["mode"] == "Land"
                and m["t"] >= window[1] - 1.0]
        if land:
            row["failsafe_in_hover"] = failsafe_in_window(modes, [window[0], land[0] - 1e-3],
                                                          expected_mode)
    # Never armed (preflight checks failed): nothing flew, so not a crash either.
    row["flown"] = "never became armable" not in str(hover.get("error", ""))
    row["crash"] = row["flown"] and (row["contacts"] > 0 or not completed)
    for key in METRICS:
        row[key] = float(metrics[key]) if key in metrics else math.nan
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per scenario: mean and worst of each metric over seeds, crash and failsafe counts."""
    out: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == name]
        agg: dict[str, Any] = {"n": len(rs), "crashes": sum(r["crash"] for r in rs),
                               "failsafes": sum(r["failsafe"] for r in rs),
                               "failsafes_in_hover": sum(r["failsafe_in_hover"] for r in rs)}
        for key in METRICS:
            vals = [r[key] for r in rs if not math.isnan(r[key])]
            agg[f"{key}_mean"] = statistics.fmean(vals) if vals else math.nan
            # min clearance: the worst is the smallest; everything else the largest.
            worst = min if key == "min_clearance_m" else max
            agg[f"{key}_worst"] = worst(vals) if vals else math.nan
        out[name] = agg
    return out


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table(batch: dict[str, Any]) -> str:
    lines = [f"Parameter set `{batch['label']}` (`{batch['params']}`), commit {batch['commit']}",
             "",
             "| scenario | seed | p95 m | rms m | worst toward wall m | min clearance m "
             "| recovery max s | crash | failsafe (in hover) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in batch["rows"]:
        lines.append(f"| {r['scenario']} | {r['seed']} | {fmt(r['p95_m'])} | {fmt(r['rms_m'])} "
                     f"| {fmt(r['worst_toward_surface_m'])} | {fmt(r['min_clearance_m'])} "
                     f"| {fmt(r['recovery_max_s'], 1)} | {'yes' if r['crash'] else 'no'} "
                     f"| {'yes' if r['failsafe'] else 'no'} "
                     f"({'yes' if r['failsafe_in_hover'] else 'no'}) |")
    lines += ["", "| scenario | p95 mean / worst m | worst toward wall mean / worst m "
              "| crashes | failsafes (in hover) |", "|---|---|---|---|---|"]
    for name, a in batch["aggregate"].items():
        lines.append(f"| {name} | {fmt(a['p95_m_mean'])} / {fmt(a['p95_m_worst'])} "
                     f"| {fmt(a['worst_toward_surface_m_mean'])} / "
                     f"{fmt(a['worst_toward_surface_m_worst'])} | {a['crashes']}/{a['n']} "
                     f"| {a['failsafes']}/{a['n']} ({a['failsafes_in_hover']}) |")
    return "\n".join(lines) + "\n"


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Markdown: per scenario, mean over seeds of ``a`` vs ``b`` with the relative change."""
    lines = [f"`{a['label']}` → `{b['label']}` (mean over seeds; worst in brackets)", "",
             "| scenario | p95 m | worst toward wall m | rms m | crashes | failsafes |",
             "|---|---|---|---|---|---|"]

    def cell(key: str, name: str) -> str:
        x, y = a["aggregate"][name], b["aggregate"][name]
        m0, m1 = x[f"{key}_mean"], y[f"{key}_mean"]
        rel = f"{100 * (m1 - m0) / m0:+.0f} %" if m0 and not math.isnan(m0) else "n/a"
        return (f"{fmt(m0)} ({fmt(x[f'{key}_worst'])}) → {fmt(m1)} ({fmt(y[f'{key}_worst'])}),"
                f" {rel}")

    for name in a["aggregate"]:
        if name not in b["aggregate"]:
            continue
        x, y = a["aggregate"][name], b["aggregate"][name]
        lines.append(f"| {name} | {cell('p95_m', name)} | {cell('worst_toward_surface_m', name)}"
                     f" | {cell('rms_m', name)} | {x['crashes']}/{x['n']} → "
                     f"{y['crashes']}/{y['n']} | {x['failsafes']}/{x['n']} → "
                     f"{y['failsafes']}/{y['n']} |")
    return "\n".join(lines) + "\n"


def pool(batches: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """Merge the rows of several batches of the same label into one batch (for pooling
    seeds flown in separate chunks), re-aggregate, write ``out`` and return it."""
    labels = {b["label"] for b in batches}
    if len(labels) != 1:
        raise ValueError(f"pooled batches must share one label, got {sorted(labels)}")
    rows = [r for b in batches for r in b["rows"]]
    seen = {(r["scenario"], r["seed"]) for r in rows}
    if len(seen) != len(rows):
        raise ValueError("pooled batches repeat a (scenario, seed) pair")
    seeds = sorted({r["seed"] for r in rows})
    pooled = {**batches[0], "pooled_from": [b.get("commit") for b in batches],
              "seeds": seeds, "rows": rows, "aggregate": aggregate(rows)}
    out.write_text(json.dumps(pooled, indent=2) + "\n")
    return pooled


def spread(batch: dict[str, Any], key: str = "p95_m") -> str:
    """Markdown: per scenario, every seed's value of ``key`` (the spread behind a mean)."""
    lines = [f"`{batch['label']}`: {key} per seed (n = {len(batch['seeds'])} seeds)", "",
             "| scenario | " + " | ".join(f"seed {s}" for s in batch["seeds"]) +
             " | mean | worst |", "|---|" + "---|" * (len(batch["seeds"]) + 2)]
    for name, agg in batch["aggregate"].items():
        by_seed = {r["seed"]: r[key] for r in batch["rows"] if r["scenario"] == name}
        cells = [fmt(by_seed.get(s, math.nan)) for s in batch["seeds"]]
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {fmt(agg[f'{key}_mean'])} | {fmt(agg[f'{key}_worst'])} |")
    return "\n".join(lines) + "\n"


def log_line(r: dict[str, Any], batch: dict[str, Any], when: str) -> str:
    status = ("not flown (never armable)" if not r.get("flown", True) else
              "crash" if r["crash"] else ("ok" if r["ok"] else "flown, check failed"))
    result = (f"truth vs setpoint: p95 {fmt(r['p95_m'])} m, rms {fmt(r['rms_m'])} m, worst "
              f"toward wall {fmt(r['worst_toward_surface_m'])} m, min clearance "
              f"{fmt(r['min_clearance_m'], 2)} m, {r['contacts']} contacts, failsafe "
              f"{'yes' if r['failsafe'] else 'no'}"
              f"{' (in hover)' if r['failsafe_in_hover'] else ''}")
    runner = batch.get("runner", "estimate")
    what = f"{runner} runner" if runner != "estimate" else "stock hover"
    extra = " " + " ".join(batch["run_args"]) if batch.get("run_args") else ""
    return (f"| {when} | {r['run']} | {batch['commit']} | eval/baselines/run_baseline "
            f"{batch['label']} ({what}{extra}): {r['scenario']} seed {r['seed']}, vio_default "
            f"estimate, `{batch['params']}`, GPS off | MacBook (Apple Silicon), Docker arm64 "
            f"| {status} | {result} | {{vs}} |")


def fly(scenario: str, seed: int, params: Path, extra: list[str],
        runner: str = "estimate") -> Path | None:
    """One flight of the runner; return its run folder."""
    module = RUNNERS[runner][0]
    cmd = [sys.executable, "-m", module, scenario,
           "--seed", str(seed), "--scenario-seed", str(seed), "--params", str(params), *extra]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("run folder:"):
            return Path(line.split(":", 1)[1].strip())
    sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
    return None


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    params = args.params.resolve().relative_to(REPO_ROOT)
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / "runs" / f"{stamp}_px4_baseline_{args.label}_{commit}"
    out.mkdir(parents=True)
    _, summary_name, expected_mode = RUNNERS[args.runner]
    batch: dict[str, Any] = {"label": args.label, "params": str(params), "commit": commit,
                             "runner": args.runner, "run_args": list(args.run_args),
                             "scenarios": list(args.scenarios), "seeds": list(args.seeds),
                             "rows": []}
    log_lines = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            folder = fly(scenario, seed, args.params, args.run_args, args.runner)
            summary_path = folder / summary_name if folder else None
            if summary_path is None or not summary_path.exists():
                print(f"{scenario} seed {seed}: no summary", flush=True)
                continue
            summary = json.loads(summary_path.read_text())
            ekf_path = folder / "ekf.json"
            ekf = json.loads(ekf_path.read_text()) if ekf_path.exists() else None
            row = row_from_summary(summary, ekf, expected_mode)
            batch["rows"].append(row)
            log_lines.append(log_line(row, batch, when))
            print(f"{scenario} seed {seed}: p95 {fmt(row['p95_m'])} m, worst toward wall "
                  f"{fmt(row['worst_toward_surface_m'])} m, crash {row['crash']}, failsafe "
                  f"{row['failsafe']}", flush=True)
            batch["aggregate"] = aggregate(batch["rows"])
            (out / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    batch["aggregate"] = aggregate(batch["rows"])
    (out / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (out / "table.md").write_text(table(batch))
    (out / "log_lines.md").write_text("\n".join(log_lines) + "\n")
    print("batch folder:", out)
    print(table(batch))
    return batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="name of the parameter set, e.g. stock or tuned_v1")
    ap.add_argument("--params", type=Path, help="PX4 .params file (EKF2 settings included)")
    ap.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--runner", choices=sorted(RUNNERS), default="estimate",
                    help="estimate: stock PX4 hover (run_estimate); hold: the Structure Hold "
                         "mode (sim2real.px4.run_gazebo)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"),
                    help="print A vs B from two batch.json files and exit")
    ap.add_argument("--pool", nargs="+", type=Path, metavar="BATCH",
                    help="merge these batch.json files (same label, disjoint seeds) into "
                         "--out, print the per-seed spread and exit")
    ap.add_argument("--out", type=Path, help="where --pool writes the pooled batch.json")
    ap.add_argument("run_args", nargs="*", help="extra run_estimate arguments, after --")
    args = ap.parse_args(argv)
    if args.pool:
        if not args.out:
            ap.error("--pool needs --out")
        pooled = pool([json.loads(p.read_text()) for p in args.pool], args.out)
        print(table(pooled))
        print(spread(pooled))
        return 0
    if args.compare:
        a, b = (json.loads(p.read_text()) for p in args.compare)
        print(compare(a, b))
        return 0
    if not args.label or not args.params:
        ap.error("--label and --params are required unless --compare is given")
    batch = run_batch(args)
    return 0 if batch["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
