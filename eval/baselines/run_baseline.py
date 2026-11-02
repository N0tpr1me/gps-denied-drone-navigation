"""Fly a PX4 parameter set over the baseline scenarios and seeds, then tabulate it.

    python -m eval.baselines.run_baseline --label stock \\
        --params sim2real/estimate_bridge/ekf2_ev.params
    python -m eval.baselines.run_baseline --label tuned_v1 \\
        --params eval/baselines/px4_tuned_v2.params
    python -m eval.baselines.run_baseline --compare runs/<stock>/batch.json runs/<tuned>/batch.json
    python -m eval.baselines.run_baseline --runner hold --label structure_hold_v3 \\
        --params eval/baselines/px4_tuned_v2.params -- --param SH_FB_EN=1

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

``--jobs N`` flies N flights at once, each container pinned to its own ``--cpus-per-job``
host CPUs and given its own Gazebo partition (``sim2real.gazebo.run_scenario.
isolation_options``); ``--jobs auto`` fits as many as the free CPUs allow. The default, one
at a time, is how every batch before 2026-11 was flown (the Mac). Rows come out in the same
order either way. SITL needs no GPU: Gazebo runs server-only with no camera, so on the GPU
server the flights use CPU cores and leave the GPUs to training.

Crash: the vehicle came within ``eval.metrics.CONTACT_RADIUS_M`` of the wall (a contact)
or the flight never completed (hover test error, no log). With ``--runner hold`` a crash is
judged on the scored hold alone, as the stock runner's is on its hover: a contact in it, or
a hold that ended early. What the module's test sequence does after it (disengage to stock
Position mode, re-engage, watchdog trip, handback; ``sim2real/px4/sitl_test.py``) is its own
column, ``post_hold_fail``: the hold ran to the end but the sequence did not (Gazebo seed 3
under ``vio_at_engage``: its estimate jump lands after the hold, and stock PX4 walks into the
wall, whatever flew the hold; ``docs/gazebo_check.md``). Failsafe: PX4 spent any part of
the flight in a failsafe (``ekf.json`` ``modes.failsafe_fraction`` > 0); ``failsafe_in_hover``
when that happened inside the scored hover window.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


HOLD_S = 60.0  # run_gazebo's --hold-s for the benchmark
HOLD_SETTLE_S = 2.0  # the engage transient run_gazebo leaves out of the scored window


def row_from_summary(summary: dict[str, Any], ekf: dict[str, Any] | None,
                     expected_mode: str = "Hold", hold_s: float = HOLD_S) -> dict[str, Any]:
    """One table row from a run's estimate_summary.json (or hold_summary.json) and
    ekf.json. ``expected_mode`` "External 1" is the hold runner: ``hold_s`` is its hold."""
    metrics = summary.get("metrics") or {}
    hover = summary.get("hover") or summary.get("flight") or {}
    # A flight that ran to the end but could not be scored is not a crash, just unscored.
    completed = "error" not in hover and summary.get("ulog_error") != "no flight.ulg" and (
        bool(metrics) or "ulog_error" in summary)
    modes = (ekf or {}).get("modes")
    fs_fraction = float(modes["failsafe_fraction"]) if modes else math.nan
    # The stock runner's summary has the window in its metrics; hold_summary.json has it
    # under "hold" (run_gazebo.score_hold).
    window = metrics.get("window_s") or (summary.get("hold") or {}).get("window_s")
    row: dict[str, Any] = {
        "scenario": summary["scenario"], "seed": summary["seed"], "run": summary["run"],
        "ok": bool(summary.get("ok")), "completed": completed,
        "contacts": int(metrics.get("contacts", 0)),
        "failsafe_fraction": fs_fraction,
        "failsafe": bool(fs_fraction > 0) if modes else False,
        "failsafe_in_hover": failsafe_in_window(modes, window, expected_mode),
        "window_s": window,
        # Gazebo's real-time factor over the hover (stock runner's hover test only): the
        # check that parallel flights each still got a whole flight's worth of CPU.
        "rtf_min": (hover.get("real_time_factor") or {}).get("min"),
        # The scripted GPS was on (profile gps block): the LOG line says so.
        "gps_on": summary.get("gps_free") is False and bool(summary.get("gps_ok")),
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
    row["post_hold_fail"] = False
    if expected_mode == "External 1" and row["flown"]:
        # The scored window is the first engaged stretch; a full one means the hold ended at
        # the commanded disengage, whatever the sequence did afterwards.
        full = bool(window) and window[1] - window[0] >= hold_s - HOLD_SETTLE_S - 1.0
        row["crash"] = row["contacts"] > 0 or not full
        row["post_hold_fail"] = full and not completed
    for key in METRICS:
        row[key] = float(metrics[key]) if key in metrics else math.nan
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per scenario: mean and worst of each metric over seeds, crash and failsafe counts."""
    out: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == name]
        agg: dict[str, Any] = {"n": len(rs), "crashes": sum(r["crash"] for r in rs),
                               "post_hold_fails": sum(r.get("post_hold_fail", False) for r in rs),
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
             "| recovery max s | crash | post-hold failed | failsafe (in hover) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in batch["rows"]:
        lines.append(f"| {r['scenario']} | {r['seed']} | {fmt(r['p95_m'])} | {fmt(r['rms_m'])} "
                     f"| {fmt(r['worst_toward_surface_m'])} | {fmt(r['min_clearance_m'])} "
                     f"| {fmt(r['recovery_max_s'], 1)} | {'yes' if r['crash'] else 'no'} "
                     f"| {'yes' if r.get('post_hold_fail') else 'no'} "
                     f"| {'yes' if r['failsafe'] else 'no'} "
                     f"({'yes' if r['failsafe_in_hover'] else 'no'}) |")
    lines += ["", "| scenario | p95 mean / worst m | worst toward wall mean / worst m "
              "| crashes | post-hold failed | failsafes (in hover) |", "|---|---|---|---|---|---|"]
    for name, a in batch["aggregate"].items():
        lines.append(f"| {name} | {fmt(a['p95_m_mean'])} / {fmt(a['p95_m_worst'])} "
                     f"| {fmt(a['worst_toward_surface_m_mean'])} / "
                     f"{fmt(a['worst_toward_surface_m_worst'])} | {a['crashes']}/{a['n']} "
                     f"| {a.get('post_hold_fails', 0)}/{a['n']} "
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


def _arg(run_args: list[str], name: str) -> str | None:
    for i, a in enumerate(run_args):
        if a == name and i + 1 < len(run_args):
            return run_args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def profile_of(run_args: list[str]) -> str:
    """The estimate profile the runner flies: its ``--profile`` argument, else its default."""
    return _arg(run_args, "--profile") or "vio_default"


def hold_s_of(run_args: list[str]) -> float:
    """The hold runner's hold: its ``--hold-s`` argument, else run_gazebo's default."""
    v = _arg(run_args, "--hold-s")
    return float(v) if v is not None else HOLD_S


def log_line(r: dict[str, Any], batch: dict[str, Any], when: str) -> str:
    status = ("not flown (never armable)" if not r.get("flown", True) else
              "crash" if r["crash"] else
              "hold ok, post-hold sequence failed" if r.get("post_hold_fail") else
              ("ok" if r["ok"] else "flown, check failed"))
    result = (f"truth vs setpoint: p95 {fmt(r['p95_m'])} m, rms {fmt(r['rms_m'])} m, worst "
              f"toward wall {fmt(r['worst_toward_surface_m'])} m, min clearance "
              f"{fmt(r['min_clearance_m'], 2)} m, {r['contacts']} contacts, failsafe "
              f"{'yes' if r['failsafe'] else 'no'}"
              f"{' (in hover)' if r['failsafe_in_hover'] else ''}")
    runner = batch.get("runner", "estimate")
    what = f"{runner} runner" if runner != "estimate" else "stock hover"
    run_args = batch.get("run_args") or []
    extra = " " + " ".join(run_args) if run_args else ""
    profile = profile_of(run_args)
    gps = "GPS on (scripted)" if r.get("gps_on") else "GPS off"
    return (f"| {when} | {r['run']} | {batch['commit']} | eval/baselines/run_baseline "
            f"{batch['label']} ({what}{extra}): {r['scenario']} seed {r['seed']}, {profile} "
            f"estimate, `{batch['params']}`, {gps} | {batch.get('host') or MAC_HOST} "
            f"| {status} | {result} | {{vs}} |")


MAC_HOST = "MacBook (Apple Silicon), Docker arm64"


def host_label(jobs: int = 1) -> str:
    """Where the batch flew, for experiments/LOG.md. ``SITL_HOST_LABEL`` overrides it (the
    GPU server's setup sets a short name)."""
    label = os.environ.get("SITL_HOST_LABEL")
    if not label:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            label = MAC_HOST
        else:
            label = f"{platform.system()} {platform.machine()} ({os.cpu_count()} CPUs), Docker"
    return label + (f", {jobs} flights in parallel" if jobs > 1 else "")


def cpu_slots(jobs: str | int, cpus_per_job: int, cpus: list[int] | None = None
              ) -> list[str | None]:
    """One ``--cpuset`` per parallel flight: disjoint blocks of ``cpus_per_job`` CPUs from
    the CPUs this process may use. ``jobs`` 1 means no pinning (the old behaviour);
    ``"auto"`` means as many blocks as fit. Refuses to oversubscribe."""
    if cpus is None:
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else \
            list(range(os.cpu_count() or 1))
    fit = len(cpus) // cpus_per_job
    n = max(fit, 1) if jobs == "auto" else int(jobs)
    if n < 1:
        raise ValueError("--jobs must be at least 1")
    if n == 1:
        return [None]
    if n > fit:
        raise ValueError(f"--jobs {n} x --cpus-per-job {cpus_per_job} needs {n * cpus_per_job} "
                         f"CPUs; {len(cpus)} available")
    blocks = [cpus[i * cpus_per_job:(i + 1) * cpus_per_job] for i in range(n)]
    return [",".join(str(c) for c in b) for b in blocks]


def fly(scenario: str, seed: int, params: Path, extra: list[str],
        runner: str = "estimate", cpuset: str | None = None,
        container: str | None = None) -> Path | None:
    """One flight of the runner; return its run folder."""
    module = RUNNERS[runner][0]
    cmd = [sys.executable, "-m", module, scenario,
           "--seed", str(seed), "--scenario-seed", str(seed), "--params", str(params), *extra]
    if cpuset:
        cmd += ["--cpuset", cpuset]
    if container:
        cmd += ["--container-name", container]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("run folder:"):
            folder = Path(line.split(":", 1)[1].strip())
            # The runner's own output, so a flight that flew but was not scored says why.
            (folder / "runner.log").write_text(proc.stdout + proc.stderr)
            return folder
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
    slots = cpu_slots(args.jobs, args.cpus_per_job)
    batch["host"] = host_label(len(slots))
    batch["jobs"] = len(slots)
    flights = [(sc, seed) for sc in args.scenarios for seed in args.seeds]
    order = {f: i for i, f in enumerate(flights)}
    free: queue.Queue[tuple[int, str | None]] = queue.Queue()
    for i, cpuset in enumerate(slots):
        free.put((i, cpuset))
    done: dict[tuple[str, int], tuple[dict[str, Any], str]] = {}
    lock = threading.Lock()

    def one(flight: tuple[str, int]) -> None:
        scenario, seed = flight
        slot, cpuset = free.get()
        try:
            when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            # Parallel flights need distinct container names; one at a time keeps the
            # runner's default name, as before.
            name = f"px4-sitl-{stamp}-{slot}" if len(slots) > 1 else None
            folder = fly(scenario, seed, args.params, args.run_args, args.runner, cpuset, name)
        finally:
            free.put((slot, cpuset))
        summary_path = folder / summary_name if folder else None
        if summary_path is None or not summary_path.exists():
            why = ""
            if folder and (folder / "runner.log").exists():
                why = "\n" + (folder / "runner.log").read_text()[-1500:]
            print(f"{scenario} seed {seed}: no summary{why}", flush=True)
            return
        summary = json.loads(summary_path.read_text())
        ekf_path = folder / "ekf.json"
        ekf = json.loads(ekf_path.read_text()) if ekf_path.exists() else None
        row = row_from_summary(summary, ekf, expected_mode, hold_s_of(args.run_args))
        print(f"{scenario} seed {seed}: p95 {fmt(row['p95_m'])} m, worst toward wall "
              f"{fmt(row['worst_toward_surface_m'])} m, crash {row['crash']}, post-hold "
              f"sequence failed {row['post_hold_fail']}, failsafe {row['failsafe']}"
              + (f", real-time factor min {row['rtf_min']}" if row["rtf_min"] else ""),
              flush=True)
        with lock:
            done[flight] = (row, when)
            keys = sorted(done, key=order.__getitem__)
            batch["rows"] = [done[k][0] for k in keys]
            batch["aggregate"] = aggregate(batch["rows"])
            (out / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")

    if len(slots) > 1:
        print(f"{len(flights)} flights, {len(slots)} at a time on CPUs {slots}", flush=True)
    with ThreadPoolExecutor(max_workers=len(slots)) as pool_:
        for fut in [pool_.submit(one, f) for f in flights]:
            fut.result()
    keys = sorted(done, key=order.__getitem__)
    log_lines = [log_line(done[k][0], batch, done[k][1]) for k in keys]
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
    ap.add_argument("--jobs", default="1",
                    help="flights at once (each pinned to its own CPUs), or 'auto'")
    ap.add_argument("--cpus-per-job", type=int, default=4,
                    help="host CPUs per parallel flight (default 4)")
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
    if args.jobs != "auto":
        try:
            args.jobs = int(args.jobs)
        except ValueError:
            ap.error("--jobs takes a number or 'auto'")
    try:
        cpu_slots(args.jobs, args.cpus_per_job)
    except ValueError as exc:
        ap.error(str(exc))
    batch = run_batch(args)
    return 0 if batch["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
