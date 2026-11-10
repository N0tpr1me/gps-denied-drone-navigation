"""Failure rates with confidence bounds for the safety manual's failure table (FMEA), from
the mock: each scripted estimate fault flown tens of thousands of times by the released
policy and the cascaded PD yardstick, the same fault at the same moment in every episode.

    python -m training.launch --config eval/configs/fault_rates_v22.yaml --smoke --device cpu
    python -m training.launch --config eval/configs/fault_rates_v22.yaml --device cuda \\
        --set "params.only=[reset]"                       # one fault per GPU job (the queue)
    python -m eval.fault_rates collect runs/<a> runs/<b> ... --out runs/fault_rates.md

Why: the field faults were flown in Gazebo on 3 seeds (``docs/field-faults.md``), which says
what happens, not how often. A buyer's safety reviewer asks for the sample size behind a
failure row. Here every fault gets ``num_envs x seeds`` episodes (20,480 in the shipped
config), so a row can say "wall contact in fewer than 1 in N episodes, 95 % bound" (the
Wilson upper bound, ``eval.gate1.wilson``; with no crash in 20,480 episodes it is about 1 in
5,500). The PD flies the same episodes, so the row also says how a stock-shaped controller
fares on the same fault.

What it measures and what it does not: the policy alone, in the mock, on the ``gz_x500_sitl``
airframe (EKF2-like fusion on the barometer). No watchdog, wall guard or handback is in the
loop, so a long silence here shows what the drone does in the seconds before the module's
5 s trip, not what the module does after it. The Gazebo field faults remain the evidence for
the handback; these numbers go beside them, not in their place (``docs/fault-rates.md``).

Inputs (``params``): ``controllers`` (as ``eval.gate1``; the PD is added as ``pd``), ``seeds``,
``num_envs``, ``base_env`` (a ``StructureHoldConfig`` mapping), ``faults`` (list of ``{name,
fmea, label, env, events, drift}``: ``env`` merges over ``base_env``; ``events`` are
``eval.gazebo_check.apply_event`` faults at ``t_s``; ``drift`` is ``{start_s, duration_s,
rate_mps, direction}``, a steady walk of the reported position fed in as the estimate's bias),
``only`` (fault names to fly; the rest are skipped), ``window_s`` (as the check; unused for the
rates). Outputs: ``metrics.jsonl``, ``report.json``, ``report.md``, ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import StructureHoldEnv
from eval.gate1 import check_not_held_out, make_factory, merge, wilson
from eval.gazebo_check import fly


def drift_offsets(spec: dict[str, Any], episode_s: float, dt: float) -> np.ndarray:
    """The reported position's bias at every step: zero until ``start_s``, then walking at
    ``rate_mps`` along ``direction`` for ``duration_s``, then held where it got to."""
    steps = int(round(episode_s / dt))
    t = np.arange(steps) * dt
    walk = np.clip(t - float(spec["start_s"]), 0.0, float(spec["duration_s"]))
    d = np.asarray(spec["direction"], dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-9)
    return (walk[:, None] * float(spec["rate_mps"]) * d[None, :]).astype(np.float32)


def one_in(upper: float) -> str:
    """``1 in N`` for an upper bound on a rate, rounded down to a round number."""
    if not math.isfinite(upper) or upper <= 0:
        return "-"
    n = 1.0 / upper
    if n < 10:
        return f"1 in {n:.1f}"
    mag = 10 ** int(math.floor(math.log10(n)))
    return f"1 in {int(math.floor(n / mag) * mag):,}"


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r["crash"])
    contacts = sum(1 for r in rows if r["reason"] == "contact")
    p95 = np.array([r.get("p95_m", math.nan) for r in rows], dtype=float)
    toward = np.array([r.get("toward_wall_m", math.nan) for r in rows], dtype=float)
    reasons: dict[str, int] = {}
    for r in rows:
        if r["crash"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    lo, hi = wilson(k, n)
    clo, chi = wilson(contacts, n)
    return {
        "episodes": n, "crashes": k, "crash_rate": k / n if n else math.nan,
        "crash_ci": [lo, hi], "crash_upper_one_in": one_in(hi), "crash_reasons": reasons,
        "contacts": contacts, "contact_rate": contacts / n if n else math.nan,
        "contact_ci": [clo, chi], "contact_upper_one_in": one_in(chi),
        "p95_m_mean": float(np.nanmean(p95)) if np.isfinite(p95).any() else math.nan,
        "toward_wall_m_mean": float(np.nanmean(toward)) if np.isfinite(toward).any() else math.nan,
        "toward_wall_m_max": float(np.nanmax(toward)) if np.isfinite(toward).any() else math.nan,
    }


def run(cfg: dict[str, Any], device: str, writer: Any) -> dict[str, Any]:
    p = cfg["params"]
    seeds = [int(s) for s in p["seeds"]]
    check_not_held_out(*seeds)
    num_envs = int(p.get("num_envs", 1024))
    window_s = float(p.get("window_s", 20.0))
    specs = [dict(c) for c in p["controllers"]] + [{"name": "pd", "policy": None}]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"controller names must be unique: {names}")
    factories = {s["name"]: make_factory(s, device) for s in specs}
    only = p.get("only")
    faults = [f for f in p["faults"] if not only or f["name"] in set(only)]
    if only and len(faults) != len(set(only)):
        known = [f["name"] for f in p["faults"]]
        raise ValueError(f"only={only}: unknown fault; the config has {known}")
    print(f"fault_rates: {len(faults)} faults x {len(seeds)} seeds x {num_envs} episodes, "
          f"controllers {names}, device {device}", flush=True)

    report: dict[str, Any] = {"seeds": seeds, "num_envs": num_envs,
                              "episodes_per_fault": num_envs * len(seeds),
                              "controllers": specs, "faults": []}
    t_start = time.perf_counter()
    for fault in faults:
        env_cfg = StructureHoldConfig.from_dict(merge(p.get("base_env") or {}, fault.get("env")))
        events = list(fault.get("events") or [])
        offsets = None
        if fault.get("drift"):
            offsets = drift_offsets(fault["drift"], env_cfg.episode_length_s, env_cfg.dt)
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for seed in seeds:
            for name in names:
                env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device)
                t0 = time.perf_counter()
                r = fly(env, factories[name](env), window_s, 0.1, events, offsets)
                for row in r:
                    writer.metric(seed, fault=fault["name"], controller=name, **row)
                rows[name].extend(r)
                d = describe(r)
                print(f"fault_rates: {fault['name']} seed {seed} {name}: crash "
                      f"{d['crashes']}/{d['episodes']}, p95 {d['p95_m_mean']:.3f} m, toward "
                      f"the wall {d['toward_wall_m_mean']:.3f} m "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
        report["faults"].append({
            "name": fault["name"], "fmea": fault.get("fmea"),
            "label": fault.get("label", fault["name"]),
            "controllers": {name: describe(rows[name]) for name in names},
        })
    report["elapsed_s"] = time.perf_counter() - t_start
    run_dir = Path(writer.run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (run_dir / "report.md").write_text(markdown(report))

    headline: dict[str, Any] = {"episodes_per_fault": report["episodes_per_fault"],
                                "faults": len(report["faults"])}
    parts = []
    for f in report["faults"]:
        for name in names:
            d = f["controllers"][name]
            headline[f"{f['name']}_{name}_crash"] = f"{d['crashes']}/{d['episodes']}"
            headline[f"{f['name']}_{name}_crash_upper"] = d["crash_ci"][1]
        g = f["controllers"][names[0]]
        parts.append(f"{f['name']}: {names[0]} {g['crashes']}/{g['episodes']} "
                     f"(under {g['crash_upper_one_in']}), pd "
                     f"{f['controllers']['pd']['crashes']}/{f['controllers']['pd']['episodes']}")
    wording = ("policy alone in the mock, no watchdog; crashes with a 95 % Wilson upper "
               "bound, the PD on the same episodes: " + "; ".join(parts))
    print(f"fault_rates: {wording} ({report['elapsed_s'] / 60:.0f} min)", flush=True)
    return {"headline": headline, "vs_baseline": wording}


def markdown(report: dict[str, Any]) -> str:
    names = [s["name"] for s in report["controllers"]]
    n = report["episodes_per_fault"]
    lines = [f"# Failure rates per fault (mock, policy alone, {n} episodes per fault)", ""]
    lines.append("Crashes are contacts with the wall, the ground, a tilt past the limit or "
                 "leaving the box; the 95 % column is the Wilson upper bound on the crash rate, "
                 "written as the rarest rate the sample can rule out. No watchdog or wall "
                 "guard flies here, so a row is what happens before the module's own trips "
                 "(docs/fault-rates.md).")
    lines.append("")
    head = "| FMEA | Fault | " + " | ".join(
        f"{nm} crashes | {nm} 95 % | {nm} p95 | {nm} toward wall" for nm in names) + " |"
    lines += [head, "|" + " --- |" * (2 + 4 * len(names))]
    for f in report["faults"]:
        parts = []
        for nm in names:
            d = f["controllers"][nm]
            parts += [f"{d['crashes']} / {d['episodes']}", d["crash_upper_one_in"],
                      f"{d['p95_m_mean']:.2f} m", f"{d['toward_wall_m_mean']:.2f} m"]
        lines.append(f"| {f.get('fmea') or '-'} | {f['label']} | " + " | ".join(parts) + " |")
    lines.append("")
    return "\n".join(lines)


def collect(run_dirs: list[Path]) -> dict[str, Any]:
    """One report from several runs of the same config (the queue flies one fault per job)."""
    out: dict[str, Any] | None = None
    for d in run_dirs:
        rep = json.loads((d / "report.json" if d.is_dir() else d).read_text())
        if out is None:
            out = {k: v for k, v in rep.items() if k != "faults"}
            out["faults"] = []
        if rep["episodes_per_fault"] != out["episodes_per_fault"]:
            raise ValueError(f"{d}: {rep['episodes_per_fault']} episodes per fault, the first "
                             f"run had {out['episodes_per_fault']}")
        seen = {f["name"] for f in out["faults"]}
        out["faults"] += [f for f in rep["faults"] if f["name"] not in seen]
    if out is None:
        raise ValueError("no runs given")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "collect":
        ap = argparse.ArgumentParser(prog="eval.fault_rates collect")
        ap.add_argument("runs", nargs="+", type=Path)
        ap.add_argument("--out", type=Path,
                        help="markdown file (default: fault_rates.md in the first run)")
        args = ap.parse_args(argv[1:])
        rep = collect(args.runs)
        first = args.runs[0] if args.runs[0].is_dir() else args.runs[0].parent
        out = args.out or first / "fault_rates.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown(rep))
        out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
        print(f"fault_rates: {len(rep['faults'])} faults -> {out}")
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
    raise SystemExit(main())
