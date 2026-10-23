"""The Gazebo-like check: fly policies in the mock the way the Gazebo benchmark flies them,
before any of them goes to the Mac for a Gazebo re-fly (docs/decisions.md 2026-10-23).

    python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu

Why it exists: v7 and v8 beat v6 in Gate 1 and then flew worse than v6 in Gazebo, because
Gate 1's mock had none of what makes v6 and v8 fall into the 1.6 Hz roll-and-yaw wobble that
lifts them 0.2 to 0.4 m in Gazebo. This check flies the ``gz_x500_sitl`` airframe (PX4's yaw
torque and gyro filters, the module's 83 Hz tick, the bridge's 0.24 m lever arm, EKF2-like
fusion; ``envs/structure_hold/airframes.py``) through the benchmark's conditions: 60 s, 1 m
from the wall, the hold point 2.4 m up, the ``vio_default`` estimate (and a clean one), calm
air and the 4 and 8 m/s scenarios of ``sim2real/gazebo/scenarios``.

Per episode, over the last ``window_s`` seconds (the first seconds are the engage transient):

* ``lift_m``: the mean height of the estimate above the hold point (what the module flies to;
  Gazebo's "offset up");
* ``wobble``: the roll action's 1.2 to 2.2 Hz band carries more than ``wobble_threshold``
  (rms, action units); Gazebo's wobbling flights sit at 0.2 to 0.3, quiet ones under 0.03;
* ``p95_m`` over the whole episode (``eval.metrics``, true centre of mass) and crashes.

A controller ``beats`` the reference (the released policy) when, pooled over the
rough-estimate cells, its p95 is lower (paired bootstrap 95 % interval of the ratio below 1),
its lift and wobble rate are no higher, its crash rate is no more than ``max_crash_increase``
above the reference's, and on the clean cell its p95 is within ``clean_p95_slack_m`` of the
reference's. Gazebo's numbers for the controllers flown there (``gazebo`` in the config) are
printed beside the mock's so the check's own fidelity stays visible.

Outputs: ``metrics.jsonl`` (one row per episode), ``report.json``, ``report.md``,
``summary.json``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import TERM_REASONS, StructureHoldEnv
from eval.gate1 import check_not_held_out, make_factory, merge
from eval.metrics import Surface, Trace, compute_metrics

TIMEOUT = TERM_REASONS.index("timeout")

DEFAULT_RULES = {
    "max_crash_increase": 0.01,
    "clean_p95_slack_m": 0.02,
}


def band_rms(x: np.ndarray, dt: float, lo: float = 1.2, hi: float = 2.2) -> np.ndarray:
    """Rms of the ``lo``..``hi`` Hz part of ``x`` (time, envs), Hann-windowed."""
    x = x - x.mean(0)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), dt)
    frac = spec[(f >= lo) & (f <= hi)].sum(0) / np.maximum(spec.sum(0), 1e-12)
    return np.sqrt(frac) * x.std(0)


def fly(env: StructureHoldEnv, controller: Any, window_s: float,
        wobble_threshold: float) -> list[dict[str, Any]]:
    """Fly each env's first episode; one row per env."""
    cfg, n, steps, dt = env.cfg, env.num_envs, env.max_episode_length, env.dt
    hold = env.hold_point.cpu().numpy().copy()
    pos = np.empty((steps + 1, n, 3), dtype=np.float32)
    est_z = np.empty((steps, n), dtype=np.float32)
    act = np.empty((steps, n, 4), dtype=np.float32)
    pos[0] = env.state.pos.cpu().numpy()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    end = np.full(n, steps, dtype=np.int64)
    reason = np.full(n, TIMEOUT, dtype=np.int64)
    obs, _ = env.get_observations()
    for k in range(steps):
        a = controller(obs)
        act[k] = a.detach().cpu().numpy()
        obs, _, done, _ = env.step(a)
        est_z[k] = (env.estimate_pos[:, 2] - env.hold_point[:, 2]).cpu().numpy()
        still = alive & ~done
        s = still.cpu().numpy()
        pos[k + 1] = np.where(s[:, None], env.state.pos.cpu().numpy(), pos[k])
        newly = (alive & done).cpu().numpy()
        if newly.any():
            end[newly] = k + 1
            reason[newly] = env.term_reason.cpu().numpy()[newly]
        alive = still
        if not bool(alive.any()):
            break

    w0 = max(0, steps - int(round(window_s / dt)))
    full = end == steps
    wob = np.full(n, np.nan)
    lift = np.full(n, np.nan)
    if full.any():
        wob[full] = band_rms(act[w0:, full, 0].astype(np.float64), dt)
        lift[full] = est_z[w0:, full].mean(0)
    surface = Surface(point=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]))
    rows = []
    for i in range(n):
        length = int(end[i])
        row: dict[str, Any] = {"env": i, "length_s": float(length * dt),
                               "crash": bool(reason[i] != TIMEOUT),
                               "reason": TERM_REASONS[int(reason[i])]}
        if full[i]:
            row.update({"lift_m": float(lift[i]), "wobble_rms": float(wob[i]),
                        "wobble": bool(wob[i] > wobble_threshold)})
        if length >= 2:
            t = np.arange(length + 1) * dt
            trace = Trace(t=t, pos=pos[: length + 1, i].astype(np.float64), pos_sp=hold[i],
                          cmd_t=t[:-1], rates_cmd=act[:length, i, :3] * cfg.max_rate_rad_s,
                          thrust_cmd=0.5 * (act[:length, i, 3] + 1.0))
            m = compute_metrics(trace, surface, contact_radius_m=cfg.contact_radius_m)
            row["p95_m"] = m["p95_m"]
        rows.append(row)
    return rows


def _col(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [r for r in rows if "lift_m" in r]
    return {
        "episodes": len(rows),
        "crash_rate": float(np.mean(_col(rows, "crash"))),
        "p95_m_mean": float(np.nanmean(_col(rows, "p95_m"))),
        "lift_m_mean": float(np.mean(_col(full, "lift_m"))) if full else float("nan"),
        "wobble_rate": float(np.mean(_col(full, "wobble"))) if full else float("nan"),
        "wobble_rms_median": float(np.median(_col(full, "wobble_rms"))) if full else float("nan"),
    }


def paired_p95_ratio(a: list[dict[str, Any]], b: list[dict[str, Any]], rng: np.random.Generator,
                     n_boot: int) -> dict[str, Any]:
    """Mean p95 of ``a`` over ``b`` on the same episodes, with a bootstrap 95 % interval."""
    pa, pb = _col(a, "p95_m"), _col(b, "p95_m")
    ok = np.isfinite(pa) & np.isfinite(pb)
    pa, pb = pa[ok], pb[ok]
    idx = rng.integers(0, len(pa), size=(n_boot, len(pa)))
    boot = pa[idx].mean(1) / pb[idx].mean(1)
    return {"ratio": float(pa.mean() / pb.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def run(cfg: dict[str, Any], device: str, writer: Any) -> dict[str, Any]:
    p = cfg["params"]
    seeds = [int(s) for s in p["seeds"]]
    check_not_held_out(*seeds)
    num_envs = int(p.get("num_envs", 128))
    window_s = float(p.get("window_s", 40.0))
    thr = float(p.get("wobble_threshold", 0.1))
    rules = {**DEFAULT_RULES, **(p.get("rules") or {})}
    specs = [dict(c) for c in p["controllers"]]
    names = [s["name"] for s in specs]
    ref = str(p["reference"])
    if ref not in names:
        raise ValueError(f"reference {ref!r} must be one of {names}")
    factories = {s["name"]: make_factory(s, device) for s in specs}
    gazebo = p.get("gazebo") or {}
    rng = np.random.default_rng(int(p.get("bootstrap_seed", 0)))
    n_boot = int(p.get("bootstrap", 2000))
    print(f"gazebo_check: {len(p['cells'])} cells x {len(seeds)} seeds x {num_envs} episodes, "
          f"controllers {names}, reference {ref}, device {device}", flush=True)

    report: dict[str, Any] = {"seeds": seeds, "num_envs": num_envs, "window_s": window_s,
                              "wobble_threshold": thr, "rules": rules, "reference": ref,
                              "controllers": specs, "gazebo": gazebo, "cells": []}
    t_start = time.perf_counter()
    flat: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cell in p["cells"]:
        env_cfg = StructureHoldConfig.from_dict(merge(p.get("base_env") or {}, cell.get("env")))
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for seed in seeds:
            for name in names:
                env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device)
                t0 = time.perf_counter()
                r = fly(env, factories[name](env), window_s, thr)
                for row in r:
                    writer.metric(seed, cell=cell["name"], controller=name, **row)
                rows[name].extend(r)
                d = describe(r)
                print(f"gazebo_check: {cell['name']} seed {seed} {name}: crash "
                      f"{d['crash_rate']:.1%}, p95 {d['p95_m_mean']:.3f} m, lift "
                      f"{d['lift_m_mean']:+.3f} m, wobble {d['wobble_rate']:.0%} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
        flat[cell["name"]] = rows
        report["cells"].append({
            "name": cell["name"], "label": cell.get("label", cell["name"]),
            "rough": bool(cell.get("rough", True)), "env": cell.get("env") or {},
            "controllers": {name: describe(rows[name]) for name in names},
            "vs_reference": {name: paired_p95_ratio(rows[name], rows[ref], rng, n_boot)
                             for name in names if name != ref},
        })

    rough = [c["name"] for c in report["cells"] if c["rough"]]
    clean = [c["name"] for c in report["cells"] if not c["rough"]]
    pooled = {name: describe([r for c in rough for r in flat[c][name]]) for name in names}
    report["pooled_rough"] = pooled
    verdicts = {}
    for name in names:
        if name == ref:
            continue
        a = [r for c in rough for r in flat[c][name]]
        b = [r for c in rough for r in flat[c][ref]]
        cmp = paired_p95_ratio(a, b, rng, n_boot)
        pa, pr = pooled[name], pooled[ref]
        checks = {
            "p95_lower": cmp["ci"][1] < 1.0,
            "lift_not_higher": pa["lift_m_mean"] <= pr["lift_m_mean"],
            "wobble_not_higher": pa["wobble_rate"] <= pr["wobble_rate"],
            "crash_ok": pa["crash_rate"] <= pr["crash_rate"] + rules["max_crash_increase"],
        }
        for c in clean:
            d = {x["name"]: x for x in report["cells"]}[c]["controllers"]
            checks[f"{c}_p95_ok"] = (d[name]["p95_m_mean"]
                                     <= d[ref]["p95_m_mean"] + rules["clean_p95_slack_m"])
        verdicts[name] = {"p95_ratio": cmp, "checks": checks, "beats": all(checks.values())}
    report["verdicts"] = verdicts
    report["elapsed_s"] = time.perf_counter() - t_start
    run_dir = Path(writer.run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (run_dir / "report.md").write_text(markdown(report))

    pr = pooled[ref]
    headline: dict[str, Any] = {"reference": ref, "episodes_per_cell": num_envs * len(seeds),
                                f"{ref}_p95_m_rough": pr["p95_m_mean"],
                                f"{ref}_lift_m_rough": pr["lift_m_mean"],
                                f"{ref}_wobble_rough": pr["wobble_rate"]}
    parts = []
    for name, v in verdicts.items():
        pa = pooled[name]
        headline.update({f"{name}_p95_m_rough": pa["p95_m_mean"],
                         f"{name}_lift_m_rough": pa["lift_m_mean"],
                         f"{name}_wobble_rough": pa["wobble_rate"],
                         f"{name}_beats_{ref}": v["beats"]})
        verdict = "beats" if v["beats"] else "does not beat"
        parts.append(f"{name} p95 {pa['p95_m_mean']:.2f} m, lift {pa['lift_m_mean']:+.2f} m, "
                     f"wobble {pa['wobble_rate']:.0%}: {verdict} {ref}")
    wording = (f"Gazebo-like check, rough-estimate cells pooled: {ref} p95 "
               f"{pr['p95_m_mean']:.2f} m, lift {pr['lift_m_mean']:+.2f} m, wobble "
               f"{pr['wobble_rate']:.0%}; " + "; ".join(parts))
    print(f"gazebo_check: {wording} ({report['elapsed_s'] / 60:.0f} min)", flush=True)
    return {"headline": headline, "vs_baseline": wording}


def markdown(report: dict[str, Any]) -> str:
    names = [c["name"] for c in report["controllers"]]
    gz = report.get("gazebo") or {}
    lines = [f"Episodes per cell and controller: {report['num_envs'] * len(report['seeds'])}; "
             f"lift and wobble over the last {report['window_s']:.0f} s; wobble = roll action "
             f"1.2-2.2 Hz rms > {report['wobble_threshold']}.", "",
             "| cell | controller | crash | p95 (m) | lift (m) | wobble "
             "| Gazebo p95 / lift / wobble |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in report["cells"]:
        for name in names:
            d = c["controllers"][name]
            g = (gz.get(name) or {}).get(c["name"])
            gtxt = (f"{g.get('p95_m', float('nan')):.2f} / {g.get('lift_m', float('nan')):+.2f} / "
                    f"{g.get('wobble', '-')}" if g else "-")
            lines.append(f"| {c['label']} | {name} | {d['crash_rate']:.1%} | "
                         f"{d['p95_m_mean']:.3f} | {d['lift_m_mean']:+.3f} | "
                         f"{d['wobble_rate']:.0%} | {gtxt} |")
    lines += ["", "Rough-estimate cells pooled:", "",
              "| controller | crash | p95 (m) | lift (m) | wobble | verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in names:
        d = report["pooled_rough"][name]
        v = report["verdicts"].get(name)
        vtxt = "reference" if v is None else (
            f"{'beats' if v['beats'] else 'does not beat'} {report['reference']} (p95 ratio "
            f"{v['p95_ratio']['ratio']:.2f}, CI {v['p95_ratio']['ci'][0]:.2f} to "
            f"{v['p95_ratio']['ci'][1]:.2f}; failed: "
            f"{', '.join(k for k, ok in v['checks'].items() if not ok) or 'none'})")
        lines.append(f"| {name} | {d['crash_rate']:.1%} | {d['p95_m_mean']:.3f} | "
                     f"{d['lift_m_mean']:+.3f} | {d['wobble_rate']:.0%} | {vtxt} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
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
