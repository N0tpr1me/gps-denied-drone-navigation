"""Gate 1 evaluation protocol: a held-out test set, the PD yardstick on the same episodes,
confidence intervals and the pass rules of ``docs/gate1.md``.

    python -m training.launch --config eval/configs/gate1.yaml --device cpu
    python -m training.launch --config eval/configs/gate1_stress.yaml --device cpu

What makes it a test set:

* **Seeds** ``HELD_OUT_SEEDS`` (9001 to 9003) build every env: hold points, start poses, wind,
  gust schedule, estimate corruption and airframe draws. No training run, checkpoint
  selection or release eval uses a seed in ``RESERVED_SEEDS`` (``training/train_ppo.py``
  refuses them), so the policy has never flown these episodes.
* **Conditions** are fixed cells, not the training distribution: mean wind exactly 0, 4 or
  8 m/s (direction drawn), the hold point exactly 1 m or 2 m from the wall, and a gust
  schedule that puts both gusts inside the 20 s episode (training draws gust starts up to
  50 s, so most training episodes see none). The estimate corruption is the standard one.
* **One episode per env, paired.** Each env flies exactly its first episode; the policy and
  the PD fly the same episodes (same seed, and nothing they do changes a draw before the
  first episode ends), so differences are measured episode by episode.

Per cell and controller: crash rate with a Wilson 95 % interval, mean per-episode p95 error
(``eval.metrics.compute_metrics`` on the true trajectory, every episode scored) with a
bootstrap 95 % interval, worst excursion toward the wall, contacts, heading error. Policy
against PD: paired bootstrap 95 % intervals of the p95 ratio and of the crash-rate
difference. The verdicts follow ``docs/gate1.md``; ``rules`` in the config sets them.

Inputs (``params``): ``controllers`` (list of ``{name, policy, fallback, yaw_guard}``; the PD
yardstick is always added as ``pd``), ``gate_controller`` (the one judged), ``seeds``,
``num_envs`` (episodes per cell and seed), ``base_env`` (a ``StructureHoldConfig`` mapping
applied to every cell), ``cells`` (list of ``{name, env, envelope, headline}``; ``env`` is
merged over ``base_env``), ``bootstrap`` (resamples), ``rules``.

Outputs: ``metrics.jsonl`` (one row per scored episode), ``report.json`` (every number),
``report.md`` (the tables for ``docs/gate1.md``) and ``summary.json`` with the verdict.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.structure_hold.baseline import CascadedPd
from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import OBS, TERM_REASONS, StructureHoldEnv
from envs.structure_hold.fallback import FallbackRule, StaleFallback, YawGuard
from eval.metrics import Surface, Trace, compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent

# The Gate 1 test seeds, and the block no training, selection or release eval may touch.
HELD_OUT_SEEDS = (9001, 9002, 9003)
RESERVED_SEEDS = range(9000, 10000)

DEFAULT_RULES = {
    "p95_headline_reduction": 0.30,  # headline cell: policy p95 at least 30 % below the PD's
    "max_pooled_crash_rate": 0.05,  # nominal cells pooled
}

TIMEOUT = TERM_REASONS.index("timeout")


def check_not_held_out(*seeds: int) -> None:
    """Refuse a seed from the Gate 1 test block (training, selection, quick evals)."""
    bad = sorted(int(s) for s in seeds if int(s) in RESERVED_SEEDS)
    if bad:
        raise ValueError(f"seeds {bad} are in the Gate 1 held-out block "
                         f"{RESERVED_SEEDS.start}-{RESERVED_SEEDS.stop - 1}; pick another")


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ---------------------------------------------------------------- flying


def fly(env: StructureHoldEnv, controller: Callable[[torch.Tensor], torch.Tensor],
        ) -> list[dict[str, Any]]:
    """Fly every env's first episode with ``controller``; one metrics row per env."""
    cfg, n, steps = env.cfg, env.num_envs, env.max_episode_length
    hold = env.hold_point.cpu().numpy().copy()
    gusts = [env.gust_windows(i) for i in range(n)]
    mass = env.mass_scale.cpu().numpy().copy()
    pos = np.empty((steps + 1, n, 3), dtype=np.float32)
    act = np.empty((steps, n, 4), dtype=np.float32)
    pos[0] = env.state.pos.cpu().numpy()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    end = np.full(n, steps, dtype=np.int64)
    reason = np.full(n, TIMEOUT, dtype=np.int64)
    yaw_err_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    obs, _ = env.get_observations()
    k = 0
    for k in range(steps):
        a = controller(obs)
        act[k] = a.detach().cpu().numpy()
        obs, _, done, _ = env.step(a)
        if bool(done.any()) and hasattr(controller, "reset_history"):
            controller.reset_history(obs, done)
        still = alive & ~done
        heading = obs[:, OBS["heading"]]
        yaw_err_sum += torch.where(still, torch.atan2(heading[:, 1], heading[:, 0]).abs(),
                                   torch.zeros_like(heading[:, 0])).double()
        p = env.state.pos.cpu().numpy()
        s = still.cpu().numpy()
        pos[k + 1] = np.where(s[:, None], p, pos[k])  # an ended episode keeps its last point
        newly = (alive & done).cpu().numpy()
        if newly.any():
            end[newly] = k + 1
            reason[newly] = env.term_reason.cpu().numpy()[newly]
        alive = still
        if not bool(alive.any()):
            break

    surface = Surface(point=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]))
    yaw_err = (yaw_err_sum.cpu().numpy() / np.maximum(end - 1, 1))
    rows = []
    for i in range(n):
        length = int(end[i])
        t = np.arange(length + 1) * env.dt
        row: dict[str, Any] = {
            "env": i, "length_s": float(length * env.dt), "crash": bool(reason[i] != TIMEOUT),
            "reason": TERM_REASONS[int(reason[i])], "hold_distance_m": float(-hold[i, 0]),
            "heading_err_deg": math.degrees(float(yaw_err[i])), "mass_scale": float(mass[i]),
        }
        if length >= 2:
            trace = Trace(t=t, pos=pos[: length + 1, i].astype(np.float64), pos_sp=hold[i],
                          cmd_t=t[:-1], rates_cmd=act[:length, i, :3] * cfg.max_rate_rad_s,
                          thrust_cmd=0.5 * (act[:length, i, 3] + 1.0))
            m = compute_metrics(trace, surface, gusts=[g for g in gusts[i] if g[0] < t[-1]],
                                contact_radius_m=cfg.contact_radius_m)
            row.update({k: m[k] for k in ("p95_m", "rms_m", "max_m", "worst_toward_surface_m",
                                          "contacts", "rate_roughness_rad_s2")})
        rows.append(row)
    return rows


def make_factory(spec: dict[str, Any], device: str) -> Callable[[StructureHoldEnv], Any]:
    """``(env) -> controller`` for a ``controllers`` entry (or the PD)."""
    if spec.get("policy") is None:
        return CascadedPd
    from training.policy_io import load_controller, read_metadata

    path = Path(spec["policy"])
    path = path if path.is_absolute() else REPO_ROOT / path
    meta = read_metadata(path)
    rule = FallbackRule.from_dict(spec["fallback"]) if spec.get("fallback") else None

    def factory(env: StructureHoldEnv):
        out = meta["output"]
        if (abs(out["max_rate_rad_s"] - env.cfg.max_rate_rad_s) > 1e-9
                or abs(out["max_thrust_n"] - env.cfg.quad.max_thrust) > 1e-9):
            raise ValueError(f"{path.name} scales actions for {out['max_rate_rad_s']} rad/s and "
                             f"{out['max_thrust_n']} N, the protocol env for "
                             f"{env.cfg.max_rate_rad_s} rad/s and {env.cfg.quad.max_thrust} N")
        ctl = load_controller(path, env.num_envs, device, meta=meta)
        if rule is not None:
            ctl = StaleFallback(env, ctl, rule)
        return YawGuard(env, ctl) if spec.get("yaw_guard") else ctl

    return factory


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _p95(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([r.get("p95_m", math.nan) for r in rows], dtype=float)


def _crash(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([r["crash"] for r in rows], dtype=float)


def describe(rows: list[dict[str, Any]], idx: np.ndarray) -> dict[str, Any]:
    """Headline numbers of one controller in one cell, with 95 % intervals."""
    p95, crash = _p95(rows), _crash(rows)
    ok = np.isfinite(p95)
    boot = np.nanmean(np.where(ok[idx], p95[idx], np.nan), axis=1)
    worst = [r.get("worst_toward_surface_m", math.nan) for r in rows]
    reasons: dict[str, int] = {}
    for r in rows:
        if r["crash"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    k, n = int(crash.sum()), len(rows)
    return {
        "episodes": n,
        "crash_rate": k / n,
        "crash_ci": wilson(k, n),
        "crash_reasons": reasons,
        "p95_m_mean": float(np.nanmean(p95)),
        "p95_ci": tuple(float(x) for x in np.nanpercentile(boot, [2.5, 97.5])),
        "p95_m_median": float(np.nanmedian(p95)),
        "worst_toward_surface_m": float(np.nanmax(worst)),
        "contact_rate": sum(1 for r in rows if r["reason"] == "contact") / n,
        "heading_err_deg_mean": float(np.mean([r["heading_err_deg"] for r in rows])),
    }


def paired(pol: list[dict[str, Any]], base: list[dict[str, Any]],
           idx: np.ndarray) -> dict[str, Any]:
    """Policy vs PD on the same episodes: p95 ratio - 1 and crash-rate difference."""
    a, b = _p95(pol), _p95(base)
    ca, cb = _crash(pol), _crash(base)

    def ratio(ii: Any) -> Any:
        return np.nanmean(a[ii], axis=-1) / np.nanmean(b[ii], axis=-1) - 1.0

    r_boot, d_boot = ratio(idx), (ca[idx] - cb[idx]).mean(axis=1)
    return {
        "p95_change": float(ratio(np.arange(len(a)))),
        "p95_change_ci": tuple(float(x) for x in np.nanpercentile(r_boot, [2.5, 97.5])),
        "crash_diff": float(ca.mean() - cb.mean()),
        "crash_diff_ci": tuple(float(x) for x in np.percentile(d_boot, [2.5, 97.5])),
    }


def per_seed(rows_by_seed: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    return [{"p95_m_mean": float(np.nanmean(_p95(r))), "crash_rate": float(_crash(r).mean())}
            for r in rows_by_seed]


# ---------------------------------------------------------------- verdicts


def judge(cell: dict[str, Any], cmp: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """The per-cell rules of docs/gate1.md for the judged controller against the PD."""
    out = {
        # Hold: significantly tighter than the PD (whole p95-ratio interval below 0).
        "hold": cmp["p95_change_ci"][1] < 0.0,
        # Safety: not significantly more crashes than the PD (difference interval reaches 0).
        "safety": cmp["crash_diff_ci"][0] <= 0.0,
    }
    if cell.get("headline"):
        out["headline"] = cmp["p95_change"] <= -float(rules["p95_headline_reduction"])
    out["pass"] = all(out.values())
    return out


# ---------------------------------------------------------------- the run


def run(cfg: dict[str, Any], device: str, writer: Any) -> dict[str, Any]:
    p = cfg["params"]
    seeds = [int(s) for s in p.get("seeds", HELD_OUT_SEEDS)]
    num_envs = int(p.get("num_envs", 1024))
    rules = {**DEFAULT_RULES, **(p.get("rules") or {})}
    specs = [dict(c) for c in p["controllers"]] + [{"name": "pd", "policy": None}]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"controller names must be unique: {names}")
    gate = str(p.get("gate_controller", specs[0]["name"]))
    if gate not in names or gate == "pd":
        raise ValueError(f"gate_controller {gate!r} must be one of the policies {names[:-1]}")
    factories = {s["name"]: make_factory(s, device) for s in specs}
    rng = np.random.default_rng(int(p.get("bootstrap_seed", 0)))
    n_boot = int(p.get("bootstrap", 2000))
    print(f"gate1: {len(p['cells'])} cells x {len(seeds)} seeds x {num_envs} episodes, "
          f"controllers {names}, judged {gate}, device {device}", flush=True)

    report: dict[str, Any] = {"seeds": seeds, "num_envs": num_envs, "rules": rules,
                              "gate_controller": gate, "controllers": specs, "cells": []}
    t_start = time.perf_counter()
    for cell in p["cells"]:
        env_cfg = StructureHoldConfig.from_dict(merge(p.get("base_env") or {}, cell.get("env")))
        rows: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in names}
        for seed in seeds:
            for name in names:
                env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device)
                t0 = time.perf_counter()
                r = fly(env, factories[name](env))
                rows[name].append(r)
                for row in r:
                    writer.metric(seed, cell=cell["name"], controller=name, **row)
                print(f"gate1: {cell['name']} seed {seed} {name}: crash "
                      f"{np.mean(_crash(r)):.1%}, p95 {np.nanmean(_p95(r)):.3f} m "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
        n_total = num_envs * len(seeds)
        idx = rng.integers(0, n_total, size=(n_boot, n_total))
        flat = {name: [x for r in rows[name] for x in r] for name in names}
        entry: dict[str, Any] = {
            "name": cell["name"], "label": cell.get("label", cell["name"]),
            "envelope": bool(cell.get("envelope", True)), "headline": bool(cell.get("headline")),
            "env": cell.get("env") or {},
            "controllers": {name: {**describe(flat[name], idx),
                                   "per_seed": per_seed(rows[name])} for name in names},
            "vs_pd": {name: paired(flat[name], flat["pd"], idx) for name in names
                      if name != "pd"},
        }
        entry["verdict"] = judge(entry, entry["vs_pd"][gate], rules)
        report["cells"].append(entry)
        c, g = entry["controllers"], entry["vs_pd"][gate]
        print(f"gate1: {cell['name']}: {gate} p95 {c[gate]['p95_m_mean']:.3f} vs PD "
              f"{c['pd']['p95_m_mean']:.3f} m ({g['p95_change']:+.0%}, CI "
              f"{g['p95_change_ci'][0]:+.0%} to {g['p95_change_ci'][1]:+.0%}); crash "
              f"{c[gate]['crash_rate']:.1%} vs {c['pd']['crash_rate']:.1%} (diff CI "
              f"{g['crash_diff_ci'][0]:+.1%} to {g['crash_diff_ci'][1]:+.1%}); "
              f"{'PASS' if entry['verdict']['pass'] else 'FAIL'}", flush=True)

    # Pooled over the envelope cells.
    env_cells = [c for c in report["cells"] if c["envelope"]]

    def pooled(name: str, key: str) -> float:
        tot = sum(c["controllers"][name]["episodes"] for c in env_cells)
        return sum(c["controllers"][name][key] * c["controllers"][name]["episodes"]
                   for c in env_cells) / tot

    pooled_crash = pooled(gate, "crash_rate")
    verdict = {
        "cells_pass": sum(c["verdict"]["pass"] for c in env_cells),
        "cells": len(env_cells),
        "pooled_crash_ok": pooled_crash <= float(rules["max_pooled_crash_rate"]),
    }
    verdict["pass"] = verdict["cells_pass"] == verdict["cells"] and verdict["pooled_crash_ok"]
    report["verdict"] = verdict
    report["pooled"] = {name: {"crash_rate": pooled(name, "crash_rate"),
                               "p95_m_mean": pooled(name, "p95_m_mean")} for name in names}
    report["elapsed_s"] = time.perf_counter() - t_start
    run_dir = Path(writer.run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (run_dir / "report.md").write_text(markdown(report))

    pg, pd = report["pooled"][gate], report["pooled"]["pd"]
    headline = {
        "gate_controller": gate, "seeds": len(seeds), "episodes_per_cell": num_envs * len(seeds),
        "verdict": "PASS" if verdict["pass"] else "FAIL",
        "cells_pass": f"{verdict['cells_pass']}/{verdict['cells']}",
        f"{gate}_p95_m_pooled": pg["p95_m_mean"], f"{gate}_crash_pooled": pg["crash_rate"],
        "pd_p95_m_pooled": pd["p95_m_mean"], "pd_crash_pooled": pd["crash_rate"],
    }
    for c in report["cells"]:
        if c["headline"]:
            headline[f"{c['name']}_p95_change"] = c["vs_pd"][gate]["p95_change"]
    wording = (f"{gate} vs cascaded PD on the same held-out episodes ({len(env_cells)} cells x "
               f"{num_envs * len(seeds)}): pooled p95 {pg['p95_m_mean']:.2f} vs "
               f"{pd['p95_m_mean']:.2f} m, crash {pg['crash_rate']:.1%} vs {pd['crash_rate']:.1%}; "
               f"{verdict['cells_pass']}/{verdict['cells']} cells pass; tuned PX4 comes with M1")
    print(f"gate1: {wording}; verdict {headline['verdict']} "
          f"({report['elapsed_s'] / 60:.0f} min)", flush=True)
    return {"headline": headline, "vs_baseline": wording}


def markdown(report: dict[str, Any]) -> str:
    """The report as Markdown tables, ready to paste into docs/gate1.md."""
    gate = report["gate_controller"]
    names = [s["name"] for s in report["controllers"]]
    lines = [
        f"Seeds {report['seeds']}, {report['num_envs']} episodes per cell and seed; "
        f"judged: `{gate}`. Intervals are 95 %.",
        "",
        "| Cell | Controller | p95 (m) | Crash | Worst toward wall (m) | Heading (deg) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in report["cells"]:
        for name in names:
            d = c["controllers"][name]
            lines.append(
                f"| {c['label']} | {name} | {d['p95_m_mean']:.3f} "
                f"[{d['p95_ci'][0]:.3f}, {d['p95_ci'][1]:.3f}] | {d['crash_rate']:.1%} "
                f"[{d['crash_ci'][0]:.1%}, {d['crash_ci'][1]:.1%}] | "
                f"{d['worst_toward_surface_m']:.2f} | {d['heading_err_deg_mean']:.1f} |")
    lines += ["", f"| Cell | {gate} p95 vs PD | Crash difference | Hold | Safety | Verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in report["cells"]:
        g, v = c["vs_pd"][gate], c["verdict"]
        extra = "" if "headline" not in v else (" (headline " +
                                                ("ok" if v["headline"] else "missed") + ")")
        lines.append(
            f"| {c['label']}{'' if c['envelope'] else ' (beyond envelope)'} | "
            f"{g['p95_change']:+.0%} [{g['p95_change_ci'][0]:+.0%}, {g['p95_change_ci'][1]:+.0%}]"
            f" | {g['crash_diff']:+.1%} [{g['crash_diff_ci'][0]:+.1%}, "
            f"{g['crash_diff_ci'][1]:+.1%}] | {'yes' if v['hold'] else 'no'} | "
            f"{'yes' if v['safety'] else 'no'} | {'PASS' if v['pass'] else 'FAIL'}{extra} |")
    v = report["verdict"]
    lines += ["", f"Overall: {v['cells_pass']}/{v['cells']} envelope cells pass, pooled crash "
              f"rate {'within' if v['pooled_crash_ok'] else 'above'} the limit: "
              f"**{'PASS' if v['pass'] else 'FAIL'}**.", ""]
    return "\n".join(lines)


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
