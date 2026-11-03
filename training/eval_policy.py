"""Score an exported policy against the cascaded PD yardstick with eval/metrics.py.

    python -m training.launch --config training/configs/structure_hold_eval.yaml \
        --smoke --device cpu

Inputs (``params``): ``policy`` (path to ``policy.onnx`` or ``policy.pt``, with its
``policy.json`` next to it; relative to the repo root), ``num_envs``, ``episodes``,
``score_envs``, ``sim``, ``seeds`` (a list; each seed builds identically seeded envs for the
policy and the PD, so both fly the same episodes: same hold points, wind, jumps and
dropouts; defaults to the config's ``seed``) and ``env`` (a ``StructureHoldConfig``
mapping: evaluate on harder wind or worse estimates than training by changing it here).
Leave ``env`` out to evaluate on the config the policy was trained with (read from
``policy.json``). Use ``eval.env`` from the training config to score on the standard
condition rather than the hardened training one.

Outputs: ``metrics.jsonl`` with one line per scored episode of each controller and one
``summary`` line per seed, ``summary.json`` with the pooled headlines over all seeds (crash
rates with their binomial standard error) and the wording for ``experiments/LOG.md``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.fallback import FallbackRule, StaleFallback, YawGuard
from training import evaluate
from training.policy_io import load_controller, read_metadata
from training.runlib import RunWriter, entry_args, load_yaml, merge

REPO_ROOT = Path(__file__).resolve().parent.parent


def pool(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-seed headlines: rates weighted by episodes ended, means by episodes scored,
    worst values as the max, plus the binomial standard error of the crash rate."""
    ended = sum(r["episodes_ended"] for r in results)
    scored = sum(r["episodes_scored"] for r in results)

    def rate(key: str) -> float:
        return (sum(r[key] * r["episodes_ended"] for r in results) / ended) if ended else math.nan

    def mean(key: str) -> float:
        vals = [(r[key], r["episodes_scored"]) for r in results
                if r["episodes_scored"] and math.isfinite(r[key])]
        n = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / n if n else math.nan

    crash = rate("crash_rate")
    return {
        "episodes_ended": ended,
        "episodes_scored": scored,
        "crash_rate": crash,
        "crash_rate_se": math.sqrt(crash * (1 - crash) / ended) if ended else math.nan,
        "contact_rate": rate("contact_rate"),
        "p95_m_mean": mean("p95_m_mean"),
        "heading_err_deg_mean": mean("heading_err_deg_mean"),
        "rms_m_mean": mean("rms_m_mean"),
        "p95_m_worst": max((r["p95_m_worst"] for r in results), default=math.nan),
        "worst_toward_surface_m": max((r["worst_toward_surface_m"] for r in results),
                                      default=math.nan),
    }


def run(cfg: dict[str, Any], device: str, writer: RunWriter) -> dict[str, Any]:
    p = cfg["params"]
    policy_path = Path(p["policy"])
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    meta = read_metadata(policy_path)
    # ``env`` is laid over the policy's own training env (policy.json env_cfg), as
    # train_ppo's eval.env is, so a config that restates only the estimate rates keeps the
    # airframe, tick and history it trained on.
    env_cfg = StructureHoldConfig.from_dict(merge(meta["env_cfg"], p.get("env")))
    num_envs, episodes = int(p.get("num_envs", 256)), int(p.get("episodes", 1))
    seeds = [int(s) for s in (p.get("seeds") or [cfg["seed"]])]
    print(f"eval: {policy_path.name} ({meta['input']['history']}-step history), {num_envs} envs "
          f"x {episodes} episode(s) of {env_cfg.episode_length_s} s, seeds {seeds}, "
          f"device {device}", flush=True)
    # Optional brake-and-hover fallback around the policy (envs/structure_hold/fallback.py):
    # `fallback: {threshold_s: 1.0}` in params; null leaves the policy bare.
    rule = FallbackRule.from_dict(p["fallback"]) if p.get("fallback") else None
    wrappers: list[StaleFallback] = []

    # `yaw_guard: true` scores what the PX4 module flies when it holds the heading itself.
    yaw_guard = bool(p.get("yaw_guard", False))

    def make_policy(env):
        ctl = load_controller(policy_path, env.num_envs, device)
        if rule is not None:
            wrappers.append(StaleFallback(env, ctl, rule))
            ctl = wrappers[-1]
        return YawGuard(env, ctl) if yaw_guard else ctl

    if rule is not None:
        print(f"eval: fallback {rule}", flush=True)
    per_seed: dict[str, list[dict[str, Any]]] = {"policy": [], "baseline": []}
    for seed in seeds:
        pol, base, wording = evaluate.compare(
            env_cfg, make_policy,
            num_envs=num_envs, episodes=episodes, score_envs=int(p.get("score_envs", 32)),
            seed=seed, device=device, sim=str(p.get("sim", "torch")),
            on_episode=lambda name, row, seed=seed: writer.metric(seed, controller=name, **row),
        )
        print(f"eval: seed {seed}: {wording}", flush=True)
        for name, r in (("policy", pol), ("baseline", base)):
            per_seed[name].append(r)
            writer.metric(seed, summary=name, **{k: r[k] for k in evaluate.HEADLINE_KEYS},
                          episodes_ended=r["episodes_ended"])
    pol, base = pool(per_seed["policy"]), pool(per_seed["baseline"])
    if pol["episodes_scored"] == 0:
        raise RuntimeError("no episode finished, so nothing was scored; run longer")
    wording = evaluate.vs_baseline(pol, base) + (
        f"; {pol['episodes_ended']} identical episodes each over {len(seeds)} seed(s), "
        f"crash rate s.e. {pol['crash_rate_se']:.1%} / {base['crash_rate_se']:.1%}")
    print(f"eval: pooled: {wording}", flush=True)
    keys = (*evaluate.HEADLINE_KEYS, "crash_rate_se", "episodes_ended")
    headline = {
        "policy": str(policy_path.relative_to(REPO_ROOT) if policy_path.is_relative_to(REPO_ROOT)
                      else policy_path),
        "seeds": len(seeds),
        "yaw_guard": yaw_guard,
        **({"fallback_threshold_s": rule.threshold_s,
            "fallback_active_fraction": float(np.mean([w.active_fraction for w in wrappers]))}
           if rule is not None else {}),
        **{f"policy_{k}": pol[k] for k in keys},
        **{f"pd_{k}": base[k] for k in keys},
    }
    # The env the policy was scored in, overrides laid in, so a run folder says what it flew.
    env_dict = json.loads(json.dumps(dataclasses.asdict(env_cfg), default=str))
    return {"headline": headline, "vs_baseline": wording, "env_cfg": env_dict}


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        torch.set_num_threads(max(1, torch.get_num_threads()))
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
