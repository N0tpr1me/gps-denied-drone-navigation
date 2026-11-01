"""Train a Route Follow policy with rsl_rl PPO on the mock simulator, export it, score it.

    python -m training.launch --config training/configs/route_follow_ppo.yaml --smoke --device cpu

The same recipe as ``train_ppo.py`` (Structure Hold), on ``envs/route_follow``: the
config's ``params`` give ``num_envs``, ``iterations``, ``num_steps_per_env``, ``history``,
``sim``, ``env`` (a ``RouteFollowConfig`` mapping), ``actor`` / ``critic`` / ``ppo`` (as
``train_ppo.DEFAULTS``), ``save_interval``, ``resume`` (a ``policy.pt`` or
``tb/model_<it>.pt``: continue for ``iterations`` more, so runs under the hour limit chain)
and ``eval`` (``num_envs``, ``episodes``, ``env`` merged over the training ``env``).

Outputs in the run folder: ``metrics.jsonl``, ``tb/``, ``policy.onnx`` + ``policy.json``
(format ``route_follow_policy/1``), ``policy.pt`` and ``summary.json``, whose headline
compares the policy with both classical yardsticks (``eval/route_follow.py``) on the eval
condition.
"""

from __future__ import annotations

import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from envs.route_follow.config import RouteFollowConfig
from envs.route_follow.env import NUM_ACTIONS, NUM_OBS, OBS, RouteFollowEnv
from eval import route_follow
from eval.gate1 import check_not_held_out
from training.policy_io import _jsonable, export_policy, load_controller
from training.route_ppo_env import RouteFollowVecEnv
from training.runlib import RunWriter, entry_args, load_yaml
from training.train_ppo import DEFAULTS as HOLD_DEFAULTS
from training.train_ppo import build_runner, build_train_cfg, merge

DEFAULTS: dict[str, Any] = {
    "num_envs": 1024,
    "iterations": 300,
    "num_steps_per_env": 48,
    "history": 3,
    "sim": "torch",
    "save_interval": 50,
    "init_at_random_ep_len": True,
    "actor": HOLD_DEFAULTS["actor"],
    "critic": HOLD_DEFAULTS["critic"],
    "ppo": HOLD_DEFAULTS["ppo"],
    "resume": None,
    "eval": {"num_envs": 128, "episodes": 1, "env": {}},
}


def policy_metadata(env_cfg: RouteFollowConfig, history: int, train_cfg: dict[str, Any],
                    extra: dict[str, Any]) -> dict[str, Any]:
    """``policy.json`` for a Route Follow policy (``training/policy_io.py`` reads
    ``input.history`` and ``input.num_obs`` from it)."""
    return {
        "format": "route_follow_policy/1",
        "input": {
            "name": "obs", "size": history * NUM_OBS, "history": history,
            "order": "oldest first; the newest observation is the last num_obs values",
            "num_obs": NUM_OBS,
            "layout": {name: [s.start, s.stop] for name, s in OBS.items()},
        },
        "output": {
            "name": "actions", "size": NUM_ACTIONS, "range": [-1.0, 1.0],
            "max_rate_rad_s": env_cfg.max_rate_rad_s, "max_thrust_n": env_cfg.quad.max_thrust,
            "yaw": "policy",
        },
        "rate_hz": round(1.0 / env_cfg.dt),
        "env_cfg": _jsonable(asdict(env_cfg)),
        "train_cfg": _jsonable(train_cfg),
        **extra,
    }


def train(cfg: dict[str, Any], run_dir: Path, device: str, writer: RunWriter) -> dict[str, Any]:
    p = merge(DEFAULTS, cfg.get("params"))
    seed = int(cfg["seed"])
    check_not_held_out(seed, seed + 1000)
    env_cfg = RouteFollowConfig.from_dict(p.get("env"))
    torch.manual_seed(seed)
    env = RouteFollowEnv(env_cfg, int(p["num_envs"]), seed=seed, device=device,
                         sim=str(p["sim"]))
    vec = RouteFollowVecEnv(env, history=int(p["history"]))
    train_cfg = build_train_cfg(p)
    runner = build_runner(vec, train_cfg, run_dir / "tb", device, writer, env_cfg.dt)
    n_params = sum(x.numel() for x in runner.alg.get_policy().parameters())
    print(f"ppo: route follow, {vec.num_envs} envs, {p['iterations']} iterations x "
          f"{p['num_steps_per_env']} steps, history {p['history']} ({vec.num_policy_obs} policy "
          f"obs, {vec.num_privileged_obs} privileged), actor {n_params:,} params, "
          f"device {device}", flush=True)
    if p.get("resume"):
        runner.load(str(Path(p["resume"])), map_location=device)
        runner.current_learning_iteration += 1
        print(f"ppo: resumed {p['resume']} at iteration {runner.current_learning_iteration}",
              flush=True)
    start = int(runner.current_learning_iteration)

    t0 = time.perf_counter()
    runner.learn(int(p["iterations"]), init_at_random_ep_len=bool(p["init_at_random_ep_len"]))
    train_s = time.perf_counter() - t0
    reward = statistics.mean(runner.logger.rewbuffer) if runner.logger.rewbuffer else float("nan")

    meta = policy_metadata(env_cfg, int(p["history"]), train_cfg, {
        "seed": seed, "iterations": int(p["iterations"]), "num_envs": vec.num_envs,
        "actor_params": n_params, "train_reward_mean": reward, "start_iteration": start,
        "resumed_from": p.get("resume"),
    })
    onnx_path = export_policy(runner, run_dir, meta)
    print(f"ppo: trained in {train_s:.0f} s, exported {onnx_path.name}", flush=True)

    e = p["eval"]
    eval_cfg = RouteFollowConfig.from_dict(merge(p.get("env") or {}, e.get("env")))
    res = route_follow.compare(
        eval_cfg, lambda ev: load_controller(onnx_path, ev.num_envs, device),
        num_envs=int(e["num_envs"]), episodes=int(e["episodes"]), seed=seed + 1000,
        device=device, sim=str(p["sim"]))
    for name, h in res.items():
        writer.metric(start + int(p["iterations"]), controller=name, **h)
    wording = route_follow.vs_baseline(res)
    print(f"ppo: eval {wording}", flush=True)
    headline = {"train_reward_mean": reward, "train_s": train_s, "actor_params": n_params,
                **{f"{name}_{k}": h[k] for name, h in res.items()
                   for k in route_follow.HEADLINE_KEYS}}
    return {"headline": headline, "vs_baseline": wording}


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        result = train(cfg, Path(args.run_dir), args.device, writer)
        writer.summary("ok", headline=result["headline"], vs_baseline=result["vs_baseline"])
        return 0
    except Exception as e:
        traceback.print_exc()
        writer.summary("failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
