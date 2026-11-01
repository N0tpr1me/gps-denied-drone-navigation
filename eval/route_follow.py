"""Score controllers on Route Follow (``envs/route_follow``), on the TRUE position.

Per finished episode (``docs/route-follow.md``, "How it is scored"):

- ``cross_track_p95_m`` / ``cross_track_max_m``: distance from the route centreline;
- ``track_p95_m``: distance from the moving target;
- ``min_clearance_m``: closest approach to any wall, floor or ceiling (centre to surface);
- ``completed``: the episode ran to its end and finished within ``COMPLETE_TOL_M`` of the
  last waypoint (the target stops there ``route.end_hold_s`` before the end);
- ``crashed`` and the reason (``TERM_REASONS``).

``score_controller`` runs ``episodes`` episode-lengths of steps; envs that crash start a new
route at once, and only finished episodes are scored. ``compare`` scores a policy and the
classical yardsticks (``envs/route_follow/baseline.py``: PD on the estimate, and PD with
range centring) on identically seeded envs.

As a run entrypoint (``python -m training.launch --config eval/configs/route_follow_baseline.yaml``)
it scores the yardsticks, and a policy when ``params.policy`` names one, on each condition
in ``params.sets`` (corridors and tunnels, with and without rangefinders).
"""

from __future__ import annotations

import copy
import math
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.route_follow.baseline import RoutePd, RoutePdGains
from envs.route_follow.config import RouteFollowConfig
from envs.route_follow.env import TERM_REASONS, RouteFollowEnv
from training.runlib import RunWriter, entry_args, load_yaml

COMPLETE_TOL_M = 0.5
HEADLINE_KEYS = ("cross_track_p95_m", "cross_track_worst_m", "track_p95_m",
                 "min_clearance_m", "completion_rate", "crash_rate", "contact_rate")


def score_controller(env: RouteFollowEnv, controller: Callable[[torch.Tensor], torch.Tensor],
                     episodes: int = 1) -> dict[str, Any]:
    n, t_max = env.num_envs, env.max_episode_length
    total = int(episodes) * t_max
    xt = torch.zeros(n, t_max, dtype=env.dtype, device=env.device)
    tr = torch.zeros_like(xt)
    clear = torch.full((n,), math.inf, dtype=env.dtype, device=env.device)
    k = torch.zeros(n, dtype=torch.long, device=env.device)
    rows: list[dict[str, Any]] = []
    obs, _ = env.get_observations()
    env_ids = torch.arange(n, device=env.device)
    for _ in range(total):
        obs, _, dones, _ = env.step(controller(obs))
        idx = k.clamp(max=t_max - 1)
        xt[env_ids, idx] = env.step_cross_track
        tr[env_ids, idx] = env.step_track_err
        clear = torch.minimum(clear, env.step_clearance)
        k += 1
        if bool(dones.any()):
            if hasattr(controller, "reset_history"):
                controller.reset_history(obs, dones)
            for i in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
                m = int(k[i])
                reason = TERM_REASONS[int(env.term_reason[i])]
                arrived = float(env.step_to_end[i]) <= COMPLETE_TOL_M
                a, b = xt[i, :m].cpu().numpy(), tr[i, :m].cpu().numpy()
                rows.append({
                    "cross_track_p95_m": float(np.percentile(a, 95)),
                    "cross_track_max_m": float(a.max()),
                    "track_p95_m": float(np.percentile(b, 95)),
                    "min_clearance_m": float(clear[i]),
                    "completed": reason == "timeout" and arrived,
                    "crashed": reason != "timeout",
                    "reason": reason,
                    "length_s": m * env.dt,
                })
            clear[dones] = math.inf
            k[dones] = 0
    return headline(rows)


def headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return float(np.mean(vals)) if vals else math.nan

    flown = [r for r in rows if not r["crashed"]]
    return {
        "episodes": len(rows),
        # Cross-track and tracking over the flights that did not crash (a crash's error is in
        # crash_rate); the clearance over all of them.
        "cross_track_p95_m": float(np.mean([r["cross_track_p95_m"] for r in flown]))
        if flown else math.nan,
        "cross_track_worst_m": float(np.max([r["cross_track_max_m"] for r in flown]))
        if flown else math.nan,
        "track_p95_m": float(np.mean([r["track_p95_m"] for r in flown])) if flown else math.nan,
        "min_clearance_m": float(np.min([r["min_clearance_m"] for r in rows])) if rows
        else math.nan,
        "completion_rate": mean("completed"),
        "crash_rate": mean("crashed"),
        "contact_rate": float(np.mean([r["reason"] == "contact" for r in rows])) if rows
        else math.nan,
    }


YARDSTICKS: dict[str, Callable[[RouteFollowEnv], Any]] = {
    "pd": lambda env: RoutePd(env),
    "pd_centring": lambda env: RoutePd(env, RoutePdGains(centering=1.0)),
}


def compare(cfg: RouteFollowConfig, policy: Callable[[RouteFollowEnv], Any] | None, *,
            num_envs: int, episodes: int, seed: int, device: str = "cpu",
            sim: str = "torch") -> dict[str, dict[str, Any]]:
    """Headline per controller (``policy`` first when given, then the yardsticks)."""
    makers = ({"policy": policy} if policy is not None else {}) | YARDSTICKS
    out = {}
    for name, make in makers.items():
        env = RouteFollowEnv(cfg, num_envs, seed=seed, device=device, sim=sim)
        out[name] = score_controller(env, make(env), episodes)
    return out


def vs_baseline(results: dict[str, dict[str, Any]]) -> str:
    """One line: each controller's cross-track p95, completion and crash rate."""
    parts = [f"{name} xtrack p95 {h['cross_track_p95_m']:.2f} m, "
             f"done {h['completion_rate']:.0%}, crash {h['crash_rate']:.0%}"
             for name, h in results.items()]
    return "; ".join(parts) + (" (mock, vs the cascaded PD yardsticks, "
                               "envs/route_follow/baseline.py)")


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, v in (over or {}).items():
        out[key] = merge(out[key], v) if isinstance(v, dict) and isinstance(out.get(key),
                                                                            dict) else v
    return out


def run(cfg: dict[str, Any], device: str, writer: RunWriter) -> dict[str, Any]:
    p = cfg["params"]
    seed = int(cfg["seed"])
    policy = None
    if p.get("policy"):
        from training.policy_io import load_controller

        path = Path(p["policy"])
        policy = lambda env: load_controller(path, env.num_envs, device)  # noqa: E731
    headline_out: dict[str, Any] = {}
    lines = []
    for i, s in enumerate(p["sets"]):
        env_cfg = RouteFollowConfig.from_dict(merge(p.get("env") or {}, s.get("env")))
        res = compare(env_cfg, policy, num_envs=int(p["num_envs"]),
                      episodes=int(p.get("episodes", 1)), seed=seed + i, device=device,
                      sim=str(p.get("sim", "torch")))
        for ctl, h in res.items():
            writer.metric(i, set=s["name"], controller=ctl, **h)
            for key in HEADLINE_KEYS:
                headline_out[f"{s['name']}/{ctl}/{key}"] = h[key]
        line = f"{s['name']}: {vs_baseline(res)}"
        print(f"route_follow: {line}", flush=True)
        lines.append(line)
    return {"headline": headline_out, "vs_baseline": " | ".join(lines)}


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        result = run(cfg, args.device, writer)
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
