"""Score a controller on the Structure Hold env with ``eval/metrics.py``.

``score_controller`` rolls ``episodes`` full episodes of ``num_envs`` envs, records the true
trajectory of the first ``score_envs`` envs and scores every finished episode (the
``EpisodeRecorder`` from ``envs/structure_hold/run.py``), then aggregates to the headline
numbers the project reports: mean and worst p95 error, worst excursion toward the wall,
contact and crash rates. ``compare`` runs the same recipe for a policy and the cascaded PD
yardstick on identically seeded envs and words the comparison for ``summary.json`` and
``experiments/LOG.md``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from envs.structure_hold.baseline import CascadedPd
from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import OBS, StructureHoldEnv
from envs.structure_hold.run import EpisodeRecorder

HEADLINE_KEYS = ("p95_m_mean", "p95_m_worst", "rms_m_mean", "worst_toward_surface_m",
                 "contact_rate", "crash_rate", "heading_err_deg_mean")


def score_controller(env: StructureHoldEnv, controller: Callable[[torch.Tensor], torch.Tensor],
                     episodes: int, score_envs: int,
                     on_episode: Callable[[dict[str, Any]], None] | None = None,
                     ) -> dict[str, Any]:
    recorder = EpisodeRecorder(env, score_envs)
    obs, _ = env.get_observations()
    rows: list[dict[str, Any]] = []
    total_steps = int(episodes) * env.max_episode_length
    counts0 = dict(env.counts)
    t0 = time.perf_counter()
    heading_err_sum = 0.0
    for step in range(total_steps):
        actions = controller(obs)
        recorder.before_step(actions)
        obs, reward, dones, extras = env.step(actions)
        heading = obs[:, OBS["heading"]]
        heading_err_sum += float(torch.atan2(heading[:, 1], heading[:, 0]).abs().mean())
        if bool(dones.any()) and hasattr(controller, "reset_history"):
            controller.reset_history(obs, dones)  # the window restarts with the new episode
        for row in recorder.after_step(dones, reward, counts0):
            rows.append(row)
            if on_episode is not None:
                on_episode({"env_step": step, **row})
    elapsed = time.perf_counter() - t0

    def agg(key: str, fn) -> float:
        vals = [r[key] for r in rows
                if key in r and isinstance(r[key], float) and math.isfinite(r[key])]
        return float(fn(vals)) if vals else math.nan

    ended = {k: env.counts[k] - counts0[k] for k in env.counts}
    total_ended = sum(ended.values())
    return {
        "episodes_scored": len(rows),
        "p95_m_mean": agg("p95_m", np.mean),
        "p95_m_worst": agg("p95_m", np.max),
        "rms_m_mean": agg("rms_m", np.mean),
        "worst_toward_surface_m": agg("worst_toward_surface_m", np.max),
        "rate_roughness_rad_s2": agg("rate_roughness_rad_s2", np.mean),
        "contact_rate": ended["contact"] / total_ended if total_ended else math.nan,
        "heading_err_deg_mean": math.degrees(heading_err_sum / total_steps),  # to the wall
        "crash_rate": (1 - ended["timeout"] / total_ended) if total_ended else math.nan,
        "episodes_ended": total_ended,
        "env_steps_per_s": total_steps * env.num_envs / elapsed,
    }


def compare(env_cfg: StructureHoldConfig, policy: Callable[..., Any], *, num_envs: int,
            episodes: int, score_envs: int, seed: int, device: str, sim: str = "torch",
            on_episode: Callable[[str, dict[str, Any]], None] | None = None,
            ) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Score ``policy`` (a factory ``(env) -> controller``) and the PD yardstick on
    identically seeded envs. Returns (policy headline, baseline headline, wording)."""
    results = {}
    for name, make in (("policy", policy), ("baseline", CascadedPd)):
        env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device, sim=sim)
        cb = (lambda row, n=name: on_episode(n, row)) if on_episode else None
        results[name] = score_controller(env, make(env), episodes, score_envs, cb)
    pol, base = results["policy"], results["baseline"]
    return pol, base, vs_baseline(pol, base)


def vs_baseline(pol: dict[str, Any], base: dict[str, Any]) -> str:
    def pct(a: float, b: float) -> str:
        if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
            return "n/a"
        return f"{(a / b - 1) * 100:+.0f}%"

    return (
        f"policy p95 {pol['p95_m_mean']:.2f} m vs PD {base['p95_m_mean']:.2f} m "
        f"({pct(pol['p95_m_mean'], base['p95_m_mean'])}); crash rate "
        f"{pol['crash_rate']:.1%} vs {base['crash_rate']:.1%}; cascaded PD in the mock "
        "(envs/structure_hold/baseline.py), tuned PX4 comes with M1"
    )
