"""Run the Structure Hold env standalone and score it with eval/metrics.py.

    python -m training.launch --config training/configs/structure_hold_rollout.yaml \
        --smoke --device cpu

or directly, with the same arguments every entrypoint takes:

    python -m envs.structure_hold.run --run-dir runs/x --config training/configs/\
structure_hold_rollout.yaml --smoke --device cpu

Inputs: the config's ``params`` block: ``num_envs``, ``episodes`` (full episodes per env),
``controller`` (``baseline``: the cascaded PD in ``baseline.py``; ``hover``: constant hover
thrust, no rates; ``random``: uniform actions), ``score_envs`` (how many envs to score with
the metrics), ``sim`` (``torch`` or ``isaac``) and ``env`` (a ``StructureHoldConfig``
mapping: wind, estimate corruption, reward weights, airframe).

Outputs, in the run folder: ``metrics.jsonl`` with one line per finished episode of a
scored env (the eval/metrics numbers plus reward and why it ended), ``summary.json`` with
the headline (mean and worst p95 error, contact rate, recovery, throughput), and the log on
stdout. Any failure writes ``status: failed`` with the error and exits 1.
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from envs.structure_hold.baseline import CascadedPd
from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import StructureHoldEnv
from eval.metrics import Surface, Trace, compute_metrics
from training.runlib import RunWriter, entry_args, load_yaml

CONTROLLERS = ("baseline", "hover", "random")


def make_controller(name: str, env: StructureHoldEnv, seed: int):
    if name == "baseline":
        return CascadedPd(env)
    if name == "hover":
        hover = torch.zeros(env.num_envs, 4, device=env.device)
        hover[:, 3] = 2.0 * env.cfg.quad.hover_thrust / env.cfg.quad.max_thrust - 1.0
        return lambda obs: hover
    if name == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        return lambda obs: 2.0 * torch.rand(env.num_envs, 4, generator=g).to(env.device) - 1.0
    raise ValueError(f"unknown controller {name!r}; choose one of {CONTROLLERS}")


class EpisodeRecorder:
    """Keeps the true position of the first ``k`` envs and scores each finished episode."""

    def __init__(self, env: StructureHoldEnv, k: int):
        self.env = env
        self.k = min(k, env.num_envs)
        self.pos: list[list[np.ndarray]] = [[] for _ in range(self.k)]
        self.actions: list[list[np.ndarray]] = [[] for _ in range(self.k)]
        self.hold: list[list[np.ndarray]] = [[] for _ in range(self.k)]
        self.gusts = [env.gust_windows(i) for i in range(self.k)]
        self.surface = Surface(point=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]))

    def before_step(self, actions: torch.Tensor) -> None:
        pos = self.env.state.pos[: self.k].cpu().numpy()
        hold = self.env.hold_point[: self.k].cpu().numpy()
        act = actions[: self.k].cpu().numpy()
        for i in range(self.k):
            self.pos[i].append(pos[i].copy())
            self.hold[i].append(hold[i].copy())  # a re-engage moves it mid-episode
            self.actions[i].append(act[i].copy())

    def after_step(self, dones: torch.Tensor, reward: torch.Tensor, counts_before: dict,
                   ) -> list[dict[str, Any]]:
        rows = []
        dt, cfg = self.env.dt, self.env.cfg
        for i in range(self.k):
            if not bool(dones[i]):
                continue
            # The env already reset env i, so its last true position (the step that ended
            # the episode, a contact included) is env.step_pos, not state.pos.
            pos = np.array(self.pos[i] + [self.env.step_pos[i].cpu().numpy()])
            hold = np.array(self.hold[i] + [self.hold[i][-1]])
            t = np.arange(len(pos)) * dt
            act = np.array(self.actions[i])
            row: dict[str, Any] = {"env": i, "length_s": float(t[-1])}
            if len(t) >= 2 and t[-1] >= 0.02:
                trace = Trace(t=t, pos=pos, pos_sp=hold, cmd_t=t[:-1],
                              rates_cmd=act[:, :3] * cfg.max_rate_rad_s,
                              thrust_cmd=0.5 * (act[:, 3] + 1.0))
                gusts = [(a, b) for a, b in self.gusts[i] if a < t[-1]]
                m = compute_metrics(trace, self.surface, gusts=gusts,
                                    contact_radius_m=cfg.contact_radius_m)
                row.update({k: v for k, v in m.items() if k != "recovery_s"})
            rows.append(row)
            self.pos[i], self.actions[i], self.hold[i] = [], [], []
            self.gusts[i] = self.env.gust_windows(i)
        return rows


def rollout(cfg: dict[str, Any], seed: int, device: str, writer: RunWriter) -> dict[str, Any]:
    p = cfg["params"]
    env_cfg = StructureHoldConfig.from_dict(p.get("env"))
    num_envs = int(p.get("num_envs", 64))
    episodes = int(p.get("episodes", 1))
    env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device,
                           sim=str(p.get("sim", "torch")))
    controller = make_controller(str(p.get("controller", "baseline")), env, seed)
    recorder = EpisodeRecorder(env, int(p.get("score_envs", 8)))
    print(f"structure_hold: {num_envs} envs, {episodes} episode(s) of "
          f"{env_cfg.episode_length_s} s at {1 / env_cfg.dt:.0f} Hz, controller "
          f"{p.get('controller', 'baseline')}, sim {env.sim.backend}, device {device}",
          flush=True)

    obs, _ = env.get_observations()
    rows: list[dict[str, Any]] = []
    total_steps = episodes * env.max_episode_length
    finished = np.zeros(num_envs, dtype=int)
    t0 = time.perf_counter()
    step = 0
    while step < total_steps:
        actions = controller(obs)
        recorder.before_step(actions)
        counts = dict(env.counts)
        obs, reward, dones, extras = env.step(actions)
        finished += dones.cpu().numpy()
        for row in recorder.after_step(dones, reward, counts):
            row["episode"] = int(finished[row["env"]])
            rows.append(row)
            writer.metric(step, **row)
        step += 1
        if step % max(1, total_steps // 10) == 0 or step == total_steps:
            rate = step * num_envs / (time.perf_counter() - t0)
            ended = {k: v for k, v in env.counts.items() if v}
            print(f"structure_hold: step {step}/{total_steps}, {rate:,.0f} env-steps/s, "
                  f"episodes ended {ended}", flush=True)
    elapsed = time.perf_counter() - t0

    def agg(key: str, fn) -> float:
        vals = [r[key] for r in rows if key in r and isinstance(r[key], float)
                and math.isfinite(r[key])]
        return float(fn(vals)) if vals else math.nan

    total_ended = sum(env.counts.values())
    headline = {
        "episodes_scored": len(rows),
        "p95_m_mean": agg("p95_m", np.mean),
        "p95_m_worst": agg("p95_m", np.max),
        "rms_m_mean": agg("rms_m", np.mean),
        "worst_toward_surface_m": agg("worst_toward_surface_m", np.max),
        "recovery_mean_s": agg("recovery_mean_s", np.mean),
        "rate_roughness_rad_s2": agg("rate_roughness_rad_s2", np.mean),
        "contact_rate": env.counts["contact"] / total_ended if total_ended else math.nan,
        "crash_rate": (1 - env.counts["timeout"] / total_ended) if total_ended else math.nan,
        "env_steps_per_s": total_steps * num_envs / elapsed,
    }
    return headline


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        headline = rollout(cfg, int(cfg["seed"]), args.device, writer)
        print("structure_hold: " + ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in headline.items()
        ), flush=True)
        if headline["episodes_scored"] == 0:
            raise RuntimeError("no episode finished, so nothing was scored; run longer")
        writer.summary("ok", headline=headline,
                       vs_baseline="cascaded PD in the mock (envs/structure_hold/baseline.py);"
                                   " tuned PX4 comes with M1")
        return 0
    except Exception as e:  # every failure lands in summary.json, then exit non-zero
        traceback.print_exc()
        writer.summary("failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
