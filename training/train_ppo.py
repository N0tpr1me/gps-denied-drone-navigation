"""Train a Structure Hold policy with rsl_rl PPO on the mock simulator, export it, score it.

    python -m training.launch --config training/configs/structure_hold_ppo.yaml --smoke --device cpu

Inputs (the config's ``params``): ``num_envs``, ``iterations``, ``num_steps_per_env``,
``history`` (observations stacked for the policy), ``sim`` (``torch`` or ``isaac``), ``env``
(a ``StructureHoldConfig`` mapping), ``actor`` / ``critic`` (``hidden_dims``, ``activation``,
``init_std``), ``ppo`` (rsl_rl's PPO keyword arguments), ``save_interval``, ``curriculum``
(``dropout_max_s``: the dropout cap training starts at, growing linearly to the env's
``estimate.dropout_max_s`` over the first ``fraction`` of the iterations; ``null`` is off),
``resume`` (a ``policy.pt`` or ``tb/model_<it>.pt`` from an earlier run: actor, critic,
optimiser and iteration counter are restored and training continues for ``iterations`` more,
so two runs under the hour limit make one longer run), ``resume_std`` (with ``resume``: set
the actor's exploration std to this value after loading; ``null`` keeps the loaded one. A
std clamped at the floor of ``actor.std_range`` gets no gradient, so a resumed policy that
reached the floor otherwise keeps that std for good), ``init_actor_onnx`` (instead of
``resume``: start the actor from a released ``.onnx``, weights and observation normaliser,
the normaliser then frozen; the critic starts fresh, so ``critic_warmup`` iterations first
train the critic alone with the actor frozen. A released policy's ``policy.pt`` lives only in
the ephemeral run folder that trained it, so this is how a release is continued later), and
``eval``
(``num_envs``, ``episodes``, ``score_envs`` and ``env``: the post-training comparison with
the PD yardstick; ``eval.env`` is merged over the training ``env``, so training can use
harder estimate corruption than the standard condition the result is reported on;
``select_every`` and ``select_envs``: the saved checkpoints at that iteration stride, plus
the final one, are scored on a selection seed and the one with the lowest crash rate, then
lowest p95, is the policy exported and reported. PPO keeps sharpening the hold after the
policy has already become careful, so the last iteration is often not the safest.)

Outputs, in the run folder: ``metrics.jsonl`` (one line per PPO iteration: mean reward,
episode length, crash rate, losses, action std, env steps per second; then one line per
scored evaluation episode), ``tb/`` (TensorBoard and rsl_rl's periodic ``model_<it>.pt``),
``policy.onnx`` + ``policy.json`` (the exported policy and its interface, see
``policy_io.py``), ``policy.pt`` (full checkpoint), ``summary.json`` with the headline
(policy and PD p95 error, crash rates, training reward) and ``vs_baseline``.
"""

from __future__ import annotations

import copy
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import NUM_OBS_V1, OBS, StructureHoldEnv
from eval.gate1 import check_not_held_out
from training import evaluate
from training.policy_io import export_policy, load_controller, policy_metadata
from training.ppo_env import DropoutCurriculum, StructureHoldVecEnv
from training.runlib import RunWriter, entry_args, load_yaml

DEFAULTS: dict[str, Any] = {
    "num_envs": 1024,
    "iterations": 300,
    "num_steps_per_env": 48,
    "history": 3,
    "num_obs": NUM_OBS_V1,  # inputs per step: 21 adds the engage time, 23 the wall range
    "sim": "torch",
    "save_interval": 50,
    "init_at_random_ep_len": True,
    # Actions are clamped to [-1, 1] by the env, so noise beyond that is free: keep the
    # exploration std bounded and skip the entropy bonus, or the std grows without limit.
    "actor": {"hidden_dims": [64, 64], "activation": "elu", "init_std": 0.25,
              "std_range": [0.01, 1.0]},
    "critic": {"hidden_dims": [128, 128], "activation": "elu"},
    "ppo": {
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.995,
        "lam": 0.97,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.0,
        "learning_rate": 1.0e-3,
        "max_grad_norm": 1.0,
        "schedule": "adaptive",
        "desired_kl": 0.01,
    },
    "curriculum": {"dropout_max_s": None, "fraction": 0.6},
    "resume": None,
    "resume_std": None,
    "init_actor_onnx": None,
    "critic_warmup": 200,
    "eval": {"num_envs": 256, "episodes": 1, "score_envs": 32, "env": {},
             "select_every": 250, "select_envs": 128},
}


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_train_cfg(p: dict[str, Any]) -> dict[str, Any]:
    """rsl_rl 5.x runner config from our flat params."""
    actor, critic = dict(p["actor"]), dict(p["critic"])
    dist = {"class_name": "GaussianDistribution", "init_std": float(actor.pop("init_std", 0.25)),
            "std_range": tuple(float(x) for x in actor.pop("std_range", (0.01, 1.0)))}
    return {
        "num_steps_per_env": int(p["num_steps_per_env"]),
        "save_interval": int(p["save_interval"]),
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "actor": {
            "class_name": "MLPModel",
            "obs_normalization": True,
            "distribution_cfg": dist,
            **actor,
        },
        "critic": {"class_name": "MLPModel", "obs_normalization": True, **critic},
        "algorithm": {"class_name": "PPO", **p["ppo"]},
    }


def make_curriculum(p: dict[str, Any]) -> DropoutCurriculum | None:
    c = p.get("curriculum") or {}
    if c.get("dropout_max_s") is None:
        return None
    env_cfg = StructureHoldConfig.from_dict(p.get("env"))
    iterations = max(1, round(float(c.get("fraction", 0.6)) * int(p["iterations"])))
    return DropoutCurriculum(float(c["dropout_max_s"]), env_cfg.estimate.dropout_max_s,
                             iterations)


def build_runner(vec: Any, train_cfg: dict[str, Any], log_dir: Path, device: str,
                 writer: RunWriter, dt: float):
    """rsl_rl's runner on any of our vec envs (``vec.cfg``, ``vec.curriculum_state()``),
    logging one ``metrics.jsonl`` row per iteration. Route Follow uses it too."""
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.utils.logger import Logger

    class JsonlLogger(Logger):
        """rsl_rl's logger plus one ``metrics.jsonl`` row per iteration."""

        def log(self, it, start_it, total_it, collect_time, learn_time, loss_dict,
                learning_rate, action_std, rnd_weight, **kw):
            crash = [float(e["crash_rate"]) for e in self.ep_extras if "crash_rate" in e]
            row = {
                "reward_mean": statistics.mean(self.rewbuffer) if self.rewbuffer else None,
                "length_s_mean": (statistics.mean(self.lenbuffer) * dt
                                  if self.lenbuffer else None),
                "crash_rate": statistics.mean(crash) if crash else None,
                "action_std": float(action_std.detach().mean()),
                "learning_rate": float(learning_rate),
                "env_steps_per_s": self.cfg["num_steps_per_env"] * self.num_envs
                / max(collect_time + learn_time, 1e-9),
                **{f"loss_{k}": float(v) for k, v in loss_dict.items()},
                **vec.curriculum_state(),
            }
            super().log(it, start_it, total_it, collect_time, learn_time, loss_dict,
                        learning_rate, action_std, rnd_weight, print_minimal=True, **kw)
            writer.metric(it, **row)
            if (it - start_it) % 10 == 0 or it + 1 == total_it:
                print(f"ppo: it {it}/{total_it} reward {row['reward_mean'] or 0:.2f} "
                      f"len {row['length_s_mean'] or 0:.1f} s crash "
                      f"{row['crash_rate'] if row['crash_rate'] is not None else 0:.2f} "
                      f"std {row['action_std']:.3f} {row['env_steps_per_s']:,.0f} steps/s",
                      flush=True)

    class Runner(OnPolicyRunner):
        def __init__(self, env, train_cfg, log_dir, device):
            super().__init__(env, train_cfg, log_dir, device)
            self.logger = JsonlLogger(
                log_dir=log_dir, cfg=self.cfg, env_cfg=env.cfg, num_envs=env.num_envs,
                is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size,
                gpu_global_rank=self.gpu_global_rank, device=self.device,
            )

    return Runner(vec, train_cfg, str(log_dir), device)


def make_runner(p: dict[str, Any], seed: int, device: str, log_dir: Path, writer: RunWriter):
    env_cfg = StructureHoldConfig.from_dict(p.get("env"))
    torch.manual_seed(seed)
    env = StructureHoldEnv(env_cfg, int(p["num_envs"]), seed=seed, device=device,
                           sim=str(p["sim"]))
    vec = StructureHoldVecEnv(env, history=int(p["history"]), curriculum=make_curriculum(p),
                              steps_per_iteration=int(p["num_steps_per_env"]),
                              num_obs=int(p["num_obs"]))
    train_cfg = build_train_cfg(p)
    runner = build_runner(vec, train_cfg, log_dir, device, writer, env_cfg.dt)
    return runner, vec, env_cfg, train_cfg


def init_actor_from_onnx(runner: Any, path: Path, history: int) -> None:
    """Load a released policy's actor (``export_onnx``'s graph: normaliser Sub and Div, then
    the MLP's Gemm layers) into the runner's actor, and freeze the normaliser there.

    When the runner's actor reads more inputs per step than the ONNX did (the 21-input
    layout from a 20-input release, or the 23-input one from v22), each step's new inputs get
    zero weights, so the actor starts out acting exactly as the release. Their normaliser:
    mean 0.5 and std 0.5 for the inputs in [0, 1] (engage time, range usable); the wall-range
    error takes the release's normaliser for the estimated x error, the quantity it measures
    on the wall instead of on the estimate."""
    import onnx
    from onnx import numpy_helper

    init = {i.name: torch.as_tensor(numpy_helper.to_array(i).copy())
            for i in onnx.load(str(path)).graph.initializer}
    div = next(n for n in onnx.load(str(path)).graph.node if n.op_type == "Div")
    actor = runner.alg.get_policy()
    norm = actor.obs_normalizer
    w0, mean, std = init["mlp.0.weight"], init["obs_normalizer._mean"], init[div.input[1]]
    std = std - norm.eps  # the graph stores std + eps
    old_k, new_k = w0.shape[1] // history, actor.mlp[0].in_features // history
    if new_k != old_k:
        if new_k < old_k or w0.shape[1] != history * old_k:
            raise ValueError(f"{path.name}: {w0.shape[1]} inputs cannot grow to "
                             f"{actor.mlp[0].in_features} over history {history}")
        names = {i: name for name, sl in OBS.items() for i in range(sl.start, sl.stop)}

        def pad(x: torch.Tensor, fill: str) -> torch.Tensor:
            """``fill``: "zero" (weights) or "norm" (the normaliser's mean or std)."""
            x = x.reshape(x.shape[0], history, old_k)
            cols = []
            for j in range(old_k, new_k):
                if fill == "zero":
                    cols.append(torch.zeros_like(x[..., 0]))
                elif names[j] == "wall_range":
                    cols.append(x[..., OBS["pos_err"].start])
                else:
                    cols.append(torch.full_like(x[..., 0], 0.5))
            extra = torch.stack(cols, dim=-1)
            return torch.cat([x, extra], dim=-1).reshape(x.shape[0], history * new_k)
        init["mlp.0.weight"] = pad(w0, "zero")
        mean, std = pad(mean, "norm"), pad(std, "norm")
    with torch.no_grad():
        for name, q in actor.mlp.named_parameters():
            q.copy_(init[f"mlp.{name}"])
        norm._mean.copy_(mean)
        norm._std.copy_(std)
        norm._var.copy_(std**2)
        norm.count.fill_(10**9)
    norm.until = int(norm.count)
    print(f"ppo: actor initialised from {path}", flush=True)


def select_checkpoint(runner, run_dir: Path, meta: dict[str, Any], eval_cfg: StructureHoldConfig,
                      p: dict[str, Any], seed: int, device: str, writer: RunWriter) -> int:
    """Score the final policy and the saved checkpoints every ``eval.select_every``
    iterations on a selection seed; load the best (lowest crash rate, then p95) into the
    runner and return its iteration."""
    e = p["eval"]
    first = int(p.get("_start_iteration", 0))
    last = first + int(p["iterations"]) - 1
    every = int(e.get("select_every", 0) or 0)
    candidates = {last: None}  # the policy now in the runner
    if every > 0:
        # Every ``every`` iterations from the start, or at multiples of it: saved
        # checkpoints sit at multiples of the save interval, and a run resumed from one
        # starts one past it, so counting from the start alone finds none.
        for it in range(first, last):
            path = run_dir / "tb" / f"model_{it}.pt"
            if ((it - first) % every == 0 or it % every == 0) and path.exists():
                candidates[it] = path
    if len(candidates) == 1:
        return last
    scores: dict[int, tuple[float, float]] = {}
    for it, path in candidates.items():
        if path is None:
            path = run_dir / "tb" / "model_final.pt"
            runner.save(str(path))
        env = StructureHoldEnv(eval_cfg, int(e.get("select_envs", 128)), seed=seed + 2000,
                               device=device, sim=str(p["sim"]))
        ctl = load_controller(path, env.num_envs, device, meta=meta)
        h = evaluate.score_controller(env, ctl, int(e["episodes"]), int(e["score_envs"]))
        scores[it] = (h["crash_rate"], h["p95_m_mean"])
        writer.metric(it, selection_crash_rate=h["crash_rate"], selection_p95_m=h["p95_m_mean"])
        print(f"ppo: checkpoint {it}: crash {h['crash_rate']:.3f}, p95 {h['p95_m_mean']:.2f} m",
              flush=True)
    best = min(scores, key=lambda it: scores[it])
    if best != last:
        runner.load(str(run_dir / "tb" / f"model_{best}.pt"))
    return best


def train(cfg: dict[str, Any], run_dir: Path, device: str, writer: RunWriter) -> dict[str, Any]:
    p = merge(DEFAULTS, cfg.get("params"))
    seed = int(cfg["seed"])
    # Training, the quick eval (seed + 1000) and checkpoint selection (seed + 2000) never
    # touch the Gate 1 test seeds (eval/gate1.py).
    check_not_held_out(seed, seed + 1000, seed + 2000)
    runner, vec, env_cfg, train_cfg = make_runner(p, seed, device, run_dir / "tb", writer)
    n_params = sum(x.numel() for x in runner.alg.get_policy().parameters())
    print(f"ppo: {vec.num_envs} envs, {p['iterations']} iterations x {p['num_steps_per_env']} "
          f"steps, history {p['history']} ({vec.num_policy_obs} policy obs, "
          f"{vec.num_privileged_obs} privileged), actor {n_params:,} params, device {device}",
          flush=True)

    if p.get("resume"):
        runner.load(str(Path(p["resume"])), map_location=device)
        runner.current_learning_iteration += 1  # the loaded iteration is done; go on after it
        print(f"ppo: resumed {p['resume']} at iteration {runner.current_learning_iteration}",
              flush=True)
        if p.get("resume_std") is not None:
            with torch.no_grad():
                runner.alg.get_policy().distribution.std_param.fill_(float(p["resume_std"]))
            print(f"ppo: exploration std reset to {float(p['resume_std'])}", flush=True)
    if p.get("init_actor_onnx"):
        if p.get("resume"):
            raise ValueError("give resume or init_actor_onnx, not both")
        init_actor_from_onnx(runner, Path(p["init_actor_onnx"]), int(p["history"]))
        if p.get("resume_std") is not None:
            with torch.no_grad():
                runner.alg.get_policy().distribution.std_param.fill_(float(p["resume_std"]))
        warm = int(p.get("critic_warmup") or 0)
        if warm > 0:
            actor = runner.alg.get_policy()
            for q in actor.parameters():
                q.requires_grad_(False)
            print(f"ppo: critic warm-up, {warm} iterations with the actor frozen", flush=True)
            runner.learn(warm, init_at_random_ep_len=bool(p["init_at_random_ep_len"]))
            runner.current_learning_iteration += 1
            for q in actor.parameters():
                q.requires_grad_(True)
            # The adaptive schedule raised the rate while the frozen actor's KL was zero.
            runner.alg.learning_rate = float(p["ppo"]["learning_rate"])
            for g in runner.alg.optimizer.param_groups:
                g["lr"] = runner.alg.learning_rate
    p["_start_iteration"] = int(runner.current_learning_iteration)

    t0 = time.perf_counter()
    runner.learn(int(p["iterations"]), init_at_random_ep_len=bool(p["init_at_random_ep_len"]))
    train_s = time.perf_counter() - t0
    reward = statistics.mean(runner.logger.rewbuffer) if runner.logger.rewbuffer else float("nan")

    e = p["eval"]
    eval_cfg = StructureHoldConfig.from_dict(merge(p.get("env") or {}, e.get("env")))
    meta = policy_metadata(env_cfg, int(p["history"]), train_cfg, {
        "seed": seed, "iterations": int(p["iterations"]), "num_envs": vec.num_envs,
        "actor_params": n_params, "train_reward_mean": reward,
        "start_iteration": p["_start_iteration"], "resumed_from": p.get("resume"),
    }, num_obs=int(p["num_obs"]))
    chosen = select_checkpoint(runner, run_dir, meta, eval_cfg, p, seed, device, writer)
    meta["selected_iteration"] = chosen
    onnx_path = export_policy(runner, run_dir, meta)
    print(f"ppo: trained in {train_s:.0f} s, exported iteration {chosen} as {onnx_path.name} "
          f"({onnx_path.stat().st_size / 1024:.0f} KB)", flush=True)

    pol, base, wording = evaluate.compare(
        eval_cfg, lambda env: load_controller(onnx_path, env.num_envs, device),
        num_envs=int(e["num_envs"]), episodes=int(e["episodes"]),
        score_envs=int(e["score_envs"]), seed=seed + 1000, device=device, sim=str(p["sim"]),
        on_episode=lambda name, row: writer.metric(int(p["iterations"]), controller=name, **row),
    )
    print(f"ppo: eval {wording}", flush=True)
    headline = {
        "train_reward_mean": reward,
        "train_s": train_s,
        "actor_params": n_params,
        "selected_iteration": chosen,
        **{f"policy_{k}": pol[k] for k in evaluate.HEADLINE_KEYS},
        **{f"pd_{k}": base[k] for k in ("p95_m_mean", "crash_rate")},
    }
    return {"headline": headline, "vs_baseline": wording, "policy": pol, "baseline": base}


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        result = train(cfg, Path(args.run_dir), args.device, writer)
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
