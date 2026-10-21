"""Score an exported policy against the cascaded PD yardstick with eval/metrics.py.

    python -m training.launch --config training/configs/structure_hold_eval.yaml \
        --smoke --device cpu

Inputs (``params``): ``policy`` (path to ``policy.onnx`` or ``policy.pt``, with its
``policy.json`` next to it; relative to the repo root), ``num_envs``, ``episodes``,
``score_envs``, ``sim``, and ``env`` (a ``StructureHoldConfig`` mapping: evaluate on harder
wind or worse estimates than training by changing it here). Leave ``env`` out to evaluate on
the config the policy was trained with (read from ``policy.json``).

Outputs: ``metrics.jsonl`` with one line per scored episode of each controller,
``summary.json`` with both headlines and the wording for ``experiments/LOG.md``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from envs.structure_hold.config import StructureHoldConfig
from training import evaluate
from training.policy_io import load_controller, read_metadata
from training.runlib import RunWriter, entry_args, load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cfg: dict[str, Any], device: str, writer: RunWriter) -> dict[str, Any]:
    p = cfg["params"]
    policy_path = Path(p["policy"])
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    meta = read_metadata(policy_path)
    env_cfg = StructureHoldConfig.from_dict(p["env"] if p.get("env") else meta["env_cfg"])
    num_envs, episodes = int(p.get("num_envs", 256)), int(p.get("episodes", 1))
    print(f"eval: {policy_path.name} ({meta['input']['history']}-step history), {num_envs} envs "
          f"x {episodes} episode(s) of {env_cfg.episode_length_s} s, device {device}", flush=True)
    pol, base, wording = evaluate.compare(
        env_cfg, lambda env: load_controller(policy_path, env.num_envs, device),
        num_envs=num_envs, episodes=episodes, score_envs=int(p.get("score_envs", 32)),
        seed=int(cfg["seed"]), device=device, sim=str(p.get("sim", "torch")),
        on_episode=lambda name, row: writer.metric(0, controller=name, **row),
    )
    print(f"eval: {wording}", flush=True)
    if pol["episodes_scored"] == 0:
        raise RuntimeError("no episode finished, so nothing was scored; run longer")
    headline = {
        "policy": str(policy_path.relative_to(REPO_ROOT) if policy_path.is_relative_to(REPO_ROOT)
                      else policy_path),
        **{f"policy_{k}": pol[k] for k in evaluate.HEADLINE_KEYS},
        **{f"pd_{k}": base[k] for k in evaluate.HEADLINE_KEYS},
    }
    return {"headline": headline, "vs_baseline": wording}


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
