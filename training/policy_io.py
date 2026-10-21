"""Save, export and load a trained Structure Hold policy.

Two files describe a policy:

* ``policy.onnx``: the deterministic actor, observation normalisation folded in. Input
  ``obs`` of shape (N, history * NUM_OBS), output ``actions`` of shape (N, 4) in [-1, 1]
  (clamp before use, as the env does). This is what the PX4 module runs.
* ``policy.json``: what the module needs to feed it: the observation layout, the history
  length and order, the action scaling, the env config it was trained with.

``policy.pt`` next to them is rsl_rl's full checkpoint (actor, critic, optimiser) for
resuming or re-exporting; it is not needed to fly.

``load_controller`` turns either file into a callable ``obs -> actions`` on the env's raw
observation (it keeps the history itself), so evaluation can score exactly the exported
artefact, not the training-time module.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.structure_hold.env import NUM_ACTIONS, NUM_OBS, OBS
from training.ppo_env import ObsHistory

Controller = Callable[[torch.Tensor], torch.Tensor]


def policy_metadata(env_cfg: Any, history: int, train_cfg: dict[str, Any],
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    layout = {name: [s.start, s.stop] for name, s in OBS.items()}
    return {
        "format": "structure_hold_policy/1",
        "input": {
            "name": "obs",
            "size": history * NUM_OBS,
            "history": history,
            "order": "oldest first; the newest observation is the last NUM_OBS values",
            "num_obs": NUM_OBS,
            "layout": layout,
            "obs_pos_clip_m": env_cfg.obs_pos_clip_m,
        },
        "output": {
            "name": "actions",
            "size": NUM_ACTIONS,
            "range": [-1.0, 1.0],
            "rates": f"actions[0:3] * {env_cfg.max_rate_rad_s} rad/s body rate setpoints "
                     "(roll, pitch, yaw)",
            "thrust": f"0.5 * (actions[3] + 1) * {env_cfg.quad.max_thrust} N collective",
            "max_rate_rad_s": env_cfg.max_rate_rad_s,
            "max_thrust_n": env_cfg.quad.max_thrust,
        },
        "rate_hz": round(1.0 / env_cfg.dt),
        "env_cfg": _jsonable(asdict(env_cfg)),
        "train_cfg": _jsonable(train_cfg),
        **(extra or {}),
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def export_onnx(actor: Any, path: Path) -> None:
    """Export rsl_rl's deterministic actor (normalisation folded in) with a free batch size."""
    module = actor.as_onnx(verbose=False).to("cpu").eval()
    dummy = (torch.zeros(1, module.input_size),)
    names = {"input_names": ["obs"], "output_names": ["actions"]}
    try:
        torch.onnx.export(module, dummy, str(path), dynamo=True, opset_version=18,
                          dynamic_shapes=({0: "batch"},), external_data=False, **names)
    except TypeError:  # older torch without the torch.export-based exporter
        torch.onnx.export(module, dummy, str(path), opset_version=17,
                          dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, **names)


def export_policy(runner: Any, run_dir: Path, metadata: dict[str, Any]) -> Path:
    """Write ``policy.onnx``, ``policy.json`` and ``policy.pt`` into ``run_dir``."""
    run_dir = Path(run_dir)
    export_onnx(runner.get_inference_policy("cpu"), run_dir / "policy.onnx")
    runner.save(str(run_dir / "policy.pt"), infos={"metadata": metadata})
    (run_dir / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir / "policy.onnx"


def read_metadata(policy_path: Path) -> dict[str, Any]:
    meta_path = Path(policy_path).with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} (written next to the policy at export)")
    return json.loads(meta_path.read_text())


class _OnnxActor:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (inp,) = self.session.get_inputs()
        self.input_name = inp.name
        self.input_size = int(inp.shape[-1])

    def __call__(self, stacked: torch.Tensor) -> torch.Tensor:
        x = stacked.detach().cpu().numpy().astype(np.float32)
        (y,) = self.session.run(None, {self.input_name: x})
        return torch.as_tensor(y, dtype=stacked.dtype, device=stacked.device)


class _TorchActor:
    """The actor rebuilt from rsl_rl's checkpoint, run deterministically."""

    def __init__(self, path: Path, meta: dict[str, Any], device: str):
        from rsl_rl.utils import resolve_class
        from tensordict import TensorDict

        train_cfg = meta["train_cfg"]
        size = int(meta["input"]["size"])
        dummy = TensorDict({"policy": torch.zeros(1, size)}, batch_size=[1])
        actor_class, actor_cfg = resolve_class(train_cfg["actor"])
        self.actor = actor_class(dummy, {"actor": ["policy"]}, "actor", NUM_ACTIONS, **actor_cfg)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.to(device).eval()
        self.input_size = size

    @torch.no_grad()
    def __call__(self, stacked: torch.Tensor) -> torch.Tensor:
        from tensordict import TensorDict

        td = TensorDict({"policy": stacked}, batch_size=[stacked.shape[0]])
        return self.actor(td)


def load_controller(policy_path: Path, num_envs: int, device: str = "cpu") -> Controller:
    """A deterministic ``obs -> actions`` callable on the env's raw observation, from a
    ``.onnx`` (run with onnxruntime, what the module ships) or ``.pt`` (rsl_rl checkpoint)."""
    policy_path = Path(policy_path)
    meta = read_metadata(policy_path)
    history = int(meta["input"]["history"])
    actor: Any
    if policy_path.suffix == ".onnx":
        actor = _OnnxActor(policy_path)
    elif policy_path.suffix == ".pt":
        actor = _TorchActor(policy_path, meta, device)
    else:
        raise ValueError(f"policy must be a .onnx or .pt file, got {policy_path.name}")
    if actor.input_size != history * NUM_OBS:
        raise ValueError(
            f"{policy_path.name} takes {actor.input_size} inputs but policy.json says "
            f"history {history} x {NUM_OBS} obs"
        )
    hist = ObsHistory(num_envs, NUM_OBS, history, device)
    state = {"fresh": True}

    def controller(obs: torch.Tensor) -> torch.Tensor:
        if state["fresh"]:
            hist.fill(obs)
            state["fresh"] = False
            stacked = hist.stacked()
        else:
            stacked = hist.push(obs)
        return actor(stacked).clamp(-1.0, 1.0)

    controller.reset_history = hist.fill  # type: ignore[attr-defined]
    return controller
