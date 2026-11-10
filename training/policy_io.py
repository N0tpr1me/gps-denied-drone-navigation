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

from envs.structure_hold.env import NUM_ACTIONS, NUM_OBS, NUM_OBS_V1, OBS
from training.ppo_env import ObsHistory

Controller = Callable[[torch.Tensor], torch.Tensor]


def policy_metadata(env_cfg: Any, history: int, train_cfg: dict[str, Any],
                    extra: dict[str, Any] | None = None,
                    num_obs: int = NUM_OBS) -> dict[str, Any]:
    layout = {name: [s.start, s.stop] for name, s in OBS.items() if s.stop <= num_obs}
    return {
        "format": "structure_hold_policy/1",
        "input": {
            "name": "obs",
            "size": history * num_obs,
            "history": history,
            "order": "oldest first; the newest observation is the last num_obs values",
            "num_obs": num_obs,
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
            # Who holds the heading: "policy" when the reward trains it (heading weight > 0),
            # "module" when the PX4 module must replace actions[2] with its own yaw loop.
            "yaw": "policy" if env_cfg.reward.heading > 0 else "module",
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
    except TypeError as e:  # older torch without the torch.export-based exporter
        if "unexpected keyword" not in str(e):
            raise
    # Some torch versions (2.7, in Isaac Lab 2.3.2's Python) take the exporter call above but
    # pin the batch to the dummy's 1; then the legacy exporter, which honours dynamic_axes.
    if not path.exists() or not _batch_is_free(path):
        legacy = {"opset_version": 17,
                  "dynamic_axes": {"obs": {0: "batch"}, "actions": {0: "batch"}}, **names}
        try:
            torch.onnx.export(module, dummy, str(path), dynamo=False, **legacy)
        except TypeError as e:  # torch before the dynamo flag
            if "unexpected keyword" not in str(e):
                raise
            torch.onnx.export(module, dummy, str(path), **legacy)


def _batch_is_free(path: Path) -> bool:
    """True when the exported model's first input has a symbolic (free) batch dimension."""
    import onnx

    dim = onnx.load(str(path)).graph.input[0].type.tensor_type.shape.dim[0]
    return not dim.HasField("dim_value")


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


def load_controller(policy_path: Path, num_envs: int, device: str = "cpu",
                    meta: dict[str, Any] | None = None) -> Controller:
    """A deterministic ``obs -> actions`` callable on the env's raw observation, from a
    ``.onnx`` (run with onnxruntime, what the module ships) or ``.pt`` (rsl_rl checkpoint).
    ``meta`` is the ``policy.json`` content; by default it is read from next to the file."""
    policy_path = Path(policy_path)
    meta = read_metadata(policy_path) if meta is None else meta
    history = int(meta["input"]["history"])
    # The policy reads the first num_obs values of the env's observation.
    num_obs = int(meta["input"].get("num_obs", NUM_OBS_V1))
    actor: Any
    if policy_path.suffix == ".onnx":
        actor = _OnnxActor(policy_path)
    elif policy_path.suffix == ".pt":
        actor = _TorchActor(policy_path, meta, device)
    else:
        raise ValueError(f"policy must be a .onnx or .pt file, got {policy_path.name}")
    if actor.input_size != history * num_obs:
        raise ValueError(
            f"{policy_path.name} takes {actor.input_size} inputs but policy.json says "
            f"history {history} x {num_obs} obs"
        )
    hist = ObsHistory(num_envs, num_obs, history, device)
    state = {"fresh": True}

    def controller(obs: torch.Tensor) -> torch.Tensor:
        obs = obs[:, :num_obs]
        if state["fresh"]:
            hist.fill(obs)
            state["fresh"] = False
            stacked = hist.stacked()
        else:
            stacked = hist.push(obs)
        return actor(stacked).clamp(-1.0, 1.0)

    def reset_history(obs: torch.Tensor, env_mask: torch.Tensor | None = None) -> None:
        hist.fill(obs[:, :num_obs], env_mask)

    controller.reset_history = reset_history  # type: ignore[attr-defined]
    return controller
