# training

`launch.py` runs any entrypoint with the project's rules enforced (free GPU, 1 GPU, 1 hour,
smoke mode, run folder, `experiments/LOG.md` line). The entrypoints:

| Entrypoint | Config | What it does |
| --- | --- | --- |
| `training.selfcheck` | `configs/selfcheck.yaml` | Machine check: a toy fit through the whole launch path |
| `envs.structure_hold.run` | `configs/structure_hold_rollout.yaml` | Roll the env with the cascaded PD yardstick and score it |
| `training.train_ppo` | `configs/structure_hold_ppo.yaml` | Train a Structure Hold policy with rsl_rl PPO, export it, score it against the PD yardstick |
| `training.eval_policy` | `configs/structure_hold_eval.yaml` | Score an exported policy (and the PD yardstick) on any env config |

```
python -m training.launch --config training/configs/structure_hold_ppo.yaml --smoke --device cpu
python -m training.launch --config training/configs/structure_hold_ppo.yaml --device cpu   # real run, logged
```

## The PPO loop

`ppo_env.py` wraps `envs.structure_hold.StructureHoldEnv` in the shape rsl_rl 5.x trains on:
the actor sees the last 3 env observations stacked (oldest first, 60 values), the critic
additionally sees what only the simulator knows (true position error and velocity, wind
force, estimate error). `train_ppo.py` builds the rsl_rl runner (64x64 ELU actor, about 8k
parameters, Gaussian actions, observation normalisation), trains, writes one
`metrics.jsonl` row per iteration (reward, episode length, crash rate, losses, action std,
env steps per second) and TensorBoard under `tb/`, then exports and scores the policy.

Every PPO default lives in `train_ppo.DEFAULTS`; a config overrides any of them under
`params`. `env` takes a `StructureHoldConfig` mapping (wind, estimate corruption, reward
weights, airframe); the defaults are the env's.

## The exported policy

A run folder ends with:

| File | What |
| --- | --- |
| `policy.onnx` | The deterministic actor, normalisation folded in. Input `obs` (N, 60), output `actions` (N, 4) in [-1, 1]. What the PX4 module runs. |
| `policy.json` | Input layout (which slice is which, history length and order), action scaling (rates = `actions[0:3] * max_rate`, thrust = `0.5 * (actions[3] + 1) * max_thrust`), the env config and training config. |
| `policy.pt` | rsl_rl's full checkpoint (actor, critic, optimiser) for resuming or re-exporting. |
| `summary.json` | Policy and PD p95 error, crash rates, and the `vs_baseline` wording that goes to `experiments/LOG.md`. |

`policy_io.load_controller(path, num_envs)` turns a `.onnx` (run with onnxruntime) or `.pt`
into a controller on the raw env observation, keeping the history buffer itself, so what is
scored is exactly what ships. Released policies live in `policies/` (ONNX + JSON, tens of KB).

Tests: `pytest training` (launch rules with a fake `nvidia-smi`; history and VecEnv shapes;
the actor is small; ONNX matches the torch actor on any batch size; the `.onnx` and `.pt`
controllers agree on the env; a mismatched history is refused; train and eval entrypoints
write their files in smoke mode).
