# envs/structure_hold

The Structure Hold task on the simulator seam: hold a quadrotor 1 to 3 m from a wall, in
wind, while the only position estimate it gets is delayed, noisy, biased, jumpy or missing.

```python
from envs.structure_hold import StructureHoldConfig, StructureHoldEnv

env = StructureHoldEnv(StructureHoldConfig(), num_envs=1024, seed=0, device="cpu", sim="torch")
obs, _ = env.get_observations()
obs, reward, dones, extras = env.step(actions)   # actions (N, 4) in [-1, 1]
```

Each step: action (body rates, thrust) → PX4-matched rate PID (`envs/sims/rate_pid.py`) →
simulator (`envs/sims`) with the wind force from `envs/disturbance` → estimate corruption
from `envs/estimate` → observation and reward. Envs that finish are reset inside `step`, and
`extras["time_outs"]` says which ended on the clock. Reward and termination use the true
state; the policy sees only the estimate. The interface is the VecEnv shape rsl_rl's runner
expects.

| | |
| --- | --- |
| World | ENU. The wall is the plane x = 0; the drone flies at x < 0, so the structure frame (x toward the surface, z up) is the world frame. |
| Hold point | `hold_distance` m from the wall (default 1 to 3), `hold_height` m up, drawn per episode. |
| Observation (20) | estimated position error and velocity in the structure frame, gravity in the body frame, heading, body rates, previous action, estimate stale flag and age. Layout in `env.py` (`OBS`). |
| Action (4) | body rate setpoints as a fraction of `max_rate_rad_s`, and thrust mapped from [-1, 1] to [0, `max_thrust`]. |
| Reward | alive bonus minus bounded penalties on true position error (saturating over `pos_scale_m`, extra for the component toward the wall), speed, body rates, action size, action change and tilt beyond `reward.tilt_free_deg`; a crash costs `reward.crash` once. Every term is bounded so a live step always beats a crash (`config.py`). |
| Episode ends | contact (clearance ≤ 0.25 m), ground, tilt over 80°, true error over 3 m, a non-finite state (counted, never silently dropped), or the clock (`episode_length_s`). |
| Randomness | one seed drives the NumPy generator (hold point, start pose, wind) and the estimator's torch generator. Same seed, same rollout. |

`baseline.py` is a classical cascaded PD on the same observation: the mock's yardstick and
proof that the task is holdable. It is not the tuned-PX4 baseline, which is PX4 itself in
SITL (M1).

## Running it standalone

```
python -m training.launch --config training/configs/structure_hold_rollout.yaml --smoke --device cpu
```

rolls the env with the baseline controller and scores the first `score_envs` envs with
`eval/metrics.py`. The run folder gets `metrics.jsonl` (one line per finished episode:
p95 and RMS error, worst excursion toward the wall, contacts, recovery after each gust,
command roughness, why the episode ended), `summary.json` (status, headline numbers,
throughput in env-steps per second, the error if it failed) and `stdout.log`. Drop
`--smoke` for the full 256-env, 20 s configuration; the launcher then also appends a line to
`experiments/LOG.md`. Config keys are documented at the top of `run.py`.

Tests: `pytest envs/structure_hold` (shapes, hold point range, hover holds in still air,
contact and non-finite handling, corruption reaches the observation, wind moves the drone,
determinism, the baseline holds within 0.5 m in mild wind, the entrypoint writes its files
and reports a bad config).
