# Conventions

These apply to all Python and C++ code. Change them only in a PR of their own.

## Frames

- **World frame in all of our Python code: ENU** (x east, y north, z up), origin at the
  start pose. Gazebo and Isaac Lab are already Z-up, so no conversion inside simulators.
- **Body frame in our Python code: FLU** (x forward, y left, z up).
- **PX4 uses NED (world) and FRD (body).** Convert only at the PX4 boundary (the SITL
  harness, the estimate injector's MAVLink output, and the C++ module), never in the
  shared core. Use the helpers in one place (`envs/frames.py`, added with the first code
  that needs it) rather than ad hoc sign flips.
- **Structure frame**: x points from the drone to the surface along the surface normal,
  z up. Observations and the wall-normal metrics use this frame.

## Units and types

- SI everywhere: m, s, kg, N, N m, rad, rad/s. Degrees only in human-facing configs and
  plots, converted on load.
- Quaternions are `(w, x, y, z)`, unit norm, rotating body to world.
- Batched tensors and arrays are `(num_envs, ...)`. Time steps are explicit `dt` arguments,
  never globals.
- Randomness takes an explicit seed or generator. No hidden global RNG state.

## Runs

- Every run (training, eval, SITL) takes one YAML config and writes one run folder under
  `runs/` (ignored by git): `config.yaml` (resolved, with git commit and seed), `stdout.log`,
  `metrics.jsonl`, `summary.json` (`status`, headline numbers, error).
- Launch through `training/launch.py`. Smoke runs never write to `experiments/LOG.md`;
  real runs always do, including failures.
