# envs

Simulation environments and task logic, all batched and backend-agnostic.

Folders (see docs/decisions.md and the execution plan):

- `disturbance/` wind and near-wall force model (NumPy core, PyTorch adapter)
- `estimate/` position-estimate corruption: delay, noise, bias, jumps, dropouts
- `sims/` rigid-body backends behind one seam: `torch_quad.py` (CPU mock, now) and
  `isaac_quad.py` (Isaac Lab, on the GPU server, later)
- `structure_hold/` the task: observations, reward, resets, the classical cascaded-PD
  yardstick and the standalone rollout entrypoint (the rate PID is in `sims/rate_pid.py`)
