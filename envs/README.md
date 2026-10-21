# envs

Simulation environments and task logic, all batched and backend-agnostic.

Planned (see docs/decisions.md and the execution plan):

- `disturbance/` wind and near-wall force model (NumPy core, PyTorch adapter)
- `estimate/` position-estimate corruption: delay, noise, bias, jumps, dropouts
- `sims/` rigid-body backends behind one seam: `torch_quad.py` (CPU mock, now) and
  `isaac_quad.py` (Isaac Lab, on the GPU server, later)
- `structure_hold/` the task: observations, reward, resets, PX4-matched rate PID
