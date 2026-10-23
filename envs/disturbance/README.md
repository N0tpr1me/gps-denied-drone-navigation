# envs/disturbance

Wind and near-wall force model shared by the PyTorch mock simulator, Isaac Lab and the
Gazebo SITL world. NumPy core (`core.py`), PyTorch adapter (`torch_adapter.py`); both run
the same code, and the tests check they agree to 1e-6.

```python
from envs.disturbance import DisturbanceConfig, sample_params, disturbance_force

cfg = DisturbanceConfig.from_dict(yaml_cfg.get("disturbance"))  # ranges; defaults are fine
p = sample_params(cfg, num_envs, seed, wall_point=(0, 0, 0), wall_normal=(1, 0, 0))
force = disturbance_force(p, t, pos, vel)  # (N, 3) N, world ENU
```

For tensors: `from envs.disturbance import torch_adapter as ta`, then
`pt = ta.to_torch(p, device, dtype)` once per reset and `ta.disturbance_force(pt, t, pos, vel)`
every step.

| Input | Shape | Unit |
| --- | --- | --- |
| `t` | scalar or `(N,)` | s since episode start |
| `pos` | `(N, 3)` | m, world ENU |
| `vel` | `(N, 3)` | m/s, world ENU |
| `wall_point`, `wall_normal` | `(3,)` or `(N, 3)` | a point on the surface, and the normal pointing into the free air (minus the structure frame's x) |

What it models: mean wind (from any direction, or a range of headings with
`mean_heading_deg`, 0 blowing along +x: into the wall in the structure hold env), gust steps at scheduled times, turbulence (seeded, exact
standard deviation, Gauss-Markov correlation time), shelter near one flat surface, and
linear plus quadratic drag on the air velocity relative to the drone. The force includes
drag from the drone's own motion, so the simulator must not add translational drag of its
own; Gazebo, which has its own, applies `disturbance_force(...) - drag_force(-vel, k_lin, k_quad)`.

Nominal values reproduce the one published data point (2.14 kg drone, 10 m/s headwind:
2.78 N at 5 m from a wall, 0.40 N at 0.5 m). Default ranges are wide until the Phase 3 fan
rig fits them. Not modelled yet: near-wall yaw torque, ground effect, an overhead slab.

Tests: `pytest envs/disturbance`.
