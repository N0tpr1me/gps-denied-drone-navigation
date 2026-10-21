# envs/sims

Rigid-body quadrotor backends behind one seam. A backend moves a body; everything that
makes the task (wind, estimate corruption, the rate loop, observations, reward) lives in
`envs/structure_hold` and never knows which backend is underneath.

```python
from envs.sims import QuadParams, QuadState, make_sim

sim = make_sim("torch", QuadParams(), num_envs=1024, dt=0.01)   # "isaac": later, GPU server
sim.reset()                                   # or reset(env_ids, QuadState(...))
state = sim.step(thrust, torque_body, force_ext_world)
```

| Call | In | Out |
| --- | --- | --- |
| `reset(env_ids=None, state=None)` | which envs and their start state (rows for those envs, or a full batch); `None` = level, at rest, at the origin | nothing |
| `step(thrust, torque_body, force_ext_world)` | `(N,)` collective thrust in N along body +z; `(N, 3)` body torque in N m (FLU); `(N, 3)` external force in N (world ENU) | `QuadState`: `pos` (N, 3) m ENU, `quat` (N, 4) w-first body-to-world, `vel` (N, 3) m/s ENU, `ang_vel` (N, 3) rad/s body FLU |

- `torch_quad.py`: the CPU mock. Semi-implicit Euler at `dt / substeps`, exact quaternion
  increment, first-order motor lag, thrust clipped to `[0, max_thrust]` and torque to
  `±max_torque`. No drag (the disturbance force already includes it), no ground, no contact.
  Deterministic for the same inputs. About 2,000 envs step in real time at 100 Hz on 4 CPU
  cores (see `test_contract.py`).
- `rate_pid.py`: a body-rate PID in the shape of PX4's `RateControl` (normalised torque,
  D on the filtered measurement, integrator freeze on saturation, PX4 default gains), so the
  policy closes the same inner loop the onboard module will.
- `isaac_quad.py` (not yet written): the Isaac Lab backend, same two calls, same
  `QuadState`. `make_sim("isaac", ...)` raises `NotImplementedError` until then.

The default `QuadParams` is a placeholder 5-inch-class airframe (0.8 kg, 2:1 thrust to
weight) until the airframe is chosen (docs/decisions.md).

Tests: `pytest envs/sims`. `tests/test_contract.py` is parametrised by backend and is the
file Isaac Lab must pass unchanged: free fall matches g, hover thrust holds still, a body
torque gives the right rotation sign and rate, an external force gives F/m, frames are ENU
and FLU, inputs are clipped, partial resets touch only their envs.
