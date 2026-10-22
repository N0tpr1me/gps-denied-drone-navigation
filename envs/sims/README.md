# envs/sims

Rigid-body quadrotor backends behind one seam. A backend moves a body; everything that
makes the task (wind, estimate corruption, the rate loop, observations, reward) lives in
`envs/structure_hold` and never knows which backend is underneath.

```python
from envs.sims import QuadParams, QuadState, make_sim

sim = make_sim("torch", QuadParams(), num_envs=1024, dt=0.01)   # or "isaac" (needs Isaac Lab)
sim.reset()                                   # or reset(env_ids, QuadState(...))
state = sim.step(thrust, torque_body, force_ext_world)
```

| Call | In | Out |
| --- | --- | --- |
| `reset(env_ids=None, state=None)` | which envs and their start state (rows for those envs, or a full batch); `None` = level, at rest, at the origin | nothing |
| `step(thrust, torque_body, force_ext_world[, torque_ext_body])` | `(N,)` collective thrust in N along body +z; `(N, 3)` body torque in N m (FLU); `(N, 3)` external force in N (world ENU); optional `(N, 3)` external body moment in N m (FLU), added after clipping and motor lag | `QuadState`: `pos` (N, 3) m ENU, `quat` (N, 4) w-first body-to-world, `vel` (N, 3) m/s ENU, `ang_vel` (N, 3) rad/s body FLU |
| `set_airframe(env_ids=None, *, mass, inertia, max_thrust, max_torque, motor_tau)` | per-env airframe from the next step on (domain randomisation); scalar or one value per listed env, a value left out returns to `QuadParams` | nothing |

- `torch_quad.py`: the CPU mock. Semi-implicit Euler at `dt / substeps`, exact quaternion
  increment, first-order motor lag, thrust clipped to `[0, max_thrust]` and torque to
  `±max_torque`. No drag (the disturbance force already includes it), no ground, no contact.
  Deterministic for the same inputs. On 4 CPU cores 4,096 envs step in 2.4 ms (sim alone)
  and 9.7 ms with the whole Structure Hold env around it, so about 4,000 envs run in real
  time at 100 Hz (`test_contract.py` checks a loose bound).
- `rate_pid.py`: a body-rate PID in the shape of PX4's `RateControl` (normalised torque,
  D on the filtered measurement, integrator freeze on saturation, PX4 default gains), so the
  policy closes the same inner loop the onboard module will.
- `isaac_quad.py`: the Isaac Lab backend (Isaac Lab 2.3.2 / Isaac Sim 5.1.0), same calls,
  same `QuadState`. One PhysX rigid body per env with the mock's mass and inertia (per env
  through `set_airframe`); clipping, motor lag and the external moment use the mock's code. Isaac Lab is imported only when the backend is built;
  without it `make_sim("isaac", ...)` raises `IsaacLabUnavailable`. Install, first-hour
  checklist and parity: `docs/isaac-lab.md`.

The default `QuadParams` is a placeholder 5-inch-class airframe (0.8 kg, 2:1 thrust to
weight). The PX4 Gazebo x500's numbers are a named airframe of the task
(`envs/structure_hold/airframes.py`, `airframe: gz_x500`).

Tests: `pytest envs/sims`. `tests/test_isaac_quad.py` checks the Isaac backend's own code
step for step against the mock over a PhysX stand-in (`tests/fake_physx.py`), so it runs
without Isaac Lab. `tests/test_contract.py` is parametrised by backend (`torch`, `isaac-glue`,
and `isaac` where Isaac Lab is installed) and is the file every backend must pass: free fall matches g, hover thrust holds still, a body
torque gives the right rotation sign and rate, an external force gives F/m, frames are ENU
and FLU, inputs are clipped, partial resets touch only their envs, per-env airframes and the
external moment act as specified.
