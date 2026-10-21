"""A classical cascaded controller on the env's observation: the mock's yardstick.

Position -> acceleration -> tilt -> body rates on the *estimated* state the policy also
sees: PD on position with an integrator against steady wind (PX4's velocity-loop
integrator does the same job), then a P attitude loop that outputs body rates. It is the
thing a learned policy has to beat in the mock before anything is claimed, and it proves the
task is holdable. It is not the tuned-PX4 baseline (that is PX4 itself in SITL, eval/); its
gains are plain hand values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.sims.torch_quad import GRAVITY, quat_rotate_inverse
from envs.structure_hold.env import OBS, StructureHoldEnv


@dataclass(frozen=True)
class BaselineGains:
    pos: float = 2.0  # 1/s^2: acceleration per metre of error
    vel: float = 2.5  # 1/s: acceleration per m/s
    integral: float = 0.6  # 1/s^3: acceleration per metre-second of accumulated error
    max_integral: float = 3.0  # m/s^2 the integrator may contribute
    max_acc: float = 6.0  # m/s^2 horizontal and vertical command limit
    tilt: float = 8.0  # rad/s per rad of tilt error
    yaw: float = 3.0  # rad/s per rad of heading error


class CascadedPd:
    def __init__(self, env: StructureHoldEnv, gains: BaselineGains | None = None):
        self.env = env
        self.g = gains or BaselineGains()
        self.mass = env.cfg.quad.mass
        self.max_thrust = env.cfg.quad.max_thrust
        self.max_rate = env.cfg.max_rate_rad_s
        self.err_integral = torch.zeros(env.num_envs, 3, dtype=env.dtype, device=env.device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        g = self.g
        err = obs[:, OBS["pos_err"]]
        vel = obs[:, OBS["vel"]]
        gravity_body = obs[:, OBS["gravity_body"]]
        heading = obs[:, OBS["heading"]]
        n = obs.shape[0]

        # Acceleration setpoint in the world (= structure) frame, then the thrust vector.
        # The integrator restarts with each episode and only acts once the start transient
        # is over, so it does not wind up while the drone is still far from the hold point.
        self.err_integral[self.env.episode_step == 0] = 0.0
        near = err.norm(dim=-1, keepdim=True) < 1.0
        self.err_integral += torch.where(near, err, torch.zeros_like(err)) * self.env.dt
        lim = g.max_integral / max(g.integral, 1e-9)
        self.err_integral.clamp_(-lim, lim)
        acc = -g.pos * err - g.vel * vel - g.integral * self.err_integral
        acc = acc.clamp(-g.max_acc, g.max_acc)
        f_world = acc * self.mass
        f_world[:, 2] += self.mass * GRAVITY
        f_norm = f_world.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_des_world = f_world / f_norm

        # Attitude: rotate the desired thrust axis into the body frame and steer body z onto it.
        quat = self.env.state.quat  # attitude is observed directly (gravity_body, heading)
        z_des_body = quat_rotate_inverse(quat, z_des_world)
        z_body = torch.zeros(n, 3, dtype=obs.dtype, device=obs.device)
        z_body[:, 2] = 1.0
        tilt_err = torch.cross(z_body, z_des_body, dim=-1)  # small-angle rotation vector
        rates = g.tilt * tilt_err
        # Heading: face the wall (+x of the structure frame).
        yaw_err = torch.atan2(heading[:, 1], heading[:, 0])
        rates[:, 2] = -g.yaw * yaw_err
        # Thrust: the component of the demanded force along the current body z.
        up_body = -gravity_body  # world up in the body frame
        thrust = (f_norm.squeeze(-1) * (z_des_body * up_body).sum(-1)).clamp(min=0.0)

        action = torch.empty(n, 4, dtype=obs.dtype, device=obs.device)
        action[:, :3] = rates / self.max_rate
        action[:, 3] = 2.0 * thrust / self.max_thrust - 1.0
        return action.clamp(-1.0, 1.0)
