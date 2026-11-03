"""Classical route followers on the env's observation: the mock's yardsticks.

``RoutePd`` is the shape of PX4's own position controller tracking a trajectory setpoint:
PD on the *estimated* error to the moving target, the target's velocity fed forward, an
integrator against a steady draft, then a P attitude loop that outputs body rates, heading
along the route. With ``centering`` > 0 it also uses the rangefinders the classical way: where
both the left and right (or up and down) readings are usable, the lateral (vertical) error
is taken from them, ``(right - left) / 2`` (the target is on the centreline, halfway between
the walls), blended with the estimate's error by ``centering``. That is the best a
hand-written controller does with the same sensors, so a learned policy is reported against
both: estimate only (what stock PX4 Mission mode does) and with centring.

These are not tuned PX4 (that is PX4 itself in SITL, ``eval/``); the gains are the Structure
Hold PD's plain hand values (``envs/structure_hold/baseline.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.route_follow.env import OBS, RouteFollowEnv
from envs.sims.torch_quad import GRAVITY, quat_rotate_inverse


@dataclass(frozen=True)
class RoutePdGains:
    pos: float = 2.0  # 1/s^2
    vel: float = 2.5  # 1/s, on the velocity error to the target's
    integral: float = 0.6  # 1/s^3
    max_integral: float = 3.0  # m/s^2
    max_acc: float = 6.0  # m/s^2
    tilt: float = 4.0  # rad/s per rad
    yaw: float = 3.0  # rad/s per rad
    centering: float = 0.0  # 0: estimate only; 1: lateral and vertical error from the ranges


class RoutePd:
    def __init__(self, env: RouteFollowEnv, gains: RoutePdGains | None = None):
        self.env = env
        self.g = gains or RoutePdGains()
        self.err_integral = torch.zeros(env.num_envs, 3, dtype=env.dtype, device=env.device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        g, env = self.g, self.env
        err = obs[:, OBS["pos_err"]].clone()  # route frame
        if g.centering > 0:
            r, ok = obs[:, OBS["ranges"]], obs[:, OBS["range_ok"]] > 0.5
            # Where the drone sits off the centreline by the walls: left positive, up positive.
            pairs = ((1, 1, 2), (2, 3, 4))  # (error axis, plus-side beam, minus-side beam)
            for axis, plus, minus in pairs:
                both = ok[:, plus] & ok[:, minus] & (r[:, plus] < env.cfg.ranges.obs_max_m) & (
                    r[:, minus] < env.cfg.ranges.obs_max_m)
                walls = 0.5 * (r[:, minus] - r[:, plus])
                err[:, axis] = torch.where(both, (1 - g.centering) * err[:, axis]
                                           + g.centering * walls, err[:, axis])
        vel = obs[:, OBS["vel"]]
        target_vel = env.to_route(env.target_vel)
        self.err_integral[env.episode_step == 0] = 0.0
        near = err.norm(dim=-1, keepdim=True) < 1.0
        self.err_integral += torch.where(near, err, torch.zeros_like(err)) * env.dt
        lim = g.max_integral / max(g.integral, 1e-9)
        self.err_integral.clamp_(-lim, lim)
        acc = -g.pos * err - g.vel * (vel - target_vel) - g.integral * self.err_integral
        acc = env.from_route(acc.clamp(-g.max_acc, g.max_acc))
        return acceleration_to_action(env, obs, acc, tilt=g.tilt, yaw=g.yaw)


def acceleration_to_action(env: RouteFollowEnv, obs: torch.Tensor, acc: torch.Tensor, *,
                           tilt: float, yaw: float) -> torch.Tensor:
    """World acceleration setpoint -> thrust and body rates, heading along the route frame's x
    (as ``envs/structure_hold/baseline.py``, which faces the wall instead)."""
    heading = obs[:, OBS["heading"]]
    n = obs.shape[0]
    mass, max_thrust, max_rate = env.cfg.quad.mass, env.cfg.quad.max_thrust, env.cfg.max_rate_rad_s
    f_world = acc * mass
    f_world[:, 2] += mass * GRAVITY
    f_norm = f_world.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z_des_body = quat_rotate_inverse(env.state.quat, f_world / f_norm)
    z_body = torch.zeros(n, 3, dtype=obs.dtype, device=obs.device)
    z_body[:, 2] = 1.0
    rates = tilt * torch.cross(z_body, z_des_body, dim=-1)
    rates[:, 2] = -yaw * torch.atan2(heading[:, 1], heading[:, 0])
    thrust = (f_norm.squeeze(-1) * z_des_body[:, 2]).clamp(min=0.0)  # along the body z
    action = torch.empty(n, 4, dtype=obs.dtype, device=obs.device)
    action[:, :3] = rates / max_rate
    action[:, 3] = 2.0 * thrust / max_thrust - 1.0
    return action.clamp(-1.0, 1.0)
