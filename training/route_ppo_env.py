"""The Route Follow env in the shape rsl_rl 5.x trains on (as ``ppo_env.py`` does for
Structure Hold): ``policy`` is the last ``history`` observations stacked, oldest first;
``privileged`` is what only the simulator knows, for the critic (asymmetric actor-critic)."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from envs.route_follow.env import NUM_OBS, RouteFollowEnv
from envs.sims.torch_quad import GRAVITY
from training.ppo_env import ObsHistory

# True error to the target (3), true velocity (3), wind / weight (3), estimate error (3), all
# in the route frame, and the true clearance (1).
PRIVILEGED_OBS = 13


class RouteFollowVecEnv:
    def __init__(self, env: RouteFollowEnv, history: int = 3):
        self.env = env
        self.cfg = env.cfg
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.history = ObsHistory(env.num_envs, NUM_OBS, history, env.device)
        self.num_policy_obs = self.history.size
        self.num_privileged_obs = PRIVILEGED_OBS
        self._weight = env.cfg.quad.mass * GRAVITY
        obs, _ = env.get_observations()
        self.history.fill(obs)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_step

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # rsl_rl randomises episode phase at the start; the target follows the clock.
        e = self.env
        e.episode_step[:] = value.to(e.episode_step.dtype)
        e.episode_time[:] = e.episode_step.to(e.dtype) * e.dt
        e._update_target()

    def privileged(self) -> torch.Tensor:
        e = self.env
        s = e.state
        return torch.cat([
            e.to_route(s.pos - e.target_pos),
            e.to_route(s.vel),
            e.to_route(e.wind_force) / self._weight,
            e.to_route(e.estimate_pos - s.pos),
            e.geo.clearance(s.pos)[:, None],
        ], dim=-1)

    def observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self.history.stacked(), "privileged": torch.nan_to_num(self.privileged())},
            batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        return self.observations()

    def reset(self) -> TensorDict:
        obs, _ = self.env.reset()
        self.history.fill(obs)
        return self.observations()

    def step(self, actions: torch.Tensor):
        obs, reward, dones, extras = self.env.step(actions)
        self.history.push(obs, dones)
        return self.observations(), reward, dones, extras

    def curriculum_state(self) -> dict[str, float]:
        return {}
