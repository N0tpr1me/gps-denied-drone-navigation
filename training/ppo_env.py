"""The Structure Hold env in the shape rsl_rl 5.x trains on, plus the observation history.

rsl_rl's runner wants ``get_observations()`` and ``step()`` to return a ``TensorDict`` of
observation *groups* that its config maps onto the actor and the critic. This wrapper adds
two things to ``envs.structure_hold.StructureHoldEnv`` without touching it:

* ``policy``: the last ``history`` env observations stacked, oldest first, so the newest
  observation is the last ``NUM_OBS`` values. A short history lets a plain MLP see how the
  estimate moves (its delay and jumps) without a recurrent network. The onboard module keeps
  the same buffer; ``ObsHistory`` is the reference for it and for the ONNX metadata.
* ``privileged``: what only the simulator knows (true position error, true velocity, the
  wind force, the estimate's error). The critic sees it, the actor never does, so the value
  estimate is sharper while the deployed policy stays honest (asymmetric actor-critic).

Episode bookkeeping (``extras["time_outs"]``, ``extras["episode"]``) passes through.

``DropoutCurriculum`` is the one training-time curriculum: the longest estimate dropout
starts short and grows linearly to the env config's value over the first part of training,
so the policy learns to coast through a stale estimate on easy cases before the long ones
(the remaining crash cause after the bounded reward) arrive. The wrapper applies it from
its own step count, so the runner needs no hook.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from envs.sims.torch_quad import GRAVITY
from envs.structure_hold.env import NUM_OBS, StructureHoldEnv

PRIVILEGED_OBS = 12  # true pos error (3), true vel (3), wind / weight (3), estimate error (3)


class ObsHistory:
    """A rolling window of the last ``length`` observations per env, oldest first.

    On an env's reset the whole window is filled with that episode's first observation, so
    the policy never sees the previous episode leaking across the boundary.
    """

    def __init__(self, num_envs: int, num_obs: int, length: int,
                 device: str | torch.device = "cpu"):
        if length < 1:
            raise ValueError("history length must be >= 1")
        self.length = int(length)
        self.num_obs = int(num_obs)
        self.buf = torch.zeros(num_envs, self.length, self.num_obs, device=device)

    @property
    def size(self) -> int:
        """Size of the stacked vector the policy consumes."""
        return self.length * self.num_obs

    def fill(self, obs: torch.Tensor, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            self.buf[:] = obs[:, None, :]
        elif bool(env_mask.any()):
            self.buf[env_mask] = obs[env_mask][:, None, :]

    def push(self, obs: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Append ``obs``; envs flagged in ``reset_mask`` get the window refilled instead."""
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1] = obs
        if reset_mask is not None:
            self.fill(obs, reset_mask)
        return self.stacked()

    def stacked(self) -> torch.Tensor:
        return self.buf.reshape(self.buf.shape[0], self.size)


class DropoutCurriculum:
    """Dropout cap in seconds as a function of the training iteration: ``start_s`` at
    iteration 0, rising linearly to ``end_s`` at ``iterations`` and flat after that."""

    def __init__(self, start_s: float, end_s: float, iterations: int):
        if start_s <= 0 or end_s < start_s or iterations < 1:
            raise ValueError("curriculum needs 0 < start_s <= end_s and iterations >= 1")
        self.start_s, self.end_s, self.iterations = float(start_s), float(end_s), int(iterations)

    def value(self, iteration: int) -> float:
        f = min(max(iteration / self.iterations, 0.0), 1.0)
        return self.start_s + f * (self.end_s - self.start_s)


class StructureHoldVecEnv:
    """rsl_rl ``VecEnv`` view of a ``StructureHoldEnv``.

    ``curriculum`` with ``steps_per_iteration`` (the runner's ``num_steps_per_env``) moves the
    env's dropout cap as training proceeds; ``curriculum_state()`` reports it for logging.
    """

    def __init__(self, env: StructureHoldEnv, history: int = 3,
                 curriculum: DropoutCurriculum | None = None, steps_per_iteration: int = 1,
                 num_obs: int = NUM_OBS, start_iteration: int = 0):
        self.env = env
        # The policy reads the first num_obs of the env's observation (NUM_OBS_V1 for the
        # 20-input layout every policy up to v21 has).
        self.num_obs = int(num_obs)
        if not 0 < self.num_obs <= NUM_OBS:
            raise ValueError(f"num_obs must be in 1..{NUM_OBS}")
        self.curriculum = curriculum
        self.steps_per_iteration = max(1, int(steps_per_iteration))
        self.env_steps = 0
        self.start_iteration = int(start_iteration)
        self.iteration = self.start_iteration
        if curriculum is not None:
            env.set_dropout_max_s(curriculum.value(self.iteration))
        self.cfg = env.cfg
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.history = ObsHistory(env.num_envs, self.num_obs, history, env.device)
        self.num_policy_obs = self.history.size
        self.num_privileged_obs = PRIVILEGED_OBS
        self._weight = env.cfg.quad.mass * GRAVITY
        obs, _ = env.get_observations()
        self.history.fill(obs[:, : self.num_obs])

    # rsl_rl's runner reads and (optionally) writes this to randomise episode phase.
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_step

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_step[:] = value.to(self.env.episode_step.dtype)
        self.env.episode_time[:] = self.env.episode_step.to(self.env.dtype) * self.env.dt
        self.env.engage_time[:] = self.env.episode_time

    def privileged(self) -> torch.Tensor:
        e = self.env
        state = e.state
        return torch.cat(
            [
                state.pos - e.hold_point,
                state.vel,
                e.wind_force / self._weight,
                e.estimate_pos - state.pos,
            ],
            dim=-1,
        )

    def observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self.history.stacked(), "privileged": torch.nan_to_num(self.privileged())},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self.observations()

    def reset(self) -> TensorDict:
        obs, _ = self.env.reset()
        self.history.fill(obs[:, : self.num_obs])
        return self.observations()

    def step(self, actions: torch.Tensor):
        obs, reward, dones, extras = self.env.step(actions)
        # A re-engage (env cfg.reengage_rate_hz) refills the window as the module's engage does.
        self.history.push(obs[:, : self.num_obs], dones | self.env.reengaged)
        self.env_steps += 1
        iteration = self.start_iteration + self.env_steps // self.steps_per_iteration
        if iteration != self.iteration:
            self.iteration = iteration
            if self.curriculum is not None:
                self.env.set_dropout_max_s(self.curriculum.value(iteration))
        return self.observations(), reward, dones, extras

    def set_start_iteration(self, iteration: int) -> None:
        """A resumed run goes on from the runner's iteration: the curriculum (indexed by
        the iteration) continues where the run it resumes left it, instead of restarting."""
        self.start_iteration = self.iteration = int(iteration)
        self.env_steps = 0
        if self.curriculum is not None:
            self.env.set_dropout_max_s(self.curriculum.value(self.iteration))

    def curriculum_state(self) -> dict[str, float]:
        return {"dropout_max_s": self.env.dropout_max_s}
