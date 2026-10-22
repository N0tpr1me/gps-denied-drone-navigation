"""Brake-and-hover fallback around any controller once the estimate has been stale too long.

The policy flies while the estimate is fresh or only briefly stale. Once the estimate has been
stale (no fresh fix) for ``threshold_s``, a classical law takes over: it brakes on the
dead-reckoned velocity (acceleration setpoint ``-brake_vel * v_est``, no position term,
because the dead-reckoned position is the least trustworthy part of the estimate) and so
settles into a level hover; it hands back the moment a fresh fix arrives. This is the mock
of the classical fallback the PX4 module would run next to the policy (``sim2real/px4``):
the rule is small enough to write in C++ unchanged, and the numbers measured here say what
it buys and what it costs in hold error.

The stale time is counted here from the observation's stale flag, as the module would count
it from estimate timestamps; the observation's own ``age`` entry is capped at 1 s and so
cannot carry a 2 s threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from envs.structure_hold.baseline import acceleration_to_action
from envs.structure_hold.env import OBS, StructureHoldEnv

Controller = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FallbackRule:
    threshold_s: float = 1.0  # take over once the estimate has been stale this long
    brake_vel: float = 2.5  # 1/s: acceleration per m/s of dead-reckoned velocity
    max_acc: float = 6.0  # m/s^2 command limit
    tilt: float = 8.0  # rad/s per rad of tilt error (as the PD yardstick)
    yaw: float = 3.0  # rad/s per rad of heading error

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FallbackRule:
        return cls(**(d or {}))


class StaleFallback:
    """Wrap ``inner`` (any ``obs -> actions`` callable) with the brake-and-hover fallback.
    The inner controller is still called every step so its history stays current for the
    hand-back."""

    def __init__(self, env: StructureHoldEnv, inner: Controller, rule: FallbackRule | None = None):
        self.env = env
        self.inner = inner
        self.rule = rule or FallbackRule()
        n = env.num_envs
        self.stale_s = torch.zeros(n, dtype=env.dtype, device=env.device)
        self.active = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.steps = 0
        self.steps_active = 0
        if hasattr(inner, "reset_history"):
            self.reset_history = inner.reset_history  # type: ignore[attr-defined]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        actions = self.inner(obs)
        r, env = self.rule, self.env
        stale = obs[:, OBS["stale"]][:, 0] > 0.5
        self.stale_s = torch.where(stale, self.stale_s + env.dt, torch.zeros_like(self.stale_s))
        self.stale_s[env.episode_step == 0] = 0.0  # a new episode starts fresh
        self.active = self.stale_s >= r.threshold_s - 1e-6
        self.steps += 1
        self.steps_active += int(self.active.sum())
        if not bool(self.active.any()):
            return actions
        vel = obs[:, OBS["vel"]]
        acc = (-r.brake_vel * vel).clamp(-r.max_acc, r.max_acc)
        brake = acceleration_to_action(env, obs, acc, tilt=r.tilt, yaw=r.yaw)
        return torch.where(self.active[:, None], brake, actions)

    @property
    def active_fraction(self) -> float:
        """Share of controller steps (over all envs) flown by the fallback."""
        return self.steps_active / max(1, self.steps * self.env.num_envs)
