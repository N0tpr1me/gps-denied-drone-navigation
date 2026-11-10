"""PyTorch adapter for the estimate corruption core: same update, tensors on any device.

Only the backend hooks differ from ``core.EstimateCorruptor``: arrays are tensors with the
given dtype and device, and draws come from a seeded ``torch.Generator``. Draws from the
two backends differ for the same seed; pass the same ``StepNoise`` to both to compare.
"""

from __future__ import annotations

from typing import Any

import torch

from envs.estimate.core import EstimateCorruptionConfig, EstimateCorruptor, StepNoise
from envs.estimate.heading import HeadingError, HeadingErrorConfig


class TorchEstimateCorruptor(EstimateCorruptor):
    xp = torch

    def __init__(
        self,
        cfg: EstimateCorruptionConfig,
        num_envs: int,
        dt: float,
        seed: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        super().__init__(cfg, num_envs, dt, seed)

    def _init_rng(self, seed: int) -> None:
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)

    def _zeros(self, shape: tuple[int, ...], integer: bool = False) -> torch.Tensor:
        dtype = torch.int64 if integer else self.dtype
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def _arange(self, n: int) -> torch.Tensor:
        return torch.arange(n, device=self.device)

    def _as_array(self, x: Any) -> torch.Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def _to_int(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int64)

    def _as_mask(self, env_ids: Any) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)] = True
        return mask

    def draw_step_noise(self) -> StepNoise:
        n, kw = self.num_envs, {"generator": self._rng, "device": self.device, "dtype": self.dtype}
        return StepNoise(
            bias=torch.randn((n, 3), **kw),
            jump_event=torch.rand(n, **kw),
            jump_size=torch.rand(n, **kw),
            jump_dir=torch.randn((n, 3), **kw),
            pos=torch.randn((n, 3), **kw),
            vel=torch.randn((n, 3), **kw),
            dropout_event=torch.rand(n, **kw),
            dropout_len=torch.rand(n, **kw),
            dropout_accel=torch.randn((n, 3), **kw),
        )

    def draw_delay_steps(self) -> torch.Tensor:
        return torch.randint(
            self.min_delay_steps,
            self.max_delay_steps + 1,
            (self.num_envs,),
            generator=self._rng,
            device=self.device,
        )

    def step(self, pos: Any, vel: Any, noise: StepNoise | None = None):
        if noise is not None:
            noise = StepNoise(*(self._as_array(x) for x in noise))
        return super().step(pos, vel, noise)


class TorchHeadingError(HeadingError):
    """``heading.HeadingError`` on tensors; draws from a seeded ``torch.Generator``."""

    xp = torch

    def __init__(self, cfg: HeadingErrorConfig, num_envs: int, dt: float, seed: int,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        super().__init__(cfg, num_envs, dt, seed)

    def _init_rng(self, seed: int) -> None:
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(seed) * 7919 + 0x4EAD)

    def _zeros(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self._rng, device=self.device, dtype=self.dtype)

    def _as_mask(self, env_ids: Any) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ids = env_ids if isinstance(env_ids, torch.Tensor) else list(env_ids)
        mask[torch.as_tensor(ids, device=self.device, dtype=torch.int64)] = True
        return mask
