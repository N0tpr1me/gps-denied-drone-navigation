"""Corrupt a clean position/velocity estimate the way a GPS-free (VIO) estimate degrades.

Pure batched transform, NumPy only. Each ``step`` takes the true position and velocity
of ``num_envs`` vehicles (ENU, m and m/s) and returns what the vehicle's estimator would
report, in this order:

1. **Delay**: the sample from ``delay_steps`` ago (per env, fixed between resets).
2. **Bias random walk** on position, clipped to ``bias_max_m`` per axis.
3. **Jumps** (relocalisation): Poisson events adding a step of 0.1 to 2 m in a random
   direction, which then decays back with time constant ``jump_decay_s``.
4. **White noise** on position and velocity.
5. **Dropouts**: Poisson events during which the output holds its last value and
   ``stale`` is True, for 0.2 to 3 s.

Randomness comes only from the generator seeded in the constructor, or from a
``StepNoise`` passed to ``step`` (how the NumPy/PyTorch parity tests share draws). The
update itself (``_step``) is written against the operations NumPy and PyTorch share, so
``torch_adapter.TorchEstimateCorruptor`` reuses it unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import numpy as np


@dataclass(frozen=True)
class EstimateCorruptionConfig:
    # Defaults are the VIO-like corruption the SITL benchmark and training share; each
    # number is justified in docs/decisions.md (estimate corruption defaults row).
    # Latency of the estimate, drawn per env at reset, rounded to whole steps.
    delay_min_s: float = 0.05
    delay_max_s: float = 0.15
    # White noise, per axis.
    pos_noise_std_m: float = 0.02
    vel_noise_std_mps: float = 0.05
    # Position bias random walk: std grows as bias_walk_std * sqrt(t), clipped per axis.
    bias_walk_std_m_per_sqrt_s: float = 0.01
    bias_max_m: float = 0.3
    # Jumps: rate in events per second, step size uniform in [min, max] metres.
    jump_rate_hz: float = 0.02
    jump_min_m: float = 0.1
    jump_max_m: float = 2.0
    # The jump offset decays back to zero with this time constant; None keeps it forever.
    jump_decay_s: float | None = 5.0
    # Dropouts: rate in events per second, duration uniform in [min, max] seconds.
    dropout_rate_hz: float = 0.02
    dropout_min_s: float = 0.2
    dropout_max_s: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EstimateCorruptionConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown estimate corruption keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def clean(cls) -> EstimateCorruptionConfig:
        """No corruption at all: the output equals the input."""
        return cls(
            delay_min_s=0.0,
            delay_max_s=0.0,
            pos_noise_std_m=0.0,
            vel_noise_std_mps=0.0,
            bias_walk_std_m_per_sqrt_s=0.0,
            bias_max_m=0.0,
            jump_rate_hz=0.0,
            dropout_rate_hz=0.0,
        )

    def validate(self, dt: float) -> None:
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}")
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None and not value >= 0:
                raise ValueError(f"{f.name} must be >= 0, got {value}")
        for lo, hi in [
            ("delay_min_s", "delay_max_s"),
            ("jump_min_m", "jump_max_m"),
            ("dropout_min_s", "dropout_max_s"),
        ]:
            if getattr(self, lo) > getattr(self, hi):
                raise ValueError(f"{lo} must be <= {hi}")
        if self.jump_decay_s is not None and not self.jump_decay_s > 0:
            raise ValueError("jump_decay_s must be positive or None")
        lo, hi = _dropout_step_range(self, dt)
        if self.dropout_rate_hz > 0 and lo > hi:
            raise ValueError(
                f"no whole number of {dt} s steps fits in "
                f"[{self.dropout_min_s}, {self.dropout_max_s}] s"
            )


def _dropout_step_range(cfg: EstimateCorruptionConfig, dt: float) -> tuple[int, int]:
    # Durations are whole steps; keep steps * dt inside the configured range.
    lo = max(1, math.ceil(cfg.dropout_min_s / dt - 1e-9))
    hi = math.floor(cfg.dropout_max_s / dt + 1e-9)
    return lo, hi


class StepNoise(NamedTuple):
    """Every random draw one step uses. Shapes are (num_envs, 3) or (num_envs,)."""

    bias: Any  # standard normal, (N, 3)
    jump_event: Any  # uniform [0, 1), (N,)
    jump_size: Any  # uniform [0, 1), (N,)
    jump_dir: Any  # standard normal, (N, 3); normalised to a direction
    pos: Any  # standard normal, (N, 3)
    vel: Any  # standard normal, (N, 3)
    dropout_event: Any  # uniform [0, 1), (N,)
    dropout_len: Any  # uniform [0, 1), (N,)


class EstimateSample(NamedTuple):
    pos: Any  # (N, 3) reported position, m, ENU
    vel: Any  # (N, 3) reported velocity, m/s, ENU
    stale: Any  # (N,) bool, True while in a dropout (pos and vel are held)
    age: Any  # (N,) seconds since the output last refreshed (0 when fresh)
    jumped: Any  # (N,) bool, a jump started this step
    dropout_started: Any  # (N,) bool, a dropout started this step


class EstimateCorruptor:
    """Batched estimate corruption on NumPy arrays (float64)."""

    xp = np

    def __init__(
        self,
        cfg: EstimateCorruptionConfig,
        num_envs: int,
        dt: float,
        seed: int,
    ):
        cfg.validate(dt)
        self.cfg = cfg
        self.num_envs = num_envs
        self.dt = float(dt)
        self._init_rng(seed)

        self.min_delay_steps = round(cfg.delay_min_s / dt)
        self.max_delay_steps = round(cfg.delay_max_s / dt)
        self.dropout_steps_min, self.dropout_steps_max = _dropout_step_range(cfg, dt)
        self.p_jump = 1.0 - math.exp(-cfg.jump_rate_hz * dt)
        self.p_dropout = 1.0 - math.exp(-cfg.dropout_rate_hz * dt)
        self.jump_keep = 1.0 if cfg.jump_decay_s is None else math.exp(-dt / cfg.jump_decay_s)
        self.bias_step_std = cfg.bias_walk_std_m_per_sqrt_s * math.sqrt(dt)

        n, length = num_envs, self.max_delay_steps + 1
        self.head = 0
        self.buf_pos = self._zeros((length, n, 3))
        self.buf_vel = self._zeros((length, n, 3))
        self.delay_steps = self._zeros((n,), integer=True)
        self.bias = self._zeros((n, 3))
        self.jump_offset = self._zeros((n, 3))
        self.held_pos = self._zeros((n, 3))
        self.held_vel = self._zeros((n, 3))
        self.dropout_left = self._zeros((n,), integer=True)
        self.age = self._zeros((n,))
        self._started = self._zeros((n,), integer=True) > 0  # all False until reset

    # Backend hooks: the only methods the PyTorch adapter overrides.

    def _init_rng(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _zeros(self, shape: tuple[int, ...], integer: bool = False) -> np.ndarray:
        return np.zeros(shape, dtype=np.int64 if integer else np.float64)

    def _arange(self, n: int) -> np.ndarray:
        return np.arange(n)

    def _as_array(self, x: Any) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def _to_int(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.int64)

    def _as_mask(self, env_ids: Any) -> np.ndarray:
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[np.asarray(env_ids)] = True
        return mask

    def draw_step_noise(self) -> StepNoise:
        r, n = self._rng, self.num_envs
        return StepNoise(
            bias=r.standard_normal((n, 3)),
            jump_event=r.random(n),
            jump_size=r.random(n),
            jump_dir=r.standard_normal((n, 3)),
            pos=r.standard_normal((n, 3)),
            vel=r.standard_normal((n, 3)),
            dropout_event=r.random(n),
            dropout_len=r.random(n),
        )

    def draw_delay_steps(self) -> np.ndarray:
        return self._rng.integers(self.min_delay_steps, self.max_delay_steps + 1, self.num_envs)

    # Shared logic.

    def reset(self, pos: Any, vel: Any, env_ids: Any = None, delay_steps: Any = None) -> None:
        """Start (or restart) envs at the given true state, with no corruption carried over.

        ``pos`` and ``vel`` are (num_envs, 3); only rows in ``env_ids`` (default: all) are
        used. ``delay_steps`` (num_envs,) overrides the random per-env delay draw.
        """
        xp = self.xp
        pos, vel = self._as_array(pos), self._as_array(vel)
        mask = self._as_mask(self._arange(self.num_envs) if env_ids is None else env_ids)
        if delay_steps is None:
            delay_steps = self.draw_delay_steps()
        delay_steps = self._to_int(self._as_array(delay_steps))
        if bool((delay_steps < 0).any()) or bool((delay_steps > self.max_delay_steps).any()):
            raise ValueError(f"delay_steps must be in [0, {self.max_delay_steps}]")
        m3 = mask[:, None]
        # Fill the whole history with the start state so the first outputs are well defined.
        self.buf_pos = xp.where(mask[None, :, None], pos[None], self.buf_pos)
        self.buf_vel = xp.where(mask[None, :, None], vel[None], self.buf_vel)
        self.delay_steps = xp.where(mask, delay_steps, self.delay_steps)
        self.bias = xp.where(m3, 0.0, self.bias)
        self.jump_offset = xp.where(m3, 0.0, self.jump_offset)
        self.held_pos = xp.where(m3, pos, self.held_pos)
        self.held_vel = xp.where(m3, vel, self.held_vel)
        self.dropout_left = xp.where(mask, 0, self.dropout_left)
        self.age = xp.where(mask, 0.0, self.age)
        self._started = self._started | mask

    def step(self, pos: Any, vel: Any, noise: StepNoise | None = None) -> EstimateSample:
        """Advance one step of ``dt`` with the true state now; return the reported state."""
        if not bool(self._started.all()):
            raise RuntimeError("call reset() for every env before step()")
        if noise is None:
            noise = self.draw_step_noise()
        return self._step(self._as_array(pos), self._as_array(vel), noise)

    def _step(self, pos: Any, vel: Any, nz: StepNoise) -> EstimateSample:
        xp, cfg = self.xp, self.cfg
        length = self.max_delay_steps + 1

        # 1. Delay: write now, read delay_steps back.
        self.head = (self.head + 1) % length
        self.buf_pos[self.head] = pos
        self.buf_vel[self.head] = vel
        read = (self.head - self.delay_steps) % length
        cols = self._arange(self.num_envs)
        d_pos = self.buf_pos[read, cols]
        d_vel = self.buf_vel[read, cols]

        # 2. Bias random walk.
        b = cfg.bias_max_m
        self.bias = xp.clip(self.bias + self.bias_step_std * nz.bias, -b, b)

        # 3. Jumps: decay the old offset, then add any new step.
        jumped = nz.jump_event < self.p_jump
        norm = xp.sqrt((nz.jump_dir * nz.jump_dir).sum(-1))[:, None]
        direction = nz.jump_dir / xp.where(norm > 1e-12, norm, 1.0)
        size = cfg.jump_min_m + (cfg.jump_max_m - cfg.jump_min_m) * nz.jump_size
        self.jump_offset = self.jump_keep * self.jump_offset + xp.where(
            jumped[:, None], size[:, None] * direction, 0.0
        )

        # 4. White noise.
        m_pos = d_pos + self.bias + self.jump_offset + cfg.pos_noise_std_m * nz.pos
        m_vel = d_vel + cfg.vel_noise_std_mps * nz.vel

        # 5. Dropouts: a new one may start only when not already in one; this step counts.
        lo, hi = self.dropout_steps_min, self.dropout_steps_max
        span = hi - lo + 1
        length_steps = lo + self._to_int(xp.clip(xp.floor(nz.dropout_len * span), 0, span - 1))
        started = (self.dropout_left == 0) & (nz.dropout_event < self.p_dropout)
        self.dropout_left = xp.where(started, length_steps, self.dropout_left)
        stale = self.dropout_left > 0
        out_pos = xp.where(stale[:, None], self.held_pos, m_pos)
        out_vel = xp.where(stale[:, None], self.held_vel, m_vel)
        self.held_pos, self.held_vel = out_pos, out_vel
        self.dropout_left = xp.where(stale, self.dropout_left - 1, self.dropout_left)
        self.age = xp.where(stale, self.age + self.dt, 0.0)

        return EstimateSample(out_pos, out_vel, stale, self.age, jumped, started)
