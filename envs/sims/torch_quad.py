"""Batched rigid-body quadrotor in PyTorch: the mock simulator behind the seam.

One body per env, semi-implicit Euler at ``dt / substeps``, a first-order motor lag on the
commanded thrust and torque, and nothing else: no drag (the disturbance model's force
already includes the drone's own drag), no ground, no contact. The task (envs/structure_hold)
decides what a crash is.

    sim = TorchQuadSim(QuadParams(), num_envs=1024, dt=0.01)
    sim.reset()                                     # all envs level at the origin
    state = sim.step(thrust, torque_body, force_ext_world)

Inputs per step are ``(N,)`` collective thrust in N along body +z, ``(N, 3)`` body torque in
N m (FLU axes) and ``(N, 3)`` external force in N (world ENU). Thrust is clipped to
``[0, max_thrust]`` and torque to ``±max_torque`` before the motor lag, as a real airframe's
mixer would. Output is a ``QuadState`` (world ENU, body FLU, quaternion w first).

The default airframe is a placeholder 5-inch-class quad (0.8 kg); the real numbers come with
the airframe choice (docs/decisions.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from envs.sims.quad_state import QuadState

GRAVITY = 9.80665  # m/s^2


@dataclass(frozen=True)
class QuadParams:
    mass: float = 0.8  # kg
    inertia: tuple[float, float, float] = (0.0035, 0.0035, 0.0060)  # kg m^2, body FLU diagonal
    max_thrust: float = 16.0  # N collective; about 2x weight for the placeholder airframe
    max_torque: tuple[float, float, float] = (0.30, 0.30, 0.05)  # N m per body axis
    motor_tau: float = 0.02  # s, first-order lag on thrust and torque; 0 disables it
    substeps: int = 2  # physics steps per ``step`` call

    def __post_init__(self) -> None:
        if self.mass <= 0 or self.max_thrust <= 0 or self.substeps < 1 or self.motor_tau < 0:
            raise ValueError("mass, max_thrust and substeps must be positive, motor_tau >= 0")
        if len(self.inertia) != 3 or min(self.inertia) <= 0:
            raise ValueError("inertia must be three positive diagonal terms")
        if len(self.max_torque) != 3 or min(self.max_torque) <= 0:
            raise ValueError("max_torque must be three positive values")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> QuadParams:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown quad params: {sorted(unknown)}")
        for key in ("inertia", "max_torque"):
            if key in d:
                d[key] = tuple(float(x) for x in d[key])
        return cls(**d)

    @property
    def hover_thrust(self) -> float:
        return self.mass * GRAVITY


# ---------------------------------------------------------------- quaternion helpers
# (w, x, y, z), unit norm, rotating body to world. Shared with the task code.


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body vectors ``v`` (N, 3) into the world frame."""
    w = q[:, :1]
    u = q[:, 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors ``v`` (N, 3) into the body frame."""
    return quat_rotate(torch.cat([q[:, :1], -q[:, 1:]], -1), v)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.cat([torch.cos(half)[:, None], axis * torch.sin(half)[:, None]], -1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(0.5 * yaw), z, z, torch.sin(0.5 * yaw)], -1)


def quat_integrate(q: torch.Tensor, ang_vel_body: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact rotation by ``ang_vel_body * dt`` (constant rate over the step), renormalised."""
    angle = ang_vel_body.norm(dim=-1) * dt
    safe = torch.where(angle > 0, angle, torch.ones_like(angle))
    axis = ang_vel_body * (dt / safe)[:, None]
    dq = quat_from_axis_angle(axis, angle)
    dq = torch.where((angle > 0)[:, None], dq, torch.tensor([1.0, 0, 0, 0], dtype=q.dtype,
                                                            device=q.device).expand_as(dq))
    out = quat_mul(q, dq)
    return out / out.norm(dim=-1, keepdim=True)


def body_z_world(q: torch.Tensor) -> torch.Tensor:
    """The body +z (thrust) axis in world coordinates, (N, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


# ---------------------------------------------------------------- the simulator


class TorchQuadSim:
    """``num_envs`` independent quadrotors stepped together. Not a torch.nn.Module; no grads."""

    backend = "torch"

    def __init__(
        self,
        params: QuadParams,
        num_envs: int,
        dt: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs must be >= 1 and dt > 0")
        self.params = params
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.h = self.dt / params.substeps
        self.inertia = torch.tensor(params.inertia, dtype=dtype, device=self.device)
        self.max_torque = torch.tensor(params.max_torque, dtype=dtype, device=self.device)
        self.gravity = torch.tensor([0.0, 0.0, -GRAVITY], dtype=dtype, device=self.device)
        # Motor lag as an exact first-order step: x += (1 - exp(-h/tau)) * (cmd - x).
        self.lag = 1.0 if params.motor_tau == 0 else 1.0 - math.exp(-self.h / params.motor_tau)
        self._state = QuadState.hover(self.num_envs, device=self.device, dtype=dtype)
        self.thrust = torch.zeros(self.num_envs, dtype=dtype, device=self.device)
        self.torque = torch.zeros(self.num_envs, 3, dtype=dtype, device=self.device)
        self.reset()

    @property
    def state(self) -> QuadState:
        return self._state

    def _ids(self, env_ids: Any) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.dtype == torch.bool:
            ids = ids.nonzero(as_tuple=False).squeeze(-1)
        return ids.long()

    def reset(self, env_ids: Any = None, state: QuadState | None = None) -> None:
        """Put ``env_ids`` (default all) at ``state``: rows for those envs only, or a full
        ``(num_envs, ...)`` state from which the rows are taken. ``None`` means level and at
        rest at the origin. Motors restart at hover thrust and zero torque."""
        ids = self._ids(env_ids)
        n = int(ids.numel())
        if n == 0:
            return
        if state is None:
            new = QuadState.hover(n, device=self.device, dtype=self.dtype)
        else:
            new = QuadState(*(torch.as_tensor(x, dtype=self.dtype, device=self.device)
                              for x in state))
            if new.num_envs == self.num_envs and n != self.num_envs:
                new = QuadState(*(x[ids] for x in new))
            if new.num_envs != n:
                raise ValueError(f"state has {new.num_envs} rows, expected {n} or {self.num_envs}")
            if not bool(new.finite().all()):
                raise ValueError("reset state must be finite")
        q = new.quat / new.quat.norm(dim=-1, keepdim=True)
        self._state.pos[ids] = new.pos
        self._state.quat[ids] = q
        self._state.vel[ids] = new.vel
        self._state.ang_vel[ids] = new.ang_vel
        self.thrust[ids] = self.params.hover_thrust
        self.torque[ids] = 0.0

    def step(self, thrust: Any, torque_body: Any, force_ext_world: Any) -> QuadState:
        """Advance every env by ``dt``. See the module docstring for shapes and units."""
        p = self.params
        thrust_cmd = torch.as_tensor(thrust, dtype=self.dtype, device=self.device)
        torque_cmd = torch.as_tensor(torque_body, dtype=self.dtype, device=self.device)
        f_ext = torch.as_tensor(force_ext_world, dtype=self.dtype, device=self.device)
        if thrust_cmd.shape != (self.num_envs,):
            raise ValueError(f"thrust must be ({self.num_envs},), got {tuple(thrust_cmd.shape)}")
        for name, x in (("torque_body", torque_cmd), ("force_ext_world", f_ext)):
            if x.shape != (self.num_envs, 3):
                raise ValueError(f"{name} must be ({self.num_envs}, 3), got {tuple(x.shape)}")
        thrust_cmd = thrust_cmd.clamp(0.0, p.max_thrust)
        torque_cmd = torch.maximum(torch.minimum(torque_cmd, self.max_torque), -self.max_torque)

        pos, quat, vel, ang_vel = self._state
        inv_m = 1.0 / p.mass
        for _ in range(p.substeps):
            self.thrust = self.thrust + self.lag * (thrust_cmd - self.thrust)
            self.torque = self.torque + self.lag * (torque_cmd - self.torque)
            # Translation: thrust along body z, gravity, external force. No drag here.
            acc = (body_z_world(quat) * self.thrust[:, None] + f_ext) * inv_m + self.gravity
            vel = vel + acc * self.h
            pos = pos + vel * self.h
            # Rotation: Euler's equation with a diagonal inertia.
            gyro = torch.cross(ang_vel, ang_vel * self.inertia, dim=-1)
            ang_acc = (self.torque - gyro) / self.inertia
            ang_vel = ang_vel + ang_acc * self.h
            quat = quat_integrate(quat, ang_vel, self.h)
        self._state = QuadState(pos, quat, vel, ang_vel)
        return self._state
