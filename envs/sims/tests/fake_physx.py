"""A stand-in for PhysX behind ``IsaacQuadSim``, so the backend's glue runs without Isaac Lab.

It keeps state the way PhysX does, not the way the mock does: positions offset by per-env
origins, angular velocity in the WORLD frame, Euler's equation with the world-frame inertia
tensor, and the attitude advanced by a world-frame rotation on the left. Integration is
semi-implicit Euler (velocity first, then position), which is what PhysX does for a free body.
A frame or sign slip in ``IsaacQuadSim`` (body vs world force, body vs world rates, origins)
therefore shows up as a mismatch against the mock, which keeps everything in the body frame.
"""

from __future__ import annotations

import torch

from envs.sims.torch_quad import GRAVITY, QuadParams, quat_from_axis_angle, quat_mul, quat_rotate


def rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class FakePhysxWorld:
    def __init__(self, params: QuadParams, num_envs: int, physics_dt: float,
                 device: str | torch.device = "cpu", spacing: float = 2.0):
        self.closed = False
        self.h = float(physics_dt)
        dev = torch.device(device)
        self.mass = torch.full((num_envs,), params.mass, device=dev)
        self.inertia = torch.diag(torch.tensor(params.inertia, device=dev)).expand(
            num_envs, 3, 3).clone()
        idx = torch.arange(num_envs, device=dev, dtype=torch.float32)
        # Isaac Lab puts env origins on a grid; any non-zero offsets exercise the subtraction.
        self.origins = torch.stack([spacing * (idx % 8), spacing * (idx // 8),
                                    torch.zeros_like(idx)], -1)
        self.pos_w = self.origins.clone()
        self.quat = torch.zeros(num_envs, 4, device=dev)
        self.quat[:, 0] = 1.0
        self.vel_w = torch.zeros(num_envs, 3, device=dev)
        self.ang_vel_w = torch.zeros(num_envs, 3, device=dev)
        self.gravity = torch.tensor([0.0, 0.0, -GRAVITY], device=dev)
        self.steps = 0

    def set_mass_properties(self, ids, mass, inertia) -> None:
        self.mass[ids] = mass
        self.inertia[ids] = torch.diag_embed(inertia)

    def write_state(self, ids, pos, quat, vel_w, ang_vel_w) -> None:
        self.pos_w[ids] = pos + self.origins[ids]
        self.quat[ids] = quat
        self.vel_w[ids] = vel_w
        self.ang_vel_w[ids] = ang_vel_w

    def step(self, force_b: torch.Tensor, torque_b: torch.Tensor) -> None:
        r = rotation_matrix(self.quat)
        acc = (r @ force_b[..., None])[..., 0] / self.mass[:, None] + self.gravity
        self.vel_w = self.vel_w + acc * self.h
        self.pos_w = self.pos_w + self.vel_w * self.h
        i_w = r @ self.inertia @ r.transpose(-1, -2)
        torque_w = (r @ torque_b[..., None])[..., 0]
        w = self.ang_vel_w
        gyro = torch.cross(w, (i_w @ w[..., None])[..., 0], dim=-1)
        ang_acc = torch.linalg.solve(i_w, (torque_w - gyro)[..., None])[..., 0]
        self.ang_vel_w = w + ang_acc * self.h
        angle = self.ang_vel_w.norm(dim=-1) * self.h
        safe = torch.where(angle > 0, angle, torch.ones_like(angle))
        dq = quat_from_axis_angle(self.ang_vel_w * (self.h / safe)[:, None], angle)
        q = quat_mul(dq, self.quat)  # world-frame increment on the left
        self.quat = q / q.norm(dim=-1, keepdim=True)
        self.steps += 1

    def read(self):
        conj = torch.cat([self.quat[:, :1], -self.quat[:, 1:]], -1)
        return (self.pos_w - self.origins, self.quat.clone(), self.vel_w.clone(),
                quat_rotate(conj, self.ang_vel_w))

    def close(self) -> None:
        self.closed = True
