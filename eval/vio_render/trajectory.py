"""A hold flight next to the structure, with the IMU it would measure (NumPy only).

The flight: ``ground_s`` at rest on the floor (a VIO starts on the ground, which is also
how Basalt likes to initialise), a smooth take-off and approach to the hold point
``hold_dist_m`` in front of the wall, then ``hold_s`` of holding there in wind with gust
steps and turbulence. A point mass with a PID position loop and a first-order lag on the
commanded acceleration (attitude and motor response) flies it, pushed by drag towards the
wind: the motion is that of a drone holding in gusts, small and smooth, not a dataset's
tour of a room. Attitude follows from the acceleration (a multicopter's thrust points
along its body z, so body z is ``a + g`` normalised; heading faces the wall with a small
wander), which makes the accelerometer and gyro exactly consistent with the position.

Everything is simulated at ``sim_hz`` and sampled at the IMU rate. Frames: world ENU, body
FLU (docs/conventions.md); quaternions (w, x, y, z), body to world. The IMU sits at the
body origin with the body axes; it measures specific force ``R^T (a + g z)`` and the body
angular velocity, each with white noise, a random-walk bias and a constant turn-on bias.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

G = 9.80665


@dataclass(frozen=True)
class FlightConfig:
    hold_dist_m: float = 1.5  # hold point distance from the wall (x = 0)
    hold_z_m: float = 0.0  # hold height, world z
    start_dist_m: float = 3.5  # where it sits on the floor before take-off
    floor_z_m: float = -2.0  # floor height (the scene's)
    ground_s: float = 3.0
    approach_s: float = 6.0
    hold_s: float = 70.0
    # Wind in world x (positive blows into the wall) and y (along it).
    wind_mps: tuple[float, float] = (4.0, 0.0)
    gusts_per_min: float = 4.0
    gust_mps: tuple[float, float] = (1.0, 4.0)
    gust_s: tuple[float, float] = (0.5, 3.0)
    turbulence_mps: float = 0.6  # rms per horizontal axis, low-pass filtered
    turbulence_tau_s: float = 1.0
    drag_per_s: float = 0.35  # a = drag * (wind - v), x500-like linear drag
    # Position loop (per axis) and the lag from commanded to actual acceleration.
    kp: float = 4.0
    kd: float = 3.6
    ki: float = 0.8
    accel_tau_s: float = 0.08
    yaw_wander_rad: float = 0.03
    sim_hz: float = 1000.0
    imu_hz: float = 200.0
    cam_hz: float = 20.0
    # IMU noise (densities, per sqrt(Hz)), bias walks and turn-on biases (std).
    accel_noise_density: float = 2.0e-3
    gyro_noise_density: float = 1.7e-4
    accel_bias_walk: float = 3.0e-3
    gyro_bias_walk: float = 2.0e-5
    accel_bias_std: float = 0.05
    gyro_bias_std: float = 0.005
    # Propeller vibration that gets through the IMU's mounting and filters, as extra white
    # noise the VIO is not told about (rms per axis). A soft-mounted flight controller IMU
    # on a 500 mm quad sees a few tenths of m/s^2 after its low-pass.
    vib_accel_rms: float = 0.3
    vib_gyro_rms: float = 0.01

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlightConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown flight keys: {sorted(unknown)}")
        data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**data)

    @property
    def duration_s(self) -> float:
        return self.ground_s + self.approach_s + self.hold_s


@dataclass
class Flight:
    """Truth and IMU at the IMU rate; camera frames are every ``cam_every`` IMU samples."""

    t: np.ndarray  # (N,) s from the start
    pos: np.ndarray  # (N, 3) world ENU, m
    vel: np.ndarray  # (N, 3) m/s
    quat: np.ndarray  # (N, 4) body to world, (w, x, y, z)
    gyro: np.ndarray  # (N, 3) rad/s, body, measured (noise and bias)
    accel: np.ndarray  # (N, 3) m/s^2, body, measured specific force
    gyro_true: np.ndarray  # (N, 3) rad/s, body
    hold_point: np.ndarray  # (3,)
    hold_start_s: float
    cam_every: int

    @property
    def cam_index(self) -> np.ndarray:
        return np.arange(0, self.t.size, self.cam_every)


def min_jerk(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, velocity and acceleration shape of a min-jerk move over s in [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return (10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4,
            60 * s - 180 * s**2 + 120 * s**3)


def quat_from_rot(r: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z), w >= 0."""
    m = r
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((m.shape[0], 4))
    # Shepperd's method, choosing the largest of w, x, y, z per sample.
    cand = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1)
    k = cand.argmax(1)
    for i in range(4):
        sel = k == i
        if not sel.any():
            continue
        a = m[sel]
        if i == 0:
            s = np.sqrt(1.0 + tr[sel]) * 2
            q[sel] = np.stack([0.25 * s, (a[:, 2, 1] - a[:, 1, 2]) / s,
                               (a[:, 0, 2] - a[:, 2, 0]) / s, (a[:, 1, 0] - a[:, 0, 1]) / s], 1)
        elif i == 1:
            s = np.sqrt(1.0 + a[:, 0, 0] - a[:, 1, 1] - a[:, 2, 2]) * 2
            q[sel] = np.stack([(a[:, 2, 1] - a[:, 1, 2]) / s, 0.25 * s,
                               (a[:, 0, 1] + a[:, 1, 0]) / s, (a[:, 0, 2] + a[:, 2, 0]) / s], 1)
        elif i == 2:
            s = np.sqrt(1.0 + a[:, 1, 1] - a[:, 0, 0] - a[:, 2, 2]) * 2
            q[sel] = np.stack([(a[:, 0, 2] - a[:, 2, 0]) / s, (a[:, 0, 1] + a[:, 1, 0]) / s,
                               0.25 * s, (a[:, 1, 2] + a[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(1.0 + a[:, 2, 2] - a[:, 0, 0] - a[:, 1, 1]) * 2
            q[sel] = np.stack([(a[:, 1, 0] - a[:, 0, 1]) / s, (a[:, 0, 2] + a[:, 2, 0]) / s,
                               (a[:, 1, 2] + a[:, 2, 1]) / s, 0.25 * s], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rot_from_quat(q: np.ndarray) -> np.ndarray:
    """(N, 4) (w, x, y, z) -> (N, 3, 3), body to world."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def log_so3(r: np.ndarray) -> np.ndarray:
    """(N, 3, 3) small rotations -> (N, 3) rotation vectors."""
    cos = np.clip((np.trace(r, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    v = np.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], 1)
    scale = np.where(ang < 1e-8, 0.5, ang / (2.0 * np.sin(np.maximum(ang, 1e-8))))
    return v * scale[:, None]


def attitude(acc: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Body-to-world rotations whose z axis is along ``acc + g`` with the given heading."""
    f = acc + np.array([0.0, 0.0, G])
    zb = f / np.linalg.norm(f, axis=1, keepdims=True)
    xc = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1)
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb, axis=1, keepdims=True)
    xb = np.cross(yb, zb)
    return np.stack([xb, yb, zb], 2)  # columns are the body axes in world


def _wind(cfg: FlightConfig, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N, 3) wind: the mean, gust steps (smoothed) and filtered turbulence, from the
    take-off on (no wind reaches a drone sitting on the floor, which keeps the start still)."""
    n = t.size
    dt = float(t[1] - t[0])
    w = np.zeros((n, 3))
    w[:, :2] = np.asarray(cfg.wind_mps)
    dur = cfg.duration_s
    k = rng.poisson(cfg.gusts_per_min * cfg.hold_s / 60.0)
    for _ in range(k):
        t0 = rng.uniform(cfg.ground_s + cfg.approach_s, dur)
        length = rng.uniform(*cfg.gust_s)
        speed = rng.uniform(*cfg.gust_mps)
        ang = rng.uniform(-np.pi / 3, np.pi / 3)  # mostly along the mean wind (or +x)
        base = np.asarray(cfg.wind_mps, dtype=float)
        head = np.arctan2(base[1], base[0]) if np.linalg.norm(base) > 0 else 0.0
        w[(t >= t0) & (t < t0 + length), :2] += speed * np.array(
            [np.cos(head + ang), np.sin(head + ang)])
    if cfg.turbulence_mps > 0:
        a = np.exp(-dt / cfg.turbulence_tau_s)
        z = rng.standard_normal((n, 2)) * cfg.turbulence_mps * np.sqrt(1 - a * a)
        turb = np.zeros((n, 2))
        for i in range(1, n):
            turb[i] = a * turb[i - 1] + z[i]
        w[:, :2] += turb
    # No wind reaches a drone sitting on the floor; it builds up over the first 2 s of the
    # climb, and gust edges take ~0.2 s to arrive (a real gust front is not a step), so the
    # acceleration, and with it the attitude, has no steps.
    w *= np.clip((t - cfg.ground_s) / 2.0, 0.0, 1.0)[:, None]
    a = np.exp(-dt / 0.2)
    for i in range(1, n):
        w[i] = a * w[i - 1] + (1 - a) * w[i]
    return w


def simulate(cfg: FlightConfig, seed: int) -> Flight:
    rng = np.random.default_rng([seed, 0xF1])
    dt = 1.0 / cfg.sim_hz
    t = np.arange(0.0, cfg.duration_s, dt)
    n = t.size
    hold = np.array([-cfg.hold_dist_m, 0.0, cfg.hold_z_m])
    start = np.array([-cfg.start_dist_m, 0.0, cfg.floor_z_m + 0.12])
    s = (t - cfg.ground_s) / cfg.approach_s
    sp, sv, sa = min_jerk(s)
    ref_p = start + sp[:, None] * (hold - start)
    ref_v = (sv / cfg.approach_s)[:, None] * (hold - start)
    ref_a = (sa / cfg.approach_s**2)[:, None] * (hold - start)
    wind = _wind(cfg, t, rng)

    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    pos[0] = start
    a_act = np.zeros(3)
    integ = np.zeros(3)
    lag = np.exp(-dt / cfg.accel_tau_s)
    for i in range(n - 1):
        on_ground = t[i] < cfg.ground_s
        if on_ground:
            a_cmd = np.zeros(3)
        else:
            err = ref_p[i] - pos[i]
            integ += err * dt
            a_cmd = (ref_a[i] + cfg.kp * err + cfg.kd * (ref_v[i] - vel[i])
                     + cfg.ki * integ)
        a_act = lag * a_act + (1 - lag) * a_cmd
        a = a_act + cfg.drag_per_s * (wind[i] - vel[i]) * np.array([1.0, 1.0, 0.3])
        if on_ground:
            a = np.zeros(3)
        acc[i] = a
        vel[i + 1] = vel[i] + a * dt
        pos[i + 1] = pos[i] + vel[i] * dt + 0.5 * a * dt * dt
    acc[-1] = acc[-2]

    # Heading: facing the wall (+x), a slow small wander once airborne.
    ph = rng.uniform(0, 2 * np.pi, 3)
    yaw = sum(cfg.yaw_wander_rad / 2 * np.sin(2 * np.pi * f * t + p)
              for f, p in zip((0.05, 0.13, 0.31), ph, strict=True))
    yaw = yaw * np.clip((t - cfg.ground_s) / cfg.approach_s, 0.0, 1.0)
    rot = attitude(acc, yaw)
    # Sample at the IMU rate. A real IMU's output is band-limited, so each sample is the
    # mean over its period: the rate is the rotation from this sample to the next over the
    # period, the specific force the mean of the sim steps in between. Integrating them
    # then lands on the truth at every sample instead of aliasing the sim's fine motion.
    every = int(round(cfg.sim_hz / cfg.imu_hz))
    idx = np.arange(0, n - every, every)
    ti, pi_, vi, ri = t[idx], pos[idx], vel[idx], rot[idx]
    rel_imu = np.einsum("nji,njk->nik", rot[idx], rot[idx + every])
    wi = log_so3(rel_imu) * cfg.imu_hz
    f_body = np.einsum("nji,nj->ni", rot, acc + np.array([0.0, 0.0, G]))
    spec = f_body[: idx[-1] + every].reshape(-1, every, 3).mean(1)

    # IMU errors: white noise (density * sqrt(rate)), bias random walk, turn-on bias.
    m = idx.size
    dti = 1.0 / cfg.imu_hz
    nrng = np.random.default_rng([seed, 0x1A])
    ab = (cfg.accel_bias_std * nrng.standard_normal(3)
          + np.cumsum(cfg.accel_bias_walk * np.sqrt(dti) * nrng.standard_normal((m, 3)), 0))
    gb = (cfg.gyro_bias_std * nrng.standard_normal(3)
          + np.cumsum(cfg.gyro_bias_walk * np.sqrt(dti) * nrng.standard_normal((m, 3)), 0))
    accel = (spec + ab + cfg.accel_noise_density * np.sqrt(cfg.imu_hz)
             * nrng.standard_normal((m, 3)) + cfg.vib_accel_rms * nrng.standard_normal((m, 3)))
    gyro = (wi + gb + cfg.gyro_noise_density * np.sqrt(cfg.imu_hz) * nrng.standard_normal((m, 3))
            + cfg.vib_gyro_rms * nrng.standard_normal((m, 3)))
    # On the floor the motors are off: no vibration before the take-off.
    still = ti < cfg.ground_s
    accel[still] = (spec + ab)[still] + cfg.accel_noise_density * np.sqrt(cfg.imu_hz) * \
        nrng.standard_normal((int(still.sum()), 3))
    gyro[still] = (wi + gb)[still] + cfg.gyro_noise_density * np.sqrt(cfg.imu_hz) * \
        nrng.standard_normal((int(still.sum()), 3))

    cam_every = int(round(cfg.imu_hz / cfg.cam_hz))
    return Flight(t=ti, pos=pi_, vel=vi, quat=quat_from_rot(ri), gyro=gyro, accel=accel,
                  gyro_true=wi, hold_point=hold, hold_start_s=cfg.ground_s + cfg.approach_s,
                  cam_every=cam_every)
