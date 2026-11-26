"""Wind and near-wall disturbance model: the force the air puts on the quadrotor.

One model, shared by the batched PyTorch simulator, Isaac Lab and the Gazebo SITL world, so
every simulator sees the same air. It has two halves:

- ``sample_params`` draws one episode's disturbance per env from a ``DisturbanceConfig``
  and a seed (NumPy, run once at reset).
- ``disturbance_force`` is a pure function of those parameters, time, position and
  velocity. It holds no state, so the answer does not depend on the simulator's step size
  or on how often it is called, and a Gazebo plugin and a training env that share a seed
  see identical wind.

Wind at time t is the sum of a mean wind, scheduled gust steps, and turbulence. The
turbulence is a sum of sinusoids with random phases whose spectrum follows a first-order
Gauss-Markov process (correlation time ``tau``), scaled so its standard deviation is exactly
the sampled ``sigma``. Near a surface the wind is sheltered by a factor

    g(d) = 1 - (1 - g0) * exp(-d / L)

where d is the distance from the surface along its normal. On top of the shelter, four
near-wall flow terms (all off by default; docs/near-wall-flow.md) shape the air the drone
sits in, again as closed-form functions of time and distance:

- windward deflection: wind blowing into the surface is blocked and turns along it (the
  stagnation flow that becomes up- and downwash on a facade);
- leeward recirculation: wind blowing away from the surface leaves a separation bubble with
  reverse flow back toward it, strongest at a distance ``wall_recirc_length``;
- wake turbulence: extra turbulence near the surface that scales with the mean wind;
- vortex shedding: a periodic crossflow at f = St * U / D in the lee of a finite structure.

A second surface (off unless ``sample_params`` places one; docs/bridge-holds.md), such as a
bridge deck overhead, adds its own shelter factor (the two multiply) and the vortices a deck
sheds from its edges in a crosswind.

The force is drag on the air velocity relative to the drone:

    v_rel = air(t, pos) - velocity,   air = g(d) * wind(t) + near-wall flow
    F = k_lin * v_rel + k_quad * |v_rel| * v_rel

The force includes the drag from the drone's own motion through still air, so a simulator
using it must not add its own translational drag. A simulator that already models body drag
(Gazebo) applies ``disturbance_force(...) - drag_force(-velocity, ...)`` instead.

Frames and units follow docs/conventions.md: world ENU, SI units, arrays ``(num_envs, ...)``.

Not modelled yet: variation along the surface (every term depends on distance only), the
yaw torque near the wall (the published case shows it changing sign, but one data point is
not enough to fit it), and the near-wall flow of the second surface beyond its shelter and
edge shedding. Ground, ceiling and wall effect on the rotors are in ``envs/ambient``. The
fan rig in Phase 3 fits the parameters; until then the ranges are wide on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

# Nominal near-wall term, fitted to the one published data point: a 2.14 kg quadrotor
# hovering in a 10 m/s headwind toward a wall felt a horizontal disturbance of 2.78 N at
# 5 m from the wall and 0.40 N at 0.5 m. With full shelter at the wall (g0 = 0) and pure
# quadratic drag, F(d) = k_quad * 10**2 * g(d)**2. The two points give L and k_quad
# (bisection on g(0.5) / g(5) = sqrt(0.40 / 2.78); the test reproduces both forces).
NOMINAL_WALL_DECAY = 1.0604895349199683  # L, m
NOMINAL_WALL_FLOOR = 0.0  # g0, wind fraction left at the surface
NOMINAL_K_QUAD = 0.028305045198112888  # N / (m/s)^2, for the 2.14 kg airframe


def _pair(value: Any, name: str) -> tuple[float, float]:
    lo, hi = (float(value), float(value)) if np.isscalar(value) else map(float, value)
    if lo > hi:
        raise ValueError(f"{name}: range ({lo}, {hi}) has min > max")
    return lo, hi


@dataclass(frozen=True)
class DisturbanceConfig:
    """Ranges the per-env disturbance is drawn from. Each range is ``(min, max)``, uniform.

    A single number in ``from_dict`` means a fixed value. Defaults follow the build plan:
    mean wind 0 to 12 m/s, gust steps, filtered turbulence, a randomised near-wall term.
    """

    # Mean wind: horizontal speed, direction uniform over the compass, no vertical mean.
    mean_speed: tuple[float, float] = (0.0, 12.0)
    # The direction the mean wind blows toward, degrees in the world frame (0 = +x, 90 = +y),
    # as the Gazebo scenarios' wind_heading_deg. The default is the whole compass. With a
    # wall at the origin facing -x (the structure hold env), 0 blows into the wall.
    mean_heading_deg: tuple[float, float] = (0.0, 360.0)
    # Gust steps: a gust adds a constant horizontal wind for its duration, then stops.
    num_gusts: int = 2
    gust_start: tuple[float, float] = (2.0, 50.0)  # s from episode start
    gust_duration: tuple[float, float] = (0.5, 3.0)  # s
    gust_speed: tuple[float, float] = (0.0, 5.0)  # m/s
    # Turbulence: standard deviation per axis (vertical is a fraction of horizontal).
    turb_sigma: tuple[float, float] = (0.0, 1.5)  # m/s, horizontal
    turb_vertical_ratio: tuple[float, float] = (0.5, 0.5)
    turb_tau: tuple[float, float] = (0.5, 3.0)  # s, correlation time
    turb_components: int = 32
    turb_omega: tuple[float, float] = (0.02, 30.0)  # rad/s, band the sinusoids cover
    # Drag on the air velocity relative to the drone.
    k_lin: tuple[float, float] = (0.0, 0.0)  # N / (m/s)
    k_quad: tuple[float, float] = (0.8 * NOMINAL_K_QUAD, 1.2 * NOMINAL_K_QUAD)
    # Near-wall shelter.
    wall_decay: tuple[float, float] = (0.5 * NOMINAL_WALL_DECAY, 2.0 * NOMINAL_WALL_DECAY)
    wall_floor: tuple[float, float] = (0.0, 0.3)
    # Near-wall flow beyond the shelter, off by default so every config written before it
    # flies the same air. Ranges are literature-shaped, not fitted (docs/near-wall-flow.md).
    # Windward deflection: the fraction of the blocked normal wind, (1 - g(d)) times the wind
    # into the surface, that turns along it, in the direction wall_deflect_angle_deg (in the
    # surface plane: 0 horizontal along it, 90 up; where on a facade the drone sits decides
    # up- or downwash, so the default draws any direction).
    wall_deflect: tuple[float, float] = (0.0, 0.0)
    wall_deflect_angle_deg: tuple[float, float] = (0.0, 360.0)
    # Leeward recirculation: reverse flow toward the surface, wall_recirc times the wind
    # blowing away from it, with the profile (d / Lr) * exp(1 - d / Lr) that peaks at Lr.
    wall_recirc: tuple[float, float] = (0.0, 0.0)
    wall_recirc_length: tuple[float, float] = (0.5, 3.0)  # Lr, m
    # Wake turbulence: standard deviation wake_ti * |mean wind| per axis at the surface,
    # decaying as exp(-d / wake_length); the turbulence spectrum, its own phases.
    wake_ti: tuple[float, float] = (0.0, 0.0)
    wake_length: tuple[float, float] = (1.0, 4.0)  # m
    # Vortex shedding: a sinusoidal crossflow of amplitude shed_amp times the mean wind
    # blowing away from the surface (the drone in the lee), at 2 pi St |mean wind| / D rad/s,
    # in the direction shed_angle_deg (as wall_deflect_angle_deg; 0 is a vertical pier's
    # sideways shedding), decaying over wake_length.
    shed_amp: tuple[float, float] = (0.0, 0.0)
    shed_width: tuple[float, float] = (0.5, 5.0)  # D, m
    shed_strouhal: tuple[float, float] = (0.1, 0.21)  # St: 0.2 a cylinder, 0.1 to 0.15 a deck
    shed_angle_deg: tuple[float, float] = (0.0, 0.0)
    # A second surface, used only when sample_params places one (surface2_point and
    # surface2_normal; docs/bridge-holds.md): a bridge deck over the drone, say. Its shelter
    # is drawn as the first surface's and the two factors multiply. As a bluff deck in a
    # crosswind it sheds vortices from its edges: a crossflow along its normal (vertical under
    # a deck) of amplitude surface2_shed_amp times the mean wind across it, at
    # 2 pi St |mean wind across| / D rad/s, decaying as exp(-d2 / surface2_wake_length).
    # Shedding is off by default.
    surface2_decay: tuple[float, float] = (0.5 * NOMINAL_WALL_DECAY, 2.0 * NOMINAL_WALL_DECAY)
    surface2_floor: tuple[float, float] = (0.0, 0.3)
    surface2_shed_amp: tuple[float, float] = (0.0, 0.0)
    surface2_shed_width: tuple[float, float] = (1.0, 3.0)  # D, m: the deck's depth
    surface2_shed_strouhal: tuple[float, float] = (0.1, 0.15)  # box-like bridge decks
    surface2_wake_length: tuple[float, float] = (1.0, 4.0)  # m
    # A steady vertical wind, m/s, + up: the draught up a sunlit facade or a shaft, the
    # downdraught off a roof edge or under a ventilation outlet. Added to the mean wind's z
    # (the mean wind is horizontal otherwise) and sheltered like the rest of it. Off by
    # default and drawn last, so every config written before it flies the same air.
    mean_vertical: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.type.startswith("tuple"):
                object.__setattr__(self, f.name, _pair(getattr(self, f.name), f.name))
        if self.num_gusts < 0 or self.turb_components < 1:
            raise ValueError("num_gusts must be >= 0 and turb_components >= 1")
        if self.turb_omega[0] <= 0 or self.turb_tau[0] <= 0 or self.wall_decay[0] <= 0:
            raise ValueError("turb_omega, turb_tau and wall_decay must be positive")
        if min(self.wall_recirc_length[0], self.wake_length[0], self.shed_width[0]) <= 0:
            raise ValueError("wall_recirc_length, wake_length and shed_width must be positive")
        if min(self.surface2_decay[0], self.surface2_shed_width[0],
               self.surface2_wake_length[0]) <= 0:
            raise ValueError("surface2_decay, surface2_shed_width and surface2_wake_length "
                             "must be positive")

    @property
    def near_wall_flow(self) -> bool:
        """True when any near-wall flow term beyond the shelter can be non-zero."""
        return max(self.wall_deflect[1], self.wall_recirc[1], self.wake_ti[1],
                   self.shed_amp[1]) > 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DisturbanceConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown disturbance config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def nominal(cls) -> DisturbanceConfig:
        """The published case: no gusts, no turbulence, nominal drag and wall term."""
        return cls(
            num_gusts=0,
            turb_sigma=(0.0, 0.0),
            k_quad=(NOMINAL_K_QUAD, NOMINAL_K_QUAD),
            wall_decay=(NOMINAL_WALL_DECAY, NOMINAL_WALL_DECAY),
            wall_floor=(NOMINAL_WALL_FLOOR, NOMINAL_WALL_FLOOR),
        )


@dataclass
class DisturbanceParams:
    """One episode's disturbance for ``num_envs`` envs. Fields are NumPy arrays, or torch
    tensors after ``envs.disturbance.torch_adapter.to_torch``."""

    mean_wind: Any  # (N, 3) m/s
    gust_start: Any  # (N, K) s
    gust_duration: Any  # (N, K) s
    gust_wind: Any  # (N, K, 3) m/s
    turb_amp: Any  # (N, 3, M) m/s
    turb_omega: Any  # (N, M) rad/s
    turb_phase: Any  # (N, 3, M) rad
    k_lin: Any  # (N,)
    k_quad: Any  # (N,)
    wall_point: Any  # (N, 3) m, a point on the surface
    wall_normal: Any  # (N, 3) unit, from the surface into the free air
    has_wall: Any  # (N,) bool; False means free air
    wall_decay: Any  # (N,) m
    wall_floor: Any  # (N,)
    # Near-wall flow (zeros when it is off or there is no wall).
    wall_deflect: Any  # (N,)
    wall_tangent: Any  # (N, 3) unit, in the surface plane: where the deflected wind goes
    wall_recirc: Any  # (N,)
    wall_recirc_length: Any  # (N,) m
    wake_amp: Any  # (N, 3, M) m/s, on turb_omega; (N, 3, 0) when the flow is off
    wake_phase: Any  # (N, 3, M) rad, as wake_amp
    wake_length: Any  # (N,) m
    shed_amp: Any  # (N,) m/s
    shed_omega: Any  # (N,) rad/s
    shed_phase: Any  # (N,) rad
    shed_dir: Any  # (N, 3) unit, in the surface plane
    # The second surface (has_surface2 False and the rest inert when there is none).
    has_surface2: Any  # (N,) bool
    surface2_point: Any  # (N, 3) m
    surface2_normal: Any  # (N, 3) unit, from the surface into the free air
    surface2_decay: Any  # (N,) m
    surface2_floor: Any  # (N,)
    surface2_shed_amp: Any  # (N,) m/s
    surface2_shed_omega: Any  # (N,) rad/s
    surface2_shed_phase: Any  # (N,) rad
    surface2_wake_length: Any  # (N,) m

    @property
    def num_envs(self) -> int:
        return self.mean_wind.shape[0]


def sample_params(
    cfg: DisturbanceConfig,
    num_envs: int,
    seed: int | np.random.Generator,
    wall_point: Any = None,
    wall_normal: Any = None,
    surface2_point: Any = None,
    surface2_normal: Any = None,
    surface2_across: Any = None,
) -> DisturbanceParams:
    """Draw one episode's disturbance per env. Same config and seed give the same params.

    ``wall_point`` and ``wall_normal`` place one flat surface per env (shape ``(3,)`` or
    ``(N, 3)``); leave them out for free air. The normal points away from the surface,
    towards the drone, so it is minus the structure frame's x axis.

    ``surface2_point`` and ``surface2_normal`` place a second surface the same way (a deck
    overhead has normal ``(0, 0, -1)``). ``surface2_across`` is the horizontal direction
    across it, the one a crosswind must blow along to shed vortices from its edges (across
    a bridge: perpendicular to its axis); left out, the whole mean wind counts, the worst
    case. The second surface draws after everything else, so leaving it out keeps the
    random stream as it was.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n, k, m = num_envs, cfg.num_gusts, cfg.turb_components

    def u(r: tuple[float, float], *shape: int) -> np.ndarray:
        return rng.uniform(r[0], r[1], size=(n, *shape))

    def horizontal(speed: np.ndarray, lo: float = 0.0, hi: float = 2.0 * math.pi) -> np.ndarray:
        heading = rng.uniform(lo, hi, size=speed.shape)
        return np.stack([speed * np.cos(heading), speed * np.sin(heading), 0.0 * speed], -1)

    mean_wind = horizontal(u(cfg.mean_speed), *map(math.radians, cfg.mean_heading_deg))
    gust_start = u(cfg.gust_start, k)
    gust_duration = u(cfg.gust_duration, k)
    gust_wind = horizontal(u(cfg.gust_speed, k))

    # Turbulence: log-spaced frequencies, jittered inside their bins so no two envs share a
    # period; amplitudes follow the Gauss-Markov spectrum 1 / (1 + (omega * tau)^2) and are
    # normalised so the variance, sum(a^2) / 2, is exactly sigma^2 on each axis.
    edges = np.geomspace(cfg.turb_omega[0], cfg.turb_omega[1], m + 1)
    omega = edges[:-1] + (edges[1:] - edges[:-1]) * rng.uniform(size=(n, m))
    tau = u(cfg.turb_tau)
    weight = (edges[1:] - edges[:-1]) / (1.0 + (omega * tau[:, None]) ** 2)
    shape = np.sqrt(2.0 * weight / weight.sum(-1, keepdims=True))  # (N, M)
    sigma_h = u(cfg.turb_sigma)
    sigma = np.stack([sigma_h, sigma_h, sigma_h * u(cfg.turb_vertical_ratio)], -1)
    turb_amp = sigma[:, :, None] * shape[:, None, :]
    turb_phase = rng.uniform(0.0, 2.0 * math.pi, size=(n, 3, m))

    has_wall = wall_normal is not None
    normal = np.broadcast_to(
        np.asarray(wall_normal if has_wall else (1.0, 0.0, 0.0), dtype=float), (n, 3)
    )
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("wall_normal must be non-zero")
    point = np.broadcast_to(
        np.asarray(wall_point if wall_point is not None else (0.0, 0.0, 0.0), dtype=float),
        (n, 3),
    )

    k_lin, k_quad = u(cfg.k_lin), u(cfg.k_quad)
    wall_decay, wall_floor = u(cfg.wall_decay), u(cfg.wall_floor)
    # The near-wall flow draws last, so turning it on leaves every draw above unchanged.
    unit_normal = normal / norm
    flow = _near_wall_flow(cfg, rng if has_wall else None, mean_wind, shape, unit_normal)
    second = _surface2(cfg, rng, n, mean_wind, surface2_point, surface2_normal,
                       surface2_across)
    if cfg.mean_vertical != (0.0, 0.0):
        mean_wind[:, 2] = u(cfg.mean_vertical)

    return DisturbanceParams(
        mean_wind=mean_wind,
        gust_start=gust_start,
        gust_duration=gust_duration,
        gust_wind=gust_wind,
        turb_amp=turb_amp,
        turb_omega=omega,
        turb_phase=turb_phase,
        k_lin=k_lin,
        k_quad=k_quad,
        wall_point=point.copy(),
        wall_normal=unit_normal,
        has_wall=np.full(n, has_wall),
        wall_decay=wall_decay,
        wall_floor=wall_floor,
        **flow,
        **second,
    )


def _surface_direction(normal: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Unit vectors (N, 3) in the plane of each surface: angle 0 is horizontal along it, 90
    is up the surface (for a horizontal surface, such as a deck underside, 90 is world +x)."""
    up = np.array([0.0, 0.0, 1.0])
    e_up = up - (normal @ up)[:, None] * normal
    flat = np.linalg.norm(e_up, axis=-1) < 1e-6
    x = np.array([1.0, 0.0, 0.0])
    e_up[flat] = x - (normal[flat] @ x)[:, None] * normal[flat]
    e_up /= np.linalg.norm(e_up, axis=-1, keepdims=True)
    e_side = np.cross(normal, e_up)
    return np.cos(angle)[:, None] * e_side + np.sin(angle)[:, None] * e_up


def _near_wall_flow(
    cfg: DisturbanceConfig,
    rng: np.random.Generator | None,
    mean_wind: np.ndarray,
    shape: np.ndarray,
    normal: np.ndarray,
) -> dict[str, np.ndarray]:
    """The near-wall flow fields. Draws nothing when the flow is off or there is no wall
    (``rng`` None), so the random stream, and every env built after the wind, is the same
    as before the flow existed."""
    n, m = shape.shape
    if rng is None or not cfg.near_wall_flow:
        # No wake sinusoids at all (M = 0), so the switched-off flow costs nothing per step.
        zero, one = np.zeros(n), np.ones(n)
        return dict(
            wall_deflect=zero, wall_tangent=np.zeros((n, 3)), wall_recirc=zero,
            wall_recirc_length=one, wake_amp=np.zeros((n, 3, 0)),
            wake_phase=np.zeros((n, 3, 0)), wake_length=one, shed_amp=zero,
            shed_omega=zero, shed_phase=zero, shed_dir=np.zeros((n, 3)),
        )

    def u(r: tuple[float, float], *size: int) -> np.ndarray:
        return rng.uniform(r[0], r[1], size=(n, *size))

    speed = np.linalg.norm(mean_wind, axis=-1)
    lee = (mean_wind * normal).sum(-1).clip(min=0.0)  # mean wind blowing away from the wall
    wall_deflect = u(cfg.wall_deflect)
    wall_tangent = _surface_direction(normal, np.radians(u(cfg.wall_deflect_angle_deg)))
    wall_recirc = u(cfg.wall_recirc)
    wall_recirc_length = u(cfg.wall_recirc_length)
    wake_amp = (u(cfg.wake_ti) * speed)[:, None, None] * shape[:, None, :] * np.ones((1, 3, 1))
    wake_phase = rng.uniform(0.0, 2.0 * math.pi, size=(n, 3, m))
    wake_length = u(cfg.wake_length)
    shed_amp = u(cfg.shed_amp) * lee
    shed_omega = 2.0 * math.pi * u(cfg.shed_strouhal) * speed / u(cfg.shed_width)
    shed_phase = rng.uniform(0.0, 2.0 * math.pi, size=n)
    shed_dir = _surface_direction(normal, np.radians(u(cfg.shed_angle_deg)))
    return dict(
        wall_deflect=wall_deflect, wall_tangent=wall_tangent, wall_recirc=wall_recirc,
        wall_recirc_length=wall_recirc_length, wake_amp=wake_amp, wake_phase=wake_phase,
        wake_length=wake_length, shed_amp=shed_amp, shed_omega=shed_omega,
        shed_phase=shed_phase, shed_dir=shed_dir,
    )


def _surface2(
    cfg: DisturbanceConfig,
    rng: np.random.Generator,
    n: int,
    mean_wind: np.ndarray,
    point: Any,
    normal: Any,
    across: Any,
) -> dict[str, np.ndarray]:
    """The second surface's fields. Draws nothing without one, so configs and scenarios
    that do not place it fly the same air from the same random stream as before."""
    if normal is None:
        zero, one = np.zeros(n), np.ones(n)
        return dict(
            has_surface2=np.zeros(n, dtype=bool), surface2_point=np.zeros((n, 3)),
            surface2_normal=np.tile([0.0, 0.0, -1.0], (n, 1)), surface2_decay=one,
            surface2_floor=one, surface2_shed_amp=zero, surface2_shed_omega=zero,
            surface2_shed_phase=zero, surface2_wake_length=one,
        )
    unit = np.broadcast_to(np.asarray(normal, dtype=float), (n, 3))
    norm = np.linalg.norm(unit, axis=-1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("surface2_normal must be non-zero")
    unit = unit / norm
    point = np.broadcast_to(
        np.asarray(point if point is not None else (0.0, 0.0, 0.0), dtype=float), (n, 3))
    if across is None:
        cross = np.linalg.norm(mean_wind, axis=-1)
    else:
        a = np.broadcast_to(np.asarray(across, dtype=float), (n, 3))
        a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
        if np.any(a_norm < 1e-9):
            raise ValueError("surface2_across must be non-zero")
        cross = np.abs((mean_wind * (a / a_norm)).sum(-1))

    def u(r: tuple[float, float]) -> np.ndarray:
        return rng.uniform(r[0], r[1], size=n)

    decay, floor = u(cfg.surface2_decay), u(cfg.surface2_floor)
    shed_amp = u(cfg.surface2_shed_amp) * cross
    shed_omega = 2.0 * math.pi * u(cfg.surface2_shed_strouhal) * cross / u(
        cfg.surface2_shed_width)
    shed_phase = rng.uniform(0.0, 2.0 * math.pi, size=n)
    wake_length = u(cfg.surface2_wake_length)
    return dict(
        has_surface2=np.ones(n, dtype=bool), surface2_point=point.copy(),
        surface2_normal=unit, surface2_decay=decay, surface2_floor=floor,
        surface2_shed_amp=shed_amp, surface2_shed_omega=shed_omega,
        surface2_shed_phase=shed_phase, surface2_wake_length=wake_length,
    )


# The functions below are written once for any array module ``xp`` (NumPy or torch): they
# use only operators, broadcasting, ``.sum(dim)``, ``.clip(min=)`` and ``xp.cos``,
# ``xp.sin``, ``xp.exp``, ``xp.sqrt``. The torch adapter calls the same code, so the two
# cannot drift.


def _time_column(t: Any, ndim: int) -> Any:
    # Scalar t broadcasts as is; per-env t of shape (N,) gets trailing axes to match.
    if t.ndim == 0:
        return t
    return t.reshape(t.shape[0], *([1] * (ndim - 1)))


def _wind(xp: Any, p: DisturbanceParams, t: Any) -> Any:
    t1 = _time_column(t, 2)  # against (N, K)
    on = (t1 >= p.gust_start) & (t1 < p.gust_start + p.gust_duration)
    gusts = (on[..., None] * p.gust_wind).sum(1)
    arg = p.turb_omega[:, None, :] * _time_column(t, 3) + p.turb_phase
    turbulence = (p.turb_amp * xp.cos(arg)).sum(-1)
    return p.mean_wind + gusts + turbulence


def _wall_distance(p: DisturbanceParams, pos: Any) -> Any:
    return ((pos - p.wall_point) * p.wall_normal).sum(-1)


def _wall_shelter(xp: Any, p: DisturbanceParams, pos: Any) -> Any:
    d = _wall_distance(p, pos).clip(min=0.0)
    return 1.0 - p.has_wall * (1.0 - p.wall_floor) * xp.exp(-d / p.wall_decay)


def _surface2_distance(p: DisturbanceParams, pos: Any) -> Any:
    return ((pos - p.surface2_point) * p.surface2_normal).sum(-1)


def _surface2_shelter(xp: Any, p: DisturbanceParams, pos: Any) -> Any:
    d = _surface2_distance(p, pos).clip(min=0.0)
    return 1.0 - p.has_surface2 * (1.0 - p.surface2_floor) * xp.exp(-d / p.surface2_decay)


def _shelter(xp: Any, p: DisturbanceParams, pos: Any) -> Any:
    # Without a second surface its factor is exactly 1, so the product is the wall's own.
    return _wall_shelter(xp, p, pos) * _surface2_shelter(xp, p, pos)


def _drag(xp: Any, v_rel: Any, k_lin: Any, k_quad: Any) -> Any:
    speed = xp.sqrt((v_rel * v_rel).sum(-1))
    return (k_lin + k_quad * speed)[:, None] * v_rel


def _air(xp: Any, p: DisturbanceParams, t: Any, pos: Any) -> Any:
    wind = _wind(xp, p, t)
    g = _wall_shelter(xp, p, pos)
    air = (g * _surface2_shelter(xp, p, pos))[:, None] * wind
    # Near-wall flow. Every term is zero when the flow is off, and adding exact zeros leaves
    # the sheltered wind bit for bit as it was.
    d = _wall_distance(p, pos).clip(min=0.0)
    w_n = (wind * p.wall_normal).sum(-1)  # > 0 blows away from the surface
    deflect = p.wall_deflect * (1.0 - g) * (-w_n).clip(min=0.0)
    s = d / p.wall_recirc_length
    recirc = p.wall_recirc * s * xp.exp(1.0 - s) * w_n.clip(min=0.0)
    wake = xp.exp(-d / p.wake_length)
    arg = p.turb_omega[:, None, : p.wake_phase.shape[-1]] * _time_column(t, 3) + p.wake_phase
    gusty = (p.wake_amp * xp.cos(arg)).sum(-1)
    shed = p.shed_amp * xp.sin(p.shed_omega * t + p.shed_phase)
    near = (deflect[:, None] * p.wall_tangent - recirc[:, None] * p.wall_normal
            + wake[:, None] * (gusty + shed[:, None] * p.shed_dir))
    # The second surface's edge shedding, along its normal (zero amplitude without one).
    d2 = _surface2_distance(p, pos).clip(min=0.0)
    shed2 = (p.surface2_shed_amp * xp.sin(p.surface2_shed_omega * t + p.surface2_shed_phase)
             * xp.exp(-d2 / p.surface2_wake_length))
    return air + near + shed2[:, None] * p.surface2_normal


def _force(xp: Any, p: DisturbanceParams, t: Any, pos: Any, vel: Any) -> Any:
    return _drag(xp, _air(xp, p, t, pos) - vel, p.k_lin, p.k_quad)


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def wind_velocity(p: DisturbanceParams, t: Any) -> np.ndarray:
    """Free-stream wind (N, 3) in m/s at time ``t`` (scalar or (N,)), before any shelter."""
    return _wind(np, p, _arr(t))


def surface2_distance(p: DisturbanceParams, pos: Any) -> np.ndarray:
    """Signed distance (N,) in m from each env's second surface; meaningless where
    ``has_surface2`` is False."""
    return _surface2_distance(p, _arr(pos))


def wall_distance(p: DisturbanceParams, pos: Any) -> np.ndarray:
    """Signed distance (N,) in m from each env's surface; meaningless where ``has_wall`` is
    False."""
    return _wall_distance(p, _arr(pos))


def shelter_factor(p: DisturbanceParams, pos: Any) -> np.ndarray:
    """Fraction (N,) of the free-stream wind that reaches ``pos``: 1 in free air, the product
    of both surfaces' factors where there are two."""
    return _shelter(np, p, _arr(pos))


def air_velocity(p: DisturbanceParams, t: Any, pos: Any) -> np.ndarray:
    """Air velocity (N, 3) in m/s at ``pos`` (N, 3) m and time ``t``: the sheltered wind plus
    the near-wall flow. The force is drag on this minus the drone's velocity."""
    return _air(np, p, _arr(t), _arr(pos))


def drag_force(v_rel: Any, k_lin: Any, k_quad: Any) -> np.ndarray:
    """Drag (N, 3) in N for air velocity ``v_rel`` (N, 3) relative to the drone."""
    return _drag(np, _arr(v_rel), _arr(k_lin), _arr(k_quad))


def disturbance_force(p: DisturbanceParams, t: Any, pos: Any, vel: Any) -> np.ndarray:
    """Aerodynamic force (N, 3) in N, world ENU, on drones at ``pos`` (N, 3) m moving at
    ``vel`` (N, 3) m/s, at time ``t`` s (scalar or (N,))."""
    return _force(np, p, _arr(t), _arr(pos), _arr(vel))
