"""The surroundings beyond wind and the estimate: air density, the battery, rain, and what a
surface next to the rotors does to their thrust.

``envs/disturbance`` is the wind and ``envs/estimate`` what the sensors get wrong. This is
the rest of the physical environment that changes how the airframe answers a command
(``docs/environment-factors.md`` has the whole list and where each factor lives):

- **Air density** from the site's temperature, pressure altitude and humidity
  (``air_density``). Rotor thrust and drag go with density, so a hot day at altitude leaves
  less thrust and less torque for the same command, and the wind pushes less.
- **Battery**: the pack's voltage over the voltage the airframe's thrust was measured at
  (rotor speed goes with voltage, thrust with its square), and the sag under load (the
  voltage drops with the current, so the top of the throttle gives less than its share; a
  cold pack sags more).
- **Rain**: a film of water (mass), rotors that lose a little thrust in heavy rain, and the
  drops' own momentum (a few hundredths of a newton: kept so the table can say it is small).
- **Rotors near a surface** (``surface_gain``, ``wall_pull``): more thrust for the same power
  close to the ground (Cheeseman and Bennett) or under a ceiling, which pulls the vehicle up
  into a deck; a pull toward a wall beside the rotors; and the unsteady push of the vehicle's
  own downwash coming back off those surfaces.

Like ``envs/disturbance`` it has two halves: ``sample_ambient`` draws one episode's numbers
per env (NumPy, at reset), and the functions of distance are written against the operations
NumPy and PyTorch share, so the env calls them on tensors.

What PX4 hides: its hover-thrust estimator follows the mass and the thrust per command
within seconds, and the module scales the policy's thrust through it. So density, battery
level and rain do not show as a thrust offset in a hold. They show as less headroom (max
thrust over weight), less torque per unit of rate-loop output (a slower rate loop) and, for
the sag, a thrust curve that bends at the top. The env applies them that way.

Everything is off by default, and the draws come from a generator of their own, so a config
without an ``ambient:`` block flies exactly the episodes it flew before. The ranges are
physics or literature-shaped, not fitted to a flight: the partner flights fit them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

SEA_LEVEL_PRESSURE_PA = 101325.0
R_DRY = 287.058  # J/(kg K)
# kg/m^3: dry air, 15 degC, sea level (ISA's 1.225), from the same gas law as air_density so
# the standard day's density ratio is exactly 1.
STANDARD_DENSITY = SEA_LEVEL_PRESSURE_PA / (R_DRY * 288.15)
R_VAPOUR = 461.495  # J/(kg K)
RAIN_FALL_SPEED_MPS = 9.0  # terminal speed of a large drop
WATER_DENSITY = 1000.0  # kg/m^3

# A 50 mm/h downpour, the rate the rain terms are stated at (assumed, not measured): a film
# of 2 % of the vehicle's mass, and rotors that lose 5 % of their thrust.
RAIN_REF_MM_H = 50.0
RAIN_FILM_MASS_FRAC = 0.02
RAIN_THRUST_LOSS_FRAC = 0.05

# Cheeseman and Bennett's ground effect: thrust / free-air thrust = 1 / (1 - (R / 4z)^2).
GROUND_COEF = 1.0 / 16.0


def air_density(temperature_c: Any, altitude_m: Any = 0.0, humidity: Any = 0.0) -> Any:
    """Air density in kg/m^3 at a site: its air temperature (degC), pressure altitude (m,
    the ISA pressure at that height) and relative humidity (0 to 1). Humid air is lighter:
    water vapour displaces the heavier dry air."""
    t = np.asarray(temperature_c, dtype=np.float64)
    h = np.asarray(altitude_m, dtype=np.float64)
    rh = np.asarray(humidity, dtype=np.float64)
    pressure = SEA_LEVEL_PRESSURE_PA * (1.0 - 2.25577e-5 * h) ** 5.25588
    saturation = 610.94 * np.exp(17.625 * t / (t + 243.04))  # Magnus, Pa
    vapour = rh * saturation
    kelvin = t + 273.15
    return (pressure - vapour) / (R_DRY * kelvin) + vapour / (R_VAPOUR * kelvin)


def _pair(value: Any, name: str) -> tuple[float, float]:
    lo, hi = ((float(value), float(value)) if isinstance(value, (int, float))
              else (float(value[0]), float(value[1])))
    if lo > hi:
        raise ValueError(f"ambient {name}: min {lo} > max {hi}")
    return lo, hi


@dataclass(frozen=True)
class AmbientConfig:
    """Ranges one episode's surroundings are drawn from, each ``(min, max)``, uniform; a
    single number is a fixed value. The defaults are the standard day at nominal voltage,
    no rain and rotors that feel no surface: nothing is drawn and nothing changes."""

    # Air: density from these three (air_density). 15 degC, sea level and dry is 1.225.
    temperature_c: tuple[float, float] = (15.0, 15.0)
    altitude_m: tuple[float, float] = (0.0, 0.0)  # pressure altitude of the site
    humidity: tuple[float, float] = (0.0, 0.0)  # relative, 0 to 1
    # Battery: the pack's voltage over the one the airframe's max thrust was measured at
    # (nominal, 3.7 V a cell: a full pack is 1.14, one at 3.5 V a cell 0.95, thrust x0.89),
    # and the share of thrust lost to sag at full throttle (0.05 warm, 0.15 cold).
    battery_voltage: tuple[float, float] = (1.0, 1.0)
    battery_sag: tuple[float, float] = (0.0, 0.0)
    # Rain rate in mm/h (2 light, 10 heavy, 50 a downpour).
    rain_mm_h: tuple[float, float] = (0.0, 0.0)
    rain_area_m2: float = 0.15  # the planform the drops land on (an x500 with its props)
    # Rotors near a surface. rotor_radius_m is the radius the effect scales with and 0 is
    # off: one 10 inch prop is 0.13 m, and a quadrotor feels the ground from further away
    # than one rotor does (its rotors share the cushion), so the range reaches past it.
    rotor_radius_m: tuple[float, float] = (0.0, 0.0)
    ground_coef: tuple[float, float] = (GROUND_COEF, GROUND_COEF)
    # Under a ceiling the same form with a larger coefficient (the rotor's inflow is
    # starved, and the measured effect is stronger than the ground's at the same gap).
    ceiling_coef: tuple[float, float] = (GROUND_COEF, 0.15)
    # A pull toward a wall beside the rotors, as a share of thrust times (R / distance)^2.
    wall_pull: tuple[float, float] = (0.0, 0.0)
    # The vehicle's own downwash coming back off the surfaces: a random force with standard
    # deviation this share of thrust times how near the surfaces are (the sum of
    # (R / distance)^2 over ground, ceiling and wall, at most 1), correlated over
    # unsteady_tau_s.
    unsteady: tuple[float, float] = (0.0, 0.0)
    unsteady_tau_s: float = 0.3
    # The barometer's static port in moving air reads the air's dynamic pressure as height:
    # this many metres per (m/s)^2 of airspeed past the vehicle. The whole dynamic pressure
    # is 0.051 m per (m/s)^2; a port in foam inside a shell sees a few per cent of it. The
    # env passes the change since the engage to the fusion (envs/estimate/ekf.py), so it
    # only acts when the barometer is the height reference.
    baro_wind_m_per_mps2: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for f in fields(self):
            if str(f.type).startswith("tuple"):
                object.__setattr__(self, f.name, _pair(getattr(self, f.name), f.name))
        if self.temperature_c[0] < -60.0 or self.temperature_c[1] > 60.0:
            raise ValueError("ambient temperature_c must lie in -60 to 60")
        if self.altitude_m[0] < -500.0 or self.altitude_m[1] > 6000.0:
            raise ValueError("ambient altitude_m must lie in -500 to 6000")
        if self.humidity[0] < 0.0 or self.humidity[1] > 1.0:
            raise ValueError("ambient humidity must lie in 0 to 1")
        if not 0.5 <= self.battery_voltage[0] or self.battery_voltage[1] > 1.2:
            raise ValueError("ambient battery_voltage must lie in 0.5 to 1.2")
        if self.battery_sag[0] < 0.0 or self.battery_sag[1] > 0.5:
            raise ValueError("ambient battery_sag must lie in 0 to 0.5")
        if self.rain_mm_h[0] < 0.0 or self.rain_mm_h[1] > 200.0:
            raise ValueError("ambient rain_mm_h must lie in 0 to 200")
        if self.rotor_radius_m[0] < 0.0 or self.rotor_radius_m[1] > 1.0:
            raise ValueError("ambient rotor_radius_m must lie in 0 to 1")
        for name in ("ground_coef", "ceiling_coef"):
            lo, hi = getattr(self, name)
            if lo < 0.0 or hi > 0.25:
                raise ValueError(f"ambient {name} must lie in 0 to 0.25")
        for name in ("wall_pull", "unsteady", "baro_wind_m_per_mps2"):
            if getattr(self, name)[0] < 0.0:
                raise ValueError(f"ambient {name} must be >= 0")
        if self.rain_area_m2 < 0.0 or self.unsteady_tau_s <= 0.0:
            raise ValueError("ambient rain_area_m2 must be >= 0 and unsteady_tau_s positive")

    @property
    def air(self) -> bool:
        return (self.temperature_c, self.altitude_m, self.humidity) != (
            (15.0, 15.0), (0.0, 0.0), (0.0, 0.0))

    @property
    def surfaces(self) -> bool:
        """True when the rotors can feel a surface."""
        return self.rotor_radius_m[1] > 0.0

    @property
    def baro_wind(self) -> bool:
        return self.baro_wind_m_per_mps2[1] > 0.0

    @property
    def active(self) -> bool:
        return (self.air or self.surfaces or self.baro_wind
                or self.battery_voltage != (1.0, 1.0) or self.battery_sag[1] > 0.0
                or self.rain_mm_h[1] > 0.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AmbientConfig:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown ambient keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class AmbientParams:
    """One episode's surroundings for ``num_envs`` envs, arrays of shape ``(N,)``."""

    density_ratio: np.ndarray  # air density / 1.225: scales drag
    # Thrust and torque for the same command, relative to the standard day at nominal voltage:
    # density x voltage^2 x what the rain leaves.
    power: np.ndarray
    mass_scale: np.ndarray  # 1 + the water film
    battery_sag: np.ndarray  # share of thrust lost at full throttle
    down_force_n: np.ndarray  # the rain's momentum, N, downward
    rotor_radius_m: np.ndarray  # 0: the rotors feel no surface
    ground_coef: np.ndarray
    ceiling_coef: np.ndarray
    wall_pull: np.ndarray
    unsteady: np.ndarray
    baro_wind_m_per_mps2: np.ndarray
    # What was drawn, for logs and tests.
    temperature_c: np.ndarray
    altitude_m: np.ndarray
    humidity: np.ndarray
    battery_voltage: np.ndarray
    rain_mm_h: np.ndarray


def sample_ambient(cfg: AmbientConfig, num_envs: int,
                   seed: int | np.random.Generator) -> AmbientParams:
    """Draw one episode's surroundings per env. Same config and seed give the same params."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def u(r: tuple[float, float]) -> np.ndarray:
        return rng.uniform(r[0], r[1], size=num_envs)

    temperature, altitude, humidity = u(cfg.temperature_c), u(cfg.altitude_m), u(cfg.humidity)
    voltage, sag, rain = u(cfg.battery_voltage), u(cfg.battery_sag), u(cfg.rain_mm_h)
    density_ratio = air_density(temperature, altitude, humidity) / STANDARD_DENSITY
    wet = rain / RAIN_REF_MM_H
    rain_flux = rain / 3.6e6  # mm/h to m/s of water
    return AmbientParams(
        density_ratio=density_ratio,
        power=density_ratio * voltage ** 2 * (1.0 - RAIN_THRUST_LOSS_FRAC * wet).clip(min=0.5),
        mass_scale=1.0 + RAIN_FILM_MASS_FRAC * wet,
        battery_sag=sag,
        down_force_n=WATER_DENSITY * rain_flux * RAIN_FALL_SPEED_MPS * cfg.rain_area_m2,
        rotor_radius_m=u(cfg.rotor_radius_m),
        ground_coef=u(cfg.ground_coef),
        ceiling_coef=u(cfg.ceiling_coef),
        wall_pull=u(cfg.wall_pull),
        unsteady=u(cfg.unsteady),
        baro_wind_m_per_mps2=u(cfg.baro_wind_m_per_mps2),
        temperature_c=temperature, altitude_m=altitude, humidity=humidity,
        battery_voltage=voltage, rain_mm_h=rain,
    )


# The functions below use only operators and ``.clip``, so they take NumPy arrays or torch
# tensors alike (the env calls them on tensors).

# The closest the formulas are trusted: at a quarter of a radius Cheeseman and Bennett's
# form has stopped following the measurements, and the env has called a contact long before.
_MAX_NEARNESS = 2.0  # R / distance


def nearness(radius: Any, distance: Any) -> Any:
    """``R / distance`` clipped to 0 .. 2; 0 where there is no surface (distance inf) or the
    effect is off (radius 0)."""
    return (radius / distance.clip(min=1e-3)).clip(min=0.0, max=_MAX_NEARNESS)


def surface_gain(radius: Any, distance: Any, coef: Any) -> Any:
    """Thrust over free-air thrust for rotors ``distance`` from a surface they blow at (the
    ground) or draw from (a ceiling): ``1 / (1 - coef (R / distance)^2)``. 1 far away."""
    x = nearness(radius, distance)
    return 1.0 / (1.0 - coef * x * x)


def sag_gain(sag: Any, throttle: Any, hover_throttle: Any) -> Any:
    """What the battery's sag leaves of a thrust command at ``throttle`` (thrust over max
    thrust), relative to hover: PX4's hover-thrust estimate has absorbed the loss at
    ``hover_throttle``, so the gain is 1 there, below 1 above it and above 1 below it."""
    return (1.0 - sag * throttle) / (1.0 - sag * hover_throttle)


def hover_headroom(power: Any, mass_scale: Any, thrust_to_weight: float) -> Any:
    """Max thrust over weight in these surroundings, for an airframe whose standard-day
    value is ``thrust_to_weight``. At 1 it cannot hover."""
    return thrust_to_weight * power / mass_scale


def describe(p: AmbientParams) -> dict[str, float]:
    """Means of what was drawn, for a run's log."""
    return {name: float(np.mean(getattr(p, name))) for name in (
        "density_ratio", "power", "mass_scale", "battery_sag", "temperature_c",
        "altitude_m", "humidity", "battery_voltage", "rain_mm_h")}


def density_altitude_m(density_ratio: float) -> float:
    """The ISA height with this density ratio: the number a pilot knows a hot, high site by."""
    return (1.0 - density_ratio ** (1.0 / 4.25588)) / 2.25577e-5 if density_ratio > 0 else math.inf
