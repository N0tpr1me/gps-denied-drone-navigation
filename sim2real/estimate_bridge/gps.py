"""A scripted GPS beside the vision estimate, for flights with GPS on (``docs/field-faults.md``,
"GPS on").

The benchmark flies GPS-free. Drone makers asked what Structure Hold does when a GPS or RTK
receiver is on as well (``docs/market-research.md``): an RTK fix that drops to a float
solution and moves by half a metre, a multipath jump, a glitch of a few samples, a receiver
that goes quiet under a bridge and comes back somewhere else. PX4's own simulated GPS cannot
do any of that, so the bridge sends its own: Gazebo's truth, in the world's latitude and
longitude, with seeded noise and these scripted faults, as a MAVLink HIL_GPS message that
PX4 turns into a ``sensor_gps`` instance (``MAV_USEHILGPS`` 1). Gazebo's own receiver is set
to report no fix (``SIM_GPS_USED`` 0, ``gps_on.params``), so this is the only GPS PX4 has.

``GpsSource`` is the receiver: its rate, its nominal fix (``rtk_fixed``, ``rtk_float`` or
``fix3d``, which set the accuracy it reports and its noise), the noise draw, and whether
EKF2 fuses it (``fused``; false leaves it in the log only, as the truth an outdoor flight
is scored on: ``rtk_truth_*`` profiles, ``eval/rtk_truth.py``). The faults,
timed from the bridge's fault anchor like the field faults:

``gps_float``
    The fix drops to RTK float for ``duration_s`` (0: for good): the reported accuracy and the
    noise go up and the solution sits ``offset_m`` off along ``direction`` (ENU, normalised).
``gps_jump``
    The position steps ``offset_m`` along ``direction`` at ``at_s`` and stays there, or comes
    back after ``duration_s`` (a glitch of a few samples when it is short).
``gps_loss``
    Nothing is sent for ``duration_s``; when the receiver comes back its solution is
    ``offset_m`` off along ``direction`` from then on (the hand-over to a different
    solution). 0 brings it back where it was.

Noise is drawn once per GPS tick, faults or not, so a trace replays exactly from the profile
(``verify_trace``). Faults are offsets on the truth, never on the vision estimate: what the
vision stream and the GPS disagree about is the point of these flights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GPS_FAULT_KINDS = ("gps_float", "gps_jump", "gps_loss")
# MAVLink GPS_FIX_TYPE values, and what a receiver of that kind reports as its accuracy
# (horizontal, vertical, m) and the noise this source draws (m per axis, vertical doubled).
FIXES: dict[str, tuple[int, float, float, float]] = {
    "fix3d": (3, 1.5, 3.0, 0.5),
    "rtk_float": (5, 0.5, 1.0, 0.3),
    "rtk_fixed": (6, 0.02, 0.04, 0.01),
}
# Latitude, longitude (deg) and altitude (m) of the Gazebo world's origin: the spherical
# coordinates of wall_slab.sdf, which copies PX4's default world. PX4's ground-truth topic
# is derived from them, so the bridge's vision stream and GPS share its origin.
WORLD_ORIGIN = (47.397971057728974, 8.546163739800146, 0.0)
NOISE_SEED_OFFSET = 7919  # the GPS stream's draws never overlap the corruptor's
SATELLITES = 14


@dataclass(frozen=True)
class GpsFault:
    kind: str
    at_s: float  # seconds after the bridge's fault anchor
    duration_s: float = 0.0  # 0: gps_float / gps_jump hold for the rest of the flight
    offset_m: float = 0.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)  # ENU

    def __post_init__(self) -> None:
        if self.kind not in GPS_FAULT_KINDS:
            raise ValueError(f"gps fault kind must be one of {GPS_FAULT_KINDS}, got {self.kind!r}")
        if not self.at_s >= 0:
            raise ValueError("at_s must be >= 0")
        if not self.duration_s >= 0:
            raise ValueError("duration_s must be >= 0")
        if self.kind == "gps_loss" and not self.duration_s > 0:
            raise ValueError("gps_loss needs duration_s > 0")
        if not self.offset_m >= 0:
            raise ValueError("offset_m must be >= 0")
        if len(self.direction) != 3 or not np.linalg.norm(self.direction) > 0:
            raise ValueError("direction must be a nonzero ENU 3-vector")

    def active(self, t_rel: float) -> bool:
        """Inside the fault's window; a zero duration never ends."""
        if t_rel < self.at_s:
            return False
        return self.duration_s == 0 or t_rel < self.at_s + self.duration_s

    @property
    def offset(self) -> np.ndarray:
        d = np.asarray(self.direction, dtype=np.float64)
        return d / np.linalg.norm(d) * self.offset_m

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GpsFault:
        d = dict(d)
        unknown = set(d) - {"kind", "at_s", "duration_s", "offset_m", "direction"}
        if unknown:
            raise ValueError(f"unknown gps fault keys: {sorted(unknown)}")
        if "direction" in d:
            d["direction"] = tuple(float(v) for v in d["direction"])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "at_s": self.at_s, "duration_s": self.duration_s,
                "offset_m": self.offset_m, "direction": list(self.direction)}


@dataclass(frozen=True)
class GpsSource:
    rate_hz: float = 10.0
    fix: str = "rtk_fixed"
    noise_m: float | None = None  # per-axis horizontal noise; None: the fix's own
    # False: PX4 receives and logs the receiver but EKF2 does not fuse it (the runners lay
    # gps_logged.params over the PX4 set instead of gps_on.params). That is RTK as the truth
    # an outdoor flight is scored on (eval/rtk_truth.py), with Gazebo's truth to check it.
    fused: bool = True

    def __post_init__(self) -> None:
        if not self.rate_hz > 0:
            raise ValueError("gps rate_hz must be > 0")
        if self.fix not in FIXES:
            raise ValueError(f"gps fix must be one of {tuple(FIXES)}, got {self.fix!r}")
        if self.noise_m is not None and not self.noise_m >= 0:
            raise ValueError("gps noise_m must be >= 0")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GpsSource:
        d = dict(d)
        unknown = set(d) - {"rate_hz", "fix", "noise_m", "fused"}
        if unknown:
            raise ValueError(f"unknown gps keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return {"rate_hz": self.rate_hz, "fix": self.fix, "noise_m": self.noise_m,
                "fused": self.fused}


@dataclass(frozen=True)
class GpsSample:
    pos: np.ndarray  # (3,) ENU, m: the solution the receiver reports
    vel: np.ndarray  # (3,) ENU, m/s
    fix_type: int  # 0: nothing sent this tick
    eph_m: float
    epv_m: float

    @property
    def sent(self) -> bool:
        return self.fix_type > 0


NO_GPS = GpsSample(np.full(3, np.nan), np.full(3, np.nan), 0, float("nan"), float("nan"))


class GpsInjector:
    """The receiver's output, one truth sample at a time; None between its ticks."""

    def __init__(self, source: GpsSource, faults: tuple[GpsFault, ...], seed: int):
        self.source = source
        self.faults = tuple(faults)
        self.rng = np.random.default_rng(int(seed) + NOISE_SEED_OFFSET)
        self._t0: float | None = None
        self._tick = -1

    def update(self, t: float, truth_pos: np.ndarray, truth_vel: np.ndarray,
               t_rel: float | None) -> GpsSample | None:
        """``t_rel`` is the time since the fault anchor, None before it (no fault applies)."""
        if self._t0 is None:
            self._t0 = float(t)
        tick = int(np.floor((float(t) - self._t0) * self.source.rate_hz + 1e-6))
        if tick <= self._tick:
            return None
        self._tick = tick
        fix = self.source.fix
        offset = np.zeros(3)
        silent = False
        for f in self.faults:
            if t_rel is None or t_rel < f.at_s:
                continue
            if f.kind == "gps_float" and f.active(t_rel):
                fix = "rtk_float"
                offset += f.offset
            elif f.kind == "gps_jump" and f.active(t_rel):
                offset += f.offset
            elif f.kind == "gps_loss":
                if f.active(t_rel):
                    silent = True
                else:
                    offset += f.offset  # back, somewhere else
        fix_type, eph, epv, fix_noise = FIXES[fix]
        noise = self.source.noise_m if self.source.noise_m is not None else fix_noise
        if fix != self.source.fix:  # a float fault brings its own noise
            noise = fix_noise
        draw = self.rng.standard_normal(3) * np.array([noise, noise, 2.0 * noise])
        if silent:
            return NO_GPS
        pos = np.asarray(truth_pos, dtype=np.float64) + offset + draw
        return GpsSample(pos, np.asarray(truth_vel, dtype=np.float64).copy(), fix_type, eph, epv)


def parse_gps_faults(data: Any) -> tuple[GpsFault, ...]:
    if not data:
        return ()
    if not isinstance(data, list):
        raise ValueError("gps_faults must be a list of mappings")
    return tuple(f if isinstance(f, GpsFault) else GpsFault.from_dict(f) for f in data)


def enu_to_lat_lon_alt(pos_enu, origin_lat_deg: float, origin_lon_deg: float,
                       origin_alt_m: float) -> tuple[float, float, float]:
    """A small ENU offset from the world origin as latitude, longitude (deg) and altitude,
    on the WGS84 ellipsoid Gazebo's spherical coordinates use (its local tangent plane; at
    the metre offsets flown here the two agree to well under a millimetre)."""
    a, e2 = 6378137.0, 6.69437999014e-3  # WGS84 semi-major axis and first eccentricity squared
    lat0 = np.radians(origin_lat_deg)
    w = np.sqrt(1.0 - e2 * np.sin(lat0) ** 2)
    n = a / w  # prime vertical radius of curvature
    m = a * (1.0 - e2) / w ** 3  # meridional
    lat = origin_lat_deg + np.degrees(float(pos_enu[1]) / m)
    lon = origin_lon_deg + np.degrees(float(pos_enu[0]) / (n * np.cos(lat0)))
    return float(lat), float(lon), float(origin_alt_m + pos_enu[2])
