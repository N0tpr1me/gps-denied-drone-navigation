"""RTK GPS as the truth for scoring flights outdoors, where there is no motion capture.

Indoors the truth is mocap (``eval.ulog``). At a real bridge or pier there is none, but an
RTK receiver with a fixed solution knows where its antenna is to a centimetre or two. This
module turns the receiver's samples in a PX4 log into a truth track for ``eval.metrics``,
and is honest about the two ways RTK differs from mocap:

- **It comes and goes.** Only RTK-fixed samples (``fix_type`` 6) are truth. A float or
  plain 3D solution is off by decimetres to metres, so those samples are dropped, never
  scored, and the scorer says how much of the hold was left (``covered``). Stretches
  without fixed samples are left out of the numbers (``eval.metrics`` ``max_gap_s``), not
  drawn through.
- **It may not be independent.** Truth must be logged, not fused. With GPS fused into EKF2
  (the GPS-on envelope, ``docs/field-faults.md``) the estimate follows the same receiver:
  a wrong solution is in both and never shows, and the moments the receiver loses its fix,
  which are the estimate's worst, are the moments left out of the score. ``gps_fused``
  reads what EKF2 really did from its aiding flags, and the scorer says so on every such
  run.

Read from whichever GPS topic in the log holds the most RTK-fixed samples: a ``sensor_gps``
instance (each receiver's own output) or ``vehicle_gps_position`` (the receiver PX4
selected, the same message). Logging matters here. PX4 v1.17's logger keeps ``sensor_gps``
at 1 Hz unless the high-rate sensors profile is on (``SDLOG_PROFILE`` bit 11, 2048), and
``vehicle_gps_position`` (10 Hz by default) only exists with ``SYS_HAS_GPS`` 1, which a
GPS-free set-up does not have. So a flight scored on RTK sets bit 11
(``sim2real/partner/rtk_truth.params``); 1 Hz is not a truth. Latitude
and longitude become ENU metres about an origin through ``envs.frames.geodetic_to_enu``.
The receiver stamps a sample when it arrives, a little after it was true; ``fit_delay``
measures that lag against the drone's estimate over the take-off.

The wall comes from points surveyed with the same RTK (``wall_from_survey``), as the mocap
wand gives it indoors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from envs.frames import geodetic_to_enu
from eval.metrics import GRID_DT_S, measured
from eval.ulog import (
    MIN_ALIGN_SPREAD_M,
    Topics,
    align_samples,
    fit_residual,
    fit_yaw_offset,
)
from sim2real.estimate_bridge.ulog_checks import CONTROL_MODE_BITS, ESTIMATOR_STATUS, FLAGS

SENSOR_GPS = "sensor_gps"  # further receivers: "sensor_gps#1", ... (the loader's naming)
SELECTED_GPS = "vehicle_gps_position"
RTK_TOPICS = [SENSOR_GPS, SELECTED_GPS, FLAGS, ESTIMATOR_STATUS]
FIX_RTK_FIXED = 6  # sensor_gps.fix_type: 3 is 3D, 4 differential, 5 RTK float, 6 RTK fixed
FIX_NAMES = {0: "none", 1: "none", 2: "2D", 3: "3D", 4: "differential", 5: "RTK float",
             6: "RTK fixed", 8: "extrapolated"}
# Fixed samples further apart than this are a gap: the larger of half a second (a drone
# holding position moves a few centimetres in that) and 2.5 receiver periods (one lost
# sample is not a gap).
MIN_GAP_S, GAP_PERIODS = 0.5, 2.5
# A truth sampled slower than this misses the peaks it is there to measure. RTK receivers
# run at 5 Hz or more; 1 Hz in a log is PX4's logger thinning sensor_gps, not the receiver.
MIN_RATE_HZ = 4.0
# The receiver's lag is searched over this range; a lag that improves the fit by less than
# DELAY_GAIN_M is not believed (the take-off did not move enough to show it).
DELAY_MAX_S, DELAY_STEP_S, DELAY_GAIN_M = 0.5, 0.01, 0.005
# Surveyed points closer than this (RMS) to one line are a line: a vertical wall through it.
COLLINEAR_M = 0.25
# The EKF2 aiding flags that mean GNSS is in the position estimate.
GNSS_FLAGS = ("cs_gnss_pos", "cs_gnss_vel", "cs_gps", "cs_gps_hgt")


@dataclass(frozen=True)
class RtkTruth:
    t: np.ndarray  # (N,) log seconds of the RTK-fixed samples
    pos: np.ndarray  # (N, 3) ENU metres about ``origin``
    origin: tuple[float, float, float]  # latitude, longitude (deg), altitude (m)
    source: str  # the topic instance read
    period_s: float  # the receiver's sample period
    samples: int  # every sample the receiver logged
    fixed: int  # of them, RTK fixed

    @property
    def max_gap_s(self) -> float:
        return max(MIN_GAP_S, GAP_PERIODS * self.period_s)

    def delayed(self, delay_s: float) -> RtkTruth:
        """The same track with every sample moved ``delay_s`` earlier, to when it was true."""
        return replace(self, t=self.t - float(delay_s))


def _receivers(topics: Topics) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Every GPS topic in the log, each receiver's own output first."""
    own = sorted(((k, v) for k, v in topics.items()
                  if k == SENSOR_GPS or k.startswith(SENSOR_GPS + "#")), key=lambda kv: kv[0])
    return own + [(k, topics[k]) for k in (SELECTED_GPS,) if k in topics]


def _lat_lon_alt(msg: dict[str, np.ndarray], altitude: str) -> tuple[np.ndarray, ...]:
    """Degrees and metres from either layout of ``sensor_gps``: float64 degrees and metres
    (PX4 v1.14 on), or integers in 1e-7 deg and mm (before)."""
    if "latitude_deg" in msg:
        key = "altitude_msl_m" if altitude == "msl" else "altitude_ellipsoid_m"
        return (np.asarray(msg["latitude_deg"], dtype=np.float64),
                np.asarray(msg["longitude_deg"], dtype=np.float64),
                np.asarray(msg[key], dtype=np.float64))
    key = "alt" if altitude == "msl" else "alt_ellipsoid"
    return (np.asarray(msg["lat"], dtype=np.float64) * 1e-7,
            np.asarray(msg["lon"], dtype=np.float64) * 1e-7,
            np.asarray(msg[key], dtype=np.float64) * 1e-3)


def read_rtk(topics: Topics, start_us: int, origin=None, altitude: str = "msl") -> RtkTruth:
    """The RTK-fixed samples of the log as an ENU track about ``origin`` (latitude,
    longitude, altitude; None: the first fixed sample). ``altitude`` names which of the
    receiver's two altitudes to read, ``msl`` or ``ellipsoid``: the one the origin and the
    surveyed points are in."""
    if altitude not in ("msl", "ellipsoid"):
        raise ValueError(f"rtk altitude must be msl or ellipsoid, got {altitude!r}")
    found = _receivers(topics)
    if not found:
        raise ValueError(f"log has no {SENSOR_GPS} (or {SELECTED_GPS}): the RTK receiver "
                         "was not logged")
    name, msg = max(found, key=lambda kv: int(np.sum(np.asarray(kv[1]["fix_type"])
                                                     == FIX_RTK_FIXED)))
    fix = np.asarray(msg["fix_type"], dtype=int)
    # timestamp_sample is when the receiver measured, where a driver fills it in.
    stamp = "timestamp_sample" if np.all(np.asarray(
        msg.get("timestamp_sample", [0])) > 0) else "timestamp"
    t = (np.asarray(msg[stamp], dtype=np.int64) - int(start_us)) * 1e-6
    t, first = np.unique(t, return_index=True)
    fix = fix[first]
    lat, lon, alt = (v[first] for v in _lat_lon_alt(msg, altitude))
    good = (fix == FIX_RTK_FIXED) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    if good.sum() < 2:
        seen = sorted({FIX_NAMES.get(int(f), str(int(f))) for f in np.unique(fix)})
        raise ValueError(f"{name} has no RTK-fixed samples (fix types logged: "
                         f"{', '.join(seen)}); only RTK-fixed positions are truth")
    if origin is None:
        i = int(np.argmax(good))
        origin = (float(lat[i]), float(lon[i]), float(alt[i]))
    origin = tuple(float(v) for v in origin)
    pos = geodetic_to_enu(lat[good], lon[good], alt[good], origin)
    period = float(np.median(np.diff(t))) if len(t) > 1 else math.nan
    if not period <= 1.0 / MIN_RATE_HZ:
        raise ValueError(f"{name} is logged at {1.0 / period:.1f} Hz; a truth needs "
                         f"{MIN_RATE_HZ:.0f} Hz or more. PX4 logs sensor_gps at 1 Hz unless "
                         "SDLOG_PROFILE has bit 11 (2048, high-rate sensors): load "
                         "sim2real/partner/rtk_truth.params")
    return RtkTruth(t=t[good], pos=pos, origin=origin, source=name, period_s=period,
                    samples=int(len(t)), fixed=int(good.sum()))


def covered(truth: RtkTruth, t0: float, t1: float) -> float:
    """The share of (t0, t1) that RTK-fixed samples cover: the time ``eval.metrics`` scores
    with ``max_gap_s`` set to the truth's own."""
    tg = np.arange(t0, t1 + 1e-9, GRID_DT_S)
    return float(np.mean(measured(truth.t, tg, truth.max_gap_s))) if len(tg) else 0.0


def fit_frame(topics: Topics, start_us: int, truth: RtkTruth, window: tuple[float, float]
              ) -> tuple[float, np.ndarray, float, bool]:
    """Yaw, offset and residual that carry the EKF frame into the RTK frame, fitted over
    ``window`` as the mocap path fits it, and whether the yaw was fitted.

    A take-off straight up gives the yaw fit nothing to hold on to. Mocap must refuse it
    (its frame points anywhere); RTK need not: its frame is north-up and so is the EKF's,
    by its compass. So below ``MIN_ALIGN_SPREAD_M`` of sideways motion the yaw is taken as
    zero and only the offset is fitted. What that can cost is the EKF's heading error times
    the sideways distance from the take-off to the hold point, which is small exactly when
    the take-off did not move sideways."""
    def seconds(m: dict[str, np.ndarray]) -> np.ndarray:
        return (np.asarray(m["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6

    src, dst, spread = align_samples(topics, truth.t, truth.pos, seconds, window)
    fitted = spread >= MIN_ALIGN_SPREAD_M
    psi, offset = fit_yaw_offset(src, dst) if fitted else (0.0, dst.mean(axis=0)
                                                           - src.mean(axis=0))
    return psi, offset, fit_residual(src, dst, psi, offset), fitted


def fit_delay(topics: Topics, start_us: int, truth: RtkTruth,
              window: tuple[float, float]) -> float:
    """How late the receiver's samples are stamped, seconds: the shift that best lines the
    truth up with the drone's estimate over ``window`` (the take-off, where it moves).
    0 when no shift improves the fit by ``DELAY_GAIN_M``."""
    lags = np.arange(0.0, DELAY_MAX_S + 1e-9, DELAY_STEP_S)
    rms = np.array([fit_frame(topics, start_us, truth.delayed(lag), window)[2]
                    for lag in lags])
    best = int(np.argmin(rms))
    return float(lags[best]) if rms[0] - rms[best] >= DELAY_GAIN_M else 0.0


def wall_from_survey(points, origin, drone_enu) -> tuple[np.ndarray, np.ndarray, float]:
    """The wall as ``eval.metrics.Surface`` wants it, from points surveyed on its face with
    the RTK rover: a point on it and the unit normal from the drone toward it (ENU about
    ``origin``), plus how far the points lie from the fitted surface (RMS, m).

    ``points`` are (latitude, longitude, altitude). Two points, or more along one line (the
    foot of a wall), make a vertical wall through them. Three or more spread over the face
    fit its plane, whatever its tilt: a leaning pier, the underside of a deck. Which way
    the normal points is settled by where the drone is (``drone_enu``, its hold point)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("wall.survey needs at least two points on the wall's face")
    p = geodetic_to_enu(pts[:, 0], pts[:, 1], pts[:, 2], origin)
    c = p.mean(axis=0)
    s = np.linalg.svd(p - c, compute_uv=False)
    if len(p) == 2 or s[1] / math.sqrt(len(p)) < COLLINEAR_M:
        # One line: only its horizontal direction places the wall, which stands on it.
        _, sh, vh = np.linalg.svd(p[:, :2] - c[:2])
        if sh[0] / math.sqrt(len(p)) < COLLINEAR_M / 2:
            raise ValueError("wall.survey points are above each other: give two points "
                             "apart along the wall, or three spread over its face")
        along = vh[0]
        normal = np.array([along[1], -along[0], 0.0])
        off = (p - c) @ normal
    else:
        _, _, v = np.linalg.svd(p - c)
        normal = v[2]
        off = (p - c) @ normal
    if (c - np.asarray(drone_enu, dtype=np.float64)) @ normal < 0:
        normal = -normal
    return c, normal, float(np.sqrt(np.mean(off ** 2)))


def gps_fused(topics: Topics, start_us: int, window: tuple[float, float]) -> bool | None:
    """Did EKF2 fuse GNSS position, velocity or height at any time in ``window``? From
    ``estimator_status_flags``, or the same bits in ``estimator_status`` when the logger
    dropped that topic. None when the log has neither: it cannot be checked."""
    if FLAGS in topics:
        f = topics[FLAGS]
        t = (np.asarray(f["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6
        on = np.zeros(len(t), dtype=bool)
        for key in GNSS_FLAGS:
            if key in f:
                on |= np.asarray(f[key], dtype=bool)
    elif ESTIMATOR_STATUS in topics and "control_mode_flags" in topics[ESTIMATOR_STATUS]:
        es = topics[ESTIMATOR_STATUS]
        t = (np.asarray(es["timestamp"], dtype=np.int64) - int(start_us)) * 1e-6
        bits = np.asarray(es["control_mode_flags"]).astype(np.uint64)
        on = np.zeros(len(t), dtype=bool)
        for key in GNSS_FLAGS:
            if key in CONTROL_MODE_BITS:
                on |= ((bits >> np.uint64(CONTROL_MODE_BITS[key])) & np.uint64(1)).astype(bool)
    else:
        return None
    inside = (t >= window[0]) & (t <= window[1])
    return bool(np.any(on & inside))
