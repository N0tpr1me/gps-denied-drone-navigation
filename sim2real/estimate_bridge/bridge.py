"""The bridge logic: truth samples in, corrupted estimate samples out, nothing published
during a dropout.

``EstimateBridge.update`` takes one Gazebo truth sample (sim time, ENU position) and
returns the corresponding :class:`BridgeSample`. Velocity is the finite difference of
consecutive truth positions, the same way PX4's gz bridge derives
``vehicle_local_position_groundtruth``. The corruptor is ``envs.estimate.EstimateCorruptor``
with one env, built exactly as ``envs/structure_hold`` builds its own (config, dt, seed), so
a run's corruption is reproducible from its config and seed; ``verify_trace`` proves it.

Honesty rules: a sample that the corruptor marks ``stale`` (inside a dropout) is logged but
``publish`` is False, so nothing goes to PX4 and the EKF sees a gap, not a repeated value.
Frames stay ENU here; :func:`odometry_ned` converts at the MAVLink boundary.

Fault anchor (``fault_anchor``): with ``"start"`` (the default) the corruptor's fault clock
runs from the bridge's first sample, so where a jump or dropout lands relative to the hold
depends on how long PX4 took to arm and take off (0.8 s of spread over the PR #42 and #44
flights, and seed 7's faults landed before takeoff). With ``"engage"`` the schedule is keyed
to the moment Structure Hold first engages (``update(..., engaged=True)``): before that, the
same seed's draws run with every jump and dropout suppressed, and at engage the corruptor is
rebuilt from the seed and pre-rolled ``fault_anchor_s`` of fault clock (faults suppressed),
with the delay history, bias and dropout state carried over so the stream is continuous.
From then on each step's draws are the ones ``vio_default`` makes at fault clock
``fault_anchor_s`` + time since engage, so every policy and every seed meets the same faults
at the same time after engage, and 25 s (the default) makes the in-hold faults the ones the
seed 1 to 8 batches already flew (their holds opened 24.7 to 25.5 s after the first sample).

``"altitude"`` is the same schedule keyed to the vehicle's own height instead: the anchor
is the first truth sample ``fault_anchor_alt_m`` or more above the first sample's height.
It is for the stock PX4 baseline, which never engages Structure Hold (its heartbeat never
shows the mode, so ``"engage"`` would fly it with no fault at all): the stock runner sets
``fault_anchor_alt_m`` to the hover altitude minus the hover window's 0.3 m margin
(``ulog_checks.hover_window``), the height where its scored hover begins, as the engage is
where the hold runner's scored hold begins.

Bias from the anchor (``bias_from_anchor``): the corruptor's bias random walk runs from its
first step, and the anchor carries the accumulated bias over for continuity, so the bias a
runner meets in its scored window depends on how many bridge samples came before its
anchor: the hold runner engages about 29 s after the bridge starts, the stock runner reaches
the hover height a few seconds earlier, and on the quiet seeds the bias sets the p95 floor
each controller sits on (the Gate 1 thread found this on the 8-seed v19 comparison, seeds
2, 4 and 8). With ``bias_from_anchor`` the walk is held at zero before the anchor and the
rebuilt corruptor's bias is zero at it, so from the anchor on both runners meet the same
bias series (the seed's increments from fault clock ``fault_anchor_s`` on), and the stream
is still continuous (0 to 0). The shipped ``vio_at_engage`` and ``vio_at_hover`` set it; old
traces replay with the profile saved in their run folder, which does not.

Field faults (``field_faults``, ``field_faults.py``): scripted failures copied from field
reports (a VIO reset to its origin, a VIO restart, a long link loss, a frozen estimate, a slow
drift), applied on top of the corruptor's output at a set time after the anchor. A reset that
is signalled counts up ``BridgeSample.resets``, which ``run_bridge`` sends as ODOMETRY's
``reset_counter``.

GPS (``gps``, ``gps_faults``, ``gps.py``): a profile with a ``gps`` block also gets a scripted
GPS receiver on the truth, with its own faults (RTK float, jumps, glitches, loss and the
hand-over back), which ``run_bridge`` sends as HIL_GPS; each sample carries the receiver's
output for that tick, or none (``BridgeSample.gps``).
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from envs.estimate import EstimateCorruptionConfig, EstimateCorruptor
from envs.frames import enu_to_ned
from sim2real.estimate_bridge.field_faults import FieldFault, FieldFaultInjector
from sim2real.estimate_bridge.gps import NO_GPS, GpsFault, GpsInjector, GpsSample, GpsSource

TRACE_COLUMNS = (
    "step", "t", "published", "stale", "jumped", "dropout_started", "age",
    "tx", "ty", "tz", "tvx", "tvy", "tvz",
    "ex", "ey", "ez", "evx", "evy", "evz",
    "engaged", "resets",
    "gps_fix", "gx", "gy", "gz", "gvx", "gvy", "gvz", "geph", "gepv",
)
# Trace columns added after the first flights; read as 0 when an old trace lacks them.
OPTIONAL_TRACE_COLUMNS = ("engaged", "resets", "gps_fix", "gx", "gy", "gz", "gvx", "gvy", "gvz",
                          "geph", "gepv")
FAULT_ANCHORS = ("start", "engage", "altitude")
NEVER_S = 1e9  # fault_start_s that holds every fault back for the whole flight


@dataclass(frozen=True)
class BridgeSample:
    step: int
    t: float  # sim time, s
    truth_pos: np.ndarray  # (3,) ENU, m
    truth_vel: np.ndarray  # (3,) ENU, m/s
    est_pos: np.ndarray  # (3,) ENU, what the corruptor reported
    est_vel: np.ndarray
    stale: bool
    jumped: bool
    dropout_started: bool
    age: float
    engaged: bool = False  # the flight script reported Structure Hold engaged at this sample
    resets: int = 0  # signalled frame resets so far (field faults): ODOMETRY reset_counter
    gps: GpsSample = NO_GPS  # the scripted receiver's output this tick (gps.py); NO_GPS: none

    @property
    def publish(self) -> bool:
        """True when this sample goes to PX4 (never inside a dropout)."""
        return not self.stale

    def row(self) -> list:
        return [
            self.step, repr(float(self.t)), int(self.publish), int(self.stale),
            int(self.jumped), int(self.dropout_started), repr(float(self.age)),
            *(repr(float(v)) for v in self.truth_pos), *(repr(float(v)) for v in self.truth_vel),
            *(repr(float(v)) for v in self.est_pos), *(repr(float(v)) for v in self.est_vel),
            int(self.engaged), int(self.resets),
            int(self.gps.fix_type), *(repr(float(v)) for v in self.gps.pos),
            *(repr(float(v)) for v in self.gps.vel), repr(float(self.gps.eph_m)),
            repr(float(self.gps.epv_m)),
        ]


class EstimateBridge:
    """Feed truth samples at a fixed rate; get the corrupted estimate back.

    ``dt`` is the spacing of the truth samples (1 / source rate); ``divisor`` runs the
    corruptor every ``divisor``-th sample (its dt is then ``dt * divisor``) for a slower
    VIO-like output rate. Samples skipped by the divisor return None.
    """

    def __init__(
        self,
        cfg: EstimateCorruptionConfig,
        dt: float,
        seed: int,
        divisor: int = 1,
        fault_anchor: str = "start",
        fault_anchor_s: float = 25.0,
        fault_anchor_alt_m: float | None = None,
        bias_from_anchor: bool = False,
        field_faults: tuple[FieldFault, ...] = (),
        gps: GpsSource | None = None,
        gps_faults: tuple[GpsFault, ...] = (),
    ):
        if divisor < 1:
            raise ValueError("divisor must be >= 1")
        if fault_anchor not in FAULT_ANCHORS:
            raise ValueError(f"fault_anchor must be one of {FAULT_ANCHORS}, got {fault_anchor!r}")
        if not fault_anchor_s >= 0:
            raise ValueError("fault_anchor_s must be >= 0")
        if fault_anchor == "altitude" and not (fault_anchor_alt_m is not None
                                               and fault_anchor_alt_m > 0):
            raise ValueError("fault_anchor 'altitude' needs fault_anchor_alt_m > 0 (the stock "
                             "runner fills it from the scenario's hover altitude)")
        self.cfg, self.dt, self.seed, self.divisor = cfg, float(dt), int(seed), int(divisor)
        self.fault_anchor, self.fault_anchor_s = fault_anchor, float(fault_anchor_s)
        self.fault_anchor_alt_m = None if fault_anchor_alt_m is None else float(fault_anchor_alt_m)
        self.bias_from_anchor = bool(bias_from_anchor) and fault_anchor != "start"
        step_dt = self.dt * self.divisor
        if fault_anchor in ("engage", "altitude"):
            # Same seed, same draws, no fault until the anchor rebuilds the corruptor; with
            # bias_from_anchor no bias walk either, so the bias carried over at the anchor
            # is zero for every anchor time (the draws are still consumed, step for step).
            pre = dataclasses.replace(cfg, fault_start_s=NEVER_S)
            if self.bias_from_anchor:
                pre = dataclasses.replace(pre, bias_walk_std_m_per_sqrt_s=0.0)
            self.corruptor = EstimateCorruptor(pre, 1, step_dt, self.seed)
        else:
            self.corruptor = EstimateCorruptor(cfg, 1, step_dt, self.seed)
        self.field_faults = FieldFaultInjector(tuple(field_faults), seed=self.seed)
        self.gps = None if gps is None else GpsInjector(gps, tuple(gps_faults), self.seed)
        self.anchor_step: int | None = None  # bridge step at which the schedule was anchored
        self.anchor_t: float | None = None
        self._last_t: float | None = None
        self._last_pos: np.ndarray | None = None
        self._first_z: float | None = None  # height reference for the altitude anchor
        self._first_t: float | None = None
        self._received = 0
        self._steps = 0

    @property
    def output_rate_hz(self) -> float:
        return 1.0 / (self.dt * self.divisor)

    @property
    def anchored(self) -> bool:
        return self.anchor_step is not None

    def update(self, t: float, pos_enu, engaged: bool = False) -> BridgeSample | None:
        """One truth sample in. ``engaged`` is whether Structure Hold is flying at this
        sample; with ``fault_anchor == "engage"`` the first True anchors the schedule."""
        pos = np.asarray(pos_enu, dtype=np.float64).reshape(3)
        if self._last_pos is None:
            vel = np.zeros(3)
            self._first_z, self._first_t = float(pos[2]), float(t)
            self.corruptor.reset(pos[None], vel[None])
            self.field_faults.start(pos)
        else:
            span = float(t) - self._last_t
            if span <= 0:
                raise ValueError(f"non-increasing sample time {t} after {self._last_t}")
            vel = (pos - self._last_pos) / span
        self._last_t, self._last_pos = float(t), pos
        self._received += 1
        if (self._received - 1) % self.divisor:
            return None
        if not self.anchored and (
            (self.fault_anchor == "engage" and engaged)
            or (self.fault_anchor == "altitude"
                and float(pos[2]) - self._first_z >= self.fault_anchor_alt_m)
        ):
            self._anchor(pos, vel, float(t))
        s = self.corruptor.step(pos[None], vel[None])
        t_rel = self.t_rel(t)
        est_pos, est_vel, stale = self.field_faults.apply(
            np.asarray(s.pos[0]), np.asarray(s.vel[0]), bool(s.stale[0]), t_rel)
        gps = self.gps.update(t, pos, vel, t_rel) if self.gps is not None else None
        sample = BridgeSample(
            step=self._steps, t=float(t), truth_pos=pos, truth_vel=vel,
            est_pos=est_pos, est_vel=est_vel,
            stale=stale, jumped=bool(s.jumped[0]),
            dropout_started=bool(s.dropout_started[0]), age=float(s.age[0]),
            engaged=bool(engaged), resets=self.field_faults.resets,
            gps=NO_GPS if gps is None else gps,
        )
        self._steps += 1
        return sample

    def t_rel(self, t: float) -> float | None:
        """Time since the fault anchor (the field faults' clock), None before it."""
        if self.fault_anchor == "start":
            return None if self._first_t is None else float(t) - self._first_t
        return None if self.anchor_t is None else float(t) - self.anchor_t

    def _anchor(self, pos: np.ndarray, vel: np.ndarray, t: float) -> None:
        """Rebuild the corruptor from the seed with the fault clock at ``fault_anchor_s``
        and carry the pre-engage state over, so the stream is continuous (module docstring)."""
        old = self.corruptor
        cfg = dataclasses.replace(self.cfg, fault_start_s=self.fault_anchor_s)
        new = EstimateCorruptor(cfg, 1, old.dt, self.seed)
        new.reset(pos[None], vel[None])  # draws the delay, as the old one did from the same seed
        assert int(new.delay_steps[0]) == int(old.delay_steps[0])
        for _ in range(new.fault_start_steps):  # the draws vio_default makes before the anchor
            new.step(pos[None], vel[None])
        # Continuity: the delayed history, bias, held outputs and dropout state are the
        # pre-engage stream's; the draws from here on are the anchored stream's. With
        # bias_from_anchor the carried bias is zero, so the walk starts here for every
        # anchor time; without it the pre-anchor walk (as long as the takeoff took) stays.
        new.buf_pos, new.buf_vel, new.head = old.buf_pos.copy(), old.buf_vel.copy(), old.head
        new.bias = old.bias.copy()
        new.jump_offset = old.jump_offset.copy()
        new.held_pos, new.held_vel = old.held_pos.copy(), old.held_vel.copy()
        new.last_d_vel = old.last_d_vel.copy()
        new.dropout_bias, new.dropout_left = old.dropout_bias.copy(), old.dropout_left.copy()
        new.age = old.age.copy()
        self.corruptor = new
        self.anchor_step, self.anchor_t = self._steps, t


@dataclass(frozen=True)
class Rangefinder:
    """A forward-facing rangefinder on the vehicle's body x axis, against the scenario's
    flat wall, from Gazebo truth: the input of the module's wall guard (``SH_WALL_MIN``),
    which the PX4 x500 model does not carry. ``normal_toward`` points from the vehicle's
    side to the wall (``Scenario.wall_normal_toward``), ENU. No noise: the first real
    flight logs the sensor's own (docs/logging-flight.md)."""

    wall_point: tuple[float, float, float]
    normal_toward: tuple[float, float, float]
    rate_hz: float = 20.0
    min_m: float = 0.1
    max_m: float = 12.0
    # Below this cosine between body x and the normal the beam is treated as missing the
    # wall (about 78 deg); the module itself ignores readings beyond 35 deg off axis.
    min_cos: float = 0.2

    def __post_init__(self) -> None:
        if not (self.rate_hz > 0 and 0 <= self.min_m < self.max_m):
            raise ValueError("rangefinder needs rate_hz > 0 and 0 <= min_m < max_m")
        if not np.isclose(np.linalg.norm(self.normal_toward), 1.0, atol=1e-6):
            raise ValueError("rangefinder normal_toward must be a unit vector")

    def range_m(self, pos_enu, quat_wxyz) -> float | None:
        """Distance along body x to the wall plane, or None when the beam misses it or the
        reading is outside the sensor's range."""
        w, x, y, z = (float(v) for v in quat_wxyz)
        body_x = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        n = np.asarray(self.normal_toward, dtype=float)
        cos = float(body_x @ n)
        if cos < self.min_cos:
            return None
        perp = float((np.asarray(self.wall_point, dtype=float) - np.asarray(pos_enu)) @ n)
        r = perp / cos
        return r if self.min_m <= r <= self.max_m else None

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v
                for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> Rangefinder:
        d = dict(d)
        for k in ("wall_point", "normal_toward"):
            if k in d:
                d[k] = tuple(float(v) for v in d[k])
        return cls(**d)


@dataclass(frozen=True)
class FrameMistake:
    """An integration mistake made on purpose at the MAVLink boundary, so the log health
    check (``eval/log_health.py``) can be tested on real PX4 logs: ``enu_as_ned`` sends the
    ENU numbers in the NED fields (the classic one); ``yaw_offset`` sends a VIO frame whose
    x axis is ``yaw_deg`` from north (clockwise seen from above, a NED yaw) while still
    declaring NED. Only what goes to PX4 changes; the trace keeps the true estimate."""

    kind: str
    yaw_deg: float = 0.0

    KINDS = ("enu_as_ned", "yaw_offset")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"frame mistake kind must be one of {self.KINDS}")

    @classmethod
    def from_dict(cls, d: dict) -> FrameMistake:
        unknown = set(d) - {"kind", "yaw_deg"}
        if unknown:
            raise ValueError(f"unknown frame mistake keys: {sorted(unknown)}")
        return cls(kind=str(d["kind"]), yaw_deg=float(d.get("yaw_deg", 0.0)))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "yaw_deg": self.yaw_deg}

    def apply(self, enu: np.ndarray) -> np.ndarray:
        """What is sent in the NED fields for an ENU vector."""
        if self.kind == "enu_as_ned":
            return np.asarray(enu, dtype=np.float64).copy()
        ned = enu_to_ned(enu)
        # coordinates in a frame whose x axis has NED yaw yaw_deg: rotate by -yaw_deg
        y = np.radians(self.yaw_deg)
        c, s = np.cos(y), np.sin(y)
        out = ned.copy()
        out[0] = c * ned[0] + s * ned[1]
        out[1] = -s * ned[0] + c * ned[1]
        return out


def odometry_ned(sample: BridgeSample, mistake: FrameMistake | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of the estimate in NED, for the ODOMETRY message (or, with a
    ``mistake``, what a wrongly wired VIO would send instead)."""
    if mistake is not None:
        return mistake.apply(sample.est_pos), mistake.apply(sample.est_vel)
    return enu_to_ned(sample.est_pos), enu_to_ned(sample.est_vel)


class TraceWriter:
    """Append every bridge sample to a CSV (``TRACE_COLUMNS``), flushed per row so a killed
    container still leaves a usable file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: IO[str] = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(TRACE_COLUMNS)
        self._fh.flush()

    def write(self, sample: BridgeSample) -> None:
        self._w.writerow(sample.row())
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_trace(path: str | Path) -> dict[str, np.ndarray]:
    """The trace CSV as column name -> float array (empty arrays for a header-only file)."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if len(r) == len(header)]
    missing = set(TRACE_COLUMNS) - set(header) - set(OPTIONAL_TRACE_COLUMNS)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    data = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(header)))
    out = {name: data[:, i] for i, name in enumerate(header)}
    for name in OPTIONAL_TRACE_COLUMNS:
        out.setdefault(name, np.zeros(data.shape[0]))
    return out
