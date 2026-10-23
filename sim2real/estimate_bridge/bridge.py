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

TRACE_COLUMNS = (
    "step", "t", "published", "stale", "jumped", "dropout_started", "age",
    "tx", "ty", "tz", "tvx", "tvy", "tvz",
    "ex", "ey", "ez", "evx", "evy", "evz",
    "engaged",
)
# Trace columns added after the first flights; read as 0 when an old trace lacks them.
OPTIONAL_TRACE_COLUMNS = ("engaged",)
FAULT_ANCHORS = ("start", "engage")
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
            int(self.engaged),
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
    ):
        if divisor < 1:
            raise ValueError("divisor must be >= 1")
        if fault_anchor not in FAULT_ANCHORS:
            raise ValueError(f"fault_anchor must be one of {FAULT_ANCHORS}, got {fault_anchor!r}")
        if not fault_anchor_s >= 0:
            raise ValueError("fault_anchor_s must be >= 0")
        self.cfg, self.dt, self.seed, self.divisor = cfg, float(dt), int(seed), int(divisor)
        self.fault_anchor, self.fault_anchor_s = fault_anchor, float(fault_anchor_s)
        step_dt = self.dt * self.divisor
        if fault_anchor == "engage":
            # Same seed, same draws, no fault until the anchor rebuilds the corruptor.
            pre = dataclasses.replace(cfg, fault_start_s=NEVER_S)
            self.corruptor = EstimateCorruptor(pre, 1, step_dt, self.seed)
        else:
            self.corruptor = EstimateCorruptor(cfg, 1, step_dt, self.seed)
        self.anchor_step: int | None = None  # bridge step at which the schedule was anchored
        self.anchor_t: float | None = None
        self._last_t: float | None = None
        self._last_pos: np.ndarray | None = None
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
            self.corruptor.reset(pos[None], vel[None])
        else:
            span = float(t) - self._last_t
            if span <= 0:
                raise ValueError(f"non-increasing sample time {t} after {self._last_t}")
            vel = (pos - self._last_pos) / span
        self._last_t, self._last_pos = float(t), pos
        self._received += 1
        if (self._received - 1) % self.divisor:
            return None
        if engaged and self.fault_anchor == "engage" and not self.anchored:
            self._anchor(pos, vel, float(t))
        s = self.corruptor.step(pos[None], vel[None])
        sample = BridgeSample(
            step=self._steps, t=float(t), truth_pos=pos, truth_vel=vel,
            est_pos=np.asarray(s.pos[0]), est_vel=np.asarray(s.vel[0]),
            stale=bool(s.stale[0]), jumped=bool(s.jumped[0]),
            dropout_started=bool(s.dropout_started[0]), age=float(s.age[0]),
            engaged=bool(engaged),
        )
        self._steps += 1
        return sample

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
        # pre-engage stream's; the draws from here on are the anchored stream's.
        new.buf_pos, new.buf_vel, new.head = old.buf_pos.copy(), old.buf_vel.copy(), old.head
        new.bias = old.bias.copy()
        new.jump_offset = old.jump_offset.copy()
        new.held_pos, new.held_vel = old.held_pos.copy(), old.held_vel.copy()
        new.last_d_vel = old.last_d_vel.copy()
        new.dropout_bias, new.dropout_left = old.dropout_bias.copy(), old.dropout_left.copy()
        new.age = old.age.copy()
        self.corruptor = new
        self.anchor_step, self.anchor_t = self._steps, t


def odometry_ned(sample: BridgeSample) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of the estimate in NED, for the ODOMETRY message."""
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
