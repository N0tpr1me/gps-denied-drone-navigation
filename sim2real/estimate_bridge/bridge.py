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
"""

from __future__ import annotations

import csv
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
)


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
    ):
        if divisor < 1:
            raise ValueError("divisor must be >= 1")
        self.cfg, self.dt, self.seed, self.divisor = cfg, float(dt), int(seed), int(divisor)
        self.corruptor = EstimateCorruptor(cfg, 1, self.dt * self.divisor, self.seed)
        self._last_t: float | None = None
        self._last_pos: np.ndarray | None = None
        self._received = 0
        self._steps = 0

    @property
    def output_rate_hz(self) -> float:
        return 1.0 / (self.dt * self.divisor)

    def update(self, t: float, pos_enu) -> BridgeSample | None:
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
        s = self.corruptor.step(pos[None], vel[None])
        sample = BridgeSample(
            step=self._steps, t=float(t), truth_pos=pos, truth_vel=vel,
            est_pos=np.asarray(s.pos[0]), est_vel=np.asarray(s.vel[0]),
            stale=bool(s.stale[0]), jumped=bool(s.jumped[0]),
            dropout_started=bool(s.dropout_started[0]), age=float(s.age[0]),
        )
        self._steps += 1
        return sample


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
    missing = set(TRACE_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    data = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(header)))
    return {name: data[:, i] for i, name in enumerate(header)}
