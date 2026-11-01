"""Field faults: scripted estimate failures copied from what integrators report in the field.

The random corruption in ``envs.estimate`` (delay, noise, bias walk, decaying jumps, dropouts
up to 3 s) is the training and benchmark model. The failures that PX4 and ArduPilot users
actually post about are blunter (``docs/market-research.md``, "Estimate faults, not missing
GPS"), and a buyer will ask what the drone does in each one. This module scripts them on top
of the corruptor's output, at a set time after the bridge's fault anchor, so every runner and
seed meets the same failure at the same moment:

``reset_origin``
    The VIO restarts its frame where the drone is, so its position snaps back to its origin
    (the bridge's first sample: where the VIO started, before takeoff) on ``axes``, and moves
    from there. ``signal`` (default on) bumps ODOMETRY's ``reset_counter``, as a VIO that
    reports its resets does; EKF2 then jumps its own position with it. Off, EKF2 sees one
    huge innovation instead. The ModalAI forum case: VIO "resets to 0,0,0 in the air".
``restart``
    The VIO process dies and comes back: nothing is published for ``duration_s``, then the
    same snap as ``reset_origin``.
``silence``
    The companion link or the VIO stops for ``duration_s``: nothing reaches PX4 (longer
    than the corruptor's 3 s dropouts; the module's stale-estimate trip is 5 s).
``freeze``
    The estimate stops moving but keeps arriving on time: the last position and zero
    velocity for ``duration_s``, never marked stale. A frozen camera pipeline does this; no
    staleness check can see it.
``drift``
    A slow ramp of ``rate_mps`` along ``direction`` (ENU, normalised) for ``duration_s``,
    then held: scale drift, or the slow walk-off a spoofed source produces.

Faults apply only once the bridge's anchor is set (``start``: from the first sample) and are
timed from it, so they do not depend on how long PX4 took to take off. The corruptor runs
unchanged underneath: a fault is an offset, an override or a withheld sample on its output,
and ``verify_trace`` replays it exactly because the profile carries the faults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FIELD_FAULT_KINDS = ("reset_origin", "restart", "silence", "freeze", "drift")
_NEEDS_DURATION = ("restart", "silence", "freeze", "drift")


@dataclass(frozen=True)
class FieldFault:
    kind: str
    at_s: float  # seconds after the bridge's fault anchor
    duration_s: float = 0.0  # silence / freeze / drift length; restart: the gap before the snap
    rate_mps: float = 0.0  # drift speed
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)  # drift direction, ENU
    axes: str = "xyz"  # reset_origin / restart: the axes that snap back to the origin
    signal: bool = True  # reset_origin / restart: bump ODOMETRY reset_counter

    def __post_init__(self) -> None:
        if self.kind not in FIELD_FAULT_KINDS:
            raise ValueError(f"field fault kind must be one of {FIELD_FAULT_KINDS}, "
                             f"got {self.kind!r}")
        if not self.at_s >= 0:
            raise ValueError("at_s must be >= 0")
        if not self.duration_s >= 0:
            raise ValueError("duration_s must be >= 0")
        if self.kind in _NEEDS_DURATION and not self.duration_s > 0:
            raise ValueError(f"{self.kind} needs duration_s > 0")
        if self.kind == "drift" and not self.rate_mps > 0:
            raise ValueError("drift needs rate_mps > 0")
        if len(self.direction) != 3 or not np.linalg.norm(self.direction) > 0:
            raise ValueError("direction must be a nonzero ENU 3-vector")
        if not self.axes or set(self.axes) - set("xyz"):
            raise ValueError("axes must be a non-empty subset of 'xyz'")

    @property
    def reset_at_s(self) -> float | None:
        """When this fault snaps the frame back, if it does."""
        if self.kind == "reset_origin":
            return self.at_s
        if self.kind == "restart":
            return self.at_s + self.duration_s
        return None

    def active(self, t_rel: float) -> bool:
        """Inside this fault's window (silence, freeze, restart's gap)."""
        return self.at_s <= t_rel < self.at_s + self.duration_s

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FieldFault:
        d = dict(d)
        known = {"kind", "at_s", "duration_s", "rate_mps", "direction", "axes", "signal"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown field fault keys: {sorted(unknown)}")
        if "direction" in d:
            d["direction"] = tuple(float(v) for v in d["direction"])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "at_s": self.at_s, "duration_s": self.duration_s,
                "rate_mps": self.rate_mps, "direction": list(self.direction),
                "axes": self.axes, "signal": self.signal}


class FieldFaultInjector:
    """Applies a profile's field faults to the corruptor's output, one sample at a time."""

    def __init__(self, faults: tuple[FieldFault, ...]):
        self.faults = tuple(faults)
        self.origin: np.ndarray | None = None
        self.offset = np.zeros(3)  # accumulated frame snaps
        self.resets = 0  # ODOMETRY reset_counter value to send
        self._reset_done = [False] * len(self.faults)
        self._frozen: list[np.ndarray | None] = [None] * len(self.faults)

    def start(self, origin_enu: np.ndarray) -> None:
        """Where the VIO frame starts: the bridge's first truth sample."""
        self.origin = np.asarray(origin_enu, dtype=np.float64).reshape(3).copy()

    def apply(self, est_pos: np.ndarray, est_vel: np.ndarray, stale: bool,
              t_rel: float | None) -> tuple[np.ndarray, np.ndarray, bool]:
        """The faulted (position, velocity, stale). ``t_rel`` is the time since the anchor,
        None before it (nothing applies)."""
        if not self.faults or t_rel is None:
            return est_pos, est_vel, stale
        if self.origin is None:
            raise RuntimeError("call start() before apply()")
        pos = np.asarray(est_pos, dtype=np.float64) + self.offset
        vel = np.asarray(est_vel, dtype=np.float64).copy()
        for i, f in enumerate(self.faults):
            at = f.reset_at_s
            if at is not None and not self._reset_done[i] and t_rel >= at:
                snap = np.zeros(3)
                for k, axis in enumerate("xyz"):
                    if axis in f.axes:
                        snap[k] = self.origin[k] - pos[k]
                self.offset += snap
                pos = pos + snap
                self._reset_done[i] = True
                if f.signal:
                    self.resets += 1
            if f.kind == "drift" and t_rel >= f.at_s:
                d = np.asarray(f.direction, dtype=np.float64)
                pos = pos + d / np.linalg.norm(d) * f.rate_mps * min(t_rel - f.at_s,
                                                                     f.duration_s)
        for i, f in enumerate(self.faults):
            if f.kind == "freeze":
                if f.active(t_rel):
                    if self._frozen[i] is None:
                        self._frozen[i] = pos.copy()
                    pos, vel = self._frozen[i].copy(), np.zeros(3)
            elif f.kind in ("silence", "restart") and f.active(t_rel):
                stale = True
        return pos, vel, stale


def parse_field_faults(data: Any) -> tuple[FieldFault, ...]:
    """A profile's ``field_faults`` list (None or empty: no field faults)."""
    if not data:
        return ()
    if not isinstance(data, list):
        raise ValueError("field_faults must be a list of mappings")
    return tuple(f if isinstance(f, FieldFault) else FieldFault.from_dict(f) for f in data)

