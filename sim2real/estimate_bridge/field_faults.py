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
``degrade``
    The camera sees less for ``duration_s`` (dust thrown up by the rotors, fog or steam, a
    dark underside): the VIO keeps running on fewer and worse features, so it gets noisier
    (``noise_m``, ``vel_noise_mps``), walks off faster (``walk_mps``, m/sqrt(s), kept after
    the window: a VIO has no way to take it back), and loses tracking for ``loss_s`` at
    random (``loss_hz``; nothing is sent). A laser rangefinder in the same dust or fog
    gets noisy (``range_noise_m``), misses returns (``range_loss_frac``) and sees short
    returns off the particles (``range_short_frac``); ``range_reading`` applies that. Every
    effect scales with a severity that ramps from 0 to 1 over ``ramp_s`` from ``at_s``.
    The random draws come from the profile's seed (``FieldFaultInjector(seed=...)``); the
    rangefinder's from their own stream, so ``verify_trace``, which replays the estimate
    only, still matches.
``runaway``
    The VIO locks onto something that moves (rotor-blown dust grains, the docs/vio-render.md
    finding: Basalt in heavy dust tracked the drifting grains and ran away by kilometres
    without reporting a tracking loss). Its velocity error grows at ``accel_mps2`` along
    ``direction`` for ``duration_s`` and the position integrates it, so the position and the
    velocity it reports agree with each other and disagree with the IMU. After the window
    the position error is held, as for ``drift``. ``max_speed_mps`` (0: none) caps the
    velocity error, for a VIO that runs off at a steady speed: Basalt started in drifting
    dust ran at up to 2.8 m/s on the floor before take-off (``field_runaway_prearm``).

Faults apply only once the bridge's anchor is set (``start``: from the first sample) and are
timed from it, so they do not depend on how long PX4 took to take off. The corruptor runs
unchanged underneath: a fault is an offset, an override or a withheld sample on its output,
and ``verify_trace`` replays it exactly because the profile carries the faults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FIELD_FAULT_KINDS = ("reset_origin", "restart", "silence", "freeze", "drift", "degrade",
                     "runaway")
_NEEDS_DURATION = ("restart", "silence", "freeze", "drift", "degrade", "runaway")
_DEGRADE_KEYS = ("ramp_s", "noise_m", "vel_noise_mps", "walk_mps", "loss_hz", "loss_s",
                 "range_noise_m", "range_loss_frac", "range_short_frac")
_ESTIMATE_STREAM, _RANGE_STREAM = 0xD0, 0xD1  # degrade draws: separate streams per seed


@dataclass(frozen=True)
class FieldFault:
    kind: str
    at_s: float  # seconds after the bridge's fault anchor
    duration_s: float = 0.0  # silence / freeze / drift length; restart: the gap before the snap
    rate_mps: float = 0.0  # drift speed
    accel_mps2: float = 0.0  # runaway: how fast the VIO's velocity error grows
    max_speed_mps: float = 0.0  # runaway: the velocity error stops growing here (0: no cap)
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)  # drift direction, ENU
    axes: str = "xyz"  # reset_origin / restart: the axes that snap back to the origin
    signal: bool = True  # reset_origin / restart: bump ODOMETRY reset_counter
    # degrade only (module docstring): full-severity values, reached ramp_s after at_s
    ramp_s: float = 0.0
    noise_m: float = 0.0  # extra white position noise, per axis
    vel_noise_mps: float = 0.0  # extra white velocity noise, per axis
    walk_mps: float = 0.0  # extra random walk, m/sqrt(s) per axis
    loss_hz: float = 0.0  # tracking losses per second
    loss_s: float = 0.0  # length of each loss
    range_noise_m: float = 0.0  # rangefinder white noise
    range_loss_frac: float = 0.0  # share of rangefinder readings with no return
    range_short_frac: float = 0.0  # share of readings that return short, off the particles

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
        if self.kind == "runaway" and not self.accel_mps2 > 0:
            raise ValueError("runaway needs accel_mps2 > 0")
        if self.kind != "runaway" and (self.accel_mps2 or self.max_speed_mps):
            raise ValueError("accel_mps2 and max_speed_mps apply to runaway only")
        if not self.max_speed_mps >= 0:
            raise ValueError("max_speed_mps must be >= 0")
        if len(self.direction) != 3 or not np.linalg.norm(self.direction) > 0:
            raise ValueError("direction must be a nonzero ENU 3-vector")
        if not self.axes or set(self.axes) - set("xyz"):
            raise ValueError("axes must be a non-empty subset of 'xyz'")
        values = {k: getattr(self, k) for k in _DEGRADE_KEYS}
        if any(not v >= 0 for v in values.values()):
            raise ValueError(f"degrade values must be >= 0: {values}")
        if self.kind != "degrade":
            if any(values.values()):
                raise ValueError(f"{sorted(k for k, v in values.items() if v)} "
                                 "apply to degrade only")
            return
        if not any(values[k] for k in _DEGRADE_KEYS if k != "ramp_s"):
            raise ValueError("degrade needs at least one effect")
        if (self.loss_hz > 0) != (self.loss_s > 0):
            raise ValueError("degrade needs loss_hz and loss_s together")
        if self.range_loss_frac + self.range_short_frac > 1:
            raise ValueError("range_loss_frac + range_short_frac must be <= 1")

    def severity(self, t_rel: float) -> float:
        """degrade: 0 outside the window, ramping to 1 over ramp_s inside it."""
        if not self.active(t_rel):
            return 0.0
        if self.ramp_s <= 0:
            return 1.0
        return min(1.0, (t_rel - self.at_s) / self.ramp_s)

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
        known = {"kind", "at_s", "duration_s", "rate_mps", "accel_mps2", "max_speed_mps",
                 "direction", "axes", "signal", *_DEGRADE_KEYS}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown field fault keys: {sorted(unknown)}")
        if "direction" in d:
            d["direction"] = tuple(float(v) for v in d["direction"])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = {"kind": self.kind, "at_s": self.at_s, "duration_s": self.duration_s,
             "rate_mps": self.rate_mps, "direction": list(self.direction),
             "axes": self.axes, "signal": self.signal}
        if self.kind == "degrade":  # old profiles stay as they were
            d.update({k: getattr(self, k) for k in _DEGRADE_KEYS})
        if self.kind == "runaway":
            d["accel_mps2"] = self.accel_mps2
            if self.max_speed_mps:  # old profiles stay as they were
                d["max_speed_mps"] = self.max_speed_mps
        return d


class FieldFaultInjector:
    """Applies a profile's field faults to the corruptor's output, one sample at a time."""

    def __init__(self, faults: tuple[FieldFault, ...], seed: int = 0):
        self.faults = tuple(faults)
        self.origin: np.ndarray | None = None
        self.offset = np.zeros(3)  # accumulated frame snaps
        self.resets = 0  # ODOMETRY reset_counter value to send
        self._reset_done = [False] * len(self.faults)
        self._frozen: list[np.ndarray | None] = [None] * len(self.faults)
        # degrade: the walk it has added so far, the end of the current tracking loss
        self._walk = [np.zeros(3) for _ in self.faults]
        self._loss_until = [-np.inf] * len(self.faults)
        self._last_t: float | None = None
        self._rng = np.random.default_rng([int(seed), _ESTIMATE_STREAM])
        self._range_rng = np.random.default_rng([int(seed), _RANGE_STREAM])

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
        dt = 0.0 if self._last_t is None else max(0.0, t_rel - self._last_t)
        self._last_t = t_rel
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
            if f.kind == "runaway" and t_rel >= f.at_s:
                d = np.asarray(f.direction, dtype=np.float64)
                d = d / np.linalg.norm(d)
                tau = min(t_rel - f.at_s, f.duration_s)
                # Accelerating up to the cap (if any), then running on at it.
                ramp = min(tau, f.max_speed_mps / f.accel_mps2) if f.max_speed_mps else tau
                speed = f.accel_mps2 * ramp
                pos = pos + d * (0.5 * f.accel_mps2 * ramp * ramp + speed * (tau - ramp))
                if f.active(t_rel):
                    vel = vel + d * speed
            if f.kind == "degrade":
                sev = f.severity(t_rel)
                if sev > 0:
                    # Fixed draws per sample in the window, so a replay meets the same ones.
                    z = self._rng.standard_normal(9)
                    u = self._rng.random()
                    self._walk[i] = self._walk[i] + f.walk_mps * sev * np.sqrt(dt) * z[6:9]
                    pos = pos + f.noise_m * sev * z[0:3]
                    vel = vel + f.vel_noise_mps * sev * z[3:6]
                    if t_rel >= self._loss_until[i] and u < f.loss_hz * sev * dt:
                        self._loss_until[i] = t_rel + f.loss_s
                    if t_rel < self._loss_until[i]:
                        stale = True
                pos = pos + self._walk[i]  # kept after the window
        for i, f in enumerate(self.faults):
            if f.kind == "freeze":
                if f.active(t_rel):
                    if self._frozen[i] is None:
                        self._frozen[i] = pos.copy()
                    pos, vel = self._frozen[i].copy(), np.zeros(3)
            elif f.kind in ("silence", "restart") and f.active(t_rel):
                stale = True
        return pos, vel, stale


    def range_reading(self, r: float | None, min_m: float,
                      t_rel: float | None) -> float | None:
        """A rangefinder reading through the degrade faults' dust or fog: noisier, sometimes
        missing (None), sometimes short. ``r`` is the clean reading (None: no return)."""
        if r is None or t_rel is None:
            return r
        for f in self.faults:
            if f.kind != "degrade":
                continue
            sev = f.severity(t_rel)
            if sev <= 0:
                continue
            u, z, v = self._range_rng.random(), self._range_rng.standard_normal(), \
                self._range_rng.random()
            if u < f.range_loss_frac * sev:
                return None
            if u < (f.range_loss_frac + f.range_short_frac) * sev:
                r = min_m + v * (r - min_m)  # a return off the particles in front
            else:
                r = max(min_m, r + f.range_noise_m * sev * z)
        return r


def parse_field_faults(data: Any) -> tuple[FieldFault, ...]:
    """A profile's ``field_faults`` list (None or empty: no field faults)."""
    if not data:
        return ()
    if not isinstance(data, list):
        raise ValueError("field_faults must be a list of mappings")
    return tuple(f if isinstance(f, FieldFault) else FieldFault.from_dict(f) for f in data)

