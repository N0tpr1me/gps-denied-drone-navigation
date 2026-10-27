"""Estimate profiles: which corruption a SITL run flies on, as one YAML file.

    name: vio_default
    seed: 1                 # overridden by --seed; the scenario's seed is NOT reused, so
                            # wind and estimate draws stay independent, as in training
    divisor: 1              # corruptor runs every divisor-th truth sample (50 Hz / divisor)
    estimate: {...}         # EstimateCorruptionConfig keys; "clean" for no corruption
    fault_anchor: start     # or "engage": fault clock keyed to the first Structure Hold
    fault_anchor_s: 25.0    #   engage, at this fault-clock value (bridge.py docstring);
    bias_from_anchor: false # true: bias walk held at zero until the anchor, so both runners
                            #   meet the same bias from it (bridge.py, "Bias from the anchor")
    fault_anchor_alt_m: null  # or "altitude": keyed to reaching this height above the
                            #   first sample (null: the stock runner fills it from the
                            #   scenario's hover altitude)
    rangefinder: null       # or bridge.Rangefinder keys: the bridge also sends a forward
                            #   DISTANCE_SENSOR against the wall (``run_gazebo --rangefinder``
                            #   fills it from the scenario); not part of the corruption

Shipped profiles live in ``profiles/``: ``clean`` (the T7 reference), ``vio_default``
(the ``envs.estimate`` defaults, i.e. what training uses), ``vio_after_takeoff``
(``vio_default`` with no jump or dropout in the first 24 s, while PX4 arms and takes off)
and ``vio_at_engage`` (``vio_default`` with its jumps and dropouts keyed to the engage,
fault clock 25 s at engage, so every flight meets the same faults at the same time after
engage and none before it); ``vio_at_hover`` is the same schedule keyed to reaching the
hover altitude, for the stock PX4 baseline, which never engages the mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from envs.estimate import EstimateCorruptionConfig
from sim2real.estimate_bridge.bridge import FAULT_ANCHORS, EstimateBridge, Rangefinder

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
SOURCE_RATE_HZ = 50.0  # Gazebo's /world/<w>/pose/info rate in the PX4 worlds
HOVER_MARGIN_M = 0.3  # ulog_checks.hover_window's margin: where the scored hover begins


@dataclass(frozen=True)
class EstimateProfile:
    name: str
    seed: int = 0
    divisor: int = 1
    estimate: EstimateCorruptionConfig = field(default_factory=EstimateCorruptionConfig)
    fault_anchor: str = "start"
    fault_anchor_s: float = 25.0
    fault_anchor_alt_m: float | None = None  # "altitude" anchor; None until a runner sets it
    bias_from_anchor: bool = False  # bias walk starts at the anchor (bridge.py docstring)
    rangefinder: Rangefinder | None = None  # forward range against the wall (run_bridge)

    def __post_init__(self) -> None:
        if self.fault_anchor not in FAULT_ANCHORS:
            raise ValueError(f"fault_anchor must be one of {FAULT_ANCHORS}")
        if not self.fault_anchor_s >= 0:
            raise ValueError("fault_anchor_s must be >= 0")
        if self.fault_anchor_alt_m is not None and not self.fault_anchor_alt_m > 0:
            raise ValueError("fault_anchor_alt_m must be > 0")

    @property
    def dt(self) -> float:
        """Spacing of the truth samples the bridge consumes, s."""
        return 1.0 / SOURCE_RATE_HZ

    @property
    def clean(self) -> bool:
        return self.estimate == EstimateCorruptionConfig.clean()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EstimateProfile:
        d = dict(d)
        unknown = set(d) - {"name", "seed", "divisor", "estimate", "fault_anchor",
                            "fault_anchor_s", "fault_anchor_alt_m", "bias_from_anchor",
                            "rangefinder"}
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        if "name" not in d:
            raise ValueError("profile needs a name")
        est = d.get("estimate", {})
        if est == "clean":
            cfg = EstimateCorruptionConfig.clean()
        elif est is None or isinstance(est, dict):
            cfg = EstimateCorruptionConfig.from_dict(est or {})
        else:
            raise ValueError("estimate must be a mapping of corruption keys or 'clean'")
        divisor = int(d.get("divisor", 1))
        if divisor < 1:
            raise ValueError("divisor must be >= 1")
        alt = d.get("fault_anchor_alt_m")
        rng = d.get("rangefinder")
        return cls(name=str(d["name"]), seed=int(d.get("seed", 0)), divisor=divisor,
                   estimate=cfg, fault_anchor=str(d.get("fault_anchor", "start")),
                   fault_anchor_s=float(d.get("fault_anchor_s", 25.0)),
                   fault_anchor_alt_m=None if alt is None else float(alt),
                   bias_from_anchor=bool(d.get("bias_from_anchor", False)),
                   rangefinder=None if rng is None else Rangefinder.from_dict(rng))

    def with_seed(self, seed: int) -> EstimateProfile:
        """The same profile with another seed (fault anchor and everything else kept)."""
        return replace(self, seed=seed)

    def with_anchor_altitude(self, hover_altitude_m: float,
                             margin_m: float = HOVER_MARGIN_M) -> EstimateProfile:
        """For an ``altitude`` anchor with no height set: anchor where the scored hover
        window begins, ``margin_m`` below the hover altitude (``ulog_checks.hover_window``).
        Other anchors, and a height already set, are returned unchanged."""
        if self.fault_anchor != "altitude" or self.fault_anchor_alt_m is not None:
            return self
        return replace(self, fault_anchor_alt_m=float(hover_altitude_m) - float(margin_m))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "divisor": self.divisor,
                "estimate": {k: v for k, v in self.estimate.__dict__.items()},
                "fault_anchor": self.fault_anchor, "fault_anchor_s": self.fault_anchor_s,
                "fault_anchor_alt_m": self.fault_anchor_alt_m,
                "bias_from_anchor": self.bias_from_anchor,
                "rangefinder": None if self.rangefinder is None else self.rangefinder.to_dict()}

    def bridge(self, dt: float | None = None, divisor: int | None = None) -> EstimateBridge:
        """The bridge this profile describes (``dt`` / ``divisor`` overrides for replays)."""
        return EstimateBridge(self.estimate, self.dt if dt is None else dt, self.seed,
                              self.divisor if divisor is None else divisor,
                              self.fault_anchor, self.fault_anchor_s, self.fault_anchor_alt_m,
                              self.bias_from_anchor)


def load_profile(path: str | Path) -> EstimateProfile:
    """Load a profile YAML; a bare name is looked up in ``profiles/``."""
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = PROFILE_DIR / f"{path}.yaml"
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: profile must be a mapping")
    return EstimateProfile.from_dict(data)
