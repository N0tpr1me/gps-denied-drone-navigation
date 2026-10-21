"""Estimate profiles: which corruption a SITL run flies on, as one YAML file.

    name: vio_default
    seed: 1                 # overridden by --seed; the scenario's seed is NOT reused, so
                            # wind and estimate draws stay independent, as in training
    divisor: 1              # corruptor runs every divisor-th truth sample (50 Hz / divisor)
    estimate: {...}         # EstimateCorruptionConfig keys; "clean" for no corruption

Shipped profiles live in ``profiles/``: ``clean`` (the T7 reference) and ``vio_default``
(the ``envs.estimate`` defaults, i.e. what training uses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from envs.estimate import EstimateCorruptionConfig

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
SOURCE_RATE_HZ = 50.0  # Gazebo's /world/<w>/pose/info rate in the PX4 worlds


@dataclass(frozen=True)
class EstimateProfile:
    name: str
    seed: int = 0
    divisor: int = 1
    estimate: EstimateCorruptionConfig = field(default_factory=EstimateCorruptionConfig)

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
        unknown = set(d) - {"name", "seed", "divisor", "estimate"}
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
        return cls(name=str(d["name"]), seed=int(d.get("seed", 0)), divisor=divisor,
                   estimate=cfg)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "divisor": self.divisor,
                "estimate": {k: v for k, v in self.estimate.__dict__.items()}}


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
