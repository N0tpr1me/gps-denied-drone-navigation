"""Small helpers every run entrypoint uses: arguments, config, metrics and summary.

An entrypoint is a module run as ``python -m <module> --run-dir D --config D/config.yaml
[--smoke] --device cpu|cuda``. It appends one JSON object per iteration to
``metrics.jsonl`` and writes ``summary.json`` once at the end. ``training/launch.py``
creates the run folder, starts the entrypoint, and records the outcome.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    """``over`` laid over ``base``, nested dicts merged key by key; neither is changed."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def entry_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _jsonable(value: Any) -> Any:
    # NaN and inf are not valid JSON; record them as strings so the file stays parseable.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "item"):  # NumPy and PyTorch scalars
        return _jsonable(value.item())
    return value


class RunWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = open(self.run_dir / "metrics.jsonl", "a")
        self._t0 = time.monotonic()

    def metric(self, step: int, **values: Any) -> None:
        row = {"step": step, "t": round(time.monotonic() - self._t0, 3)}
        row.update({k: _jsonable(v) for k, v in values.items()})
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()

    def summary(
        self,
        status: str,
        headline: dict[str, Any] | None = None,
        vs_baseline: str | None = None,
        error: str | None = None,
    ) -> None:
        write_summary(
            self.run_dir,
            status=status,
            headline={k: _jsonable(v) for k, v in (headline or {}).items()},
            vs_baseline=vs_baseline,
            error=error,
        )

    def close(self) -> None:
        self._metrics.close()


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_summary(run_dir: Path, **fields: Any) -> None:
    path = Path(run_dir) / "summary.json"
    data = read_summary(run_dir) or {}
    data.update({k: v for k, v in fields.items() if v is not None})
    path.write_text(json.dumps(data, indent=2) + "\n")
