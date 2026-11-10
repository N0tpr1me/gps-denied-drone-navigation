"""Score a surface pass: the truth against a hold point that moves (docs/surface-passes.md).

``eval.metrics`` already takes a setpoint per sample; a pass only needs that setpoint built
with care, and two numbers a plain hold has no use for:

- ``moving_reference``: the hold point at each truth sample, interpolated linearly between
  the logged samples. PX4 logs its position setpoint every 100 ms and holding each sample
  until the next would leave the reference up to 5 cm behind at 0.5 m/s, which reads as the
  vehicle running ahead.
- ``pass_numbers``: the error split along the wall normal (the standoff, the camera's
  distance to the surface, what an inspection cares about most; ``standoff_p95_m`` as
  ``eval/gate1.py`` reports it in the mock) and along the pass (how far the vehicle lags or
  leads the point), and how far and for how long the reference moved inside the window.

Both the module's flights (``sim2real.px4.run_gazebo``, reference from its status topic)
and stock PX4's (``sim2real.estimate_bridge.run_estimate``, reference from PX4's own
position setpoint) are scored through here, on the TRUE position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eval.metrics import Trace

MOVING_MPS = 0.02  # the reference counts as moving above this speed


def moving_reference(t: np.ndarray, ref_t: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """``ref_pos`` (M, 3), sampled at ``ref_t``, at the times ``t``: linear in between, the
    first and last sample held outside. Samples that are not finite are skipped."""
    ref_t = np.asarray(ref_t, dtype=float)
    ref_pos = np.asarray(ref_pos, dtype=float).reshape(-1, 3)
    ok = np.isfinite(ref_t) & np.all(np.isfinite(ref_pos), axis=1)
    ref_t, ref_pos = ref_t[ok], ref_pos[ok]
    if ref_t.size == 0:
        raise ValueError("no finite reference sample")
    ref_t, first = np.unique(ref_t, return_index=True)
    ref_pos = ref_pos[first]
    return np.stack([np.interp(t, ref_t, ref_pos[:, k]) for k in range(3)], axis=1)


def moving_trace(trace: Trace, ref_t: np.ndarray, ref_pos: np.ndarray,
                 dt: float = 0.002) -> Trace:
    """``trace`` again with the moving reference as its setpoint, on a uniform ``dt`` grid.

    ``eval.metrics`` holds a setpoint from one position sample to the next (right for a
    step, not for a ramp), and the truth is logged a few tens of times a second: resampling
    here first, position linear and the reference with it, leaves that hold one ``dt`` to
    smear, 1 mm at 0.5 m/s."""
    n = int(np.floor((trace.t[-1] - trace.t[0]) / dt + 1e-9)) + 1
    t = trace.t[0] + dt * np.arange(n)
    pos = np.stack([np.interp(t, trace.t, trace.pos[:, k]) for k in range(3)], axis=1)
    return Trace(t=t, pos=pos, pos_sp=moving_reference(t, ref_t, ref_pos), cmd_t=trace.cmd_t,
                 rates_cmd=trace.rates_cmd, thrust_cmd=trace.thrust_cmd,
                 meta={**trace.meta, "reference": "moving"})


def pass_numbers(trace: Trace, window: tuple[float, float], normal_toward) -> dict[str, Any]:
    """The pass in ``window``: how far the reference moved and for how long, and the error
    along the wall normal and along the pass. ``normal_toward`` points from the drone into
    the wall (the scenario's)."""
    inside = (trace.t >= window[0]) & (trace.t <= window[1])
    if inside.sum() < 2:
        raise ValueError("fewer than two samples in the window")
    t, pos, sp = trace.t[inside], trace.pos[inside], trace.pos_sp[inside]
    n = np.asarray(normal_toward, dtype=float)
    n = n / np.linalg.norm(n)
    err = pos - sp
    travel = sp[-1] - sp[0]
    length = float(np.linalg.norm(travel))
    out: dict[str, Any] = {
        "travel_m": length,
        # The reference's own motion along the normal: zero for a pass at a fixed standoff
        # (anything else is a guard moving the point, or a pass that is not parallel).
        "travel_along_normal_m": float(travel @ n),
        "standoff_p95_m": float(np.percentile(np.abs(err @ n), 95)),
        "standoff_max_m": float(np.abs(err @ n).max()),
        "standoff_max_toward_m": float(max((err @ n).max(), 0.0)),
        "error_max_m": float(np.linalg.norm(err, axis=1).max()),
    }
    # Where the vehicle ended up: its mean position over the last 3 s against the
    # reference's last point.
    last = t >= t[-1] - 3.0
    out["end_error_m"] = float(np.linalg.norm(pos[last].mean(axis=0) - sp[-1]))
    speed = np.linalg.norm(np.gradient(sp, t, axis=0), axis=1)
    moving = speed > MOVING_MPS
    out["moving_s"] = float(np.sum(np.diff(t)[moving[1:]]))
    if length > 1e-3:
        d = travel / length
        along = err @ d
        out["along_p95_m"] = float(np.percentile(np.abs(along), 95))
        # Negative: the vehicle is behind the point.
        out["along_mean_while_moving_m"] = float(along[moving].mean()) if moving.any() else 0.0
        across = err - np.outer(along, d) - np.outer(err @ n, n)
        out["across_p95_m"] = float(np.percentile(np.linalg.norm(across, axis=1), 95))
        if moving.any():
            e = np.linalg.norm(err, axis=1)
            out["p95_while_moving_m"] = float(np.percentile(e[moving], 95))
            out["p95_while_still_m"] = (float(np.percentile(e[~moving], 95))
                                        if (~moving).any() else float("nan"))
            out["moving_from_s"] = float(t[moving][0])
            out["moving_until_s"] = float(t[moving][-1])
    return out
