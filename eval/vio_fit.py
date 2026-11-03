"""Measure a real VIO's error against ground truth and fit ``envs/estimate`` to it.

    python -m training.launch --config eval/configs/vio_fit_rpg_euroc.yaml --device cpu

This replaces the logging flight (docs/logging-flight.md) with public real-drone data
(docs/decisions.md, software only, 2026-10-30). Input is pairs of time-stamped trajectories,
the VIO's output and the dataset's ground truth, one pair per flight. For each flight:

1. **Associate**: truth is interpolated to each VIO sample time (truth runs at 100 to 200 Hz,
   VIO at 10 to 30 Hz). VIO samples outside the truth's span are dropped.
2. **Align**: the VIO's world frame is its own (gravity-aligned, arbitrary yaw and origin),
   so it is rotated about z and shifted onto truth by least squares (4-DoF, as the RPG
   trajectory evaluation and every VIO paper do). ``e(t)`` = aligned VIO minus truth.
3. **Gaps** (``dropouts``): a step between VIO samples longer than ``gap_min_s`` and
   ``gap_factor`` times the median period. During a real tracking loss the VIO sends
   nothing; in the corruptor that is a dropout.
4. **Jumps**: a single-step change of ``e`` larger than ``jump_min_m`` and ``jump_k`` times
   the robust spread of single-step changes. The step is taken out of ``e`` before the next
   two measures, so a reset does not count as drift. How much of each jump is still there
   ``decay_probe_s`` later gives its decay.
5. **Drift and noise**: the error's structure function ``V(tau)``, the mean over sample pairs
   ``tau`` apart of ``(e(t + tau) - e(t))^2`` per axis. The corruptor's error is a random walk
   plus white noise, for which ``V(tau) = 2 noise^2 + walk^2 tau``. Both are fitted by
   least squares on the relative misfit over lags from one sample to ``walk_match_s`` (20 s,
   a training episode), and ``V`` at each reported horizon is printed beside the model's
   so the fit's error is visible. ``bias_max_m`` is the 95th percentile of ``|e(t + tau) -
   e(t)|`` at ``cap_horizon_s`` (60 s, the Gazebo hold). Both are measured twice: over every
   window, and over near-hover windows only (truth moved at most ``hover_speed_max_mps`` on
   average), which is what a hold sees and what the proposal uses when there are enough.
6. **Velocity** noise, only when the VIO file carries velocity (columns 9 to 11): the rms of
   VIO velocity minus truth velocity (central differences of interpolated truth).
7. **Time offset**: the lag that best lines up VIO speed with truth speed. With offline
   VIO this is the dataset's clock alignment, not latency: latency needs the VIO's processing
   time on a live run, and is not fitted here.

Errors are relative over windows (drift since engage), which is what a hold sees: the
corruptor starts its bias at zero at reset. Dataset flights move a few m/s through a room, so
drift that grows with distance travelled (scale and yaw error) shows up as drift in time
over all windows; the near-hover split keeps most of that out.

Pooled over all flights (time-weighted), the result is a proposed
``EstimateCorruptionConfig`` (``proposal``), printed beside the current defaults. Fields the
data cannot measure (delay, dropout dead-reckoning bias, velocity noise without velocity)
keep the default and are listed under ``kept``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from envs.estimate import EstimateCorruptionConfig
from training.runlib import RunWriter, entry_args, load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FitParams:
    skip_start_s: float = 0.0  # VIO initialisation transient, dropped before alignment
    gap_min_s: float = 0.2
    gap_factor: float = 3.0
    jump_min_m: float = 0.1
    jump_k: float = 8.0
    decay_probe_s: float = 5.0
    walk_match_s: float = 20.0
    cap_horizon_s: float = 60.0
    report_lags_s: tuple[float, ...] = (0.5, 1.0, 5.0, 20.0, 60.0)
    # Windows whose truth moved no faster than this on average count as near-hover.
    hover_speed_max_mps: float = 0.3
    # The near-hover fit is used only with at least this many pairs, from at least this
    # many flights, at the lag in question (walk_match_s for the walk, cap_horizon_s for
    # the cap); one flight's slow stretch is one situation, not a sample.
    hover_min_pairs: int = 200
    hover_min_flights: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitParams:
        data = dict(data)
        if "report_lags_s" in data:
            data["report_lags_s"] = tuple(data["report_lags_s"])
        return cls(**data)


@dataclass
class Trajectory:
    t: np.ndarray  # (N,) seconds
    pos: np.ndarray  # (N, 3) metres
    vel: np.ndarray | None = None  # (N, 3) m/s, when the file carries it


def load_tum(path: str | Path) -> Trajectory:
    """``t x y z qx qy qz qw [vx vy vz]`` per line, ``#`` comments (TUM / RPG format)."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"{path}: expected at least t x y z per line")
    vel = data[:, 8:11] if data.shape[1] >= 11 else None
    return Trajectory(data[:, 0], data[:, 1:4], vel)


def load_euroc_csv(path: str | Path) -> Trajectory:
    """EuRoC ``state_groundtruth_estimate0/data.csv``: ns, p xyz, q wxyz, v xyz, biases."""
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    return Trajectory(data[:, 0] * 1e-9, data[:, 1:4], data[:, 8:11])


def associate(est: Trajectory, gt: Trajectory) -> tuple[Trajectory, np.ndarray, np.ndarray]:
    """Truth position and velocity at each VIO time inside truth's span."""
    keep = (est.t >= gt.t[0]) & (est.t <= gt.t[-1])
    t = est.t[keep]
    gt_pos = np.stack([np.interp(t, gt.t, gt.pos[:, k]) for k in range(3)], axis=1)
    gt_vel_full = gt.vel if gt.vel is not None else np.gradient(gt.pos, gt.t, axis=0)
    gt_vel = np.stack([np.interp(t, gt.t, gt_vel_full[:, k]) for k in range(3)], axis=1)
    vel = est.vel[keep] if est.vel is not None else None
    return Trajectory(t, est.pos[keep], vel), gt_pos, gt_vel


def align_pos_yaw(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Rotate ``est`` about z and shift it onto ``gt`` (least squares); returns the result."""
    me, mg = est.mean(0), gt.mean(0)
    a, b = est - me, gt - mg
    # Best yaw for the horizontal part: atan2 of the summed cross and dot products.
    yaw = math.atan2(
        float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()),
        float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()),
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    shift = mg - rot @ me
    return est @ rot.T + shift, yaw, rot


def pair_indices(t: np.ndarray, lag: float, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i, j) with ``t[j] - t[i]`` within ``tol`` of ``lag``."""
    j = np.searchsorted(t, t + lag)
    j = np.clip(j, 0, len(t) - 1)
    # The nearer of j - 1 and j.
    jm = np.clip(j - 1, 0, len(t) - 1)
    pick = np.where(np.abs(t[jm] - t - lag) < np.abs(t[j] - t - lag), jm, j)
    i = np.arange(len(t))
    ok = (np.abs(t[pick] - t - lag) <= tol) & (pick > i)
    return i[ok], pick[ok]


def time_offset(t: np.ndarray, est_pos: np.ndarray, gt_vel: np.ndarray,
                max_s: float = 0.3) -> float:
    """Lag (s) of VIO speed behind truth speed, by cross-correlation on a uniform grid."""
    period = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], period)
    speed_est = np.linalg.norm(np.gradient(est_pos, t, axis=0), axis=1)
    speed_gt = np.linalg.norm(gt_vel, axis=1)
    a = np.interp(grid, t, speed_est)
    b = np.interp(grid, t, speed_gt)
    a, b = a - a.mean(), b - b.mean()
    n = int(round(max_s / period))
    best, best_lag = -np.inf, 0
    for k in range(-n, n + 1):
        if k >= 0:
            c = float((a[k:] * b[: len(b) - k]).sum())
        else:
            c = float((a[:k] * b[-k:]).sum())
        if c > best:
            best, best_lag = c, k
    return best_lag * period


@dataclass
class FlightStats:
    name: str
    duration_s: float
    period_s: float
    yaw_align_deg: float
    ate_rms_m: float
    gaps_s: list[float] = field(default_factory=list)
    jumps_m: list[float] = field(default_factory=list)
    jump_remaining: list[float] = field(default_factory=list)
    # Per lag (s): mean squared increment per axis (x, y, z) and pair count, over all
    # windows and over near-hover windows only.
    sf: dict[float, tuple[list[float], int]] = field(default_factory=dict)
    sf_hover: dict[float, tuple[list[float], int]] = field(default_factory=dict)
    # |increment| per axis at the cap horizon (for the pooled percentile), same split.
    cap_abs: list[float] = field(default_factory=list)
    cap_abs_hover: list[float] = field(default_factory=list)
    vel_sq_err: tuple[list[float], int] | None = None
    time_offset_s: float = 0.0
    path_m_per_s: float = 0.0


def measure_flight(name: str, est: Trajectory, gt: Trajectory, p: FitParams) -> FlightStats:
    est, gt_pos, gt_vel = associate(est, gt)
    if est.t.size < 20:
        raise ValueError(f"{name}: fewer than 20 VIO samples overlap the truth")
    keep = est.t >= est.t[0] + p.skip_start_s
    t, pos = est.t[keep], est.pos[keep]
    gt_pos, gt_vel = gt_pos[keep], gt_vel[keep]
    vel = est.vel[keep] if est.vel is not None else None
    if len(t) < 20:
        raise ValueError(f"{name}: fewer than 20 VIO samples overlap the truth")

    aligned, yaw, rot = align_pos_yaw(pos, gt_pos)
    e = aligned - gt_pos
    dt = np.diff(t)
    period = float(np.median(dt))
    duration = float(t[-1] - t[0])

    gap_limit = max(p.gap_min_s, p.gap_factor * period)
    gaps = dt[dt > gap_limit]

    # Jumps: single-step changes far outside the usual step-to-step spread. Steps across a
    # gap are judged against the same threshold (a reset after a loss is a jump too).
    de = np.diff(e, axis=0)
    step = np.linalg.norm(de, axis=1)
    mad = float(np.median(np.abs(step - np.median(step))))
    spread = 1.4826 * mad + 1e-9
    jump_idx = np.flatnonzero(step > max(p.jump_min_m, np.median(step) + p.jump_k * spread))
    jumps = step[jump_idx]
    remaining = []
    for k in jump_idx:
        later = np.searchsorted(t, t[k + 1] + p.decay_probe_s)
        if later >= len(t):
            continue
        # Share of the jump still in the error decay_probe_s later, along the jump.
        u = de[k] / max(step[k], 1e-12)
        remaining.append(float((e[later] - e[k]) @ u / step[k]))
    # Take the jump steps out so they do not count as drift.
    de_clean = de.copy()
    de_clean[jump_idx] = 0.0
    e_clean = np.vstack([e[:1], e[:1] + np.cumsum(de_clean, axis=0)])

    lags = sorted(
        set(_short_lags(period, 1.0))
        | {p.walk_match_s, p.cap_horizon_s, *p.report_lags_s}
    )
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))])
    sf: dict[float, tuple[list[float], int]] = {}
    sf_hover: dict[float, tuple[list[float], int]] = {}
    cap_abs: list[float] = []
    cap_abs_hover: list[float] = []
    for lag in lags:
        i, j = pair_indices(t, lag, tol=0.5 * period)
        if len(i) == 0:
            continue
        d = e_clean[j] - e_clean[i]
        slow = (path[j] - path[i]) <= p.hover_speed_max_mps * lag
        sf[lag] = ((d * d).mean(0).tolist(), int(len(i)))
        if slow.any():
            sf_hover[lag] = ((d[slow] * d[slow]).mean(0).tolist(), int(slow.sum()))
        if lag == p.cap_horizon_s:
            cap_abs = np.abs(d).ravel().tolist()
            cap_abs_hover = np.abs(d[slow]).ravel().tolist()

    vel_sq = None
    if vel is not None:
        dv = vel @ rot.T - gt_vel
        vel_sq = ((dv * dv).mean(0).tolist(), int(len(dv)))

    return FlightStats(
        name=name,
        duration_s=duration,
        period_s=period,
        yaw_align_deg=math.degrees(yaw),
        ate_rms_m=float(np.sqrt((e * e).sum(1).mean())),
        gaps_s=gaps.tolist(),
        jumps_m=jumps.tolist(),
        jump_remaining=remaining,
        sf=sf,
        sf_hover=sf_hover,
        cap_abs=cap_abs,
        cap_abs_hover=cap_abs_hover,
        vel_sq_err=vel_sq,
        time_offset_s=time_offset(t, aligned, gt_vel),
        path_m_per_s=float(path[-1]) / duration,
    )


def _short_lags(period: float, max_s: float) -> list[float]:
    n_max = max(2, int(round(max_s / period)))
    steps = sorted(set(np.unique(np.round(np.geomspace(1, n_max, 8))).astype(int).tolist()))
    return [round(k * period, 6) for k in steps]


def pool_sf(flights: list[FlightStats], hover: bool = False) -> dict[float, tuple[float, int]]:
    """Pair-weighted mean squared increment (averaged over x, y, z) and pairs, per lag."""
    acc: dict[float, list[float]] = {}
    for f in flights:
        for lag, (msq, n) in (f.sf_hover if hover else f.sf).items():
            a = acc.setdefault(lag, [0.0, 0])
            a[0] += float(np.mean(msq)) * n
            a[1] += n
    return {lag: (s / n, int(n)) for lag, (s, n) in sorted(acc.items()) if n > 0}


def fit_noise_walk(sf: dict[float, float], walk_match_s: float) -> tuple[float, float]:
    """Noise and walk of ``V(tau) = 2 noise^2 + walk^2 tau`` fitted to a measured ``V``.

    Least squares on the relative error over every lag up to ``walk_match_s``, so a 0.5 s
    lag counts as much as a 20 s one. Real VIO error is not exactly a walk plus white noise
    (it is smooth from one sample to the next and has a correlated part over seconds), so
    no single pair fits every horizon; the report shows the misfit per lag.
    """
    lags = np.array([lag for lag in sf if lag <= walk_match_s + 1e-9])
    if len(lags) < 2:
        raise ValueError("need at least two lags up to walk_match_s")
    v = np.array([sf[lag] for lag in lags])
    w = 1.0 / np.maximum(v, 1e-12)
    a_mat = np.stack([np.ones_like(lags), lags], axis=1) * w[:, None]
    (c0, c1), *_ = np.linalg.lstsq(a_mat, v * w, rcond=None)
    if c0 < 0:  # no white noise: refit the walk alone
        c0, c1 = 0.0, float((lags * w * v * w).sum() / (lags * w * lags * w).sum())
    if c1 < 0:  # no walk: noise alone
        c0, c1 = float((w * v * w).sum() / (w * w).sum()), 0.0
    return math.sqrt(c0 / 2.0), math.sqrt(c1)


def model_sf(noise: float, walk: float, lag: float) -> float:
    return 2.0 * noise**2 + walk**2 * lag


def _nearest(sf: dict[float, Any], lag: float) -> float | None:
    near = [k for k in sf if abs(k - lag) < 1e-6]
    return near[0] if near else None


def _flights_at(flights: list[FlightStats], hover: bool, lag: float) -> int:
    return sum(1 for f in flights if _nearest(f.sf_hover if hover else f.sf, lag) is not None)


def summarise(flights: list[FlightStats], p: FitParams,
              base: EstimateCorruptionConfig) -> dict[str, Any]:
    total_s = sum(f.duration_s for f in flights)
    fits: dict[str, dict[str, Any]] = {}
    for label, hover in (("all", False), ("hover", True)):
        pooled = pool_sf(flights, hover)
        if not pooled:
            continue
        sf = {lag: v for lag, (v, _) in pooled.items()}
        noise, walk = fit_noise_walk(sf, p.walk_match_s)
        cap_pool = np.concatenate(
            [np.asarray(f.cap_abs_hover if hover else f.cap_abs) for f in flights] or [[]]
        )
        match = _nearest(pooled, p.walk_match_s)
        cap_lag = _nearest(pooled, p.cap_horizon_s)
        fits[label] = {
            "pos_noise_std_m": noise,
            "bias_walk_std_m_per_sqrt_s": walk,
            "bias_max_m": float(np.percentile(cap_pool, 95)) if cap_pool.size else None,
            "pairs_at_walk_match": pooled[match][1] if match is not None else 0,
            "flights_at_walk_match": _flights_at(flights, hover, p.walk_match_s),
            "cap_pairs": pooled[cap_lag][1] if cap_lag is not None else 0,
            "flights_at_cap": _flights_at(flights, hover, p.cap_horizon_s),
            "drift_rms_m": {
                f"{lag:g}": {
                    "measured": math.sqrt(v),
                    "model": math.sqrt(model_sf(noise, walk, lag)),
                    "current_default": math.sqrt(
                        model_sf(base.pos_noise_std_m, base.bias_walk_std_m_per_sqrt_s, lag)
                    ),
                    "pairs": n,
                }
                for lag, (v, n) in pooled.items()
                if any(abs(lag - r) < 1e-6 for r in p.report_lags_s)
            },
        }
    # Near-hover windows are what a hold sees; fall back to all windows without enough.
    def enough(fit: dict[str, Any], pairs: str, flights_n: str) -> bool:
        return fit[pairs] >= p.hover_min_pairs and fit[flights_n] >= p.hover_min_flights

    use = "hover"
    if "hover" not in fits or not enough(fits["hover"], "pairs_at_walk_match",
                                         "flights_at_walk_match"):
        use = "all"
    chosen = fits[use]
    changes: dict[str, Any] = {
        "pos_noise_std_m": chosen["pos_noise_std_m"],
        "bias_walk_std_m_per_sqrt_s": chosen["bias_walk_std_m_per_sqrt_s"],
    }
    cap_from = use if enough(chosen, "cap_pairs", "flights_at_cap") else "all"
    cap = fits[cap_from]["bias_max_m"]
    if cap is not None:
        changes["bias_max_m"] = cap

    gaps = [g for f in flights for g in f.gaps_s]
    jumps = [j for f in flights for j in f.jumps_m]
    remaining = [r for f in flights for r in f.jump_remaining]
    # No event seen is not evidence of none (a textured lab is the easy case), so a fault
    # with no events keeps its default rather than being switched off.
    if jumps:
        changes["jump_rate_hz"] = len(jumps) / total_s
        changes["jump_min_m"] = float(min(jumps))
        changes["jump_max_m"] = float(max(jumps))
    if remaining:
        frac = float(np.median(remaining))
        changes["jump_decay_s"] = (
            None if frac >= 0.95 else p.decay_probe_s / -math.log(max(frac, 1e-3))
        )
    if gaps:
        changes["dropout_rate_hz"] = len(gaps) / total_s
        changes["dropout_min_s"] = float(min(gaps))
        changes["dropout_max_s"] = float(max(gaps))
    vel = [f.vel_sq_err for f in flights if f.vel_sq_err is not None]
    if vel:
        s = sum(float(np.mean(m)) * n for m, n in vel)
        changes["vel_noise_std_mps"] = math.sqrt(s / sum(n for _, n in vel))
    kept = sorted({f for f in asdict(base)} - set(changes))
    proposal = replace(base, **changes)
    return {
        "flights": len(flights),
        "total_s": total_s,
        "used_windows": use,
        "cap_windows": cap_from,
        "proposal": asdict(proposal),
        "current": asdict(base),
        "kept": kept,
        "fits": fits,
        "gaps_s": gaps,
        "jumps_m": jumps,
        "jump_remaining": remaining,
        "per_flight": [_flight_row(f, p) for f in flights],
    }


def _flight_row(f: FlightStats, p: FitParams) -> dict[str, Any]:
    sf = {lag: float(np.mean(m)) for lag, (m, _) in f.sf.items()}
    row: dict[str, Any] = {
        "name": f.name,
        "duration_s": f.duration_s,
        "rate_hz": 1.0 / f.period_s,
        "ate_rms_m": f.ate_rms_m,
        "path_m_per_s": f.path_m_per_s,
        "time_offset_s": f.time_offset_s,
        "gaps": len(f.gaps_s),
        "jumps": len(f.jumps_m),
    }
    try:
        noise, walk = fit_noise_walk(sf, p.walk_match_s)
        row.update(pos_noise_std_m=noise, bias_walk_std_m_per_sqrt_s=walk)
    except ValueError:
        pass
    for lag in p.report_lags_s:
        if lag in sf:
            row[f"drift_rms_{lag:g}s_m"] = math.sqrt(sf[lag])
    return row


# ---------------------------------------------------------------- inputs


def ensure_source(src: dict[str, Any]) -> Path:
    """A source's folder: ``dir`` (relative to the repo, e.g. VIO output already under
    ``runs/``), or a git source cloned at its pinned commit into ``runs/vio_data``."""
    if "dir" in src:
        root = REPO_ROOT / src["dir"]
        if not root.is_dir():
            raise FileNotFoundError(f"{src['name']}: no folder {root}")
        return root
    root = REPO_ROOT / "runs" / "vio_data" / src["name"]
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                        src["git"], str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "-q", src["commit"], "--",
                    *src.get("paths", ["."])], check=True)
    return root


def find_pairs(root: Path, pattern: str, est_name: str,
               gt_name: str) -> list[tuple[str, Path, Path]]:
    pairs = []
    for est in sorted(root.glob(f"{pattern}/{est_name}")):
        gt = est.parent / gt_name
        if gt.exists():
            pairs.append((est.parent.name, est, gt))
    return pairs


def format_report(result: dict[str, Any]) -> str:
    lines = [f"{result['flights']} flights, {result['total_s']:.0f} s of VIO"]
    lines.append(f"{'flight':32s} {'Hz':>5s} {'ATE':>6s} {'m/s':>5s} {'noise':>7s} "
                 f"{'walk':>7s} {'d20s':>6s} {'d60s':>6s} {'gaps':>4s} {'jumps':>5s} "
                 f"{'dt':>6s}")
    for r in result["per_flight"]:
        lines.append(
            f"{r['name']:32s} {r['rate_hz']:5.1f} {r['ate_rms_m']:6.3f} "
            f"{r['path_m_per_s']:5.2f} {r.get('pos_noise_std_m', float('nan')):7.4f} "
            f"{r.get('bias_walk_std_m_per_sqrt_s', float('nan')):7.4f} "
            f"{r.get('drift_rms_20s_m', float('nan')):6.3f} "
            f"{r.get('drift_rms_60s_m', float('nan')):6.3f} {r['gaps']:4d} {r['jumps']:5d} "
            f"{r['time_offset_s']:6.3f}"
        )
    for label, fit in result["fits"].items():
        lines.append(f"{label} windows: drift rms per axis since engage "
                     "(measured / fitted model / current default, pairs):")
        for lag, c in fit["drift_rms_m"].items():
            lines.append(f"  {lag:>4s} s: {c['measured']:.3f} / {c['model']:.3f} / "
                         f"{c['current_default']:.3f} m  ({c['pairs']})")
    lines.append(f"proposal from {result['used_windows']} windows, cap from "
                 f"{result['cap_windows']} windows (current -> fitted):")
    for k, v in result["proposal"].items():
        cur = result["current"][k]
        if v != cur:
            lines.append(f"  {k}: {cur} -> {_fmt(v)}")
    lines.append(f"kept (not measurable from this data): {', '.join(result['kept'])}")
    return "\n".join(lines)


def _fmt(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    params = cfg["params"]
    fit = FitParams.from_dict(params.get("fit", {}))
    writer = RunWriter(args.run_dir)
    try:
        flights: list[FlightStats] = []
        for src in params["sources"]:
            root = ensure_source(src)
            pairs = find_pairs(root, src["pattern"], src["estimate"], src["truth"])
            if not pairs:
                raise RuntimeError(f"{src['name']}: no trajectory pairs under {src['pattern']}")
            if args.smoke:
                pairs = pairs[:1]
            for name, est_path, gt_path in pairs:
                # Truth as TUM text, or EuRoC's own state_groundtruth_estimate0/data.csv.
                load_truth = load_euroc_csv if src.get("truth_format") == "euroc" else load_tum
                f = measure_flight(f"{src['name']}/{name}", load_tum(est_path),
                                   load_truth(gt_path), fit)
                flights.append(f)
                writer.metric(len(flights), flight=f.name, ate_rms_m=f.ate_rms_m,
                              gaps=len(f.gaps_s), jumps=len(f.jumps_m))
        result = summarise(flights, fit, EstimateCorruptionConfig())
        (Path(args.run_dir) / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(format_report(result), flush=True)
        prop = result["proposal"]
        writer.summary(
            "ok",
            headline={
                "flights": result["flights"],
                "vio_s": round(result["total_s"]),
                "pos_noise_std_m": round(prop["pos_noise_std_m"], 4),
                "bias_walk_std_m_per_sqrt_s": round(prop["bias_walk_std_m_per_sqrt_s"], 4),
                "bias_max_m": round(prop["bias_max_m"], 3),
                "jump_rate_hz": round(prop["jump_rate_hz"], 4),
                "dropout_rate_hz": round(prop["dropout_rate_hz"], 4),
            },
            vs_baseline="current envs/estimate defaults: noise 0.02 m, walk 0.01 m/sqrt(s), "
                        "cap 0.3 m, jumps and dropouts 0.02 Hz",
        )
        return 0
    except Exception as exc:  # recorded in summary.json, then re-raised for the log
        writer.summary("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
