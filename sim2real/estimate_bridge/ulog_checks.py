"""What PX4 made of the injected estimate, from the flight's ulog.

Reads, with pyulog: the EKF output (``vehicle_local_position``) against simulator truth
(``vehicle_local_position_groundtruth``), the EKF's aiding flags
(``estimator_status_flags``), the vision samples EKF2 processed
(``estimator_aid_src_ev_pos``: every sample, with fusion and rejection flags; the logger
rate-limits ``vehicle_visual_odometry`` so that copy is only a fallback) and the flight
modes seen (``vehicle_status``). Everything is reported in ENU and SI.

The questions it answers for the LOG entry: did EKF2 fuse vision and never GPS, how far
was its estimate from the truth, did it dead-reckon or reset during dropouts and jumps,
did PX4 fail over (a mode change it was not asked for), and what latency did it see.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from envs.frames import ned_to_enu

ESTIMATE = "vehicle_local_position"
TRUTH = "vehicle_local_position_groundtruth"
FLAGS = "estimator_status_flags"
VISION = "vehicle_visual_odometry"
EV_POS = "estimator_aid_src_ev_pos"
STATUS = "vehicle_status"
TOPICS = [ESTIMATE, TRUTH, FLAGS, VISION, EV_POS, STATUS]

# nav_state values of PX4 v1.17 (vehicle_status.msg)
NAV_STATE = {0: "Manual", 1: "Altitude", 2: "Position", 3: "Mission", 4: "Hold", 5: "Return",
             12: "Descend", 13: "Termination", 14: "Offboard", 15: "Stabilized",
             17: "Takeoff", 18: "Land", 19: "Follow", 20: "Precision land", 21: "Orbit",
             22: "VTOL takeoff", 23: "External 1"}


def load_topics(path: str | Path) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    try:
        from pyulog import ULog
    except ImportError as e:  # optional dependency: pip install -e .[ulog]
        raise ImportError("reading .ulg needs pyulog: pip install -e .[ulog]") from e
    log = ULog(str(path), message_name_filter_list=TOPICS)
    topics = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    return topics, int(log.start_timestamp)


def _seconds(msg: dict[str, np.ndarray], start_us: int) -> np.ndarray:
    return (np.asarray(msg["timestamp"], dtype=np.int64) - start_us) * 1e-6


def _in_window(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    return np.ones(t.size, dtype=bool) if window is None else (t >= window[0]) & (t <= window[1])


def _xyz(msg: dict[str, np.ndarray]) -> np.ndarray:
    return ned_to_enu(np.stack([msg["x"], msg["y"], msg["z"]], 1))


def estimate_error(topics, start_us: int, window: tuple[float, float] | None) -> dict[str, Any]:
    """EKF position error against truth on the truth's time base, inside ``window``."""
    if ESTIMATE not in topics or TRUTH not in topics:
        return {"error": "log lacks vehicle_local_position or its groundtruth"}
    est, truth = topics[ESTIMATE], topics[TRUTH]
    te, pe = _seconds(est, start_us), _xyz(est)
    tt, pt = _seconds(truth, start_us), _xyz(truth)
    te, first = np.unique(te, return_index=True)
    pe = pe[first]
    if window is not None:
        keep = (tt >= window[0]) & (tt <= window[1])
        tt, pt = tt[keep], pt[keep]
    if tt.size < 2:
        return {"error": "no truth samples in the window"}
    interp = np.stack([np.interp(tt, te, pe[:, i]) for i in range(3)], 1)
    err = interp - pt
    norm = np.linalg.norm(err, axis=1)
    in_win = _in_window(te, window)

    def resets(key: str) -> int | None:
        if key not in est or not np.any(in_win):
            return None
        return int(np.ptp(np.asarray(est[key])[first][in_win]))

    return {
        "samples": int(tt.size),
        "rms_m": float(np.sqrt(np.mean(norm**2))),
        "p95_m": float(np.percentile(norm, 95)),
        "max_m": float(norm.max()),
        "mean_bias_enu_m": [float(v) for v in err.mean(0)],
        "xy_resets": resets("xy_reset_counter"),
        "z_resets": resets("z_reset_counter"),
        "xy_invalid_fraction": (float(1.0 - np.mean(np.asarray(est["xy_valid"])[first][in_win]))
                                if "xy_valid" in est and np.any(in_win) else None),
    }


def aiding(topics, start_us: int) -> dict[str, Any]:
    """Fraction of the log each EKF2 aiding source was active, from estimator_status_flags."""
    if FLAGS not in topics:
        return {"error": "log lacks estimator_status_flags"}
    f = topics[FLAGS]
    t = _seconds(f, start_us)
    out: dict[str, Any] = {"samples": int(t.size)}
    for key, name in [("cs_ev_pos", "ev_pos"), ("cs_ev_vel", "ev_vel"), ("cs_ev_hgt", "ev_hgt"),
                      ("cs_ev_yaw", "ev_yaw"), ("cs_gnss_pos", "gnss_pos"),
                      ("cs_gnss_vel", "gnss_vel"), ("cs_gps", "gps"), ("cs_baro_hgt", "baro_hgt"),
                      ("cs_inertial_dead_reckoning", "inertial_dead_reckoning"),
                      ("cs_mag_hdg", "mag_hdg"), ("cs_mag_3d", "mag_3d")]:
        if key in f:
            out[f"{name}_fraction"] = float(np.mean(f[key]))
    out["gnss_ever_used"] = bool(any(
        np.any(f[k]) for k in ("cs_gnss_pos", "cs_gnss_vel", "cs_gps") if k in f))
    return out


def vision_samples(topics, start_us: int, gap_s: float = 0.1,
                   window: tuple[float, float] | None = None) -> dict[str, Any]:
    """The vision samples EKF2 processed: rate, gaps, fusion and rejection, latency."""
    if EV_POS in topics:
        v, source = topics[EV_POS], EV_POS
    elif VISION in topics:
        v, source = topics[VISION], VISION
    else:
        return {"error": "log has neither estimator_aid_src_ev_pos nor vehicle_visual_odometry"}
    t = _seconds(v, start_us)
    t, first = np.unique(t, return_index=True)
    dt = np.diff(t)
    gaps = dt[dt > gap_s]
    lat = (np.asarray(v["timestamp"], dtype=np.int64)
           - np.asarray(v["timestamp_sample"], dtype=np.int64))[first] * 1e-6
    out: dict[str, Any] = {
        "source": source,
        "samples": int(t.size),
        "rate_hz": float((t.size - 1) / (t[-1] - t[0])) if t.size > 1 else None,
        "gaps_over_0p1s": int(gaps.size),
        "longest_gap_s": float(gaps.max()) if gaps.size else 0.0,
        "gap_total_s": float(gaps.sum()) if gaps.size else 0.0,
        "stamp_minus_sample_mean_s": float(lat.mean()) if lat.size else None,
        "stamp_minus_sample_max_s": float(lat.max()) if lat.size else None,
    }
    if source == EV_POS:
        in_win = _in_window(t, window)
        fused = np.asarray(v["fused"], dtype=float)[first][in_win]
        rejected = np.asarray(v["innovation_rejected"], dtype=float)[first][in_win]
        innov = np.stack([v["innovation[0]"], v["innovation[1]"]], 1)[first][in_win]
        ratio = np.stack([v["test_ratio[0]"], v["test_ratio[1]"]], 1)[first][in_win]
        out.update({
            "fused_fraction": float(fused.mean()) if fused.size else None,
            "rejected_fraction": float(rejected.mean()) if rejected.size else None,
            "innovation_rms_m": (float(np.sqrt(np.nanmean((innov**2).sum(1))))
                                 if innov.size else None),
            "innovation_max_m": float(np.nanmax(np.abs(innov))) if innov.size else None,
            "test_ratio_max": float(np.nanmax(ratio)) if ratio.size else None,
        })
    return out


def modes(topics, start_us: int) -> dict[str, Any]:
    """Flight modes in order of appearance, with failsafe flags if any were raised."""
    if STATUS not in topics:
        return {"error": "log lacks vehicle_status"}
    s = topics[STATUS]
    t = _seconds(s, start_us)
    nav = np.asarray(s["nav_state"], dtype=int)
    change = np.concatenate([[True], nav[1:] != nav[:-1]])
    seq = [{"t": round(float(ti), 2), "mode": NAV_STATE.get(int(n), f"nav_state {n}")}
           for ti, n in zip(t[change], nav[change], strict=True)]
    out: dict[str, Any] = {"sequence": seq}
    if "failsafe" in s:
        out["failsafe_fraction"] = float(np.mean(s["failsafe"]))
    return out


def hover_window(topics, start_us: int, altitude: float, margin: float = 0.3,
                 settle_s: float = 2.0) -> tuple[float, float] | None:
    """Time span where the TRUE height is within ``margin`` of the hover altitude, trimmed
    by ``settle_s`` at the start, from the groundtruth topic."""
    if TRUTH not in topics:
        return None
    truth = topics[TRUTH]
    t, p = _seconds(truth, start_us), _xyz(truth)
    z0 = p[0, 2]
    at_alt = np.abs((p[:, 2] - z0) - altitude) <= margin
    if not np.any(at_alt):
        return None
    idx = np.flatnonzero(at_alt)
    t0, t1 = float(t[idx[0]]) + settle_s, float(t[idx[-1]])
    return (t0, t1) if t1 > t0 else None


def analyse(path: str | Path, altitude: float, window: tuple[float, float] | None = None
            ) -> dict[str, Any]:
    topics, start_us = load_topics(path)
    if window is None:
        window = hover_window(topics, start_us, altitude)
    out: dict[str, Any] = {"ulog": str(path), "window_s": list(window) if window else None}
    out["estimate_error"] = estimate_error(topics, start_us, window)
    out["aiding"] = aiding(topics, start_us)
    out["vision"] = vision_samples(topics, start_us, window=window)
    out["modes"] = modes(topics, start_us)
    out["gps_free"] = bool(not out["aiding"].get("gnss_ever_used", True))
    out["vision_fused"] = bool(out["aiding"].get("ev_pos_fraction", 0.0) > 0.5)
    return out


def finite(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else x
