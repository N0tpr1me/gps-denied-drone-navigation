"""The Gazebo-like check: fly policies in the mock the way the Gazebo benchmark flies them,
before any of them goes to the Mac for a Gazebo re-fly (docs/decisions.md 2026-10-23).

    python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu

Why it exists: v7 and v8 beat v6 in Gate 1 and then flew worse than v6 in Gazebo, because
Gate 1's mock had none of what makes v6 and v8 fall into the 1.6 Hz roll-and-yaw wobble that
lifts them 0.2 to 0.4 m in Gazebo. This check flies the ``gz_x500_sitl`` airframe (PX4's yaw
torque and gyro filters, the module's 83 Hz tick, the bridge's 0.24 m lever arm, EKF2-like
fusion; ``envs/structure_hold/airframes.py``) through the benchmark's conditions: 60 s, 1 m
from the wall, the hold point 2.4 m up, the ``vio_default`` estimate (and a clean one), calm
air and the 4 and 8 m/s scenarios of ``sim2real/gazebo/scenarios``.

Per episode, over the last ``window_s`` seconds (the first seconds are the engage transient):

* ``lift_m``: the mean height of the estimate above the hold point (what the module flies to;
  Gazebo's "offset up");
* ``wobble``: the roll action's 1.2 to 2.2 Hz band carries more than ``wobble_threshold``
  (rms, action units); Gazebo's wobbling flights sit at 0.2 to 0.3, quiet ones under 0.03;
* ``p95_m`` over the whole episode (``eval.metrics``, true centre of mass) and crashes.

A controller ``beats`` the reference (the released policy) when, pooled over the
rough-estimate cells, its p95 is lower (paired bootstrap 95 % interval of the ratio below 1),
its lift and wobble rate are no higher, its crash rate is no more than ``max_crash_increase``
above the reference's, and on the clean cell its p95 is within ``clean_p95_slack_m`` of the
reference's. Gazebo's numbers for the controllers flown there (``gazebo`` in the config) are
printed beside the mock's so the check's own fidelity stays visible.

It must also ring no more than the reference: the median of the roll action's 1.2 to 2.2 Hz
rms over the rough cells at most ``ring_slack`` above the reference's. The thresholded wobble
rate misses a ring that is still well under the threshold: v11 sat 17 % above v10 on it in
every rough cell while both scored 0 % wobble, and v11 then rang at 1.7 Hz in Gazebo (PR #49).
Above that band, ``fast_ring_rms`` is the larger of the roll and pitch rate commands' 2.2 to
6 Hz rms (rad/s); its median over the rough cells must be at most ``fast_ring_slack`` above
the reference's. v18 passed the 1.2 to 2.2 Hz rule and rang at 3.3 Hz in all 9 Gazebo
flights, commanded (PR #58): 2.2 to 5 Hz rate setpoint rms 0.118 rad/s against v10's 0.023
in Gazebo, 0.124 against 0.016 here.

Cells with a ``reengage`` block fly the benchmark's post-hold sequence
(``sim2real/px4/sitl_test.py``) instead: hold, a switch to stock Position mode, a re-engage
that captures its hold point from the estimate then (``fly_reengage``). They score crashes
after the switch, where the re-engage put its hold point, and ``reengage_fail``: a crash
after a re-engage whose hold point was at least ``sane_m`` off the wall, i.e. one the policy
is to blame for. A controller must have none, and crash no more than the reference during
the hold, in every such cell not marked ``scored: false``. Crashes in the stock phase and
after a re-engage at the wall are printed, not scored: they come with the sequence (stock
PX4 and every controller that holds the estimate walk into the wall there alike).

Cells with ``events`` replay a Gazebo flight's estimate faults: scripted jumps and dropouts
(``apply_event``) at the same moment in every env, with the random ones off. They stay out of
the pooled and clean sets and are scored on their own: mean worst toward the wall (the
largest true displacement along the wall normal, ``eval.metrics``) at most
``toward_wall_slack_m`` above the reference's, crashes no more than ``max_crash_increase``
above. Gazebo seed 1's 1.8 m jump along the wall drew v12 0.27 to 0.31 m toward the wall and
v10 0.03 to 0.07 m (PR #52); the pooled p95 had passed v12 because it chases jumps less.
Each replayed seed's p95 must also not be worse than the reference's (``p95_not_worse``: the
paired ratio's interval reaches 1 or below): v18's Gazebo win over v10 was seed 1 alone, with
seeds 2 and 3 1 to 25 % looser (PR #58), and the pooled cells draw random fault timelines.
A cell with ``bridge`` (``{seed, profile, engage_s}``) replays the flight's whole fault stream
instead (``bridge_stream``): the bridge's corruptor run offline with that seed gives the
position offset it added at every step of the hold (its slow bias walk as well as the jumps),
its dropouts and its delay; the env's own bias walk, jumps and dropouts are off and its delay is
the bridge's. The jumps alone left out seed 3's bias, 0.05 to 0.10 m toward the wall all hold,
which stands every controller back before the jumps: v10's replayed p95 there was 0.36 m
against Gazebo's 0.26, and 0.26 with the stream.

Outputs: ``metrics.jsonl`` (one row per episode), ``report.json``, ``report.md``,
``summary.json``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.structure_hold.baseline import CascadedPd
from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import TERM_REASONS, StructureHoldEnv
from eval.gate1 import check_not_held_out, make_factory, merge
from eval.metrics import Surface, Trace, compute_metrics

TIMEOUT = TERM_REASONS.index("timeout")

DEFAULT_RULES = {
    "max_crash_increase": 0.01,
    "clean_p95_slack_m": 0.02,
    "ring_slack": 0.05,  # median 1.2-2.2 Hz roll rms at most 5 % above the reference's
    "fast_ring_slack": 0.25,  # median 2.2-6 Hz roll/pitch rate rms at most 25 % above
    "toward_wall_slack_m": 0.05,  # replay cells: mean worst toward the wall
}


FAST_RING_HZ = (2.2, 6.0)  # band of ``fast_ring_rms``, above the wobble's 1.2 to 2.2 Hz


def band_rms(x: np.ndarray, dt: float, lo: float = 1.2, hi: float = 2.2) -> np.ndarray:
    """Rms of the ``lo``..``hi`` Hz part of ``x`` (time, envs), Hann-windowed."""
    x = x - x.mean(0)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), dt)
    frac = spec[(f >= lo) & (f <= hi)].sum(0) / np.maximum(spec.sum(0), 1e-12)
    return np.sqrt(frac) * x.std(0)


def apply_event(env: StructureHoldEnv, event: dict[str, Any]) -> None:
    """Start one scripted estimate fault in every env, as ``envs.estimate`` draws them: a
    ``jump`` (ENU vector, m) that then decays with ``jump_decay_s``, or a dead-reckoned
    dropout of ``dropout_s`` seconds with acceleration bias ``accel_bias`` (m/s^2)."""
    est = env.estimator
    if "jump" in event:
        est.jump_offset[:] += torch.as_tensor(event["jump"], dtype=est.jump_offset.dtype,
                                              device=est.jump_offset.device)
    if "dropout_s" in event:
        est.dropout_left[:] = max(1, int(round(float(event["dropout_s"]) / est.dt)))
        est.dropout_bias[:] = torch.as_tensor(event.get("accel_bias", [0.0, 0.0, 0.0]),
                                              dtype=est.dropout_bias.dtype,
                                              device=est.dropout_bias.device)


def bridge_stream(spec: dict[str, Any], episode_s: float, dt: float) -> dict[str, Any]:
    """One Gazebo flight's estimate faults from the bridge's corruptor, run offline with the
    flight's seed and profile (``sim2real.estimate_bridge``), engaged at ``engage_s`` of sim
    time as the benchmark engages: ``offsets`` (steps, 3), the position offset (bias walk plus
    jump offsets, ENU) at each env step of the hold; ``events``, its dropouts as
    ``apply_event`` takes them; ``delay_s``. The corruption does not depend on where the drone
    is, so every controller flown on that seed got this stream."""
    from sim2real.estimate_bridge.bridge import EstimateBridge
    from sim2real.estimate_bridge.config import load_profile

    prof = load_profile(spec.get("profile", "vio_at_engage"))
    engage_s = float(spec.get("engage_s", 25.0))
    b = EstimateBridge(prof.estimate, prof.dt, int(spec["seed"]), prof.divisor,
                       prof.fault_anchor, prof.fault_anchor_s)
    pos = np.zeros(3)
    ts: list[float] = []
    off: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    for k in range(int(round((engage_s + episode_s + 1.0) / prof.dt))):
        t = k * prof.dt
        s = b.update(t, pos, engaged=t >= engage_s)
        if s is None or t < engage_s:
            continue
        c = b.corruptor
        ts.append(t - engage_s)
        off.append(np.asarray(c.bias[0]) + np.asarray(c.jump_offset[0]))
        if s.dropout_started:
            events.append({"t_s": t - engage_s, "dropout_s": (int(c.dropout_left[0]) + 1) * c.dt,
                           "accel_bias": np.asarray(c.dropout_bias[0]).tolist()})
    # Each env step takes the bridge's last sample at or before it (a jump stays a step).
    k = np.searchsorted(ts, np.arange(int(round(episode_s / dt))) * dt + 1e-9, side="right") - 1
    return {"offsets": np.asarray(off)[np.clip(k, 0, len(off) - 1)], "events": events,
            "delay_s": float(b.corruptor.delay_steps[0]) * b.corruptor.dt}


def bridge_env(stream: dict[str, Any]) -> dict[str, Any]:
    """Env overrides for a ``bridge_stream`` replay: no faults of the env's own (the offsets
    carry the bias walk), the bridge's delay, room for a jump in the bias."""
    d = stream["delay_s"]
    return {"estimate": {"bias_walk_std_m_per_sqrt_s": 0.0, "bias_max_m": 10.0,
                         "jump_rate_hz": 0.0, "dropout_rate_hz": 0.0,
                         "delay_min_s": d, "delay_max_s": d}}


def fly(env: StructureHoldEnv, controller: Any, window_s: float,
        wobble_threshold: float, events: list[dict[str, Any]] | None = None,
        offsets: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Fly each env's first episode; one row per env. ``events`` start scripted estimate
    faults (``apply_event``) at their ``t_s``, the same moment in every env; ``offsets``
    (steps, 3) set the estimate's position bias at every step (``bridge_stream``)."""
    cfg, n, steps, dt = env.cfg, env.num_envs, env.max_episode_length, env.dt
    at: dict[int, list[dict[str, Any]]] = {}
    for e in events or []:
        at.setdefault(int(round(float(e["t_s"]) / dt)), []).append(e)
    hold = env.hold_point.cpu().numpy().copy()
    pos = np.empty((steps + 1, n, 3), dtype=np.float32)
    est_z = np.empty((steps, n), dtype=np.float32)
    act = np.empty((steps, n, 4), dtype=np.float32)
    pos[0] = env.state.pos.cpu().numpy()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    end = np.full(n, steps, dtype=np.int64)
    reason = np.full(n, TIMEOUT, dtype=np.int64)
    obs, _ = env.get_observations()
    bias = env.estimator.bias
    for k in range(steps):
        if offsets is not None:
            bias[:] = torch.as_tensor(offsets[min(k, len(offsets) - 1)], dtype=bias.dtype,
                                      device=bias.device)
        for e in at.get(k, []):
            apply_event(env, e)
        a = controller(obs)
        act[k] = a.detach().cpu().numpy()
        obs, _, done, _ = env.step(a)
        est_z[k] = (env.estimate_pos[:, 2] - env.hold_point[:, 2]).cpu().numpy()
        still = alive & ~done
        s = still.cpu().numpy()
        pos[k + 1] = np.where(s[:, None], env.state.pos.cpu().numpy(), pos[k])
        newly = (alive & done).cpu().numpy()
        if newly.any():
            end[newly] = k + 1
            reason[newly] = env.term_reason.cpu().numpy()[newly]
        alive = still
        if not bool(alive.any()):
            break

    w0 = max(0, steps - int(round(window_s / dt)))
    full = end == steps
    wob = np.full(n, np.nan)
    fast = np.full(n, np.nan)
    lift = np.full(n, np.nan)
    if full.any():
        wob[full] = band_rms(act[w0:, full, 0].astype(np.float64), dt)
        rates = act[w0:, full, :2].astype(np.float64) * cfg.max_rate_rad_s
        fast[full] = np.maximum(band_rms(rates[:, :, 0], dt, *FAST_RING_HZ),
                                band_rms(rates[:, :, 1], dt, *FAST_RING_HZ))
        lift[full] = est_z[w0:, full].mean(0)
    surface = Surface(point=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]))
    rows = []
    for i in range(n):
        length = int(end[i])
        row: dict[str, Any] = {"env": i, "length_s": float(length * dt),
                               "crash": bool(reason[i] != TIMEOUT),
                               "reason": TERM_REASONS[int(reason[i])]}
        if full[i]:
            row.update({"lift_m": float(lift[i]), "wobble_rms": float(wob[i]),
                        "wobble": bool(wob[i] > wobble_threshold),
                        "fast_ring_rms": float(fast[i])})
        if length >= 2:
            t = np.arange(length + 1) * dt
            trace = Trace(t=t, pos=pos[: length + 1, i].astype(np.float64), pos_sp=hold[i],
                          cmd_t=t[:-1], rates_cmd=act[:length, i, :3] * cfg.max_rate_rad_s,
                          thrust_cmd=0.5 * (act[:length, i, 3] + 1.0))
            m = compute_metrics(trace, surface, contact_radius_m=cfg.contact_radius_m)
            row["p95_m"] = m["p95_m"]
            row["toward_wall_m"] = m["worst_toward_surface_m"]
        rows.append(row)
    return rows


def fly_reengage(env: StructureHoldEnv, controller: Any, seq: dict[str, Any],
                 seed: int) -> list[dict[str, Any]]:
    """Fly the benchmark's post-hold sequence, one row per env.

    ``controller`` holds until ``disengage_s``. Then stock Position mode (the PD) takes over,
    holding the estimate where the switch found it and moving that setpoint with each
    horizontal EKF2 reset, as PX4's flight tasks do. After ``stock_s`` the module re-engages
    as ``StructureHold::engage`` does: hold point = the estimate now, last action zeroed, the
    policy's history refilled. ``guard_s`` > 0 holds the re-engage off until that long after
    the last horizontal reset (a module guard under discussion). ``jump_m`` > 0 adds a vision
    jump toward the wall, decaying as ``vio_default``'s do, ``jump_lead_s`` (a range, drawn
    per env) before the switch: Gazebo seed 3's, which put the v11 re-engages on the wall.
    """
    n, dt, dev = env.num_envs, env.dt, env.device
    steps = env.max_episode_length
    k_dis = int(round(float(seq["disengage_s"]) / dt))
    k_re = k_dis + int(round(float(seq["stock_s"]) / dt))
    guard_k = int(round(float(seq.get("guard_s", 0.0)) / dt))
    sane_m = float(seq.get("sane_m", 0.6))
    jump_m = float(seq.get("jump_m", 0.0))
    rng = np.random.default_rng(seed * 7919 + 17)
    lead = rng.uniform(*[float(x) for x in seq.get("jump_lead_s", [0.0, 0.0])], size=n)
    k_jump = k_dis - np.round(lead / dt).astype(np.int64)
    stock = CascadedPd(env)
    fusion = env.fusion
    count = fusion.xy_reset_count.clone() if fusion is not None else None
    mode = torch.zeros(n, dtype=torch.long, device=dev)  # 0 hold, 1 stock, 2 re-engaged
    last_reset = np.full(n, -(10 ** 9), dtype=np.int64)
    reset_lead = np.full(n, np.nan)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    end = np.full(n, steps, dtype=np.int64)
    reason = np.full(n, TIMEOUT, dtype=np.int64)
    k_eng = np.full(n, -1, dtype=np.int64)
    capture = np.full(n, np.nan)  # the re-engage's hold point: distance off the wall
    true_at = np.full(n, np.nan)  # and the drone's, true, at that moment
    min_clear = np.full(n, np.inf)
    obs, _ = env.get_observations()
    for k in range(steps):
        if k == k_dis:
            mode[:] = 1
            env.hold_point[:] = env.estimate_pos
            stock.err_integral.zero_()
            env._observe()
            obs = env.obs_buf
            back = (last_reset > k - int(round(5.0 / dt)))
            reset_lead[back] = (k - last_reset[back]) * dt
        if k >= k_re:
            ready = (k - last_reset) >= guard_k
            eng = (mode == 1) & alive & torch.as_tensor(ready, device=dev)
            if bool(eng.any()):
                env.hold_point[eng] = env.estimate_pos[eng]
                env.last_action[eng] = 0.0
                mode[eng] = 2
                e = eng.cpu().numpy()
                k_eng[e] = k
                capture[e] = env.clearance(env.hold_point).cpu().numpy()[e]
                true_at[e] = env.clearance().cpu().numpy()[e]
                env._observe()
                obs = env.obs_buf
                if hasattr(controller, "reset_history"):
                    controller.reset_history(obs, eng)
        if jump_m > 0.0:
            hit = torch.as_tensor(k_jump == k, device=dev)
            if bool(hit.any()):
                env.estimator.jump_offset[hit, 0] += jump_m  # +x: toward the wall
        a = torch.where((mode == 1)[:, None], stock(obs), controller(obs))
        obs, _, done, _ = env.step(a)
        if count is not None:
            new = fusion.xy_reset_count != count
            count = fusion.xy_reset_count.clone()
            last_reset[new.cpu().numpy()] = k
            shift = new & (mode == 1)
            if bool(shift.any()):
                env.hold_point[shift, :2] += fusion.xy_reset_step[shift, :2]
        still = alive & ~done
        c = env.clearance().cpu().numpy()
        after = (still & (mode == 2)).cpu().numpy()
        min_clear[after] = np.minimum(min_clear[after], c[after])
        newly = (alive & done).cpu().numpy()
        if newly.any():
            end[newly] = k + 1
            reason[newly] = env.term_reason.cpu().numpy()[newly]
        alive = still
        if not bool(alive.any()):
            break

    rows = []
    for i in range(n):
        crash = bool(reason[i] != TIMEOUT)
        engaged = bool(k_eng[i] >= 0)
        sane = bool(engaged and capture[i] >= sane_m)
        crash_after = crash and engaged and end[i] > k_eng[i]
        row: dict[str, Any] = {
            "env": i, "length_s": float(end[i] * dt), "crash": crash,
            "reason": TERM_REASONS[int(reason[i])],
            "crash_before_disengage": crash and end[i] <= k_dis,
            "reengaged": engaged, "reengage_s": float(k_eng[i] * dt) if engaged else None,
            "capture_m": float(capture[i]) if engaged else None,
            "true_at_capture_m": float(true_at[i]) if engaged else None,
            "sane_capture": sane,
            "reengage_fail": bool(crash_after and sane),
            "crash_after_wall_capture": bool(crash_after and not sane),
            "min_clearance_m": float(min_clear[i]) if np.isfinite(min_clear[i]) else None,
            "reset_lead_s": float(reset_lead[i]) if np.isfinite(reset_lead[i]) else None,
        }
        if jump_m > 0.0:
            row["jump_lead_s"] = float(lead[i])
        rows.append(row)
    return rows


def describe_reengage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Crashes after the switch, where the re-engages put their hold point, and the crash
    rate split by how long before the switch the last horizontal reset came (Gazebo seed 3:
    within 0.4 s of it for v11, which struck the wall, 1.3 s for v10, which did not)."""
    eng = [r for r in rows if r["reengaged"]]
    cap = np.array([r["capture_m"] for r in eng], dtype=np.float64)
    clear = np.array([r["min_clearance_m"] for r in eng if r["min_clearance_m"] is not None])
    out: dict[str, Any] = {
        "episodes": len(rows),
        "crash_rate": float(np.mean(_col(rows, "crash"))),
        "hold_crash_rate": float(np.mean(_col(rows, "crash_before_disengage"))),
        "reengage_fail": int(sum(r["reengage_fail"] for r in rows)),
        "wall_captures": int(sum(r["reengaged"] and not r["sane_capture"] for r in rows)),
        "crash_after_wall_capture": int(sum(r["crash_after_wall_capture"] for r in rows)),
        "never_reengaged": int(sum(not r["reengaged"] and not r["crash"] for r in rows)),
        "capture_m_p5": float(np.percentile(cap, 5)) if cap.size else float("nan"),
        "min_clearance_m_p5": float(np.percentile(clear, 5)) if clear.size else float("nan"),
    }
    for name, lo, hi in (("reset_under_1s", 0.0, 1.0), ("reset_1s_plus", 1.0, 99.0)):
        rs = [r for r in rows if r["reset_lead_s"] is not None and lo <= r["reset_lead_s"] < hi]
        out[f"crash_rate_{name}"] = float(np.mean([r["crash"] for r in rs])) if rs else None
        out[f"episodes_{name}"] = len(rs)
    return out


def _col(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [r for r in rows if "lift_m" in r]
    toward = _col(rows, "toward_wall_m")
    return {
        "episodes": len(rows),
        "crash_rate": float(np.mean(_col(rows, "crash"))),
        "p95_m_mean": float(np.nanmean(_col(rows, "p95_m"))),
        "toward_wall_m_mean": (float(np.nanmean(toward)) if np.isfinite(toward).any()
                               else float("nan")),
        "lift_m_mean": float(np.mean(_col(full, "lift_m"))) if full else float("nan"),
        "wobble_rate": float(np.mean(_col(full, "wobble"))) if full else float("nan"),
        "wobble_rms_median": float(np.median(_col(full, "wobble_rms"))) if full else float("nan"),
        "fast_ring_median": (float(np.median(_col(full, "fast_ring_rms"))) if full
                             else float("nan")),
    }


def paired_p95_ratio(a: list[dict[str, Any]], b: list[dict[str, Any]], rng: np.random.Generator,
                     n_boot: int) -> dict[str, Any]:
    """Mean p95 of ``a`` over ``b`` on the same episodes, with a bootstrap 95 % interval."""
    pa, pb = _col(a, "p95_m"), _col(b, "p95_m")
    ok = np.isfinite(pa) & np.isfinite(pb)
    pa, pb = pa[ok], pb[ok]
    idx = rng.integers(0, len(pa), size=(n_boot, len(pa)))
    boot = pa[idx].mean(1) / pb[idx].mean(1)
    return {"ratio": float(pa.mean() / pb.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def run(cfg: dict[str, Any], device: str, writer: Any) -> dict[str, Any]:
    p = cfg["params"]
    seeds = [int(s) for s in p["seeds"]]
    check_not_held_out(*seeds)
    num_envs = int(p.get("num_envs", 128))
    window_s = float(p.get("window_s", 40.0))
    thr = float(p.get("wobble_threshold", 0.1))
    rules = {**DEFAULT_RULES, **(p.get("rules") or {})}
    specs = [dict(c) for c in p["controllers"]]
    names = [s["name"] for s in specs]
    ref = str(p["reference"])
    if ref not in names:
        raise ValueError(f"reference {ref!r} must be one of {names}")
    factories = {s["name"]: make_factory(s, device) for s in specs}
    gazebo = p.get("gazebo") or {}
    rng = np.random.default_rng(int(p.get("bootstrap_seed", 0)))
    n_boot = int(p.get("bootstrap", 2000))
    print(f"gazebo_check: {len(p['cells'])} cells x {len(seeds)} seeds x {num_envs} episodes, "
          f"controllers {names}, reference {ref}, device {device}", flush=True)

    report: dict[str, Any] = {"seeds": seeds, "num_envs": num_envs, "window_s": window_s,
                              "wobble_threshold": thr, "rules": rules, "reference": ref,
                              "controllers": specs, "gazebo": gazebo,
                              "gazebo_baseline": p.get("gazebo_baseline"), "cells": [],
                              "reengage": []}
    t_start = time.perf_counter()
    scale = float(p.get("reengage_scale", 1.0))  # the smoke run shrinks the sequence
    flat: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cell in p["cells"]:
        if cell.get("reengage"):
            report["reengage"].append(run_reengage_cell(p, cell, names, factories, seeds,
                                                        num_envs, scale, device, writer))
            continue
        env_over = merge(p.get("base_env") or {}, cell.get("env"))
        events, offsets = list(cell.get("events") or []), None
        if cell.get("bridge"):
            c0 = StructureHoldConfig.from_dict(env_over)
            stream = bridge_stream(cell["bridge"], c0.episode_length_s, c0.dt)
            env_over = merge(env_over, bridge_env(stream))
            events, offsets = events + stream["events"], stream["offsets"]
        env_cfg = StructureHoldConfig.from_dict(env_over)
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for seed in seeds:
            for name in names:
                env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device)
                t0 = time.perf_counter()
                r = fly(env, factories[name](env), window_s, thr, events, offsets)
                for row in r:
                    writer.metric(seed, cell=cell["name"], controller=name, **row)
                rows[name].extend(r)
                d = describe(r)
                print(f"gazebo_check: {cell['name']} seed {seed} {name}: crash "
                      f"{d['crash_rate']:.1%}, p95 {d['p95_m_mean']:.3f} m, lift "
                      f"{d['lift_m_mean']:+.3f} m, wobble {d['wobble_rate']:.0%}, toward the "
                      f"wall {d['toward_wall_m_mean']:.3f} m "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
        flat[cell["name"]] = rows
        report["cells"].append({
            "name": cell["name"], "label": cell.get("label", cell["name"]),
            "rough": bool(cell.get("rough", True)), "env": cell.get("env") or {},
            "replay": bool(cell.get("events") or cell.get("bridge")),
            "scored": bool(cell.get("scored", True)),
            "controllers": {name: describe(rows[name]) for name in names},
            "vs_reference": {name: paired_p95_ratio(rows[name], rows[ref], rng, n_boot)
                             for name in names if name != ref},
        })

    rough = [c["name"] for c in report["cells"] if c["rough"] and not c["replay"]]
    clean = [c["name"] for c in report["cells"] if not c["rough"] and not c["replay"]]
    replay = [c for c in report["cells"] if c["replay"] and c["scored"]]
    pooled = {name: describe([r for c in rough for r in flat[c][name]]) for name in names}
    report["pooled_rough"] = pooled
    verdicts = {}
    for name in names:
        if name == ref:
            continue
        a = [r for c in rough for r in flat[c][name]]
        b = [r for c in rough for r in flat[c][ref]]
        cmp = paired_p95_ratio(a, b, rng, n_boot)
        pa, pr = pooled[name], pooled[ref]
        checks = {
            "p95_lower": cmp["ci"][1] < 1.0,
            "lift_not_higher": pa["lift_m_mean"] <= pr["lift_m_mean"],
            "wobble_not_higher": pa["wobble_rate"] <= pr["wobble_rate"],
            "ring_not_higher": (pa["wobble_rms_median"]
                                <= pr["wobble_rms_median"] * (1.0 + rules["ring_slack"])),
            "fast_ring_not_higher": (pa["fast_ring_median"]
                                     <= pr["fast_ring_median"] * (1.0 + rules["fast_ring_slack"])),
            "crash_ok": pa["crash_rate"] <= pr["crash_rate"] + rules["max_crash_increase"],
        }
        for c in [c for c in report["reengage"] if c["scored"]]:
            d = c["controllers"]
            checks[f"{c['name']}_no_reengage_fail"] = d[name]["reengage_fail"] == 0
            checks[f"{c['name']}_hold_crash_ok"] = (d[name]["hold_crash_rate"]
                                                    <= d[ref]["hold_crash_rate"]
                                                    + rules["max_crash_increase"])
        for c in clean:
            d = {x["name"]: x for x in report["cells"]}[c]["controllers"]
            checks[f"{c}_p95_ok"] = (d[name]["p95_m_mean"]
                                     <= d[ref]["p95_m_mean"] + rules["clean_p95_slack_m"])
        for c in replay:
            d = c["controllers"]
            checks[f"{c['name']}_toward_wall_ok"] = (
                d[name]["toward_wall_m_mean"]
                <= d[ref]["toward_wall_m_mean"] + rules["toward_wall_slack_m"])
            checks[f"{c['name']}_crash_ok"] = (d[name]["crash_rate"]
                                               <= d[ref]["crash_rate"]
                                               + rules["max_crash_increase"])
            checks[f"{c['name']}_p95_not_worse"] = c["vs_reference"][name]["ci"][0] <= 1.0
        verdicts[name] = {"p95_ratio": cmp, "checks": checks, "beats": all(checks.values())}
    report["verdicts"] = verdicts
    report["elapsed_s"] = time.perf_counter() - t_start
    run_dir = Path(writer.run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (run_dir / "report.md").write_text(markdown(report))

    pr = pooled[ref]
    headline: dict[str, Any] = {"reference": ref, "episodes_per_cell": num_envs * len(seeds),
                                f"{ref}_p95_m_rough": pr["p95_m_mean"],
                                f"{ref}_lift_m_rough": pr["lift_m_mean"],
                                f"{ref}_wobble_rough": pr["wobble_rate"]}
    parts = []
    for name, v in verdicts.items():
        pa = pooled[name]
        headline.update({f"{name}_p95_m_rough": pa["p95_m_mean"],
                         f"{name}_lift_m_rough": pa["lift_m_mean"],
                         f"{name}_wobble_rough": pa["wobble_rate"],
                         f"{name}_beats_{ref}": v["beats"]})
        verdict = "beats" if v["beats"] else "does not beat"
        parts.append(f"{name} p95 {pa['p95_m_mean']:.2f} m, lift {pa['lift_m_mean']:+.2f} m, "
                     f"wobble {pa['wobble_rate']:.0%}: {verdict} {ref}")
    wording = (f"Gazebo-like check, rough-estimate cells pooled: {ref} p95 "
               f"{pr['p95_m_mean']:.2f} m, lift {pr['lift_m_mean']:+.2f} m, wobble "
               f"{pr['wobble_rate']:.0%}; " + "; ".join(parts))
    print(f"gazebo_check: {wording} ({report['elapsed_s'] / 60:.0f} min)", flush=True)
    return {"headline": headline, "vs_baseline": wording}


def run_reengage_cell(p: dict[str, Any], cell: dict[str, Any], names: list[str],
                      factories: dict[str, Any], seeds: list[int], num_envs: int,
                      scale: float, device: str, writer: Any) -> dict[str, Any]:
    """Fly one ``reengage`` cell (``fly_reengage``) for every controller and seed."""
    seq = dict(cell["reengage"])
    for key in ("disengage_s", "stock_s", "after_s", "guard_s"):
        if key in seq:
            seq[key] = float(seq[key]) * scale
    if "jump_lead_s" in seq:
        seq["jump_lead_s"] = [float(x) * scale for x in seq["jump_lead_s"]]
    total = seq["disengage_s"] + seq["stock_s"] + seq["after_s"]
    env_cfg = StructureHoldConfig.from_dict(merge(merge(p.get("base_env") or {}, cell.get("env")),
                                                  {"episode_length_s": total}))
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for seed in seeds:
        for name in names:
            env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device)
            t0 = time.perf_counter()
            r = fly_reengage(env, factories[name](env), seq, seed)
            for row in r:
                writer.metric(seed, cell=cell["name"], controller=name, **row)
            rows[name].extend(r)
            d = describe_reengage(r)
            print(f"gazebo_check: {cell['name']} seed {seed} {name}: crash {d['crash_rate']:.1%}, "
                  f"re-engage failures {d['reengage_fail']}, hold points under "
                  f"{seq.get('sane_m', 0.6)} m off the wall {d['wall_captures']} (crashed "
                  f"{d['crash_after_wall_capture']}), closest after re-engage p5 "
                  f"{d['min_clearance_m_p5']:.2f} m ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    return {"name": cell["name"], "label": cell.get("label", cell["name"]),
            "scored": bool(cell.get("scored", True)),
            "reengage": cell["reengage"], "env": cell.get("env") or {},
            "controllers": {name: describe_reengage(rows[name]) for name in names}}


def markdown(report: dict[str, Any]) -> str:
    names = [c["name"] for c in report["controllers"]]
    gz = report.get("gazebo") or {}
    lines = [f"Episodes per cell and controller: {report['num_envs'] * len(report['seeds'])}; "
             f"lift and wobble over the last {report['window_s']:.0f} s; wobble = roll action "
             f"1.2-2.2 Hz rms > {report['wobble_threshold']}.", "",
             "| cell | controller | crash | p95 (m) | lift (m) | wobble | toward the wall (m) "
             "| Gazebo p95 / lift / wobble / toward the wall / fast ring |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in report["cells"]:
        for name in names:
            d = c["controllers"][name]
            g = (gz.get(name) or {}).get(c["name"])
            toward = f"{g['toward_m']:.2f}" if g and "toward_m" in g else "-"
            fast = f"{g['fast_ring']:.3f}" if g and "fast_ring" in g else "-"
            lift = f"{g['lift_m']:+.2f}" if g and "lift_m" in g else "-"
            gtxt = (f"{g.get('p95_m', float('nan')):.2f} / {lift} / "
                    f"{g.get('wobble', '-')} / {toward} / {fast}" if g else "-")
            lines.append(f"| {c['label']} | {name} | {d['crash_rate']:.1%} | "
                         f"{d['p95_m_mean']:.3f} | {d['lift_m_mean']:+.3f} | "
                         f"{d['wobble_rate']:.0%} | {d['toward_wall_m_mean']:.3f} | {gtxt} |")
    if report.get("gazebo_baseline"):
        lines += ["", f"Gazebo, stock PX4: {report['gazebo_baseline']}"]
    lines += ["", "Rough-estimate cells pooled (replay cells apart):", "",
              "| controller | crash | p95 (m) | lift (m) | wobble | ring (median rms) "
              "| fast ring (median, rad/s) | toward the wall (m) | verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        d = report["pooled_rough"][name]
        v = report["verdicts"].get(name)
        vtxt = "reference" if v is None else (
            f"{'beats' if v['beats'] else 'does not beat'} {report['reference']} (p95 ratio "
            f"{v['p95_ratio']['ratio']:.2f}, CI {v['p95_ratio']['ci'][0]:.2f} to "
            f"{v['p95_ratio']['ci'][1]:.2f}; failed: "
            f"{', '.join(k for k, ok in v['checks'].items() if not ok) or 'none'})")
        lines.append(f"| {name} | {d['crash_rate']:.1%} | {d['p95_m_mean']:.3f} | "
                     f"{d['lift_m_mean']:+.3f} | {d['wobble_rate']:.0%} | "
                     f"{d['wobble_rms_median']:.4f} | {d['fast_ring_median']:.3f} | "
                     f"{d['toward_wall_m_mean']:.3f} | {vtxt} |")
    if report.get("reengage"):
        lines += ["", "Hold, stock Position mode, re-engage (crash = any after the switch; "
                  "re-engage failure = a crash after a re-engage whose hold point was a sane "
                  "distance off the wall):", "",
                  "| cell | controller | crash | re-engage failures | hold points at the wall "
                  "(crashed) | crash, last reset < 1 s before the switch | 1 s or more | "
                  "closest after re-engage, p5 (m) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        pct = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
        for c in report["reengage"]:
            for name in names:
                d = c["controllers"][name]
                lines.append(
                    f"| {c['label']} | {name} | {d['crash_rate']:.1%} | {d['reengage_fail']} | "
                    f"{d['wall_captures']} ({d['crash_after_wall_capture']}) | "
                    f"{pct(d['crash_rate_reset_under_1s'])} of {d['episodes_reset_under_1s']} | "
                    f"{pct(d['crash_rate_reset_1s_plus'])} of {d['episodes_reset_1s_plus']} | "
                    f"{d['min_clearance_m_p5']:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from training.runlib import RunWriter, entry_args, load_yaml

    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        result = run(cfg, args.device, writer)
        writer.summary("ok", headline=result["headline"], vs_baseline=result["vs_baseline"])
        return 0
    except Exception as e:  # every failure lands in summary.json, then exit non-zero
        traceback.print_exc()
        writer.summary("failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
