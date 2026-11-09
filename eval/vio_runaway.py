"""The VIO runaway check in the mock: how often it trips on a healthy VIO, and how soon on one
that runs away (``docs/field-faults.md``, "Heavy dust").

The module's ``RunawayFix`` (``sim2real/px4/structure_hold/Watchdog.hpp``, ``SH_WD_VIO_DV``)
compares how much the VIO's velocity changed over 3 s with how much the IMU says the vehicle's
velocity changed. ``RunawayFix`` below is the same check, line for line, batched over envs in
NumPy (``tests/test_vio_runaway.py`` holds the two to the same answers). This script flies the
Structure Hold mock (``envs/structure_hold``, the x500 as SITL flies it, EKF2-like fusion)
with a controller, records each env's raw VIO fix (before the fusion) and an IMU (the true
acceleration plus white noise and a constant bias per env, which the check must tolerate),
and runs the check over it. It does not hand anything back: the episode carries on as the
controller flies it, so a set with a runaway also shows what happens without the check
(every controller follows its estimate; the 3 m box ends the episode).

Per set (``params.sets``, each an ``env`` override on ``params.base_env``) and controller:

- ``trips``: envs whose check tripped, and the false-trip rate per flight hour with a 95 %
  upper bound (a set without a runaway: every trip is false);
- with a runaway (``env.estimate.runaway_at_s``): the share caught, the time from the
  runaway's start to the trip (median, p95, max), how far the vehicle had truly moved from
  the hold point at the trip, and the share that left the 3 m box or crashed and when.

    python -m training.launch --config eval/configs/vio_runaway.yaml --device cpu
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from typing import Any

import numpy as np

# The mock (torch) is imported where it is flown, so the check and the log replay below
# run without it (a partner's ulog on a laptop).

SLOT_US = 250_000
SLOTS = 12
WINDOW_US = 3_000_000
IMU_GAP_US = 500_000
FIX_GAP_US = 2_000_000
TRIP_SLOTS = 2


class RunawayFix:
    """``Watchdog.hpp`` ``RunawayFix`` for ``n`` vehicles at once; times in integer
    microseconds, one call per stream sample with a mask of the vehicles it applies to."""

    def __init__(self, n: int, limit_mps: float):
        self.n, self.limit = n, float(limit_mps)
        self.v_imu = np.zeros((n, 3))
        self.imu_t = np.zeros(n, dtype=np.int64)
        self.fix_t = np.zeros(n, dtype=np.int64)
        self.slot = np.zeros((n, 3))
        self.slot_n = np.zeros(n, dtype=np.int64)
        self.slot_t = np.zeros(n, dtype=np.int64)
        self.ring = np.zeros((n, SLOTS + 1, 3))
        self.ring_t = np.zeros((n, SLOTS + 1), dtype=np.int64)
        self.head = np.zeros(n, dtype=np.int64)
        self.count = np.zeros(n, dtype=np.int64)
        self.over = np.zeros(n, dtype=np.int64)
        self.disagreement = np.zeros(n)
        self.runaway = np.zeros(n, dtype=bool)
        self.counter = np.zeros(n, dtype=np.int64)
        self.counter_seen = np.zeros(n, dtype=bool)

    def _restart(self, m: np.ndarray) -> None:
        self.count[m] = 0
        self.over[m] = 0
        self.slot_n[m] = 0
        self.slot[m] = 0.0
        self.fix_t[m] = 0
        self.disagreement[m] = 0.0

    def inertial(self, t_us: Any, accel: np.ndarray, mask: np.ndarray | None = None) -> None:
        t = np.broadcast_to(np.asarray(t_us, dtype=np.int64), (self.n,))
        m = np.isfinite(accel).all(1) if mask is None else (mask & np.isfinite(accel).all(1))
        ok = m & (self.imu_t > 0) & (t > self.imu_t) & (t - self.imu_t < IMU_GAP_US)
        dt = (t - self.imu_t) * 1e-6
        self.v_imu[ok] += accel[ok] * dt[ok, None]
        self._restart(m & (self.imu_t > 0) & ~ok)
        self.imu_t[m] = t[m]

    def fix(self, t_us: Any, vel: np.ndarray, counter: Any = 0,
            mask: np.ndarray | None = None) -> None:
        t = np.broadcast_to(np.asarray(t_us, dtype=np.int64), (self.n,))
        c = np.broadcast_to(np.asarray(counter, dtype=np.int64), (self.n,))
        m = np.ones(self.n, dtype=bool) if mask is None else mask.copy()
        changed = m & self.counter_seen & (c != self.counter)
        self._restart(changed)
        self.runaway[changed] = False
        self.counter[m] = c[m]
        self.counter_seen[m] = True
        m &= np.isfinite(vel).all(1) & (self.imu_t > 0)
        self._restart(m & (self.fix_t > 0) & ((t <= self.fix_t) | (t - self.fix_t >= FIX_GAP_US)))
        self.fix_t[m] = t[m]
        self._close(m & (self.slot_n > 0) & (t - self.slot_t >= SLOT_US))  # starts the next
        first = m & (self.slot_n == 0)
        self.slot_t[first] = t[first]
        self.slot[m] += vel[m] - self.v_imu[m]
        self.slot_n[m] += 1

    def _close(self, m: np.ndarray) -> None:
        if not m.any():
            return
        idx = np.flatnonzero(m)
        e = self.slot[idx] / self.slot_n[idx, None]
        self.ring[idx, self.head[idx]] = e
        self.ring_t[idx, self.head[idx]] = self.slot_t[idx]
        self.slot[idx] = 0.0
        self.slot_n[idx] = 0
        self.head[idx] = (self.head[idx] + 1) % (SLOTS + 1)
        self.count[idx] = np.minimum(self.count[idx] + 1, SLOTS + 1)
        full = idx[self.count[idx] > SLOTS]
        if full.size == 0:
            return
        o = self.ring[full, self.head[full]]
        newest = (self.head[full] - 1) % (SLOTS + 1)
        span = self.ring_t[full, newest] - self.ring_t[full, self.head[full]]
        scale = np.where(span > WINDOW_US, WINDOW_US / np.maximum(span, 1), 1.0)
        d = scale * np.linalg.norm(self.ring[full, newest] - o, axis=1)
        self.disagreement[full] = d
        over = (self.limit > 0) & (d > self.limit)
        self.over[full] = np.where(over, self.over[full] + 1, 0)
        self.runaway[full] |= self.over[full] >= TRIP_SLOTS


def fly_set(env: Any, controller: Any, p: dict[str, Any], seed: int,
            runaway_at_s: float | None) -> list[dict[str, Any]]:
    """Fly every env's first episode; one row per env."""
    from envs.structure_hold.env import TERM_REASONS

    n, dt, steps = env.num_envs, env.dt, env.max_episode_length
    limit = float(p["limit_mps"])
    fix_every = max(1, int(round(1.0 / (float(p.get("fix_rate_hz", 50.0)) * dt))))
    rng = np.random.default_rng(seed)
    imu_bias = rng.normal(0.0, float(p.get("imu_bias_mps2", 0.1)), (n, 3))
    imu_noise = float(p.get("imu_noise_mps2", 0.05))
    # EKF2 corrects its acceleration toward the VIO: the vision velocity fusion moves its tilt
    # and accel bias so that the integrated acceleration follows the fix. Modelled as a
    # critically damped velocity / acceleration-offset observer on the raw fix (natural
    # period ekf_tau_s), started converged on the IMU bias. It is why a constant bias does
    # not trip the check, and why a slow runaway can hide from it: EKF2 takes the runaway
    # into its tilt (the Gazebo logs, docs/field-faults.md "Heavy dust").
    tau = float(p.get("ekf_tau_s", 2.5))
    k1, k2 = (2.0 / tau, 1.0 / (tau * tau)) if tau > 0 else (0.0, 0.0)
    c_hat = imu_bias.copy()
    v_hat = env.state.vel.cpu().numpy().astype(np.float64).copy()
    fix_dt = fix_every * dt
    check = RunawayFix(n, limit)

    # The raw fix, as the corruptor reports it (before the fusion the policy reads).
    raw: dict[str, Any] = {}
    step_estimator = env.estimator.step

    def capture(pos: Any, vel: Any, noise: Any = None) -> Any:
        out = step_estimator(pos, vel, noise)
        raw["vel"], raw["stale"] = out.vel, out.stale
        return out

    env.estimator.step = capture
    hold = env.hold_point.cpu().numpy().copy()
    alive = np.ones(n, dtype=bool)
    end_s = np.full(n, steps * dt)
    reason = np.full(n, "timeout", dtype=object)
    trip_s = np.full(n, np.nan)
    trip_move = np.full(n, np.nan)
    max_dv = np.zeros(n)
    obs, _ = env.get_observations()
    for k in range(steps):
        vel_before = env.state.vel.cpu().numpy().copy()
        a = controller(obs)
        obs, _, done, _ = env.step(a)
        done = done.cpu().numpy()
        t_us = int(round((k + 1) * dt * 1e6))
        # step_pos is the true position after the step; state.vel is the next episode's for an
        # env that just ended, so its last IMU sample is skipped.
        vel_after = env.state.vel.cpu().numpy()
        accel = (vel_after - vel_before) / dt + imu_bias + imu_noise * rng.standard_normal((n, 3))
        accel = accel - c_hat
        v_hat += accel * dt
        live = alive & ~done
        check.inertial(t_us, accel, live)
        if (k + 1) % fix_every == 0:
            v = raw["vel"].cpu().numpy().astype(np.float64)
            fresh = live & ~raw["stale"].cpu().numpy()
            innov = np.where(fresh[:, None], v - v_hat, 0.0)
            v_hat += k1 * fix_dt * innov
            c_hat -= k2 * fix_dt * innov
            check.fix(t_us, v, 0, fresh)
        max_dv = np.where(live, np.maximum(max_dv, check.disagreement), max_dv)
        new_trip = live & check.runaway & np.isnan(trip_s)
        if new_trip.any():
            trip_s[new_trip] = (k + 1) * dt
            pos = env.step_pos.cpu().numpy()
            trip_move[new_trip] = np.linalg.norm(pos[new_trip] - hold[new_trip], axis=1)
        ended = alive & done
        if ended.any():
            end_s[ended] = (k + 1) * dt
            reason[ended] = [TERM_REASONS[int(r)] for r in env.term_reason.cpu().numpy()[ended]]
        alive &= ~done
        if not alive.any():
            break
    env.estimator.step = step_estimator
    return [{"env": i, "length_s": float(end_s[i]), "reason": str(reason[i]),
             "trip_s": None if math.isnan(trip_s[i]) else float(trip_s[i]),
             "trip_move_m": None if math.isnan(trip_move[i]) else float(trip_move[i]),
             "max_dv_mps": float(max_dv[i]), "runaway_at_s": runaway_at_s} for i in range(n)]


def poisson_upper(k: int, conf: float = 0.95) -> float:
    """One-sided upper bound on a Poisson mean after ``k`` events (k = 0: -ln(1 - conf))."""
    if k == 0:
        return -math.log(1.0 - conf)
    # Wilson-Hilferty approximation of the chi-square quantile, plenty for a rate bound.
    z = 1.6448536269514722 if conf == 0.95 else 2.3263478740408408
    nu = 2 * (k + 1)
    return 0.5 * nu * (1 - 2 / (9 * nu) + z * math.sqrt(2 / (9 * nu))) ** 3


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    hours = sum(r["length_s"] for r in rows) / 3600.0
    at = rows[0]["runaway_at_s"] if rows else None
    trips = [r for r in rows if r["trip_s"] is not None]
    out: dict[str, Any] = {"episodes": n, "flight_h": round(hours, 2), "trips": len(trips),
                           "max_dv_p99_mps": float(np.percentile([r["max_dv_mps"] for r in rows],
                                                                 99)) if rows else None}
    ended = [r for r in rows if r["reason"] != "timeout"]
    out["ended_early"] = len(ended)
    out["end_reasons"] = {k: sum(r["reason"] == k for r in ended)
                          for k in sorted({r["reason"] for r in ended})}
    if at is None:
        out["false_trips_per_h"] = len(trips) / hours if hours > 0 else None
        out["false_trips_per_h_95"] = poisson_upper(len(trips)) / hours if hours > 0 else None
        return out
    before = [r for r in trips if r["trip_s"] < at]
    caught = [r for r in trips if r["trip_s"] >= at]
    out["tripped_before_runaway"] = len(before)
    out["caught"] = len(caught)
    if caught:
        delay = np.array([r["trip_s"] - at for r in caught])
        move = np.array([r["trip_move_m"] for r in caught])
        out["time_to_trip_s"] = {"median": float(np.median(delay)),
                                 "p95": float(np.percentile(delay, 95)), "max": float(delay.max())}
        out["moved_at_trip_m"] = {"median": float(np.median(move)),
                                  "p95": float(np.percentile(move, 95)), "max": float(move.max())}
        # Caught before the episode ended (a box exit or crash before the trip is a miss).
        out["caught_before_end"] = sum(r["trip_s"] <= r["length_s"] for r in caught)
    if ended:
        t_end = np.array([r["length_s"] - at for r in ended])
        out["time_to_end_s"] = {"median": float(np.median(t_end)), "min": float(t_end.min())}
    return out


def run(cfg: dict[str, Any], device: str, writer: Any) -> dict[str, Any]:
    import torch

    from envs.structure_hold.config import StructureHoldConfig
    from envs.structure_hold.env import StructureHoldEnv
    from eval.gate1 import check_not_held_out, make_factory
    from training.runlib import merge

    p = cfg["params"]
    seeds = [int(s) for s in p["seeds"]]
    check_not_held_out(*seeds)
    num_envs = int(p.get("num_envs", 128))
    specs = [dict(c) for c in p["controllers"]]
    factories = {s["name"]: make_factory(s, device) for s in specs}
    report: dict[str, Any] = {"limit_mps": float(p["limit_mps"]), "seeds": seeds,
                              "num_envs": num_envs, "sets": []}
    t0 = time.perf_counter()
    step = 0
    for s in p["sets"]:
        env_over = merge(p["base_env"], s.get("env"))
        if p.get("episode_length_s") is not None:  # the smoke's shorter flights
            env_over["episode_length_s"] = float(p["episode_length_s"])
        env_cfg = StructureHoldConfig.from_dict(env_over)
        at = env_cfg.estimate.runaway_at_s if env_cfg.estimate.runaway_accel_mps2 > 0 else None
        entry: dict[str, Any] = {"name": s["name"], "runaway_at_s": at,
                                 "runaway_accel_mps2": env_cfg.estimate.runaway_accel_mps2,
                                 "controllers": {}}
        for spec in specs:
            rows: list[dict[str, Any]] = []
            for seed in seeds:
                torch.manual_seed(seed)
                env = StructureHoldEnv(env_cfg, num_envs, seed=seed, device=device)
                with torch.no_grad():
                    rows += fly_set(env, factories[spec["name"]](env), p, seed, at)
            d = describe(rows)
            entry["controllers"][spec["name"]] = d
            writer.metric(step, set=s["name"], controller=spec["name"], **{
                k: v for k, v in d.items() if isinstance(v, (int, float))})
            step += 1
            print(f"[{time.perf_counter() - t0:7.0f} s] {s['name']:>20} {spec['name']:>4}: "
                  f"{json.dumps(d)}", flush=True)
        report["sets"].append(entry)
    (writer.run_dir / "vio_runaway.json").write_text(json.dumps(report, indent=2))
    return report


def replay_ulog(path: str, limit_mps: float) -> dict[str, Any]:
    """Run the check over a PX4 log, as the module would have: the IMU from
    ``vehicle_local_position`` (ax, ay, az), the fix from ``vehicle_visual_odometry`` (NED
    velocity, or body-frame turned by ``vehicle_attitude``). Gives the largest disagreement
    seen, its time, and when the check would have tripped at ``limit_mps``: what a partner's
    logs need to set SH_WD_VIO_DV, and how the Gazebo flights below were judged."""
    from pyulog import ULog

    log = ULog(str(path), message_name_filter_list=[
        "vehicle_local_position", "vehicle_visual_odometry", "vehicle_attitude"])
    data = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    lp, od = data["vehicle_local_position"], data["vehicle_visual_odometry"]
    events = [(int(t), 0, k) for k, t in enumerate(lp["timestamp_sample"])]
    events += [(int(t), 1, k) for k, t in enumerate(od["timestamp_sample"])]
    events.sort()
    att = data.get("vehicle_attitude")
    f = RunawayFix(1, limit_mps)
    t0 = int(lp["timestamp"][0])
    best, best_t, trip_t = 0.0, None, None
    series = []
    for t, kind, k in events:
        if kind == 0:
            f.inertial(t, np.array([[lp["ax"][k], lp["ay"][k], lp["az"][k]]], dtype=float))
            continue
        v = np.array([od[f"velocity[{i}]"][k] for i in range(3)], dtype=float)
        frame = int(od["velocity_frame"][k])
        if frame == 3 and att is not None:  # BODY_FRD: turn it by the attitude then
            j = min(np.searchsorted(att["timestamp"], t), len(att["timestamp"]) - 1)
            w, x, y, z = (float(att[f"q[{i}]"][j]) for i in range(4))
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            v = r @ v
        elif frame != 1:
            v = np.full(3, np.nan)
        f.fix(t, v[None], int(od["reset_counter"][k]))
        d = float(f.disagreement[0])
        series.append(((t - t0) * 1e-6, d))
        if d > best:
            best, best_t = d, (t - t0) * 1e-6
        if trip_t is None and f.runaway[0]:
            trip_t = (t - t0) * 1e-6
    return {"log": str(path), "limit_mps": limit_mps, "max_dv_mps": best, "max_dv_at_s": best_t,
            "trip_at_s": trip_t, "series": series}


def headline(report: dict[str, Any]) -> dict[str, Any]:
    h: dict[str, Any] = {"limit_mps": report["limit_mps"]}
    for s in report["sets"]:
        for name, d in s["controllers"].items():
            key = f"{s['name']}/{name}"
            if s["runaway_at_s"] is None:
                h[key] = {"false_trips": d["trips"], "flight_h": d["flight_h"],
                          "per_h_95": d["false_trips_per_h_95"]}
            else:
                h[key] = {"caught": d.get("caught", 0), "of": d["episodes"],
                          "time_to_trip_s": d.get("time_to_trip_s"),
                          "ended_early": d["ended_early"]}
    return h


def main(argv: list[str] | None = None) -> int:
    from training.runlib import RunWriter, entry_args, load_yaml

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "replay":
        # python -m eval.vio_runaway replay <flight.ulg> [limit m/s]
        r = replay_ulog(argv[1], float(argv[2]) if len(argv) > 2 else 1.0)
        r.pop("series")
        print(json.dumps(r, indent=2))
        return 0
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    writer = RunWriter(args.run_dir)
    try:
        report = run(cfg, args.device, writer)
        writer.summary("ok", headline=headline(report),
                       vs_baseline="the same check over the cascaded PD yardstick's flights")
        return 0
    except Exception as e:  # every failure lands in summary.json, then exit non-zero
        traceback.print_exc()
        writer.summary("failed", error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
