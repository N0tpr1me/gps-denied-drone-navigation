"""Fit a customer's airframe for the env from one PX4 flight log and its parameter file.

    python -m training.launch --config eval/configs/airframe_fit_gz_x500.yaml --device cpu

The "airframe fitting kit" of the build plan's client package, and gap 9 of
docs/market-research.md: a drone maker brings their own drone, flies PX4's attitude
autotune (one button in QGroundControl, which the partner pack's tuned baseline needs
anyway) with the partner pack's logging profile, and sends the log, the parameter file and
the weighed mass. This returns an airframe mapping in the shape of ``gz_x500()``
(``envs/structure_hold/airframes.py``) that a training config loads with
``airframe: path/to/fit.yaml``, so the policy can be fine-tuned on their drone
(docs/airframe-fit.md). It is proven on the Gazebo x500, whose true numbers come from its
SDF.

What is fitted, all on the flight while airborne:

1. **Thrust curve and motor lag.** The accelerometer's body z reading times the mass is the
   collective thrust (aero forces are small in still air). Each rotor makes
   ``k (e + (1 - e) u)^2`` at normalised command ``u`` lagged by a first-order motor time
   constant: rotor speed linear in the command from a fraction ``e`` of full speed, thrust
   quadratic in speed, as PX4's ESC model and the env assume (``actuation.esc_min_frac``).
   ``e`` and the lag are searched on a grid, ``k`` by least squares for each; max thrust is
   ``n k``.
2. **Torque effectiveness per axis.** The raw gyro's change over short windows against the
   integral of the lagged normalised torque setpoint (``vehicle_torque_setpoint``, after
   PX4's yaw torque filter) gives ``b`` in rad/s^2 per unit command, the ratio of the env's
   ``torque_per_unit`` to the inertia, plus a constant (trim) term. The lag and a pure delay
   are searched on a grid shared by the three axes. Only ``b`` reaches the rate dynamics, so
   the inertia can be a guess (given, or the x500's scaled by mass and arm length) and
   ``torque_per_unit = b * inertia``.
3. **Torque limits.** Where PX4's mix (from the ``CA_ROTOR*`` parameters) drives the first
   rotor to full or zero command from hover, linearised as in ``gz_x500()``.
4. **Rate controller.** From the parameter file: ``MC_*RATE_P/I/D/FF`` (times ``_K``),
   ``MC_*R_INT_LIM``, ``IMU_GYRO_CUTOFF``, ``IMU_DGYRO_CUTOFF``, ``MC_YAW_TQ_CUTOFF``. The
   policy runs on top of their rate loop, so the env must run theirs.

The check, ``replay``: the fitted rotational model predicts each 0.1 s change of the gyro
from the torque commands, and its R^2 per axis says how much of the motion the model
explains. With ``truth: gz_x500`` the fit is also compared, number by number, with the
x500's SDF-derived airframe.

Logged topics needed (the partner pack's ``SDLOG_PROFILE`` 147 has them; the high-rate
bit 4 is what logs the actuator topics fast enough): ``sensor_combined``,
``actuator_motors``, ``vehicle_torque_setpoint``, ``vehicle_land_detected``, and
``hover_thrust_estimate`` for a cross-check.

Frames: everything here stays in PX4's body FRD, since only magnitudes per axis leave this
module (the env's ``torque_per_unit`` and limits are per-axis magnitudes).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.structure_hold.airframes import (
    MODULE_THRUST_RANGE,
    X500_BASE_INERTIA,
    X500_CA_ROTORS,
    airframe_config,
    px4_allocation,
)
from training.runlib import RunWriter, entry_args, load_yaml

GRAVITY = 9.80665
MOTORS = "actuator_motors"
TORQUE = "vehicle_torque_setpoint"
IMU = "sensor_combined"
LANDED = "vehicle_land_detected"
HOVER = "hover_thrust_estimate"
TOPICS = [MOTORS, TORQUE, IMU, LANDED, HOVER]
REPO_ROOT = Path(__file__).resolve().parent.parent
X500_MASS = 2.0643076923076924  # gz_x500 total mass, for the inertia guess
X500_ARM = math.hypot(0.13, 0.21)  # mean PX4 arm of 4001_gz_x500 (CA_ROTOR), m

Topics = dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class FitParams:
    rate_hz: float = 250.0  # the grid every signal is resampled to
    trim_s: float = 3.0  # skip after take-off and before landing
    window_s: float = 0.1  # gyro-change windows for the torque fit
    esc_min_grid: tuple[float, float, int] = (0.0, 0.5, 51)
    lag_grid_s: tuple[float, float, int] = (0.002, 0.08, 40)
    delay_grid_s: tuple[float, float, int] = (0.0, 0.03, 16)
    thrust_lowpass_s: float = 0.02  # against accelerometer vibration

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FitParams:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown fit keys: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _grid(spec: tuple[float, float, int]) -> np.ndarray:
    return np.linspace(spec[0], spec[1], int(spec[2]))


# ---------------------------------------------------------------- signals


@dataclass
class Flight:
    """The airborne part of one log on a uniform grid. Body FRD throughout."""

    t: np.ndarray  # (n,) s
    motors: np.ndarray  # (n, m) normalised motor commands, 0 to 1
    torque_sp: np.ndarray  # (n, 3) normalised torque setpoint
    gyro: np.ndarray  # (n, 3) rad/s, raw
    accel_z: np.ndarray  # (n,) m/s^2, specific force body z (negative when thrusting up)
    hover_thrust: float | None  # PX4's estimate, normalised collective


def _series(msg: dict[str, np.ndarray], fmt: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(msg["timestamp"], dtype=np.int64) * 1e-6
    x = np.stack([np.asarray(msg[fmt.format(k)], float) for k in range(count)], axis=1)
    t, first = np.unique(t, return_index=True)
    return t, x[first]


def _resample(t_src: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(grid, t_src, x[:, k]) for k in range(x.shape[1])], axis=1)


def airborne_window(topics: Topics, trim_s: float) -> tuple[float, float]:
    """The longest stretch the land detector says the vehicle was in the air, trimmed."""
    msg = topics[LANDED]
    t = np.asarray(msg["timestamp"], dtype=np.int64) * 1e-6
    landed = np.asarray(msg["landed"]).astype(bool)
    best, start = (0.0, 0.0), None
    for i in range(len(t)):
        if not landed[i] and start is None:
            start = t[i]
        if (landed[i] or i == len(t) - 1) and start is not None:
            end = t[i]
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    if best[1] - best[0] < 2 * trim_s + 5.0:
        raise ValueError("no airborne stretch long enough in the log")
    return best[0] + trim_s, best[1] - trim_s


def flight_from_topics(topics: Topics, p: FitParams, window: tuple[float, float] | None = None
                       ) -> Flight:
    for name in (MOTORS, TORQUE, IMU):
        if name not in topics:
            raise ValueError(f"log has no {name}: fly with SDLOG_PROFILE 147 (high rate)")
    if window is None:
        window = airborne_window(topics, p.trim_s)
    tm, motors = _series(topics[MOTORS], "control[{}]", 12)
    finite = np.all(np.isfinite(motors), axis=0)  # PX4 logs unused outputs as NaN
    used = finite & (np.abs(np.where(np.isfinite(motors), motors, 0.0)).max(axis=0) > 0)
    motors = motors[:, used]
    tt, torque = _series(topics[TORQUE], "xyz[{}]", 3)
    ti, gyro = _series(topics[IMU], "gyro_rad[{}]", 3)
    _, acc = _series(topics[IMU], "accelerometer_m_s2[{}]", 3)
    t0 = max(window[0], tm[0], tt[0], ti[0])
    t1 = min(window[1], tm[-1], tt[-1], ti[-1])
    grid = np.arange(t0, t1, 1.0 / p.rate_hz)
    hover = None
    if HOVER in topics:
        h = topics[HOVER]
        th = np.asarray(h["timestamp"], dtype=np.int64) * 1e-6
        ok = (th >= t0) & (th <= t1) & np.asarray(h.get("valid", np.ones(len(th)))).astype(bool)
        if ok.any():
            hover = float(np.median(np.asarray(h["hover_thrust"], float)[ok]))
    return Flight(t=grid, motors=np.clip(_resample(tm, motors, grid), 0.0, 1.0),
                  torque_sp=_resample(tt, torque, grid), gyro=_resample(ti, gyro, grid),
                  accel_z=_resample(ti, acc[:, 2:3], grid)[:, 0], hover_thrust=hover)


def first_order_lag(x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """``x`` (n, ...) through ``tau dy/dt = x - y`` (exact discretisation, zero-order hold),
    started at ``x[0]``."""
    a = math.exp(-dt / tau) if tau > 0 else 0.0
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * y[i - 1] + (1.0 - a) * x[i - 1]
    return y


def _delay(x: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return x
    return np.concatenate([np.repeat(x[:1], samples, axis=0), x[:-samples]], axis=0)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


# ---------------------------------------------------------------- fits


def fit_thrust(f: Flight, mass: float, p: FitParams) -> dict[str, Any]:
    """Thrust curve and motor lag (module docstring, step 1)."""
    dt = 1.0 / p.rate_hz
    thrust = -mass * _smooth(f.accel_z, round(p.thrust_lowpass_s * p.rate_hz))
    best = None
    for tau in _grid(p.lag_grid_s):
        u = first_order_lag(f.motors, tau, dt)
        for e in _grid(p.esc_min_grid):
            basis = ((e + (1.0 - e) * u) ** 2).sum(axis=1)
            k = float(basis @ thrust / (basis @ basis))
            err = float(np.mean((thrust - k * basis) ** 2))
            if best is None or err < best[0]:
                best = (err, tau, e, k)
    err, tau, e, k = best
    n = f.motors.shape[1]
    max_thrust = n * k
    u_hover = (math.sqrt(mass * GRAVITY / max_thrust) - e) / (1.0 - e)
    tau, e = float(tau), float(e)  # grid values are NumPy floats; the YAML wants plain ones
    return {"motor_tau_s": tau, "esc_min_frac": e, "rotor_k_n": k, "rotors": n,
            "max_thrust_n": max_thrust, "hover_command": u_hover,
            "thrust_rms_error_n": math.sqrt(err),
            "thrust_rms_n": float(np.std(thrust)),
            "px4_hover_thrust": f.hover_thrust}


def _window_rows(f: Flight, drive: np.ndarray, p: FitParams) -> tuple[np.ndarray, np.ndarray]:
    """Gyro change over each window and the integral of ``drive`` (n, 3) over it."""
    dt = 1.0 / p.rate_hz
    w = max(int(round(p.window_s * p.rate_hz)), 1)
    cum = np.concatenate([np.zeros((1, 3)), np.cumsum(drive, axis=0) * dt], axis=0)
    starts = np.arange(0, len(f.t) - w, w)
    dgyro = f.gyro[starts + w] - f.gyro[starts]
    integ = cum[starts + w] - cum[starts]
    return dgyro, integ


def _axis_rows(dg: np.ndarray, integ: np.ndarray, window_s: float, k: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Least squares for axis ``k``: the gyro change against the integrals of all three
    torque commands and a constant. Returns the coefficients (4,) and the residuals."""
    a = np.column_stack([integ, np.full(len(dg), window_s)])
    sol, *_ = np.linalg.lstsq(a, dg[:, k], rcond=None)
    return sol, dg[:, k] - a @ sol


def fit_torque(f: Flight, p: FitParams) -> dict[str, Any]:
    """Torque effectiveness per axis, with a lag and a delay shared by the axes (step 2).

    Each axis is fitted against all three commands, so a command that leaks into another
    axis (the Gazebo x500's yaw into roll: PX4's mix assumes arms the SDF does not have) is
    measured instead of spoiling that axis's own number. The env models only the diagonal;
    the leaks are reported in ``coupling`` as a fraction of the affected axis's own
    effectiveness."""
    dt = 1.0 / p.rate_hz
    best = None
    for tau in _grid(p.lag_grid_s):
        lagged = first_order_lag(f.torque_sp, tau, dt)
        for delay in _grid(p.delay_grid_s):
            drive = _delay(lagged, int(round(delay / dt)))
            dg, integ = _window_rows(f, drive, p)
            err, sols = 0.0, []
            for k in range(3):
                sol, r = _axis_rows(dg, integ, p.window_s, k)
                # Each axis weighted by its own variance, so yaw (small) still counts.
                err += float(np.mean(r ** 2) / max(np.var(dg[:, k]), 1e-12))
                sols.append(sol)
            if best is None or err < best[0]:
                best = (err, tau, delay, sols, dg, integ)
    err, tau, delay, sols, dg, integ = best
    names = ("roll", "pitch", "yaw")
    out: dict[str, Any] = {"lag_s": float(tau), "delay_s": float(delay), "axes": {},
                           "coupling": {}}
    for k, axis in enumerate(names):
        sol = sols[k]
        _, r = _axis_rows(dg, integ, p.window_s, k)
        # The share of the motion the other axes' commands explain.
        a = np.column_stack([integ[:, k], np.full(len(dg), p.window_s)])
        own, *_ = np.linalg.lstsq(a, dg[:, k], rcond=None)
        r_own = dg[:, k] - a @ own
        var = max(np.var(dg[:, k]), 1e-12)
        out["axes"][axis] = {"effectiveness_rad_s2": float(sol[k]), "trim_rad_s2": float(sol[3]),
                             "replay_r2": 1.0 - float(np.mean(r ** 2) / var),
                             "replay_r2_own_command_only": 1.0 - float(np.mean(r_own ** 2) / var)}
        for j, other in enumerate(names):
            if j != k:
                out["coupling"][f"{other}_to_{axis}"] = float(sol[j] / sol[k])
    return out


def read_px4_params(path: str | Path) -> dict[str, float]:
    """A QGroundControl .params file as name -> value."""
    from sim2real.estimate_bridge.params import read_params

    return {k: float(v) for k, v in read_params(path).items()}


def rate_pid_from_params(prm: dict[str, float]) -> dict[str, Any]:
    """The env's ``rate_pid`` block from PX4's rate controller parameters (step 4)."""
    axes = (("ROLL", "RR"), ("PITCH", "PR"), ("YAW", "YR"))
    defaults = {"P": (0.15, 0.15, 0.2), "I": (0.2, 0.2, 0.1), "D": (0.003, 0.003, 0.0),
                "FF": (0.0, 0.0, 0.0), "K": (1.0, 1.0, 1.0), "LIM": (0.3, 0.3, 0.3)}

    def get(axis: int, name: str) -> float:
        long, short = axes[axis]
        key = f"MC_{short}_INT_LIM" if name == "LIM" else f"MC_{long}RATE_{name}"
        return float(prm.get(key, defaults[name][axis]))

    out: dict[str, Any] = {}
    for name, key in (("P", "p"), ("I", "i"), ("D", "d")):
        out[key] = [get(a, "K") * get(a, name) for a in range(3)]
    out["ff"] = [get(a, "FF") for a in range(3)]
    out["int_lim"] = [get(a, "LIM") for a in range(3)]
    out["gyro_cutoff_hz"] = float(prm.get("IMU_GYRO_CUTOFF", 40.0))
    out["d_cutoff_hz"] = float(prm.get("IMU_DGYRO_CUTOFF", 30.0))
    out["yaw_torque_cutoff_hz"] = float(prm.get("MC_YAW_TQ_CUTOFF", 2.0))
    return out


def ca_rotors_from_params(prm: dict[str, float]) -> list[tuple[float, float, float]] | None:
    """PX4's rotor geometry ``(px, py, km)`` from ``CA_ROTOR*``, or None if not set."""
    n = int(prm.get("CA_ROTOR_COUNT", 0))
    if n < 3:
        return None
    return [(prm.get(f"CA_ROTOR{i}_PX", 0.0), prm.get(f"CA_ROTOR{i}_PY", 0.0),
             prm.get(f"CA_ROTOR{i}_KM", 0.05)) for i in range(n)]


def torque_limits(per_unit: list[float], ca_rotors: list[tuple[float, float, float]],
                  u_hover: float) -> list[float]:
    """Torque where PX4's mix first drives a rotor to 0 or 1 from hover (step 3),
    linearised: per-unit torque times the command fraction that saturates."""
    mix = px4_allocation(ca_rotors)
    out = []
    for k in range(3):
        col = mix[:, k]
        room = min((1.0 - u_hover) / c if c > 0 else -u_hover / c
                   for c in col if abs(c) > 1e-9)
        out.append(float(per_unit[k] * room))
    return out


def fit_airframe(f: Flight, mass: float, prm: dict[str, float], p: FitParams,
                 inertia: tuple[float, float, float] | None = None) -> dict[str, Any]:
    """The airframe mapping and the fit report."""
    thrust = fit_thrust(f, mass, p)
    torque = fit_torque(f, p)
    ca = ca_rotors_from_params(prm) or list(X500_CA_ROTORS)
    if inertia is None:
        arm = float(np.mean([math.hypot(px, py) for px, py, _ in ca]))
        scale = (mass / X500_MASS) * (arm / X500_ARM) ** 2
        inertia = tuple(v * scale for v in X500_BASE_INERTIA)
        inertia_source = "x500 scaled by mass and arm length (only b = torque / inertia matters)"
    else:
        inertia_source = "given"
    b = [torque["axes"][a]["effectiveness_rad_s2"] for a in ("roll", "pitch", "yaw")]
    per_unit = [float(abs(bk) * ik) for bk, ik in zip(b, inertia, strict=True)]
    limits = torque_limits(per_unit, ca, thrust["hover_command"])
    airframe = {
        "quad": {"mass": mass, "inertia": [float(v) for v in inertia],
                 "max_thrust": thrust["max_thrust_n"], "max_torque": limits,
                 "motor_tau": thrust["motor_tau_s"]},
        "rate_pid": rate_pid_from_params(prm),
        "actuation": {"torque_per_unit": per_unit, "esc_min_frac": thrust["esc_min_frac"],
                      "thrust_cmd_range": list(MODULE_THRUST_RANGE)},
        "rate_lag_s": [0.0, 0.0],
    }
    return {"airframe": airframe, "thrust": thrust, "torque": torque,
            "inertia_source": inertia_source, "ca_rotors": [list(r) for r in ca]}


def compare_to(result: dict[str, Any], name: str) -> dict[str, Any]:
    """Fitted numbers against a named airframe's (``gz_x500``), with errors in %."""
    ref = airframe_config(name)
    fit = result["airframe"]
    ref_b = [t / i for t, i in zip(ref["actuation"]["torque_per_unit"], ref["quad"]["inertia"],
                                   strict=True)]
    fit_b = [result["torque"]["axes"][a]["effectiveness_rad_s2"] for a in ("roll", "pitch", "yaw")]
    rows = {
        "max_thrust_n": (ref["quad"]["max_thrust"], fit["quad"]["max_thrust"]),
        "esc_min_frac": (ref["actuation"]["esc_min_frac"], fit["actuation"]["esc_min_frac"]),
        "motor_tau_s": (ref["quad"]["motor_tau"], fit["quad"]["motor_tau"]),
    }
    for k, axis in enumerate(("roll", "pitch", "yaw")):
        rows[f"effectiveness_{axis}_rad_s2"] = (ref_b[k], fit_b[k])
        rows[f"max_torque_over_inertia_{axis}"] = (
            ref["quad"]["max_torque"][k] / ref["quad"]["inertia"][k],
            fit["quad"]["max_torque"][k] / fit["quad"]["inertia"][k])
    return {k: {"ref": float(r), "fit": float(v), "error_pct": 100.0 * (v - r) / r}
            for k, (r, v) in rows.items()}


# ---------------------------------------------------------------- synthetic flights


def synthetic_flight(seconds: float = 40.0, rate_hz: float = 250.0, mass: float = 1.6,
                     rotor_k: float = 7.0, esc_min: float = 0.12, motor_tau: float = 0.025,
                     effectiveness: tuple[float, float, float] = (180.0, 160.0, 12.0),
                     yaw_to_roll: float = 0.05, delay_s: float = 0.004,
                     gyro_noise: float = 0.002, accel_noise: float = 0.05, seed: int = 0,
                     ca_rotors: Any = None) -> tuple[Flight, dict[str, float]]:
    """A flight with known answers, for the tests and the smoke run: square-wave torque
    steps on each axis in turn (as PX4's autotune), PX4's mix to four motors around hover,
    each motor lagged and making ``rotor_k (esc_min + (1 - esc_min) u)^2``; rates from
    ``effectiveness`` (rad/s^2 per unit, plus a yaw-to-roll leak) on the lagged, delayed
    torque command. Returns the flight and a matching parameter file (rate gains and
    ``CA_ROTOR*``)."""
    rng = np.random.default_rng(seed)
    ca = list(ca_rotors or X500_CA_ROTORS)
    dt = 1.0 / rate_hz
    n = int(seconds * rate_hz)
    t = np.arange(n) * dt
    torque = np.zeros((n, 3))
    third = n // 3
    for k, amp in enumerate((0.08, 0.08, 0.2)):
        seg = slice(k * third, (k + 1) * third)
        torque[seg, k] = amp * np.sign(np.sin(2 * math.pi * 1.5 * t[seg]))
    torque += 0.01 * rng.normal(size=torque.shape)  # the rate loop's own chatter
    collective = 0.5 + 0.03 * np.sin(2 * math.pi * 0.3 * t)  # a gentle climb and sink
    mix = px4_allocation(ca)
    u_hover = (math.sqrt(mass * GRAVITY / (len(ca) * rotor_k)) - esc_min) / (1 - esc_min)
    motors = np.clip(u_hover + (collective - 0.5)[:, None] + torque @ mix[:, :3].T, 0.0, 1.0)
    u_lag = first_order_lag(motors, motor_tau, dt)
    thrust = (rotor_k * (esc_min + (1 - esc_min) * u_lag) ** 2).sum(axis=1)
    accel_z = -thrust / mass + accel_noise * rng.normal(size=n)
    b = np.diag(effectiveness).astype(float)
    b[0, 2] = yaw_to_roll * effectiveness[0]
    drive = _delay(first_order_lag(torque, motor_tau, dt), int(round(delay_s / dt)))
    gyro = np.cumsum(drive @ b.T, axis=0) * dt
    gyro += gyro_noise * rng.normal(size=gyro.shape)
    prm = {"MC_ROLLRATE_P": 0.15, "MC_ROLLRATE_I": 0.2, "MC_ROLLRATE_D": 0.003,
           "MC_ROLLRATE_K": 1.0, "MC_PITCHRATE_P": 0.12, "MC_PITCHRATE_K": 2.0,
           "IMU_GYRO_CUTOFF": 40.0, "CA_ROTOR_COUNT": float(len(ca))}
    for i, (px, py, km) in enumerate(ca):
        prm.update({f"CA_ROTOR{i}_PX": px, f"CA_ROTOR{i}_PY": py, f"CA_ROTOR{i}_KM": km})
    return Flight(t=t, motors=motors, torque_sp=torque, gyro=gyro, accel_z=accel_z,
                  hover_thrust=None), prm


# ---------------------------------------------------------------- entrypoint


def load_topics(path: str | Path) -> Topics:
    try:
        from pyulog import ULog
    except ImportError as e:  # optional dependency: pip install -e .[ulog]
        raise ImportError("reading .ulg needs pyulog: pip install -e .[ulog]") from e
    log = ULog(str(path), message_name_filter_list=TOPICS)
    return {d.name: d.data for d in log.data_list if d.multi_id == 0}


def format_report(res: dict[str, Any]) -> str:
    th, tq, af = res["thrust"], res["torque"], res["airframe"]
    lines = [
        f"thrust: {th['rotors']} rotors, max {th['max_thrust_n']:.2f} N, esc_min_frac "
        f"{th['esc_min_frac']:.3f}, motor lag {1000 * th['motor_tau_s']:.1f} ms, hover command "
        f"{th['hover_command']:.3f} (PX4's estimate {th['px4_hover_thrust']}), rms error "
        f"{th['thrust_rms_error_n']:.3f} N of {th['thrust_rms_n']:.3f} N",
        f"torque: lag {1000 * tq['lag_s']:.1f} ms, delay {1000 * tq['delay_s']:.1f} ms",
    ]
    for axis, a in tq["axes"].items():
        lines.append(f"  {axis:5s} b {a['effectiveness_rad_s2']:8.2f} rad/s^2 per unit, trim "
                     f"{a['trim_rad_s2']:+.3f}, replay R^2 {a['replay_r2']:.3f} (own command "
                     f"only {a['replay_r2_own_command_only']:.3f})")
    big = {k: round(v, 3) for k, v in tq["coupling"].items() if abs(v) >= 0.02}
    lines.append(f"  cross-coupling (fraction of the axis's own b, 2 % or more): {big or 'none'}")
    lines.append(f"inertia: {af['quad']['inertia']} ({res['inertia_source']})")
    lines.append(f"torque_per_unit {[round(v, 3) for v in af['actuation']['torque_per_unit']]}"
                 f", max_torque {[round(v, 3) for v in af['quad']['max_torque']]}")
    if res.get("truth"):
        lines.append(f"\nagainst {res['truth_name']}:")
        for k, r in res["truth"].items():
            lines.append(f"  {k:32s} ref {r['ref']:9.4f}  fit {r['fit']:9.4f}  "
                         f"{r['error_pct']:+6.1f} %")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    params = cfg["params"]
    p = FitParams.from_dict(params.get("fit"))
    writer = RunWriter(args.run_dir)
    try:
        if args.smoke and params.get("synthetic") is not None:
            f, prm = synthetic_flight(**params["synthetic"])
        else:
            from eval.fit_wind import latest

            f = flight_from_topics(load_topics(latest(params["ulog"])), p,
                                   params.get("window_s"))
            prm = read_px4_params(REPO_ROOT / params["params_file"])
        inertia = params.get("inertia")
        res = fit_airframe(f, float(params["mass_kg"]), prm, p,
                           tuple(inertia) if inertia else None)
        if params.get("truth"):
            res["truth_name"] = params["truth"]
            res["truth"] = compare_to(res, params["truth"])
        run = Path(args.run_dir)
        (run / "fit.json").write_text(json.dumps(res, indent=2) + "\n")
        (run / "airframe.yaml").write_text(
            "# Fitted by eval/fit_airframe.py; load with airframe: <this file>\n"
            + yaml.safe_dump(res["airframe"], sort_keys=False))
        print(format_report(res), flush=True)
        head = {"max_thrust_n": round(res["thrust"]["max_thrust_n"], 2),
                "motor_tau_ms": round(1000 * res["thrust"]["motor_tau_s"], 1),
                **{f"b_{a}": round(v["effectiveness_rad_s2"], 1)
                   for a, v in res["torque"]["axes"].items()}}
        if res.get("truth"):
            head["worst_error_pct"] = round(max(abs(r["error_pct"])
                                                for r in res["truth"].values()), 1)
        writer.summary("ok", headline=head,
                       vs_baseline=f"airframe {params.get('truth')}" if params.get("truth")
                       else "none: a customer airframe has no reference")
        return 0
    except Exception as exc:  # recorded in summary.json, then re-raised for the log
        writer.summary("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
