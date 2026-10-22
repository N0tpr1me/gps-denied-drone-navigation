"""Parity between two simulator backends: does Isaac Lab move the drone the way the mock does?

    python -m eval.sim_parity --a torch --b isaac --device cuda --out runs/parity_isaac.json

Two checks, both with identical inputs on both backends, both on true state:

1. **Open loop** (``open_loop``): twelve one-second scenarios, one per env, that each isolate
   a piece of physics (hover, free fall, climb, roll / pitch pulses, yaw spin, external force,
   a tilted and spinning start that exercises the gyroscopic term and the reset frames,
   saturated commands, an aggressive all-axis sweep, the same sweep on another airframe set
   with ``set_airframe``, and an external body moment like the near-wall yaw moment). Inputs
   are a fixed function of time, so the trajectories can be compared step by step: worst
   position, velocity, attitude and body-rate difference per scenario.
2. **Closed loop** (``closed_loop``): the Structure Hold env with the cascaded PD on the
   corrupted estimate, same seed, same wind, same estimate noise, for ``seconds``. Errors
   cannot grow without bound here, so this is the check that matters for training: per env,
   the worst position difference while both copies are alive; how many envs crash in one
   backend and not the other; and the hold error (p95) of each. It runs twice: on the nominal
   airframe, and with the Gate 1 airframe randomisation and wall yaw moment on
   (``RANDOMISED``), so every env flies its own airframe.

Tolerances (``OPEN_LOOP_TOL`` per scenario, ``CLOSED_LOOP_TOL``) are written down below and
come from one rule: Isaac Lab must agree with the mock about as well as the mock agrees with
itself at an 8x finer physics step (``--a torch --b torch-fine``), which is the
discretisation error every mock result already carries; each limit is twice that measured
gap, with small floors where the gap is zero. The test that runs this check is
``eval/tests/test_sim_parity.py``; it skips when Isaac Lab is not installed.

Backends by name: ``torch`` (mock), ``isaac`` (Isaac Lab), ``torch-fine`` (mock at 8x the
substeps, the reference for the tolerances).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from envs.sims import QuadParams, QuadState, make_sim
from envs.sims.torch_quad import GRAVITY, quat_from_axis_angle, quat_mul

DT = 0.01
OPEN_LOOP_STEPS = 100  # 1 s at 100 Hz
FINE_FACTOR = 8


@dataclass(frozen=True)
class OpenLoopTolerance:
    """Worst difference over the one-second scenario."""

    pos_m: float
    vel_m_s: float
    att_rad: float
    rate_rad_s: float


@dataclass(frozen=True)
class ClosedLoopTolerance:
    """Structure Hold with the cascaded PD; position terms over envs that would not have
    crashed in either backend."""

    pos_median_m: float = 0.02  # median over envs of the worst position difference
    pos_p95_m: float = 0.10  # 95th percentile over envs of the same
    crash_mismatch: float = 0.02  # fraction of envs that would crash in one backend only
    p95_rel: float = 0.01  # relative difference of the pooled hold-error p95


SCENARIOS = ("hover", "free_fall", "climb", "roll_pulse", "pitch_pulse", "yaw_spin",
             "external_force", "tilted_spinning_start", "saturated", "aggressive",
             "other_airframe", "external_moment")

# The airframe the ``other_airframe`` scenario flies (``set_airframe``): every knob off
# nominal, inside or at the edge of the Gate 1 "combined" stress row, with unequal inertia
# scales so a swapped axis shows, and a max thrust the sweep runs into.
OTHER_AIRFRAME = {"mass": 0.8 * 1.3, "inertia": (0.0035 * 1.5, 0.0035 * 0.7, 0.0060 * 1.2),
                  "max_thrust": 14.0, "max_torque": (0.24, 0.24, 0.015), "motor_tau": 0.05}

# Per scenario: twice the gap measured between the mock and the mock at 8x finer physics
# steps (2026-10-22, CPU, float32), rounded up to two digits, with floors of 1 mm, 1 mm/s,
# 1 mrad and 0.01 rad/s where that gap is zero or below them (hover, and axes a scenario
# barely moves).
# Regenerate only in a PR that says why (docs/isaac-lab.md); never loosen to make Isaac pass.
OPEN_LOOP_TOL: dict[str, OpenLoopTolerance] = {
    "hover": OpenLoopTolerance(pos_m=0.001, vel_m_s=0.001, att_rad=0.001, rate_rad_s=0.01),
    "free_fall": OpenLoopTolerance(pos_m=0.083, vel_m_s=0.041, att_rad=0.001, rate_rad_s=0.01),
    "climb": OpenLoopTolerance(pos_m=0.025, vel_m_s=0.013, att_rad=0.001, rate_rad_s=0.01),
    "roll_pulse": OpenLoopTolerance(pos_m=0.0043, vel_m_s=0.0024, att_rad=0.0042,
                                    rate_rad_s=0.024),
    "pitch_pulse": OpenLoopTolerance(pos_m=0.0043, vel_m_s=0.0024, att_rad=0.0042,
                                     rate_rad_s=0.024),
    "yaw_spin": OpenLoopTolerance(pos_m=0.001, vel_m_s=0.001, att_rad=0.014, rate_rad_s=0.01),
    "external_force": OpenLoopTolerance(pos_m=0.0039, vel_m_s=0.001, att_rad=0.001,
                                        rate_rad_s=0.01),
    "tilted_spinning_start": OpenLoopTolerance(pos_m=0.02, vel_m_s=0.033, att_rad=0.013,
                                               rate_rad_s=0.01),
    "saturated": OpenLoopTolerance(pos_m=0.059, vel_m_s=0.092, att_rad=0.053, rate_rad_s=0.47),
    "aggressive": OpenLoopTolerance(pos_m=0.081, vel_m_s=0.076, att_rad=0.047, rate_rad_s=0.15),
    # Added 2026-10-22 with set_airframe and the external body moment, by the same rule.
    "other_airframe": OpenLoopTolerance(pos_m=0.11, vel_m_s=0.11, att_rad=0.071,
                                        rate_rad_s=0.22),
    "external_moment": OpenLoopTolerance(pos_m=0.023, vel_m_s=0.0048, att_rad=0.01,
                                         rate_rad_s=0.01),
}
# Measured gaps (two seeds, 256 envs x 10 s): median 7 to 10 mm, p95 43 mm, no crash
# mismatch, hold p95 within 0.2 %.
CLOSED_LOOP_TOL = ClosedLoopTolerance()
# With RANDOMISED on (four seeds, 256 envs x 10 s): median 13 to 16 mm, p95 0.13 to 0.17 m,
# crash mismatch 0 to 1.6 % (up to 4 envs), hold p95 within 0.6 to 1.5 %. Twice the worst
# seed, rounded up to two digits. The tail is wider than nominal because a heavy, slow-motor,
# high-inertia airframe is marginal under the PD, so step-size differences grow there.
RANDOMISED_TOL = ClosedLoopTolerance(pos_median_m=0.033, pos_p95_m=0.33, crash_mismatch=0.032,
                                     p95_rel=0.03)

SimFactory = Callable[[QuadParams, int, float, str], Any]


def factory(name: str) -> SimFactory:
    if name in ("torch", "isaac"):
        return lambda params, n, dt, device: make_sim(name, params, n, dt, device=device)
    if name == "torch-fine":
        return lambda params, n, dt, device: make_sim(
            "torch", replace(params, substeps=params.substeps * FINE_FACTOR), n, dt,
            device=device)
    raise ValueError(f"unknown backend {name!r}; choose torch, isaac or torch-fine")


# ---------------------------------------------------------------- open loop


def initial_state(params: QuadParams, device: str) -> QuadState:
    n = len(SCENARIOS)
    s = QuadState.hover(n, pos=(1.0, -2.0, 3.0), device=device)
    k = SCENARIOS.index("tilted_spinning_start")
    axis = torch.tensor([[0.6, -0.4, 0.69]], device=device)
    axis = axis / axis.norm()
    s.quat[k] = quat_from_axis_angle(axis, torch.tensor([0.5], device=device))[0]
    s.vel[k] = torch.tensor([1.0, 0.5, -0.2], device=device)
    s.ang_vel[k] = torch.tensor([0.5, -0.3, 0.8], device=device)
    # Rolled, so a body moment applied about the world axes instead would show.
    k = SCENARIOS.index("external_moment")
    s.quat[k] = quat_from_axis_angle(torch.tensor([[1.0, 0.0, 0.0]], device=device),
                                     torch.tensor([0.4], device=device))[0]
    return s


def inputs_at(step: int, params: QuadParams, device: str
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(thrust (n,), body torque (n, 3), world force (n, 3), external body torque (n, 3)) for
    every scenario at ``step``."""
    t = step * DT
    n = len(SCENARIOS)
    hover = params.hover_thrust
    thrust = torch.full((n,), hover, device=device)
    torque = torch.zeros(n, 3, device=device)
    force = torch.zeros(n, 3, device=device)
    moment = torch.zeros(n, 3, device=device)
    tmax = torch.tensor(params.max_torque, device=device)

    def pulse(axis: int) -> torch.Tensor:
        v = torch.zeros(3, device=device)
        v[axis] = 0.02 if t < 0.1 else (-0.02 if t < 0.2 else 0.0)
        return v

    i = SCENARIOS.index
    thrust[i("free_fall")] = 0.0
    thrust[i("climb")] = 1.3 * hover
    torque[i("roll_pulse")] = pulse(0)
    torque[i("pitch_pulse")] = pulse(1)
    torque[i("yaw_spin")] = torch.tensor([0.0, 0.0, 0.01], device=device)
    force[i("external_force")] = torch.tensor(
        [0.6 + 0.3 * math.sin(2 * math.pi * t), -0.3, 0.2], device=device)
    k = i("tilted_spinning_start")
    thrust[k] = hover * (1.0 + 0.2 * math.sin(2 * math.pi * 1.5 * t))
    torque[k] = 0.005 * torch.tensor([math.sin(3 * t), math.cos(2 * t), 0.5], device=device)
    force[k] = torch.tensor([0.2, 0.1, 0.0], device=device)
    k = i("saturated")
    thrust[k] = 2.0 * params.max_thrust if t < 0.3 else -1.0
    torque[k] = torch.tensor([1.0, -1.0, 1.0], device=device) * (1.0 if t < 0.05 else 0.0)
    sweep = torch.tensor([math.sin(2 * math.pi * 2.0 * t), math.sin(2 * math.pi * 1.7 * t + 1.0),
                          math.sin(2 * math.pi * 1.3 * t + 2.0)], device=device)
    k = i("aggressive")
    thrust[k] = hover * (1.0 + 0.4 * math.sin(2 * math.pi * 2.0 * t))
    torque[k] = 0.3 * tmax * sweep
    # Around its own hover point, into its lower thrust and torque limits.
    k = i("other_airframe")
    thrust[k] = OTHER_AIRFRAME["mass"] * GRAVITY * (1.0 + 0.4 * math.sin(2 * math.pi * 2.0 * t))
    torque[k] = 0.4 * tmax * sweep
    # A yaw moment like the near-wall one (Gate 1's 25 % row at the wall), plus small roll
    # and pitch parts, on top of a hover command: neither clipped nor lagged.
    k = i("external_moment")
    moment[k] = torch.tensor([0.004 * math.sin(2 * math.pi * t), -0.003, 0.0125],
                             device=device)
    return thrust, torque, force, moment


def rollout_open_loop(make: SimFactory, params: QuadParams, device: str,
                      steps: int = OPEN_LOOP_STEPS) -> dict[str, torch.Tensor]:
    """Trajectories (steps, n, ...) of every scenario on one backend."""
    sim = make(params, len(SCENARIOS), DT, device)
    sim.set_airframe([SCENARIOS.index("other_airframe")], **OTHER_AIRFRAME)
    sim.reset(state=initial_state(params, device))
    out: dict[str, list[torch.Tensor]] = {"pos": [], "quat": [], "vel": [], "ang_vel": []}
    for k in range(steps):
        s = sim.step(*inputs_at(k, params, device))
        for key in out:
            out[key].append(getattr(s, key).detach().to("cpu", torch.float64).clone())
    return {key: torch.stack(v) for key, v in out.items()}


def trajectory_errors(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]
                      ) -> dict[str, torch.Tensor]:
    """Worst difference over time, per env: position, velocity, attitude angle, body rate."""
    # Relative rotation a^-1 b; atan2 keeps small angles exact (acos of a dot product near 1
    # would put a 1e-3 rad floor under float32 quaternions).
    conj = torch.cat([a["quat"][..., :1], -a["quat"][..., 1:]], -1)
    rel = quat_mul(conj.reshape(-1, 4), b["quat"].reshape(-1, 4)).reshape(a["quat"].shape)
    angle = 2.0 * torch.atan2(rel[..., 1:].norm(dim=-1), rel[..., 0].abs())
    return {
        "pos_m": (a["pos"] - b["pos"]).norm(dim=-1).amax(0),
        "vel_m_s": (a["vel"] - b["vel"]).norm(dim=-1).amax(0),
        "att_rad": angle.amax(0),
        "rate_rad_s": (a["ang_vel"] - b["ang_vel"]).norm(dim=-1).amax(0),
    }


def open_loop(make_a: SimFactory, make_b: SimFactory, params: QuadParams | None = None,
              device: str = "cpu",
              tol: dict[str, OpenLoopTolerance] | OpenLoopTolerance | None = None
              ) -> dict[str, Any]:
    """Worst differences per scenario and the failures against ``tol`` (per scenario, or
    one tolerance for all; default ``OPEN_LOOP_TOL``)."""
    params = params or QuadParams()
    tol = OPEN_LOOP_TOL if tol is None else tol
    if isinstance(tol, OpenLoopTolerance):
        tol = dict.fromkeys(SCENARIOS, tol)
    err = trajectory_errors(rollout_open_loop(make_a, params, device),
                            rollout_open_loop(make_b, params, device))
    limits = {name: asdict(tol[name]) for name in SCENARIOS}
    per_scenario = {
        name: {q: float(err[q][i]) for q in err} for i, name in enumerate(SCENARIOS)
    }
    failures = [
        f"{name}: {q} {v:.3g} > {limits[name][q]:g}"
        for name, row in per_scenario.items() for q, v in row.items() if v > limits[name][q]
    ]
    return {"ok": not failures, "failures": failures, "tolerance": limits,
            "worst": {q: float(err[q].max()) for q in err}, "per_scenario": per_scenario}


# ---------------------------------------------------------------- closed loop


# The closed loop runs with the env's crash rules switched off, so no env is ever reset:
# a reset draws new random numbers, and a crash that lands one step apart in the two backends
# would shift every later draw for every env and make the runs incomparable. Whether an env
# WOULD have crashed is judged afterwards with the standard rules (``would_crash``).
NO_TERMINATION = {"max_pos_error_m": 1.0e6, "min_height_m": -1.0e6, "max_tilt_deg": 180.0,
                  "contact_radius_m": -1.0e6}

# The second closed-loop run: the airframe and wall yaw moment of the Gate 1 "combined" stress
# row (eval/configs/gate1_stress.yaml), so each env flies its own airframe through
# ``set_airframe`` and feels a yaw moment through the external body torque. The row's
# estimate settings are left out: they do not touch the simulator.
RANDOMISED: dict[str, Any] = {
    "airframe_rand": {"mass_scale": [0.8, 1.3], "inertia_scale": [0.7, 1.5],
                      "hover_thrust_error": [-0.1, 0.1], "motor_tau_s": [0.02, 0.05]},
    "wall_yaw_torque": {"peak_nm": [0.0, 0.0125]},
}


def _closed_loop_run(sim: str, num_envs: int, seconds: float, seed: int, device: str,
                     env: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    from envs.sims.torch_quad import body_z_world
    from envs.structure_hold import StructureHoldConfig, StructureHoldEnv
    from envs.structure_hold.baseline import CascadedPd

    overrides: dict[str, Any] = {**(env or {}), **NO_TERMINATION}
    if sim == "torch-fine":
        overrides["quad"] = {"substeps": QuadParams().substeps * FINE_FACTOR}
    cfg = StructureHoldConfig.from_dict(overrides)
    rules = StructureHoldConfig()  # the standard crash rules, applied afterwards
    steps = int(round(seconds / cfg.dt))
    if steps >= cfg.max_episode_length:
        raise ValueError("closed-loop parity compares one episode: seconds < episode length")
    hold = StructureHoldEnv(cfg, num_envs, seed=seed, device=device,
                            sim="torch" if sim == "torch-fine" else sim)
    ctl = CascadedPd(hold)
    obs, _ = hold.get_observations()
    cos_tilt = math.cos(math.radians(rules.max_tilt_deg))
    pos, err = [], []
    crash = torch.zeros(num_envs, dtype=torch.bool, device=hold.device)
    for _ in range(steps):
        s = hold.state
        e = (s.pos - hold.hold_point).norm(dim=-1)
        pos.append(s.pos.detach().cpu().double())
        err.append(e.detach().cpu().double())
        crash |= ((e > rules.max_pos_error_m) | (s.pos[:, 2] <= rules.min_height_m)
                  | (body_z_world(s.quat)[:, 2] < cos_tilt)
                  | (-s.pos[:, 0] <= rules.contact_radius_m) | ~s.finite())
        obs, _, dones, _ = hold.step(ctl(obs))  # the PD restarts its integrator itself
        if bool(dones.any()):
            raise RuntimeError("an env ended inside the parity window; the runs would diverge")
    return {"pos": torch.stack(pos), "err": torch.stack(err), "crash": crash.cpu()}


def closed_loop(sim_a: str = "torch", sim_b: str = "isaac", *, num_envs: int = 256,
                seconds: float = 10.0, seed: int = 0, device: str = "cpu",
                tol: ClosedLoopTolerance = CLOSED_LOOP_TOL,
                env: dict[str, Any] | None = None) -> dict[str, Any]:
    """``env``: Structure Hold config overrides for both runs (``RANDOMISED`` for the second
    run); the crash rules are always switched off during the run and judged afterwards."""
    a = _closed_loop_run(sim_a, num_envs, seconds, seed, device, env)
    b = _closed_loop_run(sim_b, num_envs, seconds, seed, device, env)
    worst = (a["pos"] - b["pos"]).norm(dim=-1).amax(0)  # per env, over the window
    calm = ~(a["crash"] | b["crash"])  # envs that would not have crashed in either
    w = worst[calm] if bool(calm.any()) else worst

    def p95(run: dict[str, torch.Tensor], keep: torch.Tensor) -> float:
        e = run["err"][:, keep]
        return float(torch.quantile(e.flatten(), 0.95)) if e.numel() else math.nan

    p95_a, p95_b = p95(a, calm), p95(b, calm)
    rel = abs(p95_b - p95_a) / p95_a if p95_a > 0 else math.inf
    stats = {
        "pos_median_m": float(w.median()),
        "pos_p95_m": float(torch.quantile(w, 0.95)),
        "crash_mismatch": float((a["crash"] ^ b["crash"]).double().mean()),
        "p95_rel": rel,
    }
    limits = asdict(tol)
    failures = [f"{k} {v:.3g} > {limits[k]:g}" for k, v in stats.items() if not v <= limits[k]]
    return {"ok": not failures, "failures": failures, "tolerance": limits, **stats,
            "hold_p95_m": {sim_a: p95_a, sim_b: p95_b},
            "would_crash": {sim_a: float(a["crash"].double().mean()),
                            sim_b: float(b["crash"].double().mean())},
            "num_envs": num_envs, "seconds": seconds, "seed": seed}


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="torch", help="reference backend (default torch)")
    ap.add_argument("--b", default="isaac", help="backend under test (default isaac)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-envs", type=int, default=256, help="closed loop envs")
    ap.add_argument("--seconds", type=float, default=10.0, help="closed loop horizon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="write the full report here as JSON")
    args = ap.parse_args(argv)

    ol = open_loop(factory(args.a), factory(args.b), device=args.device)
    print(f"open loop, {args.b} vs {args.a} on {args.device} (worst over 1 s / limit):")
    for name, row in ol["per_scenario"].items():
        lim = ol["tolerance"][name]
        cells = "  ".join(f"{q} {v:.1e}/{lim[q]:.1e}" for q, v in row.items())
        print(f"  {name:<22} {cells}")
    runs = {}
    for name, env, tol in (("nominal", None, CLOSED_LOOP_TOL),
                           ("randomised", RANDOMISED, RANDOMISED_TOL)):
        cl = closed_loop(args.a, args.b, num_envs=args.num_envs, seconds=args.seconds,
                         seed=args.seed, device=args.device, tol=tol, env=env)
        runs[name] = cl
        print(f"closed loop, Structure Hold + PD, {name} airframe, {args.num_envs} envs x "
              f"{args.seconds:g} s: worst position gap median {cl['pos_median_m']:.2e} m, "
              f"p95 {cl['pos_p95_m']:.2e} m; crash mismatch {cl['crash_mismatch']:.1%}; "
              f"hold p95 {cl['hold_p95_m']} (rel {cl['p95_rel']:.1%})")
    failures = ol["failures"] + [f"{name} {f}" for name, cl in runs.items()
                                 for f in cl["failures"]]
    ok = not failures
    for f in failures:
        print(f"  FAIL {f}")
    print("parity: PASS" if ok else "parity: FAIL (see docs/isaac-lab.md, 'If parity fails')")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"a": args.a, "b": args.b, "device": args.device,
                                        "ok": ok, "open_loop": ol, "closed_loop": runs["nominal"],
                                        "closed_loop_randomised": runs["randomised"]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
