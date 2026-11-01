"""Record what the Python side computes, for the C++ unit test of the policy controller.

Writes ``tests/PolicyTestVectors.hpp`` with three tables, all taken from the real Python
code (not re-derived here), so ``tests/test_policy.cpp`` proves the module matches training:

* ``MLP``: stacked policy inputs -> onnxruntime outputs of the released policy;
* ``OBS``: mock states (ENU / FLU) with their NED / FRD equivalents -> the observation
  ``StructureHoldEnv._observe`` writes;
* ``FALLBACK``: those observations and attitudes -> the brake-and-hover action of
  ``envs.structure_hold.fallback`` (``acceleration_to_action`` on ``-brake_vel * vel``).

The structure frame of the mock is the ENU world (x east toward the wall), which at the
PX4 boundary is a NED heading of +90 deg; the NED / FRD columns are produced with
``envs/frames.py``. Run after a policy release or an env change and commit the result:

    python -m sim2real.px4.tools.make_test_vectors training/policies/structure_hold_v2.onnx
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from envs.frames import enu_to_ned, flu_to_frd
from envs.sims import QuadState
from envs.structure_hold.baseline import acceleration_to_action
from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import NUM_OBS_V1, StructureHoldEnv
from envs.structure_hold.fallback import FallbackRule
from training.policy_io import read_metadata

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "tests" / "PolicyTestVectors.hpp"
T_ENU_TO_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
T_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) body -> world, as envs/sims/torch_quad quat_rotate."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def random_quat(rng: np.random.Generator, max_tilt_deg: float = 40.0) -> np.ndarray:
    tilt = math.radians(rng.uniform(0, max_tilt_deg))
    yaw = rng.uniform(-0.6, 0.6)
    axis = np.array([rng.normal(), rng.normal(), 0.0])
    axis /= np.linalg.norm(axis)
    q_tilt = np.array([math.cos(tilt / 2), *(axis * math.sin(tilt / 2))])
    q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_tilt
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _rows(a: np.ndarray, per_line: int = 8) -> str:
    vals = [f"{np.float32(v)!s}f" for v in a.reshape(-1).astype(np.float32)]
    chunks = (vals[k:k + per_line] for k in range(0, len(vals), per_line))
    return "\n".join("\t" + ", ".join(c) + "," for c in chunks)


def _table(name: str, a: np.ndarray, comment: str = "") -> str:
    tail = f"  // {comment}" if comment else ""
    return f"static constexpr float {name} = {{{tail}\n{_rows(a)}\n}};"


def make(onnx_path: Path, out_path: Path, n: int = 12, seed: int = 0) -> str:
    import onnxruntime as ort

    meta = read_metadata(onnx_path)
    history = int(meta["input"]["history"])
    num_obs = int(meta["input"].get("num_obs", NUM_OBS_V1))  # the policy's slice of the obs
    rng = np.random.default_rng(seed)
    # The policy's own airframe (policy.json env_cfg), so the fallback vectors use the same
    # mass and thrust the header's MASS_KG / MAX_THRUST_N carry.
    cfg = StructureHoldConfig.from_dict(meta.get("env_cfg"))
    env = StructureHoldEnv(cfg, num_envs=n, seed=seed)

    # --- states in the mock's frames
    hold = np.stack([-rng.uniform(1.0, 3.0, n), np.zeros(n), np.full(n, 2.0)], 1)
    pos = hold + rng.normal(0, 0.5, (n, 3))
    vel = rng.normal(0, 1.0, (n, 3))
    quat = np.stack([random_quat(rng) for _ in range(n)])
    ang_vel = rng.normal(0, 1.5, (n, 3))
    last_action = rng.uniform(-1, 1, (n, 4))
    stale = rng.random(n) < 0.5
    age = np.where(stale, rng.uniform(0.2, 2.5, n), 0.0)  # the env's age: 0 while fresh
    engage = rng.uniform(0.0, 6.0, n)  # past the cap on some
    # Forward range: a reading, its age (some too old) and the latched reference; the random
    # attitudes put the beam off the wall on many, so both flag values occur.
    range_meas = rng.uniform(0.3, 4.0, n)
    range_age = np.where(rng.random(n) < 0.25, 0.5, rng.uniform(0.0, 0.1, n))
    range_ref = rng.uniform(0.7, 3.0, n)
    pos[0] = hold[0] + np.array([-7.0, 0.3, 0.1])  # beyond the 5 m clip
    stale[0], age[0] = True, 3.0

    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    state = QuadState(pos=t(pos), quat=t(quat), vel=t(vel), ang_vel=t(ang_vel))
    env.sim.reset(torch.arange(n), state)
    env.hold_point[:] = t(hold)
    env.estimate_pos[:] = t(pos)
    env.estimate_vel[:] = t(vel)
    env.estimate_stale[:] = torch.as_tensor(stale)
    env.estimate_age[:] = t(age)
    env.engage_time[:] = t(engage)
    env.range_present[:] = True
    env.range_meas[:] = t(range_meas)
    env.range_age[:] = t(range_age)
    env.range_ref[:] = t(range_ref)
    env.last_action[:] = t(last_action)
    # On airframes with PX4's gyro filter the env observes the filtered rate the rate loop
    # last read (rate_pid.gyro). The module reads vehicle_angular_velocity, which is that
    # filtered rate, so the vectors describe the sample -> observation mapping with the
    # filter state at the sampled rate.
    env.rate_pid.gyro[:] = t(ang_vel)
    env._observe()
    obs = env.obs_buf[:, :num_obs].clone()
    # What the module hands the policy for the range: its usable flag (WallGuard::usable and
    # the off-axis test) and the unclipped error, so the C++ clip is checked too.
    _, range_ok = env.range_input()
    range_ok = range_ok.numpy()
    range_err = np.where(range_ok, range_ref - range_meas * env._beam_cos().numpy(), 0.0)

    rule = FallbackRule(threshold_s=0.5)
    acc = (-rule.brake_vel * obs[:, 3:6]).clamp(-rule.max_acc, rule.max_acc)
    fallback = acceleration_to_action(env, obs, acc, tilt=rule.tilt, yaw=rule.yaw).numpy()

    # --- the same states at the PX4 boundary
    pos_ned = enu_to_ned(pos)
    vel_ned = enu_to_ned(vel)
    hold_ned = enu_to_ned(hold)
    rates_frd = flu_to_frd(ang_vel)
    r_frd_ned = np.stack([T_ENU_TO_NED @ quat_to_rotmat(q) @ T_FRD_TO_FLU for q in quat])

    # --- MLP vectors: inputs shaped like real stacked observations
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    base = np.tile(obs.numpy(), (1, history))
    x = (base + rng.normal(0, 0.1, base.shape)).astype(np.float32)
    x[0] = 0.0
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    stale_words = [str(bool(v)).lower() for v in stale]
    out = [
        "// GENERATED by sim2real/px4/tools/make_test_vectors.py from the Python env and",
        f"// training/policies/{onnx_path.name}. Do not edit.",
        "#pragma once",
        "",
        "namespace structure_hold { namespace testvec",
        "{",
        "",
        f"static constexpr int N = {n};",
        f"static constexpr int HISTORY = {history};",
        f"static constexpr int NUM_OBS = {num_obs};",
        "// The mock's structure x (east) is a NED heading of +90 deg.",
        "static constexpr float STRUCTURE_YAW_NED = 1.5707963267948966f;",
        f"static constexpr float FALLBACK_THRESHOLD_S = {rule.threshold_s!r}f;",
        "// The Python FallbackRule the FALLBACK_ACTION table was made with.",
        f"static constexpr float FALLBACK_BRAKE_VEL = {rule.brake_vel!r}f;",
        f"static constexpr float FALLBACK_MAX_ACC = {rule.max_acc!r}f;",
        f"static constexpr float FALLBACK_TILT = {rule.tilt!r}f;",
        f"static constexpr float FALLBACK_YAW = {rule.yaw!r}f;",
        "",
        _table("HOLD_NED[N * 3]", hold_ned),
        _table("POS_NED[N * 3]", pos_ned),
        _table("VEL_NED[N * 3]", vel_ned),
        _table("R_FRD_NED[N * 9]", r_frd_ned, "row-major"),
        _table("RATES_FRD[N * 3]", rates_frd),
        _table("LAST_ACTION[N * 4]", last_action),
        "static constexpr bool STALE[N] = {" + ", ".join(stale_words) + "};",
        _table("AGE_S[N]", age),
        _table("ENGAGE_S[N]", engage),
        "static constexpr bool RANGE_OK[N] = {" + ", ".join(
            "true" if x else "false" for x in range_ok) + "};",
        _table("RANGE_ERR_M[N]", range_err),
        _table("OBS[N * NUM_OBS]", obs.numpy(), "StructureHoldEnv._observe"),
        _table("FALLBACK_ACTION[N * 4]", fallback, "acceleration_to_action"),
        "",
        _table("MLP_IN[N * HISTORY * NUM_OBS]", x),
        _table("MLP_OUT[N * 4]", y, "onnxruntime"),
        "",
        "}} // namespace structure_hold::testvec",
        "",
    ]
    text = "\n".join(out)
    out_path.write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("onnx", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    make(args.onnx, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
