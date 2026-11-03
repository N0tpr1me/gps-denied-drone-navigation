"""Split a Gate 1 cell's hold error into the settle after engage and the steady hold.

Gate 1 scores the p95 of the whole 20 s episode, which starts with the 0.2 m per axis start
scatter (``start_offset_std_m``), so a controller that settles slowly scores like one that
holds loosely. This flies controllers on one cell of a Gate 1 config (on a scratch seed, never
the held-out ones) and prints, per controller: the p95 over the episode and after 5 s, the
mean error at fixed times after engage, the estimate's own error and the mean offset per axis
after 5 s (structure frame: x toward the wall). A diagnostic printout, not a run.

    python -m eval.settle_split --config eval/configs/gate1_x500_ekf_vio_fit_v19.yaml \
        --cells w0_d1,w8_d2 --policy v19=training/policies/structure_hold_v19.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from envs.structure_hold.config import StructureHoldConfig
from envs.structure_hold.env import StructureHoldEnv
from eval.gate1 import check_not_held_out, make_factory, merge

TIMES_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
STEADY_S = 5.0


def split(env: StructureHoldEnv, controller) -> dict[str, float]:
    n = env.num_envs
    steps = env.max_episode_length - 1  # the last step resets every env
    hold = env.hold_point.clone()
    true_err = np.zeros((steps, n, 3), np.float32)
    est_err = np.zeros((steps, n, 3), np.float32)
    alive = torch.ones(n, dtype=torch.bool)
    obs, _ = env.get_observations()
    with torch.no_grad():
        for k in range(steps):
            obs, _, done, _ = env.step(controller(obs))
            alive &= ~done
            true_err[k] = (env.state.pos - hold).numpy()
            est_err[k] = (env.estimate_pos - env.state.pos).numpy()
    a = alive.numpy()  # crashed episodes are left out: this is about the hold, not crashes
    true_err, est_err = true_err[:, a], est_err[:, a]
    t = (np.arange(steps) + 1) * env.dt  # true_err[k] is the state after step k
    e = np.linalg.norm(true_err, axis=-1)
    late = t >= STEADY_S
    out = {"p95_all": np.percentile(e, 95, axis=0).mean(),
           "p95_steady": np.percentile(e[late], 95, axis=0).mean(),
           "est_err_p95_steady": np.percentile(np.linalg.norm(est_err[late], axis=-1), 95,
                                               axis=0).mean(),
           "alive": a.mean()}
    out.update({f"e_{x:g}s": e[max(0, int(round(x / env.dt)) - 1)].mean()
                for x in TIMES_S if x < t[-1]})
    out.update({f"offset_{ax}": true_err[late, :, i].mean() for i, ax in enumerate("xyz")})
    return {k: float(v) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="a Gate 1 config (eval.gate1)")
    ap.add_argument("--cells", default=None, help="comma-separated cell names (default all)")
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH",
                    help="a policy (.onnx, or .pt with a .json next to it); the PD always flies")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=8001)
    ap.add_argument("--env", default=None, help="YAML mapping merged over each cell's env")
    args = ap.parse_args(argv)
    check_not_held_out(args.seed)
    p = yaml.safe_load(open(args.config))["params"]
    specs = [{"name": s.split("=", 1)[0], "policy": s.split("=", 1)[1]} for s in args.policy]
    specs.append({"name": "pd", "policy": None})
    cells = None if args.cells is None else set(args.cells.split(","))
    over = yaml.safe_load(args.env) if args.env else None
    for cell in p["cells"]:
        if cells is not None and cell["name"] not in cells:
            continue
        cfg = StructureHoldConfig.from_dict(merge(merge(p.get("base_env") or {},
                                                        cell.get("env")), over))
        for spec in specs:
            env = StructureHoldEnv(cfg, args.num_envs, seed=args.seed, device="cpu")
            r = split(env, make_factory(spec, "cpu")(env))
            curve = " ".join(f"{x:g}s {r[f'e_{x:g}s']:.3f}" for x in TIMES_S if f"e_{x:g}s" in r)
            print(f"{cell['name']} {spec['name']:>8}: p95 {r['p95_all']:.3f} m, after "
                  f"{STEADY_S:g} s {r['p95_steady']:.3f} m (estimate error "
                  f"{r['est_err_p95_steady']:.3f}); mean error {curve}; offset x/y/z "
                  f"{r['offset_x']:+.3f}/{r['offset_y']:+.3f}/{r['offset_z']:+.3f} m; "
                  f"alive {r['alive']:.1%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
