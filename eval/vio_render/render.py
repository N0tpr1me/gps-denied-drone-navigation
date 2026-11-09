"""Render hold flights in clear air, fog, dust and low light, run Basalt on each, and measure
how the real VIO's error changes (docs/vio-render.md).

    python -m training.launch --config eval/configs/vio_render_wall.yaml --device cpu
    python -m training.launch --config eval/configs/vio_render_wall.yaml --smoke --device cpu

Per seed: one hold flight (``trajectory.py``) is rendered once per frame by the configured
renderer (``raster``: NumPy, any machine; ``isaac``: Isaac Sim, the GPU server), and every
condition is made from that same clean frame (``degrade.py``), so the conditions differ only
in the air and the light, never in the flight. Each condition becomes one EuRoC dataset;
Basalt (``eval/vio_basalt``, Docker) runs on it offline; ``eval/vio_fit.py`` measures the
error against the flight's truth. Pooled over seeds, each condition gets the same fit as the
EuRoC flights had (drift, cap, noise, velocity noise, jumps, gaps), printed beside the clear
condition (the reference), the ``envs/estimate`` defaults and the ``degrade`` field fault's
made-up numbers, which this is meant to replace with measured ones.

Outputs in the run folder: ``data/s<seed>/<condition>/`` with Basalt's output and the truth
(images deleted unless ``keep_images``), ``fit.json`` per condition, ``metrics.jsonl`` one
row per flight, ``summary.json``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from envs.estimate import EstimateCorruptionConfig
from eval import vio_fit
from eval.vio_render import degrade, euroc
from eval.vio_render.camera import RigConfig, world_from_cam
from eval.vio_render.scene import build_scene
from eval.vio_render.trajectory import Flight, FlightConfig, rot_from_quat, simulate
from training.runlib import RunWriter, entry_args, load_yaml

# The degrade field fault's VIO numbers (sim2real/estimate_bridge/profiles/
# field_degrade_at_engage.yaml): what the Gazebo dust/fog/low-light flights assumed.
DEGRADE_FAULT = {"pos_noise_std_m": 0.05, "vel_noise_std_mps": 0.2,
                 "bias_walk_std_m_per_sqrt_s": 0.1, "loss_hz": 0.25, "loss_s": 0.5}


# ---------------------------------------------------------------- rendering


class _Job:
    """Everything a render worker needs, built once per process."""

    def __init__(self, params: dict[str, Any], flight: Flight, seed: int, out: Path):
        self.rig = RigConfig.from_dict(params.get("rig", {}))
        self.fcfg = FlightConfig.from_dict(params.get("flight", {}))
        self.camera = degrade.CameraModel.from_dict(params.get("camera", {}))
        self.conds = [degrade.Condition.from_dict(c) for c in params["conditions"]]
        self.flight = flight
        self.seed = seed
        self.out = out
        self.renderer_name = params.get("renderer", "raster")
        self.scene_cfg = params.get("scene", {"kind": "wall"})
        self.supersample = int(params.get("supersample", 1))
        self.calib_err = euroc.CalibError.from_dict(params.get("calib_error", {}))
        self.renderer = None
        self.dust: list[degrade.DustCloud | None] = []

    def start(self) -> None:
        scene = build_scene(self.scene_cfg)
        if self.renderer_name == "raster":
            from eval.vio_render.raster import RasterRenderer

            self.renderer = RasterRenderer(scene, self.rig, self.supersample)
        elif self.renderer_name == "isaac":
            from eval.vio_render.isaac_backend import IsaacRenderer

            self.renderer = IsaacRenderer(scene, self.rig, self.out / "isaac_scene")
        else:
            raise ValueError(f"unknown renderer {self.renderer_name!r}; raster or isaac")
        self.dust = [degrade.DustCloud(c, self.fcfg.wind_mps, self.seed * 1000 + i)
                     if c.dust_per_m3 > 0 else None for i, c in enumerate(self.conds)]

    def frame(self, k: int) -> list[tuple[float, float]]:
        """Render IMU sample ``k`` (a camera frame) under every condition; returns each
        condition's (exposure ms, gain)."""
        fl = self.flight
        rot_wb = rot_from_quat(fl.quat[k : k + 1])[0]
        poses = [world_from_cam(self.rig, cam, fl.pos[k], rot_wb) for cam in (0, 1)]
        if hasattr(self.renderer, "render_pair"):  # Isaac: both cameras in one step
            frames = self.renderer.render_pair(poses)
        else:
            frames = [self.renderer.render(rot_wc, centre) for rot_wc, centre in poses]
        clean = [(rad, rng, rot_wc, centre)
                 for (rad, rng), (rot_wc, centre) in zip(frames, poses, strict=True)]
        stamp = euroc.stamp_ns(fl.t[k] + self.calib_err.time_offset_ms * 1e-3)
        exposures = []
        for i, cond in enumerate(self.conds):
            lens = [degrade.apply(cond, rad, rng, dust=self.dust[i], rig=self.rig,
                                  rot_wc=rot_wc, centre=centre, t=float(fl.t[k]))
                    for rad, rng, rot_wc, centre in clean]
            # One exposure for both cameras (a synchronised stereo pair), set by the left.
            exp_ms, gain = self.camera.exposure(float(lens[0].mean()))
            root = self.out / cond.name / "mav0"
            for cam in (0, 1):
                img = lens[cam]
                if self.camera.motion_blur:
                    r_bc, _ = self.rig.body_from_cam(cam)
                    du, dv = degrade.blur_px(self.rig, fl.gyro_true[k], r_bc, exp_ms)
                    img = degrade.motion_blur(img, du, dv)
                noise = np.random.default_rng([self.seed, i, k, cam, 0x5E])
                png = euroc.png_gray(self.camera.capture(img, exp_ms, gain, noise))
                (root / f"cam{cam}" / "data" / f"{stamp}.png").write_bytes(png)
            exposures.append((exp_ms, gain))
        return exposures


_WORKER: _Job | None = None


def _worker_init(job: _Job) -> None:
    global _WORKER
    _WORKER = job
    _WORKER.start()


def _worker_frames(ks: list[int]) -> list[tuple[int, list[tuple[float, float]]]]:
    assert _WORKER is not None
    return [(k, _WORKER.frame(k)) for k in ks]


def render_flight(params: dict[str, Any], flight: Flight, seed: int, out: Path,
                  workers: int, log=print) -> dict[str, Any]:
    """Write one EuRoC dataset per condition for ``flight``; returns render stats."""
    job = _Job(params, flight, seed, out)
    for cond in job.conds:
        for cam in ("cam0", "cam1"):
            (out / cond.name / "mav0" / cam / "data").mkdir(parents=True, exist_ok=True)
    frames = [int(k) for k in flight.cam_index]
    t0 = time.monotonic()
    results: dict[int, list[tuple[float, float]]] = {}
    if workers <= 1 or job.renderer_name == "isaac":  # Isaac: one scene, this process
        _worker_init(job)
        for n, k in enumerate(frames):
            results[k] = _WORKER.frame(k)  # type: ignore[union-attr]
            if n % 200 == 0:
                log(f"  s{seed}: frame {n}/{len(frames)}  {time.monotonic() - t0:.0f} s")
    else:
        chunks = [frames[i : i + 20] for i in range(0, len(frames), 20)]
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_worker_init, initargs=(job,)) as pool:
            for n, part in enumerate(pool.imap_unordered(_worker_frames, chunks)):
                results.update(dict(part))
                if n % 10 == 0:
                    log(f"  s{seed}: {len(results)}/{len(frames)} frames  "
                        f"{time.monotonic() - t0:.0f} s")
    off = job.calib_err.time_offset_ms * 1e-3
    stamps = [euroc.stamp_ns(flight.t[k] + off) for k in frames]
    stats = {}
    for i, cond in enumerate(job.conds):
        euroc.finish_dataset(out / cond.name, stamps, flight, job.rig, job.fcfg,
                             job.calib_err, seed)
        exp = np.array([results[k][i] for k in frames])
        stats[cond.name] = {"exposure_ms_median": float(np.median(exp[:, 0])),
                            "gain_median": float(np.median(exp[:, 1]))}
    stats["render_s"] = time.monotonic() - t0
    return stats


# ---------------------------------------------------------------- Basalt


def run_basalt(dataset: Path, out: Path, image: str, docker: str = "docker") -> dict[str, Any]:
    """Basalt stereo-inertial on one dataset folder (read only); writes
    ``stamped_traj_estimate.txt`` (TUM + velocity, eval/vio_basalt's patch), ``truth.csv``
    and ``basalt.log`` into ``out``."""
    dataset, out = dataset.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [*docker.split(), "run", "--rm", "-v", f"{dataset}:/data:ro", "-v", f"{out}:/work",
           "-w", "/work", image,
           "basalt_vio", "--dataset-path", "/data", "--dataset-type", "euroc",
           "--cam-calib", "/data/basalt_calib.json", "--config-path",
           "/data/basalt_config.json", "--show-gui", "0", "--use-imu", "1",
           "--save-trajectory", "tum"]
    t0 = time.monotonic()
    with open(out / "basalt.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    wall = time.monotonic() - t0
    traj = out / "trajectory.txt"
    if proc.returncode != 0 or not traj.exists():
        tail = (out / "basalt.log").read_text()[-600:]
        raise RuntimeError(f"basalt failed on {dataset} (exit {proc.returncode}): {tail}")
    traj.replace(out / "stamped_traj_estimate.txt")
    shutil.copy(dataset / "mav0" / "state_groundtruth_estimate0" / "data.csv",
                out / "truth.csv")
    for junk in out.glob("stats_*.ubjson"):
        junk.unlink()
    return {"basalt_s": wall}


def basalt_available(image: str, docker: str = "docker") -> bool:
    try:
        out = subprocess.run([*docker.split(), "image", "inspect", image],
                             capture_output=True, check=False)
    except FileNotFoundError:
        return False
    return out.returncode == 0


# ---------------------------------------------------------------- scoring


def score_condition(flights: list[vio_fit.FlightStats], fit: vio_fit.FitParams,
                    frames_expected: int, poses: list[int]) -> dict[str, Any]:
    result = vio_fit.summarise(flights, fit, EstimateCorruptionConfig())
    result["pose_share"] = float(np.mean([p / frames_expected for p in poses]))
    return result


def _hover_drift(result: dict[str, Any], lag: str) -> float | None:
    for label in ("hover", "all"):
        fitd = result["fits"].get(label)
        if fitd and lag in fitd["drift_rms_m"] and fitd["drift_rms_m"][lag]["pairs"] > 0:
            return fitd["drift_rms_m"][lag]["measured"]
    return None


def condition_row(name: str, result: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    prop = result["proposal"]
    per = result["per_flight"]
    return {
        "condition": name,
        "flights": result["flights"],
        "ate_rms_m": float(np.mean([r["ate_rms_m"] for r in per])),
        "ate_max_m": float(np.max([r["ate_rms_m"] for r in per])),
        "drift_20s_m": _hover_drift(result, "20"),
        "drift_60s_m": _hover_drift(result, "60"),
        "pos_noise_std_m": prop["pos_noise_std_m"],
        "vel_noise_std_mps": prop["vel_noise_std_mps"],
        "bias_walk_std_m_per_sqrt_s": prop["bias_walk_std_m_per_sqrt_s"],
        "bias_max_m": prop["bias_max_m"],
        "jumps": int(sum(r["jumps"] for r in per)),
        "gaps": int(sum(r["gaps"] for r in per)),
        "pose_share": result["pose_share"],
        "exposure_ms": stats.get("exposure_ms_median"),
        "gain": stats.get("gain_median"),
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    head = (f"{'condition':18s} {'ATE':>6s} {'ATEmax':>6s} {'d20s':>6s} {'d60s':>6s} "
            f"{'noise':>6s} {'vnoise':>6s} {'walk':>6s} {'cap':>6s} {'jumps':>5s} "
            f"{'gaps':>4s} {'poses':>5s} {'exp':>5s} {'gain':>5s}")
    lines = [head]

    def f(v: Any, w: int = 6, d: int = 3) -> str:
        return f"{v:{w}.{d}f}" if isinstance(v, int | float) and v is not None else f"{'-':>{w}s}"

    for r in rows:
        lines.append(
            f"{r['condition']:18s} {f(r['ate_rms_m'])} {f(r['ate_max_m'])} "
            f"{f(r['drift_20s_m'])} {f(r['drift_60s_m'])} {f(r['pos_noise_std_m'], 6, 4)} "
            f"{f(r['vel_noise_std_mps'])} {f(r['bias_walk_std_m_per_sqrt_s'], 6, 4)} "
            f"{f(r['bias_max_m'])} {r['jumps']:5d} {r['gaps']:4d} {f(r['pose_share'], 5, 2)} "
            f"{f(r['exposure_ms'], 5, 1)} {f(r['gain'], 5, 1)}")
    lines.append("modelled degrade fault (Gazebo field faults): noise 0.05 m, velocity noise "
                 "0.2 m/s, walk 0.1 m/sqrt(s), a 0.5 s loss every 4 s")
    d = EstimateCorruptionConfig()
    lines.append(f"envs/estimate defaults: noise {d.pos_noise_std_m} m, velocity noise "
                 f"{d.vel_noise_std_mps} m/s, walk {d.bias_walk_std_m_per_sqrt_s} m/sqrt(s), "
                 f"cap {d.bias_max_m} m")
    return "\n".join(lines)


# ---------------------------------------------------------------- entrypoint


def main(argv: list[str] | None = None) -> int:
    args = entry_args(argv)
    cfg = load_yaml(args.config)
    params = cfg["params"]
    writer = RunWriter(args.run_dir)
    run_dir = Path(args.run_dir)
    try:
        fcfg = FlightConfig.from_dict(params.get("flight", {}))
        fit = vio_fit.FitParams.from_dict(params.get("fit", {}))
        seeds = [int(s) for s in params.get("seeds", [cfg["seed"]])]
        if any(9000 <= s <= 9999 for s in seeds):
            raise ValueError("seeds 9000 to 9999 are held out for Gate 1")
        workers = params.get("workers", "auto")
        workers = (len(os.sched_getaffinity(0)) if workers == "auto" else int(workers))
        bcfg = params.get("basalt", {})
        image = bcfg.get("image", "dronenav-basalt")
        docker = os.environ.get("DOCKER", bcfg.get("docker", "docker"))
        run_vio = bool(bcfg.get("run", True))
        if run_vio and not basalt_available(image, docker):
            if args.smoke:
                print(f"basalt image {image!r} not found: smoke renders only", flush=True)
                run_vio = False
            else:
                raise RuntimeError(f"basalt image {image!r} not found; build eval/vio_basalt "
                                   "(cloud: eval/vio_basalt/cloud_build.sh)")
        # Renders from earlier runs (one per seed is fine): Isaac renders in its own
        # container, Basalt runs on the host.
        from_run = params.get("from_run")
        sources: list[Path] = []
        if from_run is not None:
            runs = from_run.split(",") if isinstance(from_run, str) else list(from_run)
            for r in (r.strip() for r in runs):
                p = Path(r) if Path(r).is_absolute() else vio_fit.REPO_ROOT / r
                if not (p / "data").is_dir():
                    raise FileNotFoundError(f"from_run {p}: no data/ folder of renders")
                sources.append(p)
        conds = [degrade.Condition.from_dict(c) for c in params["conditions"]]
        names = [c.name for c in conds]
        if len(set(names)) != len(names):
            raise ValueError("condition names must be unique")
        print(f"renderer {params.get('renderer', 'raster')}, {len(seeds)} seeds x "
              f"{len(conds)} conditions, {fcfg.duration_s:.0f} s flights, {workers} workers",
              flush=True)

        per_cond: dict[str, list[vio_fit.FlightStats]] = {n: [] for n in names}
        poses: dict[str, list[int]] = {n: [] for n in names}
        stats_all: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
        frames_expected = 0
        step = 0
        for seed in seeds:
            flight = simulate(fcfg, seed)
            frames_expected = int(flight.cam_index.size)
            hold = flight.t >= flight.hold_start_s
            err = np.linalg.norm(flight.pos[hold] - flight.hold_point, axis=1)
            print(f"s{seed}: flight {flight.t[-1]:.0f} s, {frames_expected} frames, hold p95 "
                  f"{np.percentile(err, 95):.3f} m", flush=True)
            out = run_dir / "data" / f"s{seed}"
            if from_run is None:
                stats = render_flight(params, flight, seed, out, workers,
                                      log=lambda m: print(m, flush=True))
                (out / "render_stats.json").write_text(json.dumps(stats, indent=2))
                print(f"s{seed}: rendered {len(conds)} conditions in {stats['render_s']:.0f} s",
                      flush=True)
                src = out
            else:  # images rendered by an earlier run (Isaac, in its own container)
                found = [p / "data" / f"s{seed}" for p in sources
                         if (p / "data" / f"s{seed}").is_dir()]
                if not found:
                    raise FileNotFoundError(f"seed {seed}: no renders in {from_run}")
                src = found[0]
                stats = json.loads((src / "render_stats.json").read_text())
            for cond in conds:
                ds = out / cond.name
                stats_all[cond.name].append(stats[cond.name])
                if not run_vio:
                    continue
                done = src / cond.name / "stamped_traj_estimate.txt"
                if from_run is not None and done.exists():
                    # Basalt already ran on it (a per-seed run): pool its output as it is.
                    ds.mkdir(parents=True, exist_ok=True)
                    for f in ("stamped_traj_estimate.txt", "truth.csv", "basalt.log"):
                        if (src / cond.name / f).exists() and src != out:
                            shutil.copy(src / cond.name / f, ds / f)
                    b = {"basalt_s": float("nan")}
                else:
                    b = run_basalt(src / cond.name, ds, image, docker)
                est = vio_fit.load_tum(ds / "stamped_traj_estimate.txt")
                truth = vio_fit.load_euroc_csv(ds / "truth.csv")
                fs = vio_fit.measure_flight(f"{cond.name}/s{seed}", est, truth, fit)
                per_cond[cond.name].append(fs)
                poses[cond.name].append(int(est.t.size))
                step += 1
                writer.metric(step, seed=seed, condition=cond.name, ate_rms_m=fs.ate_rms_m,
                              jumps=len(fs.jumps_m), gaps=len(fs.gaps_s),
                              poses=int(est.t.size), frames=frames_expected,
                              basalt_s=round(b["basalt_s"], 1), **stats[cond.name])
                print(f"  {cond.name:16s} ATE {fs.ate_rms_m:.3f} m, {est.t.size} poses, "
                      f"{len(fs.jumps_m)} jumps, {len(fs.gaps_s)} gaps, basalt "
                      f"{b['basalt_s']:.0f} s", flush=True)
                if from_run is None and not params.get("keep_images", False):
                    for cam in ("cam0", "cam1"):
                        shutil.rmtree(ds / "mav0" / cam / "data", ignore_errors=True)

        if not run_vio:
            writer.summary("ok", headline={"rendered_conditions": len(conds),
                                           "seeds": len(seeds), "basalt": "not run"},
                           vs_baseline="none: renders only (Basalt runs in a second run "
                                       "with params.from_run pointing here)")
            return 0

        rows, fits = [], {}
        for cond in conds:
            med = {k: float(np.median([s[k] for s in stats_all[cond.name]]))
                   for k in ("exposure_ms_median", "gain_median")}
            res = score_condition(per_cond[cond.name], fit, frames_expected, poses[cond.name])
            fits[cond.name] = res
            rows.append(condition_row(cond.name, res, med))
        (run_dir / "fit.json").write_text(json.dumps(
            {"conditions": fits, "rows": rows, "conditions_cfg": [asdict(c) for c in conds],
             "degrade_fault": DEGRADE_FAULT}, indent=2, default=str) + "\n")
        print(format_table(rows), flush=True)
        clear = rows[0]
        worst = max(rows, key=lambda r: r["ate_rms_m"])
        writer.summary(
            "ok",
            headline={
                "conditions": len(rows),
                "seeds": len(seeds),
                f"{clear['condition']}_ate_m": round(clear["ate_rms_m"], 3),
                "worst_condition": worst["condition"],
                "worst_ate_m": round(worst["ate_rms_m"], 3),
                "worst_walk": round(worst["bias_walk_std_m_per_sqrt_s"], 4),
                "jumps_total": int(sum(r["jumps"] for r in rows)),
                "gaps_total": int(sum(r["gaps"] for r in rows)),
            },
            vs_baseline=f"each condition against {clear['condition']} (same flights), the "
                        "envs/estimate defaults and the degrade field fault's assumed numbers",
        )
        return 0
    except Exception as exc:
        writer.summary("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
