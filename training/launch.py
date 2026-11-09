"""Start one run with the project's rules enforced.

    python -m training.launch --config training/configs/selfcheck.yaml --smoke --device cpu

What it does, in order:

1. Loads the YAML config (``name``, ``entrypoint``, ``seed``, ``expected_hours``, ``gpus``,
   ``params``, and an optional ``smoke`` block of param overrides).
2. Refuses more than 1 GPU, or an expected run over 1 hour, without ``--approved``.
3. With ``--device cuda``, reads ``nvidia-smi`` and refuses unless enough GPUs are free
   (little memory used, low utilisation, no compute processes). ``--device cpu`` skips this.
   ``--gpu N`` pins the run to GPU N instead (still refused if N is busy); the queue runner
   (``training/queue.py``) uses it so two runs started together never pick the same card.
4. Refuses a real (non-smoke) run on a dirty git tree unless ``--allow-dirty``, because the
   log line must name the code that ran. Lines appended to ``experiments/LOG.md`` do not
   count: every real run adds one, and a chain of runs must not refuse its own second part.
   ``--set params.KEY=VALUE`` (or ``--set seed=N``) overrides one value, parsed as YAML,
   after the smoke block; the overrides are written into ``config.yaml`` and the log line.
   ``--sim torch|isaac`` overrides the config's ``params.sim`` (refused for a config that
   does not choose a simulator, and Isaac with ``--device cpu``, since Isaac Sim needs the GPU).
5. Creates ``runs/<time>_<name>_s<seed>_<commit>[-dirty][-smoke]/`` with the resolved
   ``config.yaml``, runs the entrypoint, tees its output to ``stdout.log``, and enforces a
   timeout.
6. Finalises ``summary.json`` (``ok``, ``failed`` or ``timeout``) and, for real runs only,
   appends one line to ``experiments/LOG.md``.

Exit codes: 0 ok, 1 run failed or timed out, 2 refused by a rule, 3 no free GPU.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.runlib import load_yaml, read_summary, write_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "experiments" / "LOG.md"
DEFAULT_RUNS = REPO_ROOT / "runs"

MAX_GPUS_WITHOUT_APPROVAL = 1
MAX_HOURS_WITHOUT_APPROVAL = 1.0
SMOKE_TIMEOUT_HOURS = 0.25
# A GPU counts as free when another job is clearly not using it.
FREE_MEMORY_USED_MIB = 1024
FREE_UTILISATION_PCT = 10

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_NO_GPU = 0, 1, 2, 3


class Refused(Exception):
    def __init__(self, message: str, code: int = EXIT_REFUSED):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- GPUs


@dataclass
class Gpu:
    index: int
    uuid: str
    memory_used_mib: int
    memory_total_mib: int
    utilisation_pct: int
    busy_processes: int = 0

    @property
    def free(self) -> bool:
        return (
            self.busy_processes == 0
            and self.memory_used_mib < FREE_MEMORY_USED_MIB
            and self.utilisation_pct < FREE_UTILISATION_PCT
        )


def parse_gpus(query_gpu_csv: str, query_apps_csv: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu``
    and ``--query-compute-apps=gpu_uuid,pid`` output (csv, noheader, nounits)."""
    busy: dict[str, int] = {}
    for line in query_apps_csv.strip().splitlines():
        if line.strip():
            uuid = line.split(",")[0].strip()
            busy[uuid] = busy.get(uuid, 0) + 1
    gpus = []
    for line in query_gpu_csv.strip().splitlines():
        if not line.strip():
            continue
        index, uuid, used, total, util = (f.strip() for f in line.split(","))
        gpus.append(
            Gpu(
                index=int(index),
                uuid=uuid,
                memory_used_mib=int(float(used)),
                memory_total_mib=int(float(total)),
                # Some GPUs report "[N/A]" utilisation; treat that as busy to be safe.
                utilisation_pct=int(float(util)) if util.replace(".", "").isdigit() else 100,
                busy_processes=busy.get(uuid, 0),
            )
        )
    return gpus


def query_gpus(nvidia_smi: str = "nvidia-smi") -> list[Gpu]:
    if shutil.which(nvidia_smi) is None:
        raise Refused(
            "nvidia-smi not found, so GPU use cannot be checked. "
            "Use --device cpu on a machine without an NVIDIA GPU.",
            EXIT_NO_GPU,
        )
    fmt = ["--format=csv,noheader,nounits"]
    gpu_q = [nvidia_smi, "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", *fmt]
    app_q = [nvidia_smi, "--query-compute-apps=gpu_uuid,pid", *fmt]
    try:
        gpu_out = subprocess.run(gpu_q, capture_output=True, text=True, check=True, timeout=30)
        app_out = subprocess.run(app_q, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise Refused(f"nvidia-smi failed: {e}", EXIT_NO_GPU) from e
    return parse_gpus(gpu_out.stdout, app_out.stdout)


def pick_free_gpus(gpus: list[Gpu], count: int) -> list[Gpu]:
    free = [g for g in gpus if g.free]
    if len(free) < count:
        table = "\n".join(
            f"  GPU {g.index}: {g.memory_used_mib}/{g.memory_total_mib} MiB, "
            f"{g.utilisation_pct}% util, {g.busy_processes} processes"
            + ("  (free)" if g.free else "  (busy)")
            for g in gpus
        )
        raise Refused(f"Need {count} free GPU(s), found {len(free)}:\n{table}", EXIT_NO_GPU)
    return free[:count]


def pin_gpus(gpus: list[Gpu], wanted: str, count: int) -> list[Gpu]:
    """``--gpu 2`` (or ``2,3``): exactly those GPUs, each of which must be free."""
    try:
        indices = [int(x) for x in wanted.split(",") if x.strip()]
    except ValueError as e:
        raise Refused(f"--gpu {wanted!r}: expected GPU indices like 2 or 2,3") from e
    if len(indices) != count:
        raise Refused(f"--gpu {wanted}: the config asks for {count} GPU(s)")
    by_index = {g.index: g for g in gpus}
    chosen = []
    for i in indices:
        g = by_index.get(i)
        if g is None:
            raise Refused(f"--gpu {i}: no such GPU", EXIT_NO_GPU)
        if not g.free:
            raise Refused(f"--gpu {i}: busy ({g.memory_used_mib} MiB used, "
                          f"{g.utilisation_pct}% util, {g.busy_processes} processes)", EXIT_NO_GPU)
        chosen.append(g)
    return chosen


# ---------------------------------------------------------------- config and rules


REQUIRED_KEYS = ("name", "entrypoint", "seed", "expected_hours")


def resolve_config(raw: dict[str, Any], smoke: bool) -> dict[str, Any]:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise Refused(f"config is missing required keys: {', '.join(missing)}")
    cfg = copy.deepcopy(raw)
    cfg.setdefault("gpus", 1)
    cfg.setdefault("params", {})
    smoke_overrides = cfg.pop("smoke", None) or {}
    if smoke:
        cfg["params"].update(smoke_overrides)
    return cfg


SIMS = ("torch", "isaac")


def apply_sim(cfg: dict[str, Any], sim: str | None, device: str) -> None:
    """``--sim``: run the same recipe on another simulator backend (envs/sims)."""
    if sim is None:
        return
    if "sim" not in cfg["params"]:
        raise Refused(f"--sim {sim}: this config has no params.sim, so it does not use a simulator")
    if sim == "isaac" and device != "cuda":
        raise Refused("--sim isaac needs --device cuda: Isaac Sim does not start without the GPU")
    cfg["params"]["sim"] = sim


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> list[str]:
    """``--set``: ``params.a.b=VALUE`` or ``seed=N``, the value parsed as YAML. Only params and
    the seed can change, so the hour and GPU rules always read the committed config."""
    applied = []
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise Refused(f"--set {item!r}: expected KEY=VALUE")
        value = yaml.safe_load(raw) if raw.strip() else None
        if key == "seed":
            cfg["seed"] = int(value)
        elif key.startswith("params.") and len(key) > len("params."):
            node = cfg["params"]
            *parents, leaf = key.split(".")[1:]
            for name in parents:
                # A number steps into a list that is already there (``params.controllers.1
                # .policy``: the queue points a check at the policy a training job exported);
                # anything else makes or enters a mapping.
                if isinstance(node, list) and name.isdigit():
                    node = node[int(name)]
                    continue
                if not isinstance(node.get(name), dict | list):
                    node[name] = {}
                node = node[name]
            if isinstance(node, list):
                if not leaf.isdigit() or int(leaf) >= len(node):
                    raise Refused(f"--set {key}: {leaf!r} is not an index into a list of "
                                  f"{len(node)}")
                node[int(leaf)] = value
            else:
                node[leaf] = value
        else:
            raise Refused(f"--set {key}: only params.* and seed can be overridden")
        applied.append(f"{key}={raw.strip()}")
    return applied


def check_rules(cfg: dict[str, Any], *, device: str, smoke: bool, approved: bool) -> None:
    gpus = int(cfg["gpus"]) if device == "cuda" else 0
    if smoke and gpus > 1:
        raise Refused("smoke runs use at most 1 GPU; set gpus: 1")
    if gpus > MAX_GPUS_WITHOUT_APPROVAL and not approved:
        raise Refused(
            f"config asks for {gpus} GPUs; more than {MAX_GPUS_WITHOUT_APPROVAL} needs Zain's "
            "go-ahead first. Re-run with --approved once agreed."
        )
    hours = float(cfg["expected_hours"])
    if not smoke and hours > MAX_HOURS_WITHOUT_APPROVAL and not approved:
        raise Refused(
            f"expected_hours is {hours}; runs over {MAX_HOURS_WITHOUT_APPROVAL} h need Zain's "
            "go-ahead first. Re-run with --approved once agreed."
        )


def timeout_seconds(cfg: dict[str, Any], smoke: bool) -> float:
    if smoke:
        return SMOKE_TIMEOUT_HOURS * 3600
    hours = float(cfg.get("timeout_hours", 2 * float(cfg["expected_hours"])))
    return max(hours, SMOKE_TIMEOUT_HOURS) * 3600


# ---------------------------------------------------------------- git and run folder


# Real runs append to it, so a tree dirty only there still runs the committed code.
APPEND_ONLY_PATHS = ("experiments/LOG.md",)


def is_dirty(porcelain: str) -> bool:
    """True when ``git status --porcelain`` shows anything but appended log lines."""
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:].strip()
        if status.strip() == "M" and path in APPEND_ONLY_PATHS:
            continue
        return True
    return False


def git_state(repo: Path = REPO_ROOT) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "nogit", True
    return commit, is_dirty(status)


def make_run_dir(runs_root: Path, cfg: dict[str, Any], commit: str, dirty: bool, smoke: bool,
                 now: dt.datetime) -> Path:
    name = (
        f"{now:%Y%m%d-%H%M%S}_{cfg['name']}_s{cfg['seed']}_{commit[:7]}"
        + ("-dirty" if dirty else "")
        + ("-smoke" if smoke else "")
    )
    run_dir = runs_root / name
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = runs_root / f"{name}-{n}"
    run_dir.mkdir(parents=True)
    return run_dir


# ---------------------------------------------------------------- running


def run_entrypoint(cmd: list[str], env: dict[str, str], log_path: Path, timeout_s: float,
                   echo: bool = True) -> tuple[int | None, bool]:
    """Run ``cmd``, tee its output to ``log_path``. Returns (returncode, timed_out)."""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        reader.join(timeout=10)
    return proc.returncode, timed_out


def format_headline(headline: dict[str, Any] | None) -> str:
    if not headline:
        return "-"
    parts = []
    for k, v in headline.items():
        parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts)


def append_log_line(log_path: Path, *, when: dt.datetime, run_dir: Path, commit: str,
                    dirty: bool, config_path: Path, device: str, summary: dict[str, Any],
                    sim: str | None = None, overrides: list[str] | None = None) -> None:
    def cell(text: str) -> str:
        return str(text).replace("|", "/").replace("\n", " ")

    status = summary.get("status", "failed")
    result = format_headline(summary.get("headline"))
    if status != "ok" and summary.get("error"):
        result = f"{result}; error: {summary['error']}" if result != "-" else summary["error"]
    row = [
        when.strftime("%Y-%m-%d %H:%M"),
        f"`{run_dir.name}`",
        f"`{commit[:7]}`" + (" (dirty)" if dirty else ""),
        f"`{config_path.as_posix()}`" + (f" --sim {sim}" if sim else "")
        + "".join(f" --set {o}" for o in overrides or []),
        device,
        status.upper() if status != "ok" else "ok",
        result,
        summary.get("vs_baseline", "none yet"),
    ]
    with open(log_path, "a") as f:
        f.write("| " + " | ".join(cell(c) for c in row) + " |\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="a few iterations; never logged")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--sim", choices=SIMS, default=None,
                   help="simulator backend, overriding the config's params.sim")
    p.add_argument("--gpu", default=None,
                   help="pin to these GPU indices (comma list, one per config gpus) if free")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override params.KEY (dotted) or seed, value parsed as YAML; repeatable")
    p.add_argument("--approved", action="store_true",
                   help="Zain has agreed to a run over 1 hour or on more than 1 GPU")
    p.add_argument("--allow-dirty", action="store_true",
                   help="allow a real run with uncommitted changes (logged as dirty)")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--nvidia-smi", default="nvidia-smi", help=argparse.SUPPRESS)
    p.add_argument("--quiet", action="store_true", help="do not echo the run's output")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = resolve_config(load_yaml(args.config), args.smoke)
        apply_sim(cfg, args.sim, args.device)
        overrides = apply_overrides(cfg, args.overrides)
        check_rules(cfg, device=args.device, smoke=args.smoke, approved=args.approved)
        commit, dirty = git_state()
        if dirty and not args.smoke and not args.allow_dirty:
            raise Refused(
                "uncommitted changes: commit first so the LOG.md line names the code that ran, "
                "or pass --allow-dirty"
            )
        env = dict(os.environ)
        if args.device == "cuda":
            gpus = query_gpus(args.nvidia_smi)
            chosen = (pin_gpus(gpus, args.gpu, int(cfg["gpus"])) if args.gpu is not None
                      else pick_free_gpus(gpus, int(cfg["gpus"])))
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g.index) for g in chosen)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
    except Refused as e:
        print(f"launch refused: {e}", file=sys.stderr)
        return e.code
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"launch refused: cannot read config: {e}", file=sys.stderr)
        return EXIT_REFUSED

    started = dt.datetime.now(dt.timezone.utc)
    run_dir = make_run_dir(args.runs_dir, cfg, commit, dirty, args.smoke, started)
    resolved = dict(cfg)
    resolved["_run"] = {
        "config_file": args.config.as_posix(),
        "commit": commit,
        "dirty": dirty,
        "smoke": args.smoke,
        "device": args.device,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "overrides": overrides,
        "started_utc": started.isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
    config_out = run_dir / "config.yaml"
    config_out.write_text(yaml.safe_dump(resolved, sort_keys=False))

    cmd = [sys.executable, "-m", cfg["entrypoint"], "--run-dir", str(run_dir),
           "--config", str(config_out), "--device", args.device]
    if args.smoke:
        cmd.append("--smoke")
    print(f"launch: {run_dir}", flush=True)
    t0 = time.monotonic()
    returncode, timed_out = run_entrypoint(
        cmd, env, run_dir / "stdout.log", timeout_seconds(cfg, args.smoke), echo=not args.quiet
    )
    duration = round(time.monotonic() - t0, 1)

    summary = read_summary(run_dir) or {}
    if timed_out:
        status, error = "timeout", f"killed after {timeout_seconds(cfg, args.smoke):.0f} s"
    elif returncode != 0:
        status = "failed"
        error = summary.get("error") or f"exit code {returncode}; see stdout.log"
    elif summary.get("status") != "ok":
        status = "failed"
        error = summary.get("error") or "entrypoint exited 0 but wrote no ok summary.json"
    else:
        status, error = "ok", None
    write_summary(run_dir, status=status, error=error, returncode=returncode,
                  duration_s=duration, smoke=args.smoke)
    summary = read_summary(run_dir) or {}

    if not args.smoke:
        append_log_line(args.log, when=started, run_dir=run_dir, commit=commit, dirty=dirty,
                        config_path=args.config, device=args.device, summary=summary,
                        sim=args.sim, overrides=overrides)
    print(f"launch: {status} in {duration} s -> {run_dir / 'summary.json'}", flush=True)
    return EXIT_OK if status == "ok" else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
