"""Run a queue of jobs on the GPU server: every free GPU busy, SITL flights on the other cores.

    python -m training.queue training/queues/server_week1.yaml --dry-run --gpus 4  # the plan
    python -m training.queue training/queues/server_week1.yaml --smoke            # every smoke
    python -m training.queue training/queues/server_week1.yaml                   # for real
    python -m training.queue training/queues/server_week1.yaml --approved        # with Zain's yes

The queue file (``training/queues/*.yaml``, docs/gpu-server-plan.md) lists jobs in priority
order. Each job is either a ``launch`` config (run through ``training/launch.py``, so the
project's rules, the run folder and the ``experiments/LOG.md`` line come with it) or a shell
``cmd`` (SITL batches through ``eval.baselines.run_baseline``, Isaac Lab through
``training/isaac_run.sh``, the machine report).

How jobs are placed:

- ``gpu: true`` jobs get one free GPU each (``nvidia-smi``, the same test as launch.py),
  pinned with ``launch --gpu N`` (``{gpu}`` in a ``cmd``), so two jobs never share a card,
  and ``gpu_job_cpus`` CPU cores of their own.
- Every other job runs on the CPU lane, one at a time, on all the cores the GPU jobs do not
  hold, minus ``reserve_cpus``. A SITL batch with ``--jobs auto`` fills that lane by itself
  (run_baseline reads the cores it was given), and one at a time keeps container names
  apart.
- A job starts once everything in its ``after`` list finished ok; jobs earlier in the file
  go first when both could start. A failed job blocks the jobs after it, not the rest.
- ``chain: N`` runs a launch config N times, each part resuming the previous part's
  ``policy.pt`` (or, after a timeout, its newest ``tb/model_<it>.pt``), so a long training is
  a chain of runs under the hour rule. ``chain_set`` overrides apply from the second part
  (for example ``params.init_actor_onnx: null``: a resumed part must not re-initialise).
- ``{run:ID}`` in a ``set`` value or a ``cmd`` becomes the run folder of job ID's last good
  part (``runs/...``), so a scoring job can name the policy a training job just exported.
- ``needs_approval: true``, or ``hours`` over 1, means Zain must agree first: the job is
  skipped unless the queue runs with ``--approved`` (which launch.py then also gets).
- ``enabled: false`` lists a job without running it (it waits on a result, such as a fine-
  tune that is only worth training if a check fails).

``--smoke`` runs each launch job once with ``--smoke`` (no chain, never logged; ``set`` values
that name another job's run, or replace a param the config's smoke block sets such as
``params.num_envs``, are left out, since the smoke block stands on its own)
and each cmd job's ``smoke_cmd`` if it has one; ``--device cpu`` does the same without GPUs
(the cloud container's check; GPU-only cmd jobs are skipped). State lives in
``runs/queue_<name>[-smoke]/state.json``; re-running the same queue skips what finished ok
and continues a chain from its last good part, so a stopped queue resumes where it was.

Exit code: 0 when every job that could run finished ok, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training import launch

REPO_ROOT = launch.REPO_ROOT
DEFAULT_RUNS = launch.DEFAULT_RUNS
POLL_S = 5.0
START_GAP_S = 3.0  # between two starts: run folders and container names are time-stamped
RUN_DIR_LINE = re.compile(r"^launch: (\S+)$")


# ---------------------------------------------------------------- the queue file


@dataclass
class Job:
    id: str
    gpu: bool = False
    launch: str | None = None
    cmd: str | None = None
    smoke_cmd: str | None = None
    set: dict[str, Any] = field(default_factory=dict)
    chain: int = 1
    chain_set: dict[str, Any] = field(default_factory=dict)
    after: list[str] = field(default_factory=list)
    hours: float = 0.5
    needs_approval: bool = False
    enabled: bool = True
    why: str = ""

    @property
    def requires_approval(self) -> bool:
        return self.needs_approval or self.hours / max(self.chain, 1) > 1.0


@dataclass
class Queue:
    name: str
    jobs: list[Job]
    reserve_cpus: int = 4
    gpu_job_cpus: int = 4


def load_queue(path: Path) -> Queue:
    raw = yaml.safe_load(path.read_text()) or {}
    jobs = []
    for item in raw.get("jobs") or []:
        unknown = set(item) - set(Job.__dataclass_fields__)
        if unknown:
            raise ValueError(f"job {item.get('id')}: unknown keys {sorted(unknown)}")
        jobs.append(Job(**item))
    ids = [j.id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("job ids must be unique")
    for j in jobs:
        if (j.launch is None) == (j.cmd is None):
            raise ValueError(f"job {j.id}: give exactly one of launch or cmd")
        if j.chain < 1 or (j.chain > 1 and j.launch is None):
            raise ValueError(f"job {j.id}: chain needs a launch config and must be 1 or more")
        missing = [a for a in j.after if a not in ids]
        if missing:
            raise ValueError(f"job {j.id}: after names unknown jobs {missing}")
        if any(ids.index(a) > ids.index(j.id) for a in j.after):
            raise ValueError(f"job {j.id}: after must name earlier jobs")
        if j.launch is not None and not (REPO_ROOT / j.launch).exists():
            raise ValueError(f"job {j.id}: no such config {j.launch}")
    return Queue(name=str(raw.get("name") or path.stem), jobs=jobs,
                 reserve_cpus=int(raw.get("reserve_cpus", 4)),
                 gpu_job_cpus=int(raw.get("gpu_job_cpus", 4)))


# ---------------------------------------------------------------- choosing what runs next


def runnable(job: Job, *, smoke: bool, device: str, approved: bool) -> str | None:
    """None if the job may run in this mode, else why it is skipped."""
    if not job.enabled:
        return "disabled (waits on a result; switch it on in the queue file)"
    if smoke and job.cmd is not None and job.smoke_cmd is None:
        return "no smoke_cmd"
    if device == "cpu" and job.gpu and job.cmd is not None:
        return "needs a GPU"
    if job.requires_approval and not approved and not smoke:
        return "needs Zain's go-ahead (--approved)"
    return None


def next_jobs(jobs: list[Job], status: dict[str, str], free_gpu_lanes: int,
              cpu_lane_free: bool) -> list[Job]:
    """The jobs to start now, in file order: deps done ok, a lane free for each."""
    chosen = []
    for j in jobs:
        if status.get(j.id) != "pending":
            continue
        if any(status.get(a) != "ok" for a in j.after):
            continue
        if j.gpu and free_gpu_lanes > 0:
            chosen.append(j)
            free_gpu_lanes -= 1
        elif not j.gpu and cpu_lane_free:
            chosen.append(j)
            cpu_lane_free = False
    return chosen


def block_dependents(jobs: list[Job], status: dict[str, str]) -> None:
    """Pending jobs whose dependency failed or was skipped can never run."""
    changed = True
    while changed:
        changed = False
        for j in jobs:
            if status.get(j.id) == "pending" and any(
                    status.get(a) in ("failed", "blocked", "skipped") for a in j.after):
                status[j.id] = "blocked"
                changed = True


def simulate(queue: Queue, gpus: int, *, approved: bool, smoke: bool = False,
             device: str = "cuda") -> list[dict[str, Any]]:
    """The schedule the queue would follow if every job took its ``hours``: one row per job
    with its lane and start/end in hours from the queue's start."""
    status = {}
    for j in queue.jobs:
        status[j.id] = "pending" if runnable(j, smoke=smoke, device=device,
                                             approved=approved) is None else "skipped"
    block_dependents(queue.jobs, status)
    lanes_free_at = [0.0] * gpus
    cpu_free_at = 0.0
    done_at: dict[str, float] = {}
    rows = []
    for j in queue.jobs:  # file order is priority order; deps are earlier in the file
        if status[j.id] != "pending":
            rows.append({"job": j, "lane": "-", "start": None, "end": None,
                         "status": status[j.id]})
            continue
        ready = max([done_at[a] for a in j.after], default=0.0)
        if j.gpu:
            if gpus == 0:
                rows.append({"job": j, "lane": "-", "start": None, "end": None,
                             "status": "no GPU"})
                continue
            k = min(range(gpus), key=lambda i: max(lanes_free_at[i], ready))
            start = max(lanes_free_at[k], ready)
            lanes_free_at[k] = start + j.hours
            lane = f"GPU {k}"
        else:
            start = max(cpu_free_at, ready)
            cpu_free_at = start + j.hours
            lane = "CPU"
        done_at[j.id] = start + j.hours
        rows.append({"job": j, "lane": lane, "start": start, "end": start + j.hours,
                     "status": "planned"})
    return rows


def plan_table(queue: Queue, rows: list[dict[str, Any]], gpus: int) -> str:
    lines = [f"Queue {queue.name}: {gpus} GPU lane(s) + 1 CPU lane", "",
             "| Job | Lane | Start h | End h | Parts | Zain's yes | Status | Why |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        j = r["job"]
        when = (f"{r['start']:.1f}", f"{r['end']:.1f}") if r["start"] is not None else ("-", "-")
        lines.append(f"| {j.id} | {r['lane']} | {when[0]} | {when[1]} | {j.chain} | "
                     f"{'yes' if j.requires_approval else '-'} | {r['status']} | {j.why} |")
    ends = [r["end"] for r in rows if r["end"] is not None]
    gpu_hours = sum(r["job"].hours for r in rows if r["end"] is not None and r["job"].gpu)
    if ends:
        span = max(ends)
        busy = gpu_hours / (span * gpus) if span and gpus else 0.0
        lines += ["", f"All planned jobs done after about {span:.1f} h; "
                      f"GPUs busy {100 * busy:.0f} % of that time ({gpu_hours:.1f} GPU-hours)."]
    return "\n".join(lines)


# ---------------------------------------------------------------- running


def free_gpu_indices(nvidia_smi: str) -> list[int]:
    return [g.index for g in launch.query_gpus(nvidia_smi) if g.free]


def cpu_blocks(total: list[int], gpu_lanes: int, per_gpu: int, reserve: int
               ) -> tuple[list[list[int]], list[int]]:
    """Fixed CPU cores for each GPU lane, then the CPU lane's cores (minus the reserve)."""
    usable = total[reserve:] if len(total) > reserve else total
    blocks = []
    for i in range(gpu_lanes):
        block = usable[i * per_gpu:(i + 1) * per_gpu]
        blocks.append(block or usable[:1])
    rest = usable[gpu_lanes * per_gpu:] or usable[-max(1, len(usable) // 4):]
    return blocks, rest


def run_path(run_dir: str) -> Path:
    """A run folder as recorded in the state: relative to the repo, or absolute."""
    p = Path(run_dir)
    return p if p.is_absolute() else REPO_ROOT / p


def resume_point(run_dir: Path) -> Path | None:
    """Where the next part of a chain resumes: the run's ``policy.pt``, or after a timeout
    its newest periodic checkpoint."""
    final = run_dir / "policy.pt"
    if final.exists():
        return final
    ckpts = sorted((run_dir / "tb").glob("model_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit()
                   else -1)
    return ckpts[-1] if ckpts else None


RUN_REF = re.compile(r"\{run:([A-Za-z0-9_.-]+)\}")


def expand_runs(text: str, run_dirs: dict[str, str]) -> str:
    """``{run:ID}`` -> that job's last good run folder; a job without one is an error."""
    def sub(m: re.Match) -> str:
        jid = m.group(1)
        if jid not in run_dirs:
            raise KeyError(f"{{run:{jid}}}: job {jid} has no finished run folder")
        return run_dirs[jid]
    return RUN_REF.sub(sub, text)


def set_args(values: dict[str, Any]) -> list[str]:
    out = []
    for k, v in values.items():
        # One line whatever its length: the override goes into argv and the LOG.md row as is.
        dumped = yaml.safe_dump(v, default_flow_style=True, width=float("inf")).strip()
        out += ["--set", f"{k}={dumped}"]
        if out[-1].endswith("\n..."):
            out[-1] = out[-1][:-4]
    return out


def launch_command(job: Job, *, part: int, smoke: bool, device: str, gpu: int | None,
                   approved: bool, resume: Path | None,
                   run_dirs: dict[str, str] | None = None,
                   passthrough: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "training.launch", "--config", str(job.launch),
           "--device", device, "--quiet", *(passthrough or [])]
    if gpu is not None and device == "cuda":
        cmd += ["--gpu", str(gpu)]
    if smoke:
        cmd.append("--smoke")
    if approved and job.requires_approval:
        cmd.append("--approved")
    values = dict(job.set)
    if part > 1:
        values.update(job.chain_set)
        values["params.resume"] = str(resume)
    if smoke:  # the smoke block replaces what points at files; a 3-iteration policy is no test
        values = {k: v for k, v in values.items()
                  if not (isinstance(v, str) and RUN_REF.search(v))}
        # and its sizes stand: a job that asks for 2048 drones still smokes on the block's 8
        raw = yaml.safe_load((REPO_ROOT / job.launch).read_text()) or {}
        sized = {f"params.{k}" for k in raw.get("smoke") or {}}
        values = {k: v for k, v in values.items() if k not in sized}
    values = {k: expand_runs(v, run_dirs or {}) if isinstance(v, str) else v
              for k, v in values.items()}
    return cmd + set_args(values)


def shell_command(job: Job, *, smoke: bool, gpu: int | None,
                  run_dirs: dict[str, str] | None = None) -> list[str]:
    text = job.smoke_cmd if smoke else job.cmd
    assert text is not None
    text = text.replace("{gpu}", "" if gpu is None else str(gpu))
    return ["bash", "-c", expand_runs(text, run_dirs or {})]


@dataclass
class Running:
    job: Job
    proc: subprocess.Popen
    log: Path
    part: int
    gpu: int | None
    started: float


class Runner:
    def __init__(self, queue: Queue, *, smoke: bool, device: str, approved: bool,
                 runs_dir: Path, nvidia_smi: str = "nvidia-smi", max_gpus: int | None = None,
                 only: list[str] | None = None, retry_failed: bool = False,
                 log: Path | None = None, allow_dirty: bool = False):
        self.q, self.smoke, self.device, self.approved = queue, smoke, device, approved
        self.nvidia_smi = nvidia_smi
        # launch.py gets the same run folder, log and nvidia-smi as the queue.
        self.passthrough = ["--runs-dir", str(runs_dir), "--nvidia-smi", nvidia_smi]
        if log is not None:
            self.passthrough += ["--log", str(log)]
        if allow_dirty:
            self.passthrough.append("--allow-dirty")
        self.dir = runs_dir / f"queue_{queue.name}{'-smoke' if smoke else ''}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = self.load_state()
        self.jobs = [j for j in queue.jobs if not only or j.id in only]
        for j in self.jobs:
            s = self.state.setdefault(j.id, {"status": "pending", "parts": []})
            if s["status"] in ("running", "blocked", "skipped") or (
                    retry_failed and s["status"] == "failed"):
                s["status"] = "pending"  # a stopped queue picks these up again
            why = runnable(j, smoke=smoke, device=device, approved=approved)
            if s["status"] == "pending" and why:
                s["status"], s["why"] = "skipped", why
        if only:  # jobs outside --only count as done for ordering
            for j in queue.jobs:
                if j.id not in only:
                    self.state.setdefault(j.id, {"status": "ok", "parts": []})
        if device == "cuda":
            gpus = free_gpu_indices(nvidia_smi)
        else:
            gpus = list(range(1))  # one virtual lane: GPU jobs run one at a time on the CPU
        if max_gpus is not None:
            gpus = gpus[:max_gpus]
        self.gpus = gpus
        cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else \
            list(range(os.cpu_count() or 1))
        self.gpu_cpus, self.cpu_lane = cpu_blocks(cores, len(gpus), queue.gpu_job_cpus,
                                                  queue.reserve_cpus)
        self.running: list[Running] = []

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    def status(self) -> dict[str, str]:
        return {jid: s["status"] for jid, s in self.state.items()}

    def run_dirs(self) -> dict[str, str]:
        """Each job's last good run folder, for ``{run:ID}``."""
        out = {}
        for jid, s in self.state.items():
            good = [p["run_dir"] for p in s.get("parts", [])
                    if p.get("status") == "ok" and p.get("run_dir")]
            if good:
                out[jid] = good[-1]
        return out

    def say(self, text: str) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
        line = f"[queue {stamp}] {text}"
        print(line, flush=True)
        with open(self.dir / "queue.log", "a") as f:
            f.write(line + "\n")

    def start(self, job: Job, gpu_lane: int | None) -> None:
        s = self.state[job.id]
        done_parts = [p for p in s["parts"] if p.get("status") == "ok"]
        part = len(done_parts) + 1
        gpu = self.gpus[gpu_lane] if gpu_lane is not None and self.device == "cuda" else None
        if job.launch is not None:
            resume = None
            if part > 1:
                resume = resume_point(run_path(done_parts[-1]["run_dir"]))
                if resume is None:
                    s["status"], s["why"] = "failed", "previous part left no checkpoint"
                    return
            try:
                cmd = launch_command(job, part=part, smoke=self.smoke, device=self.device,
                                     gpu=gpu, approved=self.approved, resume=resume,
                                     run_dirs=self.run_dirs(), passthrough=self.passthrough)
            except KeyError as e:
                s["status"], s["why"] = "failed", str(e)
                self.say(f"{job.id}: {e}")
                self.save_state()
                return
        else:
            try:
                cmd = shell_command(job, smoke=self.smoke, gpu=gpu, run_dirs=self.run_dirs())
            except KeyError as e:
                s["status"], s["why"] = "failed", str(e)
                self.say(f"{job.id}: {e}")
                self.save_state()
                return
        cores = self.gpu_cpus[gpu_lane] if gpu_lane is not None else self.cpu_lane
        log = self.dir / "logs" / f"{job.id}_p{part}.log"
        env = dict(os.environ)
        if gpu is not None:
            env["QUEUE_GPU"] = str(gpu)
        with open(log, "w") as out:
            out.write("$ " + shlex.join(cmd) + "\n")
            out.flush()
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                preexec_fn=(lambda: os.sched_setaffinity(0, cores))
                if hasattr(os, "sched_setaffinity") else None)
        s["status"] = "running"
        s.setdefault("gpu_lane", gpu_lane)
        self.running.append(Running(job, proc, log, part, gpu_lane, time.monotonic()))
        where = f"GPU {gpu}" if gpu is not None else (
            f"GPU lane {gpu_lane}" if gpu_lane is not None else "CPU lane")
        shown = log.relative_to(REPO_ROOT) if log.is_relative_to(REPO_ROOT) else log
        self.say(f"start {job.id} part {part}/{1 if self.smoke else job.chain} on {where} "
                 f"({len(cores)} cores) -> {shown}")
        self.save_state()

    def finish(self, r: Running) -> None:
        s = self.state[r.job.id]
        code = r.proc.returncode
        run_dir = None
        for line in r.log.read_text(errors="replace").splitlines():
            m = RUN_DIR_LINE.match(line.strip())
            if m:
                run_dir = m.group(1)
                break
        if run_dir is not None:
            p = Path(run_dir)
            run_dir = str(p.relative_to(REPO_ROOT)) if p.is_absolute() and p.is_relative_to(
                REPO_ROOT) else str(p)
        part = {"part": r.part, "returncode": code, "run_dir": run_dir,
                "minutes": round((time.monotonic() - r.started) / 60, 1), "log": str(r.log)}
        # A timed-out chain part still leaves checkpoints; the next part resumes from them.
        timed_out = run_dir is not None and r.job.chain > 1 and \
            (launch.read_summary(run_path(run_dir)) or {}).get("status") == "timeout" and \
            resume_point(run_path(run_dir)) is not None
        part["status"] = "ok" if code == 0 or timed_out else "failed"
        s["parts"].append(part)
        parts_needed = 1 if self.smoke else r.job.chain
        ok_parts = sum(1 for p in s["parts"] if p["status"] == "ok")
        if part["status"] == "failed":
            s["status"] = "failed"
        elif ok_parts >= parts_needed:
            s["status"] = "ok"
        else:
            s["status"] = "pending"  # the next part goes on the same lane next round
        self.say(f"{r.job.id} part {r.part}: {part['status']} (exit {code}, "
                 f"{part['minutes']} min){' ' + run_dir if run_dir else ''}")
        self.save_state()

    def run(self) -> int:
        self.say(f"{len(self.jobs)} jobs, GPU lanes {self.gpus if self.device == 'cuda' else 'cpu'}"
                 f", CPU lane {len(self.cpu_lane)} cores, smoke={self.smoke}")
        for j in self.jobs:
            if self.state[j.id]["status"] == "skipped":
                self.say(f"skip {j.id}: {self.state[j.id].get('why')}")
        try:
            while True:
                for r in [r for r in self.running if r.proc.poll() is not None]:
                    self.running.remove(r)
                    self.finish(r)
                status = self.status()
                block_dependents(self.jobs, status)
                for jid, st in status.items():
                    if st == "blocked" and self.state[jid]["status"] != "blocked":
                        self.state[jid]["status"] = "blocked"
                        self.say(f"blocked {jid}: a job it waits on did not finish ok")
                busy_lanes = {r.gpu for r in self.running if r.gpu is not None}
                # A chain keeps its lane between parts, so its checkpoints stay on one GPU.
                free_lanes = [i for i in range(len(self.gpus)) if i not in busy_lanes]
                cpu_free = not any(r.gpu is None for r in self.running)
                started = False
                for j in next_jobs(self.jobs, self.status(), len(free_lanes), cpu_free):
                    lane = free_lanes.pop(0) if j.gpu else None
                    self.start(j, lane)
                    started = True
                    time.sleep(START_GAP_S)
                if not self.running and not started:
                    break
                time.sleep(POLL_S)
        except KeyboardInterrupt:
            self.say("stopping: terminating running jobs (re-run to resume)")
            for r in self.running:
                r.proc.terminate()
            for r in self.running:
                r.proc.wait()
                self.state[r.job.id]["status"] = "pending"
            self.save_state()
            return 1
        self.save_state()
        lines = ["", "| Job | Status | Parts | Last run |", "| --- | --- | --- | --- |"]
        for j in self.jobs:
            s = self.state[j.id]
            last = s["parts"][-1]["run_dir"] if s["parts"] else s.get("why", "-")
            lines.append(f"| {j.id} | {s['status']} | {len(s['parts'])} | {last} |")
        table = "\n".join(lines)
        (self.dir / "summary.md").write_text(table.strip() + "\n")
        print(table, flush=True)
        bad = [j.id for j in self.jobs if self.state[j.id]["status"] in ("failed", "blocked")]
        return 1 if bad else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("queue", type=Path)
    p.add_argument("--dry-run", action="store_true", help="print the planned schedule only")
    p.add_argument("--gpus", type=int, default=None,
                   help="--dry-run: GPU lanes to plan for (default: the free GPUs now, or 4)")
    p.add_argument("--smoke", action="store_true", help="every job's smoke version, no chains")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--approved", action="store_true",
                   help="Zain agreed to the jobs marked as needing it (passed on to launch.py)")
    p.add_argument("--max-gpus", type=int, default=None, help="use at most this many GPUs")
    p.add_argument("--only", default=None, help="comma list of job ids to run")
    p.add_argument("--retry-failed", action="store_true", help="run failed jobs again")
    p.add_argument("--allow-dirty", action="store_true",
                   help="passed to launch.py: real runs on a tree with uncommitted changes")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--log", type=Path, default=None, help="passed to launch.py")
    p.add_argument("--nvidia-smi", default="nvidia-smi", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = load_queue(args.queue)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        print(f"queue: cannot read {args.queue}: {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        gpus = args.gpus
        if gpus is None:
            try:
                gpus = len(free_gpu_indices(args.nvidia_smi))
            except launch.Refused:
                gpus = 4
        rows = simulate(queue, gpus, approved=args.approved, smoke=args.smoke,
                        device=args.device)
        print(plan_table(queue, rows, gpus))
        return 0
    try:
        runner = Runner(queue, smoke=args.smoke, device=args.device, approved=args.approved,
                        runs_dir=args.runs_dir, nvidia_smi=args.nvidia_smi,
                        max_gpus=args.max_gpus,
                        only=args.only.split(",") if args.only else None,
                        retry_failed=args.retry_failed, log=args.log,
                        allow_dirty=args.allow_dirty)
    except launch.Refused as e:
        print(f"queue refused: {e}", file=sys.stderr)
        return e.code
    if args.device == "cuda" and not runner.gpus and any(j.gpu for j in runner.jobs):
        print("queue: no free GPU; GPU jobs wait (use --device cpu to check on CPU)",
              file=sys.stderr)
        return launch.EXIT_NO_GPU
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
