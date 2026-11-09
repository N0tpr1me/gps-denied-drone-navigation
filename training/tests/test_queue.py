import json
from pathlib import Path

import pytest
import yaml

from training import launch, queue
from training.tests.test_launch import GPU_IDLE, fake_nvidia_smi

QUEUES = Path(launch.REPO_ROOT, "training", "queues")


def write_queue(tmp_path: Path, jobs: list[dict], **top) -> Path:
    cfg_path = tmp_path / "fake_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "fake", "entrypoint": "training.tests.fake_entry", "seed": 3,
        "expected_hours": 0.01, "gpus": 1, "params": {"mode": "ok"},
    }))
    for j in jobs:
        if j.get("launch") == "CFG":
            j["launch"] = str(cfg_path)
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "jobs": jobs, **top}))
    return path


def run(path: Path, tmp_path: Path, *extra: str) -> int:
    smi = fake_nvidia_smi(tmp_path, GPU_IDLE)
    log = tmp_path / "LOG.md"
    if not log.exists():
        log.write_text("| Date | Run |\n| --- | --- |\n")
    return queue.main([str(path), "--runs-dir", str(tmp_path / "runs"), "--nvidia-smi", smi,
                       "--log", str(log), "--allow-dirty", *extra])


def run_config(run_dir: str) -> dict:
    return yaml.safe_load((queue.run_path(run_dir) / "config.yaml").read_text())


def state(tmp_path: Path, smoke: bool = False) -> dict:
    name = "queue_t-smoke" if smoke else "queue_t"
    return json.loads((tmp_path / "runs" / name / "state.json").read_text())


# ---------------------------------------------------------------- the queue file


def test_every_committed_queue_loads_and_plans():
    for path in QUEUES.glob("*.yaml"):
        q = queue.load_queue(path)
        assert q.jobs, path
        rows = queue.simulate(q, 4, approved=True)
        planned = [r for r in rows if r["status"] == "planned"]
        assert planned, path
        # Every launch job's config has a smoke block (the rule for training scripts) and
        # every GPU job the plan can place; disabled jobs say why they wait.
        for j in q.jobs:
            if j.launch:
                raw = yaml.safe_load((launch.REPO_ROOT / j.launch).read_text())
                assert "smoke" in raw, j.id
            if not j.enabled:
                assert j.why, j.id
        table = queue.plan_table(q, rows, 4)
        assert "GPUs busy" in table


def test_every_committed_queue_set_applies_to_its_config():
    # A `set` that points into a list (params.controllers.1) or names a config key that is not
    # there would only fail on the server; launch's own override code catches it here.
    for path in QUEUES.glob("*.yaml"):
        for j in queue.load_queue(path).jobs:
            if not (j.launch and j.set):
                continue
            cfg = yaml.safe_load((launch.REPO_ROOT / j.launch).read_text())
            args = queue.set_args(j.set)
            assert not any("\n" in a for a in args), j.id  # each ends up in a LOG.md row
            launch.apply_overrides(cfg, args[1::2])


def test_bad_queue_files_are_refused(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        queue.load_queue(write_queue(tmp_path, [{"id": "a"}]))
    with pytest.raises(ValueError, match="unknown jobs"):
        queue.load_queue(write_queue(tmp_path, [{"id": "a", "cmd": "true", "after": ["z"]}]))
    with pytest.raises(ValueError, match="earlier"):
        queue.load_queue(write_queue(tmp_path, [{"id": "a", "cmd": "true", "after": ["b"]},
                                                {"id": "b", "cmd": "true"}]))
    with pytest.raises(ValueError, match="unique"):
        queue.load_queue(write_queue(tmp_path, [{"id": "a", "cmd": "true"},
                                                {"id": "a", "cmd": "true"}]))
    with pytest.raises(ValueError, match="no such config"):
        queue.load_queue(write_queue(tmp_path, [{"id": "a", "launch": "nope.yaml"}]))


def test_simulate_fills_gpu_lanes_and_respects_after(tmp_path):
    path = write_queue(tmp_path, [
        {"id": "a", "gpu": True, "launch": "CFG", "hours": 0.9},
        {"id": "b", "gpu": True, "launch": "CFG", "hours": 0.5},
        {"id": "c", "gpu": True, "launch": "CFG", "hours": 0.5, "after": ["b"]},
        {"id": "d", "cmd": "true", "hours": 0.5},
        {"id": "e", "cmd": "true", "hours": 0.5, "after": ["d"]},
        {"id": "big", "gpu": True, "launch": "CFG", "hours": 3.0},
    ])
    q = queue.load_queue(path)
    rows = {r["job"].id: r for r in queue.simulate(q, 2, approved=False)}
    assert rows["a"]["lane"] == "GPU 0" and rows["b"]["lane"] == "GPU 1"
    assert rows["c"]["lane"] == "GPU 1" and rows["c"]["start"] == 0.5
    assert rows["d"]["lane"] == "CPU" and rows["e"]["start"] == 0.5
    assert rows["big"]["status"] == "skipped"  # over 1 h without --approved
    rows = {r["job"].id: r for r in queue.simulate(q, 2, approved=True)}
    assert rows["big"]["status"] == "planned" and rows["big"]["start"] == 0.9


def test_chain_hours_are_per_part():
    j = queue.Job(id="x", launch="c", chain=3, hours=2.4)
    assert not j.requires_approval
    assert queue.Job(id="y", launch="c", chain=1, hours=1.5).requires_approval


# ---------------------------------------------------------------- running


def test_runs_jobs_in_order_with_deps_and_records_state(tmp_path):
    marker = tmp_path / "order.txt"
    path = write_queue(tmp_path, [
        {"id": "first", "cmd": f"echo first >> {marker}", "hours": 0.01},
        {"id": "train", "gpu": True, "launch": "CFG", "hours": 0.01},
        {"id": "second", "cmd": f"echo second >> {marker}", "after": ["first"], "hours": 0.01},
        {"id": "bad", "cmd": "exit 3", "hours": 0.01},
        {"id": "after_bad", "cmd": f"echo never >> {marker}", "after": ["bad"], "hours": 0.01},
    ])
    assert run(path, tmp_path, "--device", "cpu") == 1
    s = state(tmp_path)
    assert s["first"]["status"] == "ok" and s["second"]["status"] == "ok"
    assert s["train"]["status"] == "ok" and "_fake_s3_" in s["train"]["parts"][0]["run_dir"]
    assert len((tmp_path / "LOG.md").read_text().splitlines()) == 3  # one real run logged
    assert s["bad"]["status"] == "failed" and s["after_bad"]["status"] == "blocked"
    assert marker.read_text().split() == ["first", "second"]
    # Re-running skips what finished and retries nothing unless asked.
    assert run(path, tmp_path, "--device", "cpu") == 1
    assert marker.read_text().split() == ["first", "second"]
    assert run(path, tmp_path, "--device", "cpu", "--retry-failed") == 1
    assert state(tmp_path)["bad"]["parts"].__len__() == 2


def test_chain_resumes_from_the_previous_part(tmp_path, monkeypatch):
    # fake_entry writes no policy.pt; stand one in so the chain has a checkpoint to resume.
    real_start = queue.Runner.start

    def start(self, job, lane):
        for p in self.state[job.id]["parts"]:
            if p.get("run_dir"):
                (queue.run_path(p["run_dir"]) / "policy.pt").write_text("ckpt")
        real_start(self, job, lane)

    monkeypatch.setattr(queue.Runner, "start", start)
    path = write_queue(tmp_path, [
        {"id": "chain", "gpu": True, "launch": "CFG", "chain": 2, "hours": 0.02,
         "chain_set": {"params.init_actor_onnx": None}},
    ])
    assert run(path, tmp_path, "--device", "cpu") == 0
    parts = state(tmp_path)["chain"]["parts"]
    assert len(parts) == 2 and all(p["status"] == "ok" for p in parts)
    cfg2 = run_config(parts[1]["run_dir"])
    assert cfg2["params"]["resume"].endswith("policy.pt")
    assert cfg2["params"]["init_actor_onnx"] is None
    assert any(o.startswith("params.resume=") for o in cfg2["_run"]["overrides"])
    cfg1 = run_config(parts[0]["run_dir"])
    assert "resume" not in cfg1["params"]


def test_run_reference_points_at_the_other_jobs_folder(tmp_path):
    path = write_queue(tmp_path, [
        {"id": "train", "gpu": True, "launch": "CFG", "hours": 0.01},
        {"id": "score", "gpu": True, "launch": "CFG", "after": ["train"], "hours": 0.01,
         "set": {"params.policy": "{run:train}/policy.onnx"}},
        {"id": "shell", "cmd": "echo {run:train} > " + str(tmp_path / "ref.txt"),
         "after": ["train"], "hours": 0.01},
    ])
    assert run(path, tmp_path, "--device", "cpu") == 0
    s = state(tmp_path)
    train_dir = s["train"]["parts"][0]["run_dir"]
    cfg = run_config(s["score"]["parts"][0]["run_dir"])
    assert cfg["params"]["policy"] == f"{train_dir}/policy.onnx"
    assert (tmp_path / "ref.txt").read_text().strip() == train_dir


def test_smoke_runs_every_job_once_without_chains_or_logging(tmp_path):
    path = write_queue(tmp_path, [
        {"id": "chain", "gpu": True, "launch": "CFG", "chain": 3, "hours": 2.0},
        {"id": "big", "cmd": "exit 1", "smoke_cmd": "true", "hours": 5.0},
        {"id": "nosmoke", "cmd": "exit 1", "hours": 0.1},
        {"id": "off", "cmd": "exit 1", "smoke_cmd": "true", "enabled": False, "hours": 0.1},
    ])
    assert run(path, tmp_path, "--device", "cpu", "--smoke") == 0
    s = state(tmp_path, smoke=True)
    assert s["chain"]["status"] == "ok" and len(s["chain"]["parts"]) == 1
    assert s["chain"]["parts"][0]["run_dir"].endswith("-smoke")
    assert s["big"]["status"] == "ok"
    assert s["nosmoke"]["status"] == "skipped" and s["off"]["status"] == "skipped"


def test_approval_gates_long_jobs(tmp_path):
    path = write_queue(tmp_path, [{"id": "long", "cmd": "true", "hours": 3.0}])
    assert run(path, tmp_path, "--device", "cpu") == 0
    assert state(tmp_path)["long"]["status"] == "skipped"
    assert run(path, tmp_path, "--device", "cpu", "--approved") == 0
    assert state(tmp_path)["long"]["status"] == "ok"


def test_gpu_jobs_pin_distinct_free_gpus(tmp_path):
    path = write_queue(tmp_path, [
        {"id": "a", "gpu": True, "launch": "CFG", "hours": 0.01},
        {"id": "b", "gpu": True, "launch": "CFG", "hours": 0.01},
    ])
    assert run(path, tmp_path, "--device", "cuda") == 0
    s = state(tmp_path)
    seen = set()
    for jid in ("a", "b"):
        cfg = run_config(s[jid]["parts"][0]["run_dir"])
        seen.add(cfg["_run"]["cuda_visible_devices"])
    assert seen == {"0", "1"}


def test_dry_run_prints_the_plan(tmp_path, capsys):
    path = write_queue(tmp_path, [{"id": "a", "gpu": True, "launch": "CFG", "hours": 0.5}])
    assert queue.main([str(path), "--dry-run", "--gpus", "2"]) == 0
    out = capsys.readouterr().out
    assert "2 GPU lane(s)" in out and "| a | GPU 0 |" in out
