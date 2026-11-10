import json
import stat
from pathlib import Path

import pytest
import yaml

from training import launch

LOG_HEADER = "| Date | Run |\n| --- | --- |\n"
GPU_IDLE = "0, GPU-aaa, 3, 24564, 0\n1, GPU-bbb, 3, 24564, 0\n"
GPU_ONE_BUSY = "0, GPU-aaa, 18000, 24564, 97\n1, GPU-bbb, 3, 24564, 0\n"


def write_config(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "name": "fake",
        "entrypoint": "training.tests.fake_entry",
        "seed": 3,
        "expected_hours": 0.01,
        "gpus": 1,
        "params": {"mode": "ok"},
    }
    cfg.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def fake_nvidia_smi(tmp_path: Path, gpus_csv: str, apps_csv: str = "") -> str:
    script = tmp_path / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  --query-gpu=*) printf '%s' '{gpus_csv}' ;;\n"
        f"  --query-compute-apps=*) printf '%s' '{apps_csv}' ;;\n"
        "esac\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def dirs(tmp_path):
    log = tmp_path / "LOG.md"
    log.write_text(LOG_HEADER)
    return {"runs": tmp_path / "runs", "log": log}


def run(cfg: Path, dirs, *extra: str) -> int:
    return launch.main(["--config", str(cfg), "--runs-dir", str(dirs["runs"]),
                        "--log", str(dirs["log"]), "--quiet", *extra])


def only_run_dir(dirs) -> Path:
    (run_dir,) = list(dirs["runs"].iterdir())
    return run_dir


# ---------------------------------------------------------------- GPU parsing


def test_parse_gpus_marks_busy_by_memory_util_and_processes():
    gpus = launch.parse_gpus(GPU_ONE_BUSY, "")
    assert [g.free for g in gpus] == [False, True]
    gpus = launch.parse_gpus(GPU_IDLE, "GPU-bbb, 1234\n")
    assert [g.free for g in gpus] == [True, False]
    assert gpus[1].busy_processes == 1


def test_parse_gpus_treats_unknown_utilisation_as_busy():
    (gpu,) = launch.parse_gpus("0, GPU-aaa, 3, 24564, [N/A]\n", "")
    assert not gpu.free


def test_pick_free_gpus_refuses_when_not_enough_free():
    gpus = launch.parse_gpus(GPU_ONE_BUSY, "")
    assert [g.index for g in launch.pick_free_gpus(gpus, 1)] == [1]
    with pytest.raises(launch.Refused) as e:
        launch.pick_free_gpus(gpus, 2)
    assert e.value.code == launch.EXIT_NO_GPU
    assert "GPU 0" in str(e.value) and "(busy)" in str(e.value)


# ---------------------------------------------------------------- rules


def test_more_than_one_gpu_needs_approval(tmp_path, dirs):
    cfg = write_config(tmp_path, gpus=2)
    smi = fake_nvidia_smi(tmp_path, GPU_IDLE)
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", smi, "--allow-dirty") == 2
    assert not dirs["runs"].exists()


def test_run_over_one_hour_needs_approval(tmp_path, dirs):
    cfg = write_config(tmp_path, expected_hours=3)
    assert run(cfg, dirs, "--device", "cpu", "--allow-dirty") == 2
    # Smoke runs are short by definition, so the hour rule does not apply to them.
    assert run(cfg, dirs, "--device", "cpu", "--smoke") == 0


def test_approved_lifts_the_rules():
    cfg = launch.resolve_config(
        {"name": "x", "entrypoint": "m", "seed": 0, "expected_hours": 5, "gpus": 2}, smoke=False
    )
    launch.check_rules(cfg, device="cuda", smoke=False, approved=True)


def test_smoke_never_uses_more_than_one_gpu():
    cfg = launch.resolve_config(
        {"name": "x", "entrypoint": "m", "seed": 0, "expected_hours": 0.1, "gpus": 2}, smoke=True
    )
    with pytest.raises(launch.Refused):
        launch.check_rules(cfg, device="cuda", smoke=True, approved=True)


def test_missing_config_keys_are_refused(tmp_path, dirs):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\n")
    assert run(path, dirs, "--device", "cpu", "--smoke") == 2


def test_cuda_refused_when_gpus_busy_or_nvidia_smi_missing(tmp_path, dirs):
    cfg = write_config(tmp_path)
    busy = fake_nvidia_smi(tmp_path, "0, GPU-aaa, 18000, 24564, 97\n")
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", busy, "--smoke") == 3
    missing = str(tmp_path / "no-such-nvidia-smi")
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", missing, "--smoke") == 3


def test_cuda_pins_the_free_gpu(tmp_path, dirs):
    cfg = write_config(tmp_path)
    smi = fake_nvidia_smi(tmp_path, GPU_ONE_BUSY)
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", smi, "--smoke") == 0
    resolved = yaml.safe_load((only_run_dir(dirs) / "config.yaml").read_text())
    assert resolved["_run"]["cuda_visible_devices"] == "1"


def test_real_run_refused_on_dirty_tree(tmp_path, dirs, monkeypatch):
    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234def", True))
    cfg = write_config(tmp_path)
    assert run(cfg, dirs, "--device", "cpu") == 2
    assert run(cfg, dirs, "--device", "cpu", "--allow-dirty") == 0


# ---------------------------------------------------------------- running and logging


def test_smoke_run_writes_run_folder_and_skips_log(tmp_path, dirs):
    cfg = write_config(tmp_path, smoke={"extra": 1})
    assert run(cfg, dirs, "--device", "cpu", "--smoke") == 0
    run_dir = only_run_dir(dirs)
    assert run_dir.name.endswith("-smoke") and "_fake_s3_" in run_dir.name
    for name in ("config.yaml", "stdout.log", "metrics.jsonl", "summary.json"):
        assert (run_dir / name).exists(), name
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert resolved["params"]["extra"] == 1 and "smoke" not in resolved
    assert resolved["_run"]["smoke"] is True and resolved["_run"]["cuda_visible_devices"] == ""
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["headline"]["smoke_seen"] is True
    assert "fake_entry mode=ok smoke=True device=cpu" in (run_dir / "stdout.log").read_text()
    assert dirs["log"].read_text() == LOG_HEADER


def test_real_run_appends_one_log_line(tmp_path, dirs, monkeypatch):
    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234def", False))
    cfg = write_config(tmp_path)
    assert run(cfg, dirs, "--device", "cpu") == 0
    lines = dirs["log"].read_text().splitlines()
    assert len(lines) == 3
    row = lines[-1]
    assert "`abc1234`" in row and "| ok |" in row and "value=1" in row and "+0% vs fake" in row


@pytest.mark.parametrize(
    "mode, expected_error",
    [("fail", "fake failure"), ("crash", "exit code 1"), ("no_summary", "no ok summary.json")],
)
def test_failed_runs_are_recorded(tmp_path, dirs, monkeypatch, mode, expected_error):
    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234def", False))
    cfg = write_config(tmp_path, params={"mode": mode})
    assert run(cfg, dirs, "--device", "cpu") == 1
    summary = json.loads((only_run_dir(dirs) / "summary.json").read_text())
    assert summary["status"] == "failed" and expected_error in summary["error"]
    assert "| FAILED |" in dirs["log"].read_text().splitlines()[-1]


def test_timeout_kills_the_run(tmp_path, dirs, monkeypatch):
    monkeypatch.setattr(launch, "SMOKE_TIMEOUT_HOURS", 2 / 3600)
    cfg = write_config(tmp_path, params={"mode": "sleep"})
    assert run(cfg, dirs, "--device", "cpu", "--smoke") == 1
    summary = json.loads((only_run_dir(dirs) / "summary.json").read_text())
    assert summary["status"] == "timeout"


def test_selfcheck_smoke_end_to_end(dirs):
    cfg = launch.REPO_ROOT / "training" / "configs" / "selfcheck.yaml"
    assert run(cfg, dirs, "--device", "cpu", "--smoke") == 0
    run_dir = only_run_dir(dirs)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["headline"]["iterations"] == 20
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 20 and metrics[-1]["loss"] < metrics[0]["loss"]


# ---------------------------------------------------------------- --sim (simulator backend)


def test_sim_overrides_params_and_shows_in_the_log(tmp_path, dirs, monkeypatch):
    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234def", False))
    cfg = write_config(tmp_path, params={"mode": "ok", "sim": "isaac"})
    assert run(cfg, dirs, "--device", "cpu", "--sim", "torch") == 0
    resolved = yaml.safe_load((only_run_dir(dirs) / "config.yaml").read_text())
    assert resolved["params"]["sim"] == "torch"
    assert "cfg.yaml` --sim torch |" in dirs["log"].read_text().splitlines()[-1]


def test_sim_refused_for_configs_without_a_simulator(tmp_path, dirs):
    cfg = write_config(tmp_path)  # params has no sim key
    assert run(cfg, dirs, "--device", "cpu", "--smoke", "--sim", "torch") == launch.EXIT_REFUSED
    assert not dirs["runs"].exists()


def test_isaac_needs_cuda(tmp_path, dirs):
    cfg = write_config(tmp_path, params={"mode": "ok", "sim": "torch"})
    assert run(cfg, dirs, "--device", "cpu", "--smoke", "--sim", "isaac") == launch.EXIT_REFUSED
    assert not dirs["runs"].exists()


# ---------------------------------------------------------------- --gpu, --set, LOG.md lines


def test_gpu_pins_the_named_gpu_only_when_free(tmp_path, dirs):
    cfg = write_config(tmp_path)
    smi = fake_nvidia_smi(tmp_path, GPU_ONE_BUSY)
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", smi, "--smoke", "--gpu", "0") == 3
    assert not dirs["runs"].exists()
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", smi, "--smoke", "--gpu", "1") == 0
    resolved = yaml.safe_load((only_run_dir(dirs) / "config.yaml").read_text())
    assert resolved["_run"]["cuda_visible_devices"] == "1"
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", smi, "--smoke", "--gpu", "7") == 3
    assert run(cfg, dirs, "--device", "cuda", "--nvidia-smi", smi, "--smoke", "--gpu", "0,1") == 2


def test_set_overrides_params_and_seed_and_shows_in_the_log(tmp_path, dirs, monkeypatch):
    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234def", False))
    cfg = write_config(tmp_path, params={"mode": "ok", "env": {"a": 1, "b": 2}})
    assert run(cfg, dirs, "--device", "cpu", "--set", "params.env.a=[1, 2]",
               "--set", "params.new.deep=null", "--set", "seed=11") == 0
    run_dir = only_run_dir(dirs)
    assert "_s11_" in run_dir.name
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert resolved["params"]["env"] == {"a": [1, 2], "b": 2}
    assert resolved["params"]["new"] == {"deep": None} and resolved["seed"] == 11
    assert resolved["_run"]["overrides"] == ["params.env.a=[1, 2]", "params.new.deep=null",
                                            "seed=11"]
    assert "--set params.env.a=[1, 2]" in dirs["log"].read_text().splitlines()[-1]
    # Only params and the seed: the hour and GPU rules always read the committed config.
    assert run(cfg, dirs, "--device", "cpu", "--smoke", "--set", "expected_hours=0.1") == 2
    assert run(cfg, dirs, "--device", "cpu", "--smoke", "--set", "nonsense") == 2


def test_set_steps_into_lists_by_index():
    cfg = {"params": {"controllers": [{"name": "a", "policy": "a.onnx"},
                                      {"name": "b", "policy": "b.onnx"}], "seeds": [1, 2]}}
    applied = launch.apply_overrides(cfg, ["params.controllers.1.policy=runs/x/policy.onnx",
                                           "params.seeds.0=7"])
    assert cfg["params"]["controllers"][1] == {"name": "b", "policy": "runs/x/policy.onnx"}
    assert cfg["params"]["controllers"][0]["policy"] == "a.onnx"
    assert cfg["params"]["seeds"] == [7, 2]
    assert applied == ["params.controllers.1.policy=runs/x/policy.onnx", "params.seeds.0=7"]
    with pytest.raises(launch.Refused, match="not an index"):
        launch.apply_overrides(cfg, ["params.seeds.9=1"])


def test_appended_log_lines_do_not_make_the_tree_dirty():
    assert not launch.is_dirty("")
    assert not launch.is_dirty(" M experiments/LOG.md\n")
    assert launch.is_dirty(" M experiments/LOG.md\n M training/launch.py\n")
    assert launch.is_dirty("?? experiments/LOG.md\n")
    assert launch.is_dirty("A  experiments/LOG.md\n")
