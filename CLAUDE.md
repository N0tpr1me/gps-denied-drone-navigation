# Working in this repo

Structure Hold: one onboard PX4 flight mode that holds a quadrotor 1 to 3 m from a
structure in gusts without GPS, robust to a noisy, late, jumpy or missing position
estimate, with a classical watchdog that hands back to stock PX4.

## Rules

- Commits are authored by the repo owner's git identity only. No Claude or Claude Code
  attribution anywhere: no Co-Authored-By, no Claude-Session, no "Generated with" lines
  in commits, PR bodies, comments or issues.
- Layout: `envs/` sim environments and task logic, `training/` scripts and configs,
  `eval/` metrics, baselines and the SITL harness, `sim2real/` PX4 module, Docker and
  Gazebo, `viz/` Three.js demo, `experiments/LOG.md` one entry per real run, `docs/`
  decisions and conventions.
- Every training script has a smoke mode (`--smoke`: a few iterations, 1 GPU or CPU).
  Run it before saying anything works. Launch runs through `training/launch.py`.
- Real runs are logged in `experiments/LOG.md` (date, commit, config, result, baseline).
  Every reported result is compared against the tuned-PX4 baseline.
- Run `nvidia-smi` before any GPU job; use only free GPUs. Ask before any run over 1 hour
  or on more than 1 GPU (`launch.py` enforces both).
- Never commit secrets, datasets, or files over 50 MB. Checkpoints and ulogs stay out of git.
- Frames and units follow `docs/conventions.md`. Decisions live in `docs/decisions.md`.

## Commands

```
conda env create -f environment.yml && conda activate dronenav   # or: pip install -e .[dev]
ruff check . && pytest
python -m training.launch --config training/configs/selfcheck.yaml --smoke --device cpu
```
