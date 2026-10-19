# CLAUDE.md

## Goal
Navigation software that lets a quadrotor fly reliably without GPS (indoors, tunnels, under bridges, confined spaces) using onboard sensors such as a camera and IMU. Policies are trained in simulation with RL, tested in PX4 SITL, then flown on a small real drone. Focus: civilian inspection and logistics.

## Hardware
Workstation with multiple NVIDIA RTX 4500 Ada GPUs (24 GB each), ~512 GB RAM, AMD Threadripper Pro, CUDA, Conda and Docker.
- Run `nvidia-smi` before any GPU job and only use free GPUs.
- Ask before any run expected to last over 1 hour or use more than 1 GPU.

## Stack
Isaac Lab in Docker for sim and training; PPO to start; PX4 SITL + ROS 2 for flight stack testing; Three.js demo in `viz/`; conda env `dronenav`. Open choices are tracked in `docs/decisions.md`.

## Layout
- `envs/` sim environments and tasks
- `training/` scripts and configs
- `eval/` evaluation and classical baselines
- `sim2real/` PX4 / ROS 2 integration and real-drone code
- `viz/` Three.js demo
- `experiments/LOG.md` one entry per run
- `docs/` design notes and decisions

## Rules
- Plan before changes that touch more than one folder.
- Every training script has a smoke-test mode (a few iterations, 1 GPU). Run it before saying anything works.
- Log every real run in `experiments/LOG.md` with date, git commit, config and result.
- Compare every reported result against a classical baseline.
- Never commit secrets, datasets, or checkpoints over 50 MB.
- Customer interview notes stay out of the repo unless fully anonymised.
- No Claude attribution anywhere on GitHub: no Co-Authored-By or Claude-Session trailers, no "Generated with Claude Code" lines in commits, PRs or comments.
