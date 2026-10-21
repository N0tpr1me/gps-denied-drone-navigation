# GPS-Denied Drone Navigation

Navigation software that lets a small quadrotor fly reliably without GPS (indoors, tunnels, under bridges, confined spaces) using onboard sensors such as a camera and IMU.

Focus: civilian inspection and logistics.

## Approach

1. **Simulation training** — flight control policies trained with reinforcement learning in simulation.
2. **PX4 SITL** — policies validated in PX4 software-in-the-loop.
3. **Real flight** — deployed and flown on a small real drone.

## What we are building first

**Structure Hold**: one onboard PX4 flight mode that holds a drone steady 1 to 3 m from a
wall, pier or deck underside in gusts, without GPS, on a position estimate that may be
noisy, late, jumpy or briefly missing. A learned outer loop commands PX4's own rate
controller; a plain C++ watchdog hands back to stock PX4 the moment anything looks wrong.
Every result is measured against a properly tuned stock PX4.

## Layout

| Folder | What lives there |
| --- | --- |
| `envs/` | Wind and near-wall model, estimate corruption, simulator backends, the task |
| `training/` | `launch.py` (runs anything, enforces the rules), entrypoints, configs |
| `eval/` | Metrics, the tuned-PX4 baseline, the SITL benchmark harness |
| `sim2real/` | PX4 SITL Docker image, Gazebo world and adapters, the PX4 module |
| `viz/` | Three.js log replay |
| `experiments/LOG.md` | One line per real run |
| `docs/` | `decisions.md`, `conventions.md` |

## Quick start

```
conda env create -f environment.yml && conda activate dronenav   # or: pip install -e .[dev]
ruff check . && pytest
python -m training.launch --config training/configs/selfcheck.yaml --smoke --device cpu
```

`launch.py` refuses a busy GPU, more than 1 GPU, or a run expected to exceed 1 hour unless
`--approved` is passed, writes each run to `runs/<time>_<name>_s<seed>_<commit>/`, and appends
real (non-smoke) runs to `experiments/LOG.md`. On a new GPU machine, run
`bash training/server_report.sh` first.

## Status

Phase 0 (foundations and the tuned-PX4 baseline). See `docs/decisions.md`.
