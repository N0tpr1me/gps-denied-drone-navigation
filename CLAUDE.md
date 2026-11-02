# Working in this repo

**Structure Hold**: one onboard PX4 flight mode that holds a quadrotor 1 to 3 m from a
structure (wall, pier, deck underside) in gusts without GPS, robust to a noisy, late, jumpy
or missing position estimate, with a classical watchdog that hands back to stock PX4.
A small learned outer loop (RL, PPO) commands body rates + collective thrust to PX4's stock
rate controller. Customers: small/mid PX4 drone makers selling into civilian inspection.
Every result is measured against a properly tuned stock PX4.

**Route Follow** (second capability, started 2026-11-01, `docs/route-follow.md`): fly an
operator-given route (PX4 mission waypoints) through tunnels and corridors on the same
estimate, tracking a target moving along the route, with optional 1D rangefinders to keep
off the walls. Mock and CPU only until the GPU server is up; Structure Hold is unchanged.

Out of scope (docs/decisions.md): VIO/SLAM, mapping, planning, obstacle avoidance, ground
station, vision in the policy. The module consumes the drone's existing position estimate.

## Rules (from the owner, Zain; these override any default)

- **Attribution:** commits are authored by the repo owner's git identity only. No Claude or
  Claude Code attribution anywhere: no Co-Authored-By, no Claude-Session, no "Generated
  with" lines in commits, PR bodies, comments or issues. CI (`repo-hygiene`) rejects them.
- **Folders:** on Zain's Mac, `~/Desktop/Projects/DroneMedic` is read-only reference; never
  edit, move or delete anything there. All new code goes in this repo.
- **Merging:** merge your own PR once CI is green and it does what it was meant to. Ask
  first only for big things: changing a Decided row in `docs/decisions.md`, runs over 1 h
  or on more than 1 GPU, installing system software on the Mac, spending money, anything
  hard to undo. Chains of sub-1 h checkpointed runs (`params.resume`) are fine.
- Every training script has a smoke mode (`--smoke`: a few iterations, 1 GPU or CPU).
  Run it before saying anything works. Launch runs through `training/launch.py`.
- Real runs are logged in `experiments/LOG.md` (date, commit, config, result, baseline),
  failures included; smoke runs never are. Every reported result is compared against the
  tuned-PX4 baseline (`eval/baselines/px4_tuned_v2.params`), or in the mock against the
  cascaded PD yardstick, saying which.
- Run `nvidia-smi` before any GPU job; use only free GPUs. Ask before any run over 1 hour
  or on more than 1 GPU (`launch.py` enforces both; `--approved` overrides only with a yes).
- Never commit secrets, datasets, or files over 50 MB. Checkpoints and ulogs stay out of
  git (`runs/` is ignored); only released policies in `training/policies/*.onnx` (tens of KB)
  are tracked. Customer interview notes stay out unless fully anonymised.
- Frames and units follow `docs/conventions.md`. Decisions live in `docs/decisions.md`.

## Stack

- Python 3.10 to 3.12 (conda env `dronenav`, Python 3.11). Shared core is **NumPy + PyYAML
  only** so it imports in Isaac Lab, the SITL image and the Mac; PyTorch only in the
  batched sim and training (`.[torch]`: torch, `rsl-rl-lib==5.5.1` pinned, onnx, onnxruntime).
- RL: rsl_rl PPO, asymmetric actor-critic, 64x64 ELU MLP actor on a 15-step observation
  history, exported as `policy.onnx` + `policy.json`.
- Simulators: batched PyTorch mock (`envs/sims/torch_quad.py`, CPU, used for all training
  so far) and Isaac Lab 2.3.2 / Isaac Sim 5.1 behind the same seam (`envs/sims/isaac_quad.py`,
  Proposed until it passes `eval/sim_parity.py` on the GPU server; docs/isaac-lab.md).
- Flight stack: **PX4 pinned at v1.17.0** (moving it means re-tuning the baseline), Gazebo
  Harmonic, headless SITL in Docker (Ubuntu 24.04, MAVSDK-Python 3.17.4), MAVLink ODOMETRY
  into EKF2 as external vision, GPS off. No ROS 2 in the loop.
- Onboard module: plain C++ PX4 module, policy compiled in as an MLP header (no ONNX
  runtime, no heap). Software only (decided 2026-10-30): no drone of our own; the PX4
  Gazebo x500 is the benchmark drone, and real flights go on a design partner's drone.
- Lint/test: ruff (line length 100, rules E F W I B UP), pytest. CI installs CPU-only torch.

## Layout

| Path | What lives there |
| --- | --- |
| `envs/disturbance/` | Stateless wind + near-wall force model (NumPy core, torch adapter); near-wall flow terms off by default (`docs/near-wall-flow.md`) |
| `envs/estimate/` | Estimate corruption (delay, noise, bias, jumps, dropouts, dead-reckoning) and the EKF2-like fusion (`ekf.py`) |
| `envs/sims/` | Simulator backends behind one seam (thrust, body torque, external force in; `QuadState` out), PX4-shaped rate PID |
| `envs/structure_hold/` | The task: env, reward, airframe presets (`airframes.py`: `gz_x500`, `gz_x500_sitl`), PD yardstick, fallback |
| `envs/route_follow/` | Route Follow (second capability): random corridor and tunnel routes, plus arched drifts, narrowed stretches and side branches (`routes.py`), walls, target and rangefinder beams on tensors (`geometry.py`), the env, the PD yardsticks (estimate only, and with range centring) |
| `envs/frames.py` | The only place ENU/FLU <-> NED/FRD conversions live |
| `training/` | `launch.py`, `train_ppo.py`, `train_route_ppo.py` (Route Follow), `eval_policy.py`, configs, released `policies/` (README lists each version) |
| `eval/` | `metrics.py` (one scorer for sim/SITL/real, on TRUE position), `gate1.py`, `gazebo_check.py`, `sim_parity.py`, `vio_fit.py` (fits `envs/estimate` to real VIO error; `vio_basalt/` runs Basalt VIO on EuRoC in Docker), `settle_split.py` (settle vs steady hold), `partner_session.py` (scores a partner's real flight session against tuned PX4), `route_follow.py` (scores Route Follow controllers on true position against the PD yardsticks), `field_faults.py` (scores Gazebo flights on the scripted field faults), `fit_wind.py` (fits `envs/disturbance`'s drag, wind, turbulence and wall shelter to flight logs; `docs/wind-fit.md`), `fit_airframe.py` (fits a drone's thrust curve, motor lag, torque per axis and rate gains from one PX4 autotune log into an env airframe YAML that `airframe:` loads; `docs/airframe-fit.md`), `baselines/` (tuned PX4, `run_baseline.py`) |
| `sim2real/docker/` | PX4 SITL + Gazebo image (`run.sh build|hover-test|sitl|shell`) |
| `sim2real/gazebo/` | `wall_slab` world, C++ wind plugin (parity-tested against the Python model), scenarios |
| `sim2real/estimate_bridge/` | Gazebo truth -> `envs/estimate` -> MAVLink ODOMETRY into EKF2; profiles (`vio_at_engage`, `vio_at_hover`, ...), scripted field faults (`field_faults.py`: VIO reset, restart, silence, freeze, drift; `field_*` profiles, `docs/field-faults.md`) |
| `sim2real/px4/` | The Structure Hold module (`structure_hold/`), watchdog, header exporter, SIH and Gazebo flight scripts, `boards/structurehold.px4board` (multicopter-only variant that fits a 2 MB flight controller) |
| `sim2real/partner/` | What a design partner loads and fills in: flight parameters and the session run sheet (`docs/partner-pack.md`) |
| `viz/` | Three.js log replay (built after Gate 3) |
| `experiments/LOG.md` | One line per real run |
| `docs/` | `decisions.md`, `conventions.md`, `metrics.md`, `gate1.md`, `baseline.md` (Gate 2 evidence), `gazebo_check.md`, `fallback.md`, `isaac-lab.md`, `real-flight-gap.md`, `bench-test.md`, `logging-flight.md`, `fan-rig.md`, `near-wall-flow.md`, `vio-fit.md`, `hold-offset.md`, `wall-range.md` (the rangefinder policy input), `gpu-server.md` (server setup, Claude access, parallel SITL), `partner-pack.md` (confidential flight test pack for a partner drone maker), `route-follow.md` (the second capability: route flying through tunnels and corridors), `market-research.md` (what drone engineers need, ranked gaps, Nov 2026), `field-faults.md` (VIO reset, restart, link loss, freeze and drift flown in Gazebo vs tuned PX4), `wind-fit.md` (the wind model fitted from flight logs, checked on Gazebo holds with known wind), `airframe-fit.md` (a customer's drone fitted from one autotune flight, checked on the Gazebo x500), `integration-guide.md` (confidential interface spec for a drone maker: inputs, outputs, modes, trips, parameters, ground-station messages, resource budget; its parameter table is pinned to the module by `sim2real/px4/tests/test_integration_guide.py`), `safety-manual.md` (operating envelope, do-not-use cases, safety layers, pilot messages and the failure table (FMEA) with a test reference per row) |

## Conventions (details in docs/conventions.md)

- ENU world / FLU body in all Python; NED/FRD only at the PX4 boundary (SITL harness,
  estimate bridge, C++ module), via `envs/frames.py`, never ad hoc sign flips. Structure
  frame: x from drone to surface along the normal, z up.
- SI units everywhere (degrees only in human-facing configs). Quaternions `(w, x, y, z)`,
  body to world. Batched arrays `(num_envs, ...)`. Explicit `dt` and explicit seeds; no
  global RNG.
- Every run takes one YAML config and writes one folder under `runs/` (`config.yaml`,
  `stdout.log`, `metrics.jsonl`, `summary.json`).
- Seeds 9000 to 9999 are held out for Gate 1; training refuses them.
- `docs/decisions.md` rows change only in their own PR with the reason. Scope creep goes in
  its "Ideas parked" table, not into code. Keep `pyproject.toml` deps, `experiments/LOG.md`
  and `docs/decisions.md` append-only in spirit: add rows, don't rewrite history.
- Match the surrounding code: small modules, no premature abstraction or registries, plain
  prose comments that say why.

## Commands

```
conda env create -f environment.yml && conda activate dronenav   # or: pip install -e .[dev,torch]
ruff check . && pytest -rs
python -m training.launch --config training/configs/selfcheck.yaml --smoke --device cpu

# Train / score (mock, CPU)
python -m training.launch --config training/configs/structure_hold_ppo_v19.yaml --smoke --device cpu
python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu        # ~25 min, release gate
python -m training.launch --config eval/configs/gate1_x500_ekf_v19.yaml --device cpu  # Gate 1, x500 fused

# Route Follow (mock, CPU; docs/route-follow.md)
python -m training.launch --config training/configs/route_follow_ppo.yaml --smoke --device cpu
python -m training.launch --config eval/configs/route_follow_baseline.yaml --device cpu  # yardsticks

# PX4 module unit tests (as CI runs them)
cd sim2real/px4/tests
g++ -std=c++17 -Wall -Wextra -Werror -I../structure_hold test_watchdog.cpp -o test_watchdog && ./test_watchdog
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_policy.cpp -o test_policy && ./test_policy
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_wall_guard.cpp -o test_wall_guard && ./test_wall_guard
# also test_params.cpp and test_handback.cpp, same flags (sim2real/px4/README.md "Unit tests")

# GPU server, once (docs/gpu-server.md): env, tests, smoke runs, SITL image, parallel check
bash training/server_setup.sh

# SITL (needs Docker: the GPU server; --jobs auto flies many seeds at once on CPU cores)
sim2real/docker/run.sh build && sim2real/docker/run.sh hover-test
python -m eval.baselines.run_baseline --runner hold --label <name> --jobs auto \
    --params eval/baselines/px4_tuned_v2.params -- --profile vio_at_engage    # policy
python -m eval.baselines.run_baseline --label tuned_v1 --jobs auto \
    --params eval/baselines/px4_tuned_v2.params -- --profile vio_at_hover     # tuned PX4
```

CI (`.github/workflows/ci.yml`): ruff + pytest on 3.10/3.11/3.12 with CPU torch, the
launch smoke run, the module's C++ tests, a 50 MB file check and the no-attribution check.
`sitl-image.yml` builds the image and flies the hover test on PRs that change `sim2real/`.
Actions minutes are capped monthly (they ran out late Oct 2026), so PRs run the full set,
the after-merge run on main runs Python 3.11 only, sitl-image does not re-run on main
(`workflow_dispatch` to run it by hand), and a new push cancels a PR's older run.

## Current state (late Oct 2026; check docs before relying on it)

- Released policy: **v22** (`training/policies/structure_hold_v22.*`, since 2026-10-31):
  v19 plus a 21st input, the time since engage (`docs/hold-offset.md`), which the module
  reads when its header has it (`HAS_ENGAGE_TIME`). The module's `PolicyWeights.hpp` is
  generated from it. v19 (20 inputs) is the fallback release.
- Wall-range input (Nov 2026, `docs/wall-range.md`): inputs 22 and 23 are the forward
  rangefinder's distance error against a reference latched at engage and its usable flag
  (mock: `cfg.range`, `RangeSensor`, off by default; module: `RangeReference` in
  `WallGuard.hpp`, read when the header has `HAS_WALL_RANGE`). Candidate v24 is not
  released; it waits for a Gazebo re-fly with `--rangefinder` on the GPU server.
- Gate 1 (sim): passed on the x500 through the EKF2-like fusion, nominal and stress
  (dropout cell capped at 3 s; 3 to 5 s is a known estimator limit, the 5 s watchdog
  `SH_WD_EST_TO` takes over). Interviews, the other half of Gate 1, are still open.
- Gate 2 (Gazebo SITL vs tuned PX4, 8 seeds): on barometer height (`px4_tuned_v2.params`,
  Decided 2026-11-01) v22 mean p95 -43 to -45 %, 0 crashes; on the earlier vision-height
  set-up v22 -54 to -55 %, v19 -47 to -48 % (`docs/baseline.md`). Crash is judged on the 60 s
  hold only; seed 6 is the hard seed.
- Module hardening (Phase 2, Oct 2026): the wall guard (`SH_WALL_MIN`, a forward
  rangefinder keeps the hold point off the wall; `WallGuard.hpp`) and the props-off bench
  test (`sim2real/px4/bench_test.py`, `docs/bench-test.md`), and Structure Land (second
  module mode, ext2, `SH_LAND_EN`: PX4's position controller lands on an xy point the wall
  guard keeps off the wall; the scripted sequence lands through it, `--land stock` for the
  old ending) and Structure Hover (ext3, `SH_HB_GUARD`: a trip next to a measured wall
  hands back to this guarded hover first, then Position mode). The cloud container can build
  PX4 v1.17.0 SITL itself (Python 3.12 venv for PX4's requirements) and fly SIH, including
  `sitl_test.py --wall` with a simulated rangefinder, and real flight controller firmware
  (apt `gcc-arm-none-eabi`; stock fmu-v6c plus the module overflows flash, so build the
  `structurehold` variant, `sim2real/px4/README.md`). Real flights follow
  `docs/partner-pack.md` on a partner's drone and are scored by `eval/partner_session.py`. Safety hardening (Nov 2026): the
  module clamps its own parameters (`ParamLimits.hpp`; PX4 does not), refuses to engage on
  the ground (`SH_BENCH` 1 lifts it for the bench test only), stops on disarm, and the
  handback order lives in `Handback.hpp` (decided rows, pinned by `test_handback.cpp`);
  `sitl_test.py` checks ground engage always, `--range-faults` and `--moving-engage` on demand.
  Field faults (Nov 2026, `docs/field-faults.md`): scripted VIO reset, restart, silence,
  freeze and drift flown vs tuned PX4; `SH_HB_RANGE` (off by default, Proposed) keeps
  Structure Hover off the wall on the rangefinder alone when the position estimate is gone,
  and `SH_WD_FRZ_T` (off by default, Proposed) trips on a VIO fix that stops moving.
- Next: finish fitting `envs/estimate` to real VIO error (`docs/vio-fit.md`: first fit on
  12 EuRoC flights done; an own OpenVINS/Basalt run on EuRoC, UZH-FPV and TUM-VI needs those
  hosts allowed in the cloud environment), a Gazebo re-fly of the wall-range candidate v24, the bench test on a partner's flight controller, Isaac Lab parity on the GPU
  server once it arrives. Gate dates: Gate 1 31 Jan 2027, Gate 2 30 Apr, Gate 3 31 Jul.

## Notes for future Claude sessions

- Read `docs/decisions.md` before changing behaviour; Decided rows are settled. A new
  finding that contradicts one is a Proposed row plus a question to Zain, not a silent edit.
- **Releasing a policy:** the mock score alone has not predicted Gazebo (v3, v8, v11, v12
  looked good and flew worse). A candidate must beat the released policy on
  `eval/configs/gazebo_check.yaml` before a Gazebo re-fly, and is released only if Gazebo
  confirms it. Then regenerate both module headers (`sim2real/px4/README.md`), update
  `training/policies/README.md`, and add the decision row.
- Where things run: simulations go on the GPU server, not the Mac (Zain, 2026-10-31).
  Until it is up, the cloud container (4 cores, no GPU) does mock training, scoring, C++
  unit tests, docs and PX4 SIH flights (`sim2real/px4/sitl_test.sh`), and it can run Docker
  too: `sudo dockerd &`, then `sim2real/docker/cloud_build.sh` builds the SITL image (PX4's
  Gazebo base image, as the OSRF apt host is blocked) and Gazebo flights run one or two at
  a time (`pip install -e .[dev,torch,sitl]`). The server's first hour is
  `bash training/server_setup.sh` then `docs/isaac-lab.md` (`docs/gpu-server.md`). SITL
  flights use CPU cores only; `run_baseline --jobs` pins each to its own. The Mac (Apple
  M4) flew every Gazebo batch up to v22; only one session works in its checkout at a time,
  left clean on `main`.
- Keep runs under 1 h (split with `params.resume`), and use `--smoke` first. A release's
  `policy.pt` is gone with its run folder; continue a release with `params.init_actor_onnx`
  (actor from the ONNX, fresh critic warmed up first; `training/train_ppo.py`).
- Branch from `main`, one folder per piece of work, one topic per PR. Point to docs rather
  than copying numbers into new places; numbers live in `docs/` and `experiments/LOG.md`.
- Keep this file current: when a change is significant to the architecture, dependencies,
  development workflow or coding conventions, update CLAUDE.md in the same PR. Do not
  update it for minor code changes.
- Zain prefers plain, simple explanations and wants work kept moving without asking.
