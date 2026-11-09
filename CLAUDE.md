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
The one tracked Route Follow policy is v2 (`training/policies/route_follow_v2.*`, mock only,
not released); the worlds and stress configs score it against the PD yardsticks. Its SITL
baseline (Nov 2026) is tuned PX4 Mission mode flying four fixed route worlds in Gazebo
(`sim2real/route_follow/`), scored on the mock's own walls.

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
- Auterion variant (decided 2026-11-02, `docs/auterion-app.md`): the same flight code as an
  AuterionOS app; ROS 2 Humble + `px4_ros2` inside that app only. Nothing signed up or paid
  for at Auterion (Skynode Developer program, Suite) without asking Zain.
- Lint/test: ruff (line length 100, rules E F W I B UP), pytest. CI installs CPU-only torch.

## Layout

| Path | What lives there |
| --- | --- |
| `envs/disturbance/` | Stateless wind + near-wall force model (NumPy core, torch adapter); near-wall flow terms off by default (`docs/near-wall-flow.md`); an optional second surface (a deck overhead: its shelter and edge shedding; `docs/bridge-holds.md`) |
| `envs/estimate/` | Estimate corruption (delay, noise, bias, jumps, dropouts, dead-reckoning), the EKF2-like fusion (`ekf.py`) and the heading error a compass near steel puts in the attitude (`heading.py`, `heading_error:` in an env config, off by default; `docs/heading-error.md`) |
| `envs/sims/` | Simulator backends behind one seam (thrust, body torque, external force in; `QuadState` out), PX4-shaped rate PID |
| `envs/structure_hold/` | The task: env, reward, airframe presets (`airframes.py`: `gz_x500`, `gz_x500_sitl`), PD yardstick, fallback, surface passes (`passes.py`: the hold point moving 5 m along or up the wall, `surface_pass`, off by default; `docs/surface-passes.md`) |
| `envs/route_follow/` | Route Follow (second capability): random corridor and tunnel routes, plus arched drifts, narrowed stretches and side branches (`routes.py`), walls, target and rangefinder beams on tensors (`geometry.py`), the env, the PD yardsticks (estimate only, and with range centring) |
| `envs/frames.py` | The only place ENU/FLU <-> NED/FRD conversions live |
| `training/` | `launch.py` (`--gpu N` pins a card, `--set params.KEY=VALUE` overrides one value; a numeric path segment steps into a list, `params.controllers.1.policy`), `queue.py` (runs a `queues/*.yaml` job list, `server_week1.yaml` first, then `server_week2.yaml`, `heading_error.yaml`, `route_gazebo.yaml`: every free GPU busy, SITL batches on the other cores, chains of sub-1 h parts; `docs/gpu-server-plan.md`), `isaac_parity.sh` (the Isaac Lab first hour as one command), `train_ppo.py`, `train_route_ppo.py` (Route Follow), `eval_policy.py`, configs, released `policies/` (README lists each version) |
| `eval/` | `metrics.py` (one scorer for sim/SITL/real, on TRUE position), `gate1.py`, `gazebo_check.py`, `sim_parity.py`, `vio_fit.py` (fits `envs/estimate` to real VIO error; `vio_basalt/` runs Basalt VIO on EuRoC in Docker), `vio_render/` (renders hold flights in fog, dust and low light as stereo + IMU datasets, NumPy ray caster or Isaac Sim, and runs Basalt on them; `docs/vio-render.md`), `settle_split.py` (settle vs steady hold), `partner_session.py` (scores a partner's real flight session against tuned PX4), `route_follow.py` (scores Route Follow controllers on true position against the PD yardsticks), `field_faults.py` (scores Gazebo flights on the scripted field faults), `eval_kit.py` (the evaluation kit behind `./evaluate.sh`: the five field faults flown with Structure Hold and tuned PX4 in one command, one comparison table; `docs/eval-kit.md`), `partner_report.py` (one-page HTML benchmark report from a scored partner session), `fit_wind.py` (fits `envs/disturbance`'s drag, wind, turbulence and wall shelter to flight logs; `docs/wind-fit.md`), `fit_airframe.py` (fits a drone's thrust curve, motor lag, torque per axis and rate gains from one PX4 autotune log into an env airframe YAML that `airframe:` loads; `docs/airframe-fit.md`), `hold_datasheet.py` (the hold datasheet from three `gate1` wind sweeps: the wind at which each controller still holds, per direction, distance and estimate quality; `docs/hold-datasheet.md`), `fault_rates.py` (crash rate per scripted estimate fault with a Wilson 95 % bound, tens of thousands of episodes in the mock, v22 and the PD; `docs/fault-rates.md`), `vio_runaway.py` (the module's VIO runaway check in the mock: false trips per hour and time to trip; `replay` runs it over a ulog), `log_health.py` (reads one PX4 ulog and says in plain English whether EKF2 really fuses the external position source, in the right frame, with the right heading source and `EKF2_EV_DELAY`, frame and delay measured against the IMU; `docs/log-health.md`), `baselines/` (tuned PX4, `run_baseline.py`) |
| `sim2real/docker/` | PX4 SITL + Gazebo image (`run.sh build|hover-test|sitl|shell`); `build_kit.sh` packages the evaluation kit for a buyer, binary only (Decided 2026-11-02) |
| `sim2real/gazebo/` | `wall_slab`, `bridge_deck` and `bridge_pier` worlds (a scenario names its `world`), C++ wind plugin (parity-tested against the Python model, second surface included), scenarios |
| `sim2real/estimate_bridge/` | Gazebo truth -> `envs/estimate` -> MAVLink ODOMETRY into EKF2; profiles (`vio_at_engage`, `vio_at_hover`, ...), scripted field faults (`field_faults.py`: VIO reset, restart, silence, freeze, drift, degrade (dust, fog, low light); `field_*` profiles, `docs/field-faults.md`), wiring mistakes on purpose for the log health check (`frame_mistake`; `health_*` profiles), a scripted GPS/RTK receiver beside the VIO (`gps.py`, HIL_GPS; profiles with a `gps` block fly with GPS on, `gps_on.params`, `field_gps_*` and `field_rtk_float` faults), a scripted heading error through the vision yaw (`vision_yaw.py`; profiles with a `vision_yaw` block fuse it, `vision_yaw.params`, `field_heading_*`) |
| `sim2real/px4/` | The Structure Hold module (`structure_hold/`), watchdog, keep-out (`KeepOut.hpp`), header exporter, SIH and Gazebo flight scripts, `boards/structurehold.px4board` (multicopter-only variant that fits a 2 MB flight controller) |
| `sim2real/route_follow/` | Route Follow in Gazebo: four benchmark routes (`routes/*.yaml`) drawn as worlds from `wall_slab.sdf`, the mock's route geometry in NumPy (`walls.py`), stock PX4 Mission mode flying them on the estimate bridge (`mission_flight.py`, MAVSDK), the host runner and scorer (`run_mission.py`, `--jobs`, `--rescore`); estimate profiles `route_*`; server queue `training/queues/route_gazebo.yaml` |
| `sim2real/assurance/` | The buyer's software parts list (`sbom.cdx.json`, CycloneDX, regenerated by `make_sbom.py`; CI fails when it is stale) for `docs/assurance-pack.md` |
| `sim2real/auterion/` | Structure Hold as an AuterionOS app (`docs/auterion-app.md`): a PX4 external mode in a ROS 2 node via `px4_ros2`, flying the module's own headers (`HoldCore.hpp`); AuterionOS package pinned to AuterionOS 4.6.2, SITL image and SIH flight against stock PX4 (`sitl_app_test.sh`) |
| `sim2real/partner/` | What a design partner loads and fills in: flight parameters and the session run sheet (`docs/partner-pack.md`) |
| `viz/` | Three.js log replay (built after Gate 3) |
| `experiments/LOG.md` | One line per real run |
| `docs/` | `decisions.md`, `conventions.md`, `metrics.md`, `gate1.md`, `baseline.md` (Gate 2 evidence), `gazebo_check.md`, `fallback.md`, `isaac-lab.md`, `real-flight-gap.md`, `bench-test.md`, `logging-flight.md`, `fan-rig.md`, `near-wall-flow.md`, `vio-fit.md`, `vio-render.md` (real Basalt VIO on rendered flights in fog, dust and low light), `hold-offset.md`, `wall-range.md` (the rangefinder policy input), `gpu-server.md` (server setup, Claude access, parallel SITL), `gpu-server-plan.md` (the server's first week: the job queue in order, lanes, times, go/no-go checks, what needs Zain's yes; "Week 2" is the approved eight-item follow-on), `hold-datasheet.md` (where Structure Hold breaks over wind speed, direction, distance and estimate quality), `fault-rates.md` (failure rates per FMEA fault with 95 % bounds), `partner-pack.md` (confidential flight test pack for a partner drone maker), `route-follow.md` (the second capability: route flying through tunnels and corridors), `market-research.md` (what drone engineers need, ranked gaps, Nov 2026), `field-faults.md` (VIO reset, restart, link loss, freeze and drift flown in Gazebo vs tuned PX4), `wind-fit.md` (the wind model fitted from flight logs, checked on Gazebo holds with known wind), `airframe-fit.md` (a customer's drone fitted from one autotune flight, checked on the Gazebo x500), `log-health.md` (the log health check: what it checks, how frame and delay are measured, tested on Gazebo logs with wiring mistakes made on purpose), `integration-guide.md` (confidential interface spec for a drone maker: inputs, outputs, modes, trips, parameters, ground-station messages, resource budget; its parameter table is pinned to the module by `sim2real/px4/tests/test_integration_guide.py`), `assurance-pack.md` (confidential buyer paperwork: parts list, statement of origin, standards mapping against ASTM F3201/F3269 and others, pointer to the safety manual sign-off; claims no certification), `eval-kit.md` (the one-command evaluation kit and the one-page partner benchmark report), `safety-manual.md` (operating envelope, do-not-use cases, safety layers, pilot messages and the failure table (FMEA) with a test reference per row, and a sign-off block; its messages and defaults are pinned to the module by `sim2real/px4/tests/test_safety_manual.py`), `surface-passes.md` (the 5 m passes along a wall: a moving hold point in the mock, its Gate 1 style scorer and the server training job), `heading-error.md` (a wrong heading near steel: what it does to a hold, the mock model, the Gazebo vision-yaw profiles, the server queue), `bridge-holds.md` (holds under a bridge deck and in a pier's wake: the second surface, the two Gazebo worlds, v22 vs the PD in the mock, the server queue `training/queues/bridge_holds.yaml`), `pilot-status.md` (what the pilot sees while the module flies: status lines, near-limit warnings and the live `DEBUG_FLOAT_ARRAY` "SH_STATUS" for QGroundControl; `PilotStatus.hpp`) |

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
python -m training.launch --config eval/configs/gate1_x500_ekf_pass.yaml --device cpu # surface passes

# Route Follow (mock, CPU; docs/route-follow.md)
python -m training.launch --config training/configs/route_follow_ppo.yaml --smoke --device cpu
python -m training.launch --config eval/configs/route_follow_baseline.yaml --device cpu  # yardsticks
python -m sim2real.route_follow.run_mission --label <name> --jobs auto \
    --params eval/baselines/px4_tuned_v2.params --profile route_vio --seeds 1 2   # Gazebo Mission

# PX4 module unit tests (as CI runs them)
cd sim2real/px4/tests
g++ -std=c++17 -Wall -Wextra -Werror -I../structure_hold test_watchdog.cpp -o test_watchdog && ./test_watchdog
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_policy.cpp -o test_policy && ./test_policy
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_wall_guard.cpp -o test_wall_guard && ./test_wall_guard
# also test_params.cpp, test_handback.cpp, test_range_hover.cpp, test_pilot_status.cpp and
# test_keep_out.cpp, same flags (sim2real/px4/README.md "Unit tests")
# Auterion app core (sim2real/auterion/README.md): test_hold_core.cpp, -std=c++17, same warnings,
#   -I../include -I../../../px4/structure_hold; SIH flight: sim2real/auterion/sitl_app_test.sh <PX4_DIR>

# GPU server, once (docs/gpu-server.md): env, tests, smoke runs, SITL image, parallel check
bash training/server_setup.sh
# then the first week's queue (docs/gpu-server-plan.md): --dry-run shows the plan, --smoke
# runs every job's smoke, --approved covers the jobs listed there for Zain's yes
python -m training.queue training/queues/server_week1.yaml --dry-run
python -m training.queue training/queues/server_week1.yaml --smoke

# SITL (needs Docker: the GPU server; --jobs auto flies many seeds at once on CPU cores)
sim2real/docker/run.sh build && sim2real/docker/run.sh hover-test
python -m eval.baselines.run_baseline --runner hold --label <name> --jobs auto \
    --params eval/baselines/px4_tuned_v2.params -- --profile vio_at_engage    # policy
python -m eval.baselines.run_baseline --label tuned_v1 --jobs auto \
    --params eval/baselines/px4_tuned_v2.params -- --profile vio_at_hover     # tuned PX4
./evaluate.sh        # evaluation kit: 5 field faults, Structure Hold vs tuned PX4, one table
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
  hold only; seed 6 is the hard seed. The mock fusion has a barometer-height option
  (`fusion.height_source: baro`, numbers from EKF2 on Gazebo hovers; `docs/baseline.md`
  "The mock fusion on barometer height"); `gz_x500_sitl`, and with it the release check,
  training and Route Follow, fly on it since 2026-11-03 (Decided). The Gate 1 configs state
  their own vision-height fusion block, so the Gate 1 numbers stand; every policy up to v22
  trained and was gated on vision height, and its check numbers on the barometer are in
  `docs/gazebo_check.md` "Policies judged".
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
  freeze, drift and degrade (dust, fog, low light: a noisier, walking VIO and a spoiled
  rangefinder; the wall guard's spike gate keeps short returns from moving the hold point) flown
  vs tuned PX4; `SH_HB_RANGE` (on by default since 2026-11-02) keeps
  Structure Hover off the wall on the rangefinder alone when the position estimate is gone,
  and `SH_WD_FRZ_T` (1 s, on by default since 2026-11-02) trips on a VIO fix that stops moving.
  Heavy dust (Nov 2026, `docs/field-faults.md` "Heavy dust"): `SH_WD_VIO_DV` (0.8 m/s, on by
  default) trips on a VIO whose velocity runs away from the IMU's over 3 s (`RunawayFix`) and
  hands back as if the position were lost; `eval/vio_runaway.py` scores it in the mock and
  replays a ulog (`replay <ulg>`) to set the limit on a partner's own logs.
  Keep-out (Nov 2026, `KeepOut.hpp`, `SH_KO_DIST`, -1 follows PX4's `CP_DIST`): PX4's
  collision-prevention data (`obstacle_distance`, side-facing `distance_sensor`s) keeps the
  hold point of all three modes that far from obstacles other than the structure; not
  avoidance. `structurehold.px4board` also drops the Septentrio GNSS driver so the fmu-v6x
  build still fits (99.0 % with the pilot status and keep-out).
  GPS on (2026-11-02, decided; `docs/field-faults.md` "GPS on"): the envelope allows a GPS or
  RTK receiver fused beside the VIO, with the forward rangefinder required (`SH_WALL_REQ` 1):
  on a 1.5 m GPS jump tuned PX4 hit the wall on 3 of 3 seeds, v22's wall guard held 0.70 m;
  no module code was added (safety manual 1.3). PX4 v1.17's `EKF2_HGT_REF` 1 is GPS height,
  not barometer (0); the benchmark set said 1 until 2026-11-02 and flew on the barometer only
  because there was no GPS; it says 0 now (`eval/log_health.py` warns on 1).
- Surface passes (Nov 2026, `docs/surface-passes.md`): the hold point moves 5 m along or up
  the wall at the same standoff (mock only). The velocity input is relative to the moving
  hold point, so no new input: v22 flies passes as it is and the pass training
  (`structure_hold_ppo_pass.yaml`, queue `training/queues/surface_passes.yaml`) starts from it. Scored Gate 1 style
  by `eval/configs/gate1_x500_ekf_pass.yaml`. The module and Gazebo do not fly passes yet.
- Heading error near steel (2026-11-09, `docs/heading-error.md`): in the mock v22 holds a
  15 deg wrong heading but circles into the wall at 30 deg (100 % crash, the PD holds); the
  server queue `training/queues/heading_error.yaml` trains two heading candidates. In Gazebo
  (80 flights, cloud, same day) it went the other way: v22 held all 8 drift-to-30-deg flights
  where tuned PX4 hit the wall in 5 of 8. EKF2 did not keep a 30 deg vision-yaw offset
  (14 to 22 deg mean), so Gazebo is milder than the mock's fixed error, and the 30 deg-from-
  takeoff batches are muddied by stock PX4 circling before the engage.
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
  confirms it. Then regenerate both module headers (`sim2real/px4/README.md`) and the parts list
  (`python -m sim2real.assurance.make_sbom`), update
  `training/policies/README.md`, and add the decision row.
- Where things run: simulations go on the GPU server, not the Mac (Zain, 2026-10-31). The
  server turned out to be a shared university machine (2026-11-09, `docs/gpu-server.md` "The
  server we got"): reached only by `ssh gpu` from the Mac on the VPN, at most 2 GPUs at once,
  Gazebo batches fly in cloud containers for now and the server does GPU and mock work
  (CPU training there is too slow under other users' load).
  Until it is up, the cloud container (4 cores, no GPU) does mock training, scoring, C++
  unit tests, docs and PX4 SIH flights (`sim2real/px4/sitl_test.sh`), and it can run Docker
  too: `sudo dockerd &`, then `sim2real/docker/cloud_build.sh` builds the SITL image (PX4's
  Gazebo base image, as the OSRF apt host is blocked) and Gazebo flights run one or two at
  a time (`pip install -e .[dev,torch,sitl]`). The server's first hour is
  `bash training/server_setup.sh` then `python -m training.queue training/queues/server_week1.yaml`
  (`docs/gpu-server-plan.md`: Isaac parity, v24 re-fly, Route Follow pipes, 10 to 12 m/s wind,
  the benchmark moved over, in one queue; `docs/gpu-server.md`). SITL
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
