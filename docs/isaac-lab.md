# Isaac Lab: the second simulator

Isaac Lab plugs in behind the simulator seam (`envs/sims`, docs/decisions.md): it moves the
rigid body, and everything else (rate loop, PX4-like rate lag, wind, estimate corruption,
observations, reward, PPO) is the same code that trains on the mock. This page is what to
do when the GPU server arrives, so that bringing Isaac Lab up stays inside the 5 working
days the plan allows. Nothing here has been run on Isaac Lab yet: there is no GPU machine
(see "What is verified and what is not").

## Pinned versions

| What | Pin | Why |
| --- | --- | --- |
| Isaac Lab | **2.3.2** (git tag `v2.3.2`) | Newest release on the stable 2.x line. 3.0 is beta / early access and moves the physics backend (Newton) and APIs; revisit at its GA |
| Isaac Sim | **5.1.0** | The Isaac Sim that Isaac Lab 2.3 is built on |
| Docker image | **`nvcr.io/nvidia/isaac-lab:2.3.2`** (headless; record its digest on first pull) | NVIDIA's prebuilt Isaac Lab image, Isaac Sim 5.1 inside; no image build of our own |
| Python | 3.11 (Isaac Sim 5.x's own) | Inside our 3.10 to 3.12 support; CI already tests 3.11 |
| PyTorch | 2.7.0 + CUDA 12.8, as shipped in the image | Keep Isaac's torch; our `torch>=2.6` extra accepts it |
| rsl_rl | `rsl-rl-lib==5.5.1` (ours) over the image's 3.1.2 | We train with our own rsl_rl pin (docs/decisions.md); Isaac Lab's `isaaclab_rl` wrapper is not used, so its pin does not matter |

Host requirements for Isaac Sim 5.1 (NVIDIA's requirements page): an RTX-class GPU with
RT cores and at least 16 GB of VRAM (A100 / H100 are **not supported**; the RTX 4500 Ada in
the project brief is fine), Ubuntu 22.04 or 24.04, NVIDIA driver 580.65.06 or a newer
production-branch driver, 32 GB RAM or more, Docker with the NVIDIA Container Toolkit.

## What the backend does (envs/sims/isaac_quad.py)

One PhysX rigid body per env, spawned with Isaac Lab's `InteractiveScene` and driven the way
Isaac Lab's direct-workflow quadcopter task (`Isaac-Quadcopter-Direct-v0`) drives its
Crazyflie: a body-frame force and torque written through `permanent_wrench_composer` every
physics step. The task's Crazyflie articulation is not used, because its four propeller
links add mass, inertia and spin the mock does not have, and the seam promises "one rigid
body". Mock parameters map one to one:

| Mock (`QuadParams` / env config) | Isaac Lab |
| --- | --- |
| `mass` 0.8 kg | PhysX body mass (`set_masses`, read back, fails loudly if not taken) |
| `inertia` (0.0035, 0.0035, 0.0060) kg m^2 | PhysX diagonal inertia, centre of mass at the body origin with axes = body FLU (`set_inertias`, `set_coms`, read back) |
| `max_thrust` 16 N, `max_torque` (0.3, 0.3, 0.05) N m | Clipped in `IsaacQuadSim.step` with the mock's code, per env |
| `motor_tau` 0.02 s | First-order lag in torch per physics sub-step, the mock's formula, per env |
| `set_airframe(env_ids, mass=, inertia=, max_thrust=, max_torque=, motor_tau=)` (per-env airframe, Gate 1 randomisation) | Same call and validation as the mock. Mass and inertia go into PhysX for just those envs (`set_masses` / `set_inertias` with env indices, read back); limits and motor lag are per-env tensors in torch |
| `step(..., torque_ext_body)` (near-wall yaw moment) | Added to the lagged motor torque in the body frame, neither clipped nor lagged, then written with the rest of the wrench |
| hover-thrust error, rate loop `RatePid`, rate lag `rate_lag_s` 0.1 to 0.2 s, action limit `max_rate_rad_s` 2 rad/s | Unchanged: they live in `envs/structure_hold`, above the seam |
| `substeps` 2 | PhysX `dt` = 0.01 / 2 = 5 ms, one wrench write per sub-step |
| gravity 9.80665 m/s^2 | `SimulationCfg.gravity = (0, 0, -9.80665)` (Isaac's default is 9.81) |
| no drag, no damping | `linear_damping = angular_damping = 0` (PhysX default angular damping is 0.05), sleeping and stabilisation off, gyroscopic forces on, angular velocity cap lifted, collisions off |
| frames ENU / FLU, quaternion (w, x, y, z) | Isaac Lab is Z-up with (w, x, y, z) quaternions; positions are returned relative to each env's origin, PhysX's world-frame angular velocity is rotated to body FLU |

The per-episode airframe randomisation (mass, inertia, thrust-to-weight, motor lag) and the
near-wall yaw moment of the Gate 1 work reach the backend only through `set_airframe` and the
fourth `step` argument, so the Isaac backend flies them the way the mock does; the parity
check below runs with them on as well as off. Mass and inertia change between steps the
way Isaac Lab's own `randomize_rigid_body_mass` event changes them, and the backend reads
every write back and fails loudly if PhysX did not take it.

Isaac Lab is imported only when an `IsaacQuadSim` is built, so CI and the Mac never need it;
asking for `--sim isaac` without it raises `IsaacLabUnavailable` pointing here. Isaac Lab
runs one physics scene per process: building a second `IsaacQuadSim` closes the first, and
stepping the old one fails loudly. Training and scoring build envs one after another, so
this does not bite.

## Install (on the server)

Prefer the Docker image; use the pip route only if Docker is not allowed on the server.

**Docker (recommended).**

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
docker inspect --format '{{index .RepoDigests 0}}' nvcr.io/nvidia/isaac-lab:2.3.2   # record it
mkdir -p ~/docker/isaac-sim/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}
cd ~/gps-denied-drone-navigation          # the repo, cloned on the server
docker run --name dronenav-isaac --entrypoint bash -it --gpus all --rm --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v "$PWD":/workspace/dronenav -w /workspace/dronenav \
  nvcr.io/nvidia/isaac-lab:2.3.2
```

Inside the container (the cache mounts make later starts fast; the first start compiles
shaders for several minutes):

```bash
ls /workspace                                   # Isaac Lab's folder: isaaclab or IsaacLab
export PY="/workspace/isaaclab/isaaclab.sh -p"  # adjust to what ls showed
$PY -m pip freeze | grep -E '^(torch|numpy)==' > /tmp/keep.txt
$PY -m pip install -e '.[torch,dev]' -c /tmp/keep.txt   # keep Isaac's torch and numpy<2
$PY -c "import isaaclab, envs.sims.isaac_quad as q; print('isaac ok', q.isaac_available())"
```

`pip` warns that `isaaclab_rl` wants `rsl-rl-lib==3.1.2`: expected and harmless (we do not
use that wrapper).

**pip (no Docker).** Needs glibc 2.35+ (`ldd --version`) and a Python 3.11 env:

```bash
conda create -n dronenav-isaac python=3.11 -y && conda activate dronenav-isaac
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
(cd ~/IsaacLab && ./isaaclab.sh --install none)    # Isaac Lab core and assets, no RL wrappers
pip install -e '.[torch,dev]' -c <(pip freeze | grep -E '^(torch|numpy)==')
export PY=python
```

## First hour on the new server

Run in order; stop and post the output in the project if a step fails.

1. **Who else is on it.** `nvidia-smi`. Note which GPUs are free; use only a free one
   (launch.py checks this too).
2. **Machine report.** `bash training/server_report.sh | tee server_report.txt` and paste it
   in the project. Check against the host requirements above: GPU model has RT cores and
   16 GB+, driver 580.65.06 or newer, Ubuntu 22.04/24.04, Docker usable without sudo,
   `nvidia-ctk` present, nvcr.io reachable. A data-centre GPU without RT cores (A100, H100)
   means Isaac Sim will not run: stop and tell Zain (the decided fallback is the batched
   PyTorch sim, which runs on any CUDA GPU).
3. **Image.** `docker pull nvcr.io/nvidia/isaac-lab:2.3.2`, start it as above, install the
   repo into Isaac's Python, record the image digest.
4. **Isaac starts headless.** `$PY /workspace/isaaclab/scripts/tutorials/00_sim/log_time.py --headless`
   (the stock smoke test; it exits after a few seconds).
5. **Repo tests in Isaac's Python.**
   `$PY -m pytest -rs envs/sims eval/tests/test_sim_parity.py training/tests/test_isaac_ready.py`.
   The contract tests now include `isaac` (CPU physics pipeline) and
   `test_isaac_lab_matches_the_mock` runs on the GPU. Nothing in that list may say SKIPPED for
   an Isaac reason.
6. **Parity report.** `$PY -m eval.sim_parity --a torch --b isaac --device cuda --out runs/parity_isaac.json`.
   Open loop on the placeholder airframe and on the x500, then three closed loops: the PD on
   the nominal airframe, the PD with the Gate 1 airframe randomisation on, and v22 on its own
   x500 task. Must end `parity: PASS` (a few minutes). Keep the JSON; paste the printed table
   in the project. `--a torch --b torch-fine` prints the reference gaps the limits come from.
7. **Mock on the GPU, for comparison.**
   `$PY -m training.launch --config training/configs/structure_hold_ppo_v22.yaml --sim torch --smoke --device cuda`
8. **Isaac training smoke run, 1 GPU, a few iterations.**
   `$PY -m training.launch --config training/configs/structure_hold_ppo_v22.yaml --sim isaac --smoke --device cuda`.
   This is the released recipe (v22: x500, fusion, engage-time input, starting from v19's
   ONNX). Must finish `launch: ok`; `runs/*-smoke/stdout.log` shows `env_steps_per_s` for
   both simulators. While it runs, `nvidia-smi` should show the process on the GPU launch picked.
9. **v22 scored on Isaac.**
   `$PY -m training.launch --config training/configs/structure_hold_eval_v22.yaml --sim isaac --device cuda`
   (8 min on 4 CPU cores). On the mock (run `20261101-012947`, `experiments/LOG.md`), 3 seeds
   x 512 identical 60 s episodes: v22 p95 0.211 / 0.206 / 0.202 m, crash 0 / 0.4 / 0 %
   (pooled 0.206 m, 0.1 %); the PD 0.61 / 0.56 / 0.70 m, crash 1.6 / 1.6 / 2.0 %. The
   scoring is deterministic: a second run gave the same numbers to the digit. On Isaac, v22's
   p95 per seed must land within the mock's 0.20 to 0.21 m spread (a few mm either side is
   the physics step, as in the parity check) and its crash rate stay under 1 %.
10. **Tell Zain**: parity numbers, smoke steps/s on both backends, v22's score on both, and
   the image digest for the decisions row below. A real Isaac run (not smoke) follows the
   usual rules: logged in `experiments/LOG.md` by launch.py, compared against the PD and
   tuned-PX4 baselines, and anything over 1 hour or more than 1 GPU needs Zain's go-ahead.

## M3: the first Isaac training

Nothing about the task is ported, because nothing has to be: the Structure Hold env, the
wind and near-wall flow, the estimate corruption and EKF2-like fusion, the airframe
randomisation, re-engages and the engage-time input all sit above the seam and run
unchanged on whichever backend is underneath (`training/tests/test_isaac_ready.py` steps the
v22 task on the Isaac backend's own code and the mock side by side, crashes, resets and
re-engages included, and they agree to float32 rounding). So M3 is the released recipe on
the other backend, judged the way every candidate is:

1. Train: `$PY -m training.launch --config training/configs/structure_hold_ppo_v22.yaml --sim isaac --device cuda`
   (v22's own recipe, 1 GPU; on the mock it took ~25 min on 4 CPU cores, so it is under the
   1 h rule; check the smoke run's steps/s first).
2. Score the result with `structure_hold_eval_v22.yaml` pointed at the new `policy.onnx`,
   on the mock (`--sim torch`): a policy trained on Isaac must hold the mock's task about as
   well as v22 does. The mock score alone has never been enough to release a policy, so it
   then goes through `eval/configs/gazebo_check.yaml` and a Gazebo re-fly like any candidate.
3. Isaac is worth keeping if it trains as well per iteration and faster per hour; the
   speed lets later runs use more envs or longer training inside the 1 h rule. If it does
   not, the decided fallback stands (the batched PyTorch sim on the GPU).

What may need attention on the GPU (cannot be measured on CPU): the env syncs the GPU with
the CPU a few times per step (episode counters, `done` checks, re-engage draws in NumPy).
That is correct but costs speed at thousands of envs; profile before changing it, and keep
the results identical when changing it (`test_isaac_ready.py` checks it).

## Parity: what is checked and the tolerances

`eval/sim_parity.py` (test: `eval/tests/test_sim_parity.py`) feeds both backends identical
inputs and compares true state.

- **Open loop:** twelve one-second scenarios, one per env: hover, free fall, climb, roll and
  pitch torque pulses, yaw spin, external force, a tilted / moving / spinning start
  (gyroscopic term, reset frames), saturated commands (clipping, motor lag), an aggressive
  all-axis sweep, the same kind of sweep on another airframe set with `set_airframe` (mass
  x1.3, unequal inertia scales, 50 ms motors, lower thrust and yaw limits it runs into), and
  an external body moment like the near-wall yaw moment on a rolled body. Worst difference
  over the second, per scenario. It runs on the placeholder airframe and again on the x500
  (below).
- **Closed loop:** Structure Hold with the cascaded PD on the corrupted estimate, same seed,
  wind and estimate noise on both, 256 envs x 10 s. The env's crash rules are switched off
  for this run so no env resets (a reset draws new random numbers; one crash a step apart
  in the two backends would change every later draw for every env). Whether an env would
  have crashed is judged afterwards with the standard rules. Reported: per env, the worst
  position difference (median and p95 over envs that would not have crashed); the share of
  envs that would crash in one backend only; the hold-error p95 of each. It runs twice: on
  the nominal airframe, and with the airframe and wall yaw moment of the Gate 1 "combined"
  stress row on (`RANDOMISED`: mass x0.8 to 1.3, inertia x0.7 to 1.5, hover-thrust error
  +/-10 %, motors 20 to 50 ms, yaw moment up to 25 % of yaw authority), so every env flies its
  own airframe through `set_airframe`. A third run flies v22 on its own x500 task (below).

The tolerances come from one rule: Isaac must agree with the mock at least as well as the
mock agrees with itself at an 8x finer physics step (the discretisation error every mock
number already carries), with about a 2x margin. Measured mock-vs-8x-finer gaps, CPU:

Open loop, worst difference over 1 s (mock vs mock at 8x finer step; the limit is twice it,
with floors of 1 mm, 1 mm/s, 1 mrad, 0.01 rad/s):

| Scenario | Position | Velocity | Attitude | Body rate |
| --- | --- | --- | --- | --- |
| hover | 0 | 0 | 0 | 0 |
| free_fall | 41 mm | 20 mm/s | 0 | 0 |
| climb | 12 mm | 6.2 mm/s | 0 | 0 |
| roll_pulse, pitch_pulse | 2.1 mm | 1.2 mm/s | 2.1 mrad | 0.012 rad/s |
| yaw_spin | 0 | 0 | 7.0 mrad | 0.0035 rad/s |
| external_force | 1.9 mm | 0 | 0 | 0 |
| tilted_spinning_start | 9.9 mm | 16 mm/s | 6.1 mrad | 0.0038 rad/s |
| saturated | 29 mm | 46 mm/s | 26 mrad | 0.23 rad/s |
| aggressive | 40 mm | 38 mm/s | 23 mrad | 0.074 rad/s |
| other_airframe | 51 mm | 52 mm/s | 35 mrad | 0.11 rad/s |
| external_moment | 11 mm | 2.4 mm/s | 5.0 mrad | 0.0021 rad/s |

Closed loop, 256 envs x 10 s, seeds 0 and 1: worst position gap median 7 to 10 mm and p95
43 mm, no crash mismatch, hold p95 within 0.2 %. Limits: median 20 mm, p95 100 mm, crash
mismatch 2 % of envs, hold p95 within 1 %. Isaac at the same 5 ms physics step and the same
semi-implicit integration should sit well inside these; the Isaac backend's own code over
the PhysX stand-in is within 1e-5 of the mock.

Closed loop with the airframe randomisation on, 256 envs x 10 s, seeds 0 to 3: median 13 to
16 mm, p95 0.13 to 0.17 m, crash mismatch 0 to 1.6 % (up to 4 envs), hold p95 within 0.6 to
1.5 %. Its own limits, twice the worst seed: median 33 mm, p95 0.33 m, crash mismatch 3.2 %,
hold p95 within 3 %. The tail is wider than nominal because a heavy, slow-motor, high-inertia
airframe is marginal under the PD, so small step-size differences grow there. The Isaac
backend's own code, driven by the env (per-env airframes into the PhysX stand-in, the yaw
moment through `step`), is within 1e-5 of the mock here too.

On the x500 (added 2026-11-01). The open-loop scenarios run again on the PX4 Gazebo x500's
rigid body (2.06 kg, 34 N, `envs/structure_hold/airframes.py`), with every fixed force
scaled by the mass ratio and every fixed torque by the axis's inertia ratio, so each
scenario asks the same accelerations of it (on the placeholder the scale is exactly 1 and
nothing changed). Mock vs 8x finer, and the limits (twice it, same floors):

| Scenario | Position | Velocity | Attitude | Body rate |
| --- | --- | --- | --- | --- |
| hover, free_fall, climb, yaw_spin, external_force | as the placeholder | | | |
| roll_pulse, pitch_pulse | 2.1 mm | 1.2 mm/s | 2.1 mrad | 0.012 rad/s |
| tilted_spinning_start | 10 mm | 17 mm/s | 6.2 mrad | 0.0040 rad/s |
| saturated | 23 mm | 24 mm/s | 13 mrad | 0.12 rad/s |
| aggressive | 41 mm | 38 mm/s | 24 mrad | 0.077 rad/s |
| other_airframe | 51 mm | 52 mm/s | 37 mrad | 0.12 rad/s |
| external_moment | 11 mm | 2.8 mm/s | 5.0 mrad | 0.0024 rad/s |

The third closed loop (`X500_V22`) is the task the released policy trains on, flown by v22
itself: `gz_x500_sitl` (12 ms tick, PX4's gyro and yaw filters, the bridge's lever arm, the
EKF2-like fusion), v22's airframe randomisation and wall yaw moment, wind toward the wall,
and the near-wall flow terms at their evaluation ranges. Everything Structure Hold sends
through the seam is in it. Mock vs 8x finer, 256 envs x 10 s, seeds 0 to 3: median 2.0 mm,
p95 6.8 to 8.1 mm, no crash mismatch, hold p95 within 0.07 % (v22 holds 0.28 to 0.32 m p95
there). Limits: median 4.1 mm and p95 17 mm (twice the worst seed), crash mismatch 2 % and
hold p95 within 1 % (the nominal run's floors). v22 on the x500 is much tighter than the PD
on the placeholder because the x500 is heavier and slower to rotate, so a step-size
difference moves it less. The Isaac backend's own code over the PhysX stand-in is within
1 mm of the mock here.

The check has teeth: in CI it fails a backend whose mass is 5 % off, whose roll inertia is
10 % off, whose motor lag is doubled, that swaps the roll and pitch torque axes, that
ignores `set_airframe` or only its motor lag, or that applies the external moment about the
world axes (`test_parity_catches_a_wrong_backend`; the x500 set catches a 5 % mass error and
an ignored `set_airframe` too), and the Isaac backend's own code passes it at 1e-4
over a PhysX stand-in that keeps state the way PhysX does (`envs/sims/tests/fake_physx.py`).

## If parity fails

Read the per-scenario table first; the scenario that fails points at the cause.

| Fails | Look at |
| --- | --- |
| `free_fall` | gravity (must be 9.80665, not Isaac's 9.81) or the physics dt (`substeps`) |
| `hover`, `climb` | mass not taken (the backend reads it back; check the log), thrust direction |
| `roll_pulse`, `pitch_pulse`, `yaw_spin` | inertia or centre-of-mass frame not taken; torque applied in the world frame instead of the body |
| `external_force` | the world force must be rotated into the body frame (`body_wrench`) |
| `tilted_spinning_start` | reset frames (body rates must be written to PhysX as world rates); gyroscopic forces flag |
| `aggressive` | angular damping not 0, angular velocity cap, sleep / stabilisation thresholds |
| `saturated` | clipping or motor lag; this is our code and passes in CI, so suspect a version change |
| `other_airframe` (the others pass) | per-env mass or inertia not taken by PhysX (the backend reads them back; check the log), or env indices off by one in `set_mass_properties` |
| `external_moment` | the moment must be added in the body frame, before `body_wrench` |
| nominal closed loop passes, randomised fails | same as `other_airframe`; or the PhysX mass write landed after the step it was meant for |
| open loop passes, closed loop fails | the two runs did not see the same randomness: both must run on the same device (`--device cuda`); or many envs sit at a crash boundary (rerun with `--num-envs 1024`) |
| everything off by a small steady drift | PhysX put bodies to sleep or the TGS solver applies external forces differently: try `PhysxCfg(enable_external_forces_every_iteration=True)` |

Rules: fix the cause, do not loosen a tolerance to pass; changing a tolerance is a
docs/decisions.md row. If Isaac Lab cannot pass parity and run the PPO smoke training
within the 5 working days, the decided fallback applies (keep training on the batched
PyTorch sim on the GPU) and Zain is told.

## What is verified and what is not

Verified here (CI, no GPU): the backend's glue (frames, clipping, motor lag, per-env
airframes, the external moment, resets, env origins) step for step against the mock over a
PhysX stand-in, and driven by the Structure Hold env with the airframe randomisation on; the contract tests on that
glue; the lazy import (the repo never imports Isaac Lab unless asked); the parity check and
its tolerances on the mock (`python -m eval.sim_parity --a torch --b torch-fine` passes);
`launch.py --sim` (override, refusals, LOG line); the PPO smoke run through
`launch --sim torch`. And, since 2026-11-01, over the PhysX stand-in with Isaac Lab's one
scene per process enforced (`training/tests/test_isaac_ready.py`): the v22 task step for
step against the mock, crashes, resets and re-engages included; the v22 training recipe
with `sim: isaac` (critic warm-up, training, checkpoint selection, export, PD comparison,
each building its envs one after another); the eval entry with `sim: isaac`.

Not verified until the server: anything that touches Isaac Lab itself. The calls the
backend makes were read from the v2.3.2 sources (`InteractiveScene`, `RigidObject`,
`permanent_wrench_composer`, `root_physx_view.set_masses / set_inertias / set_coms`,
`SimulationContext` teardown as in Isaac Lab's own `build_simulation_context`), but have not
run. Isaac Sim was not installed in the cloud container: it needs Python 3.11 and an RTX GPU
and is tens of GB, so it is not practical for CPU-only checks; the Isaac tests skip cleanly
there with the reason printed (`pytest -rs`).

## Proposed decisions row (pinned versions)

Appended to docs/decisions.md as **Proposed**; it becomes Decided, with the image digest
added, once steps 1 to 8 of the first-hour checklist pass:

| Decision | Status | Date | Why / what would change it |
| --- | --- | --- | --- |
| Isaac Lab pinned at 2.3.2 on Isaac Sim 5.1.0, run from `nvcr.io/nvidia/isaac-lab:2.3.2` (digest recorded on first pull); our rsl_rl pin and PPO recipe unchanged; the Isaac backend is one PhysX rigid body per env with the mock's mass, inertia, motor lag and clipping (`envs/sims/isaac_quad.py`), accepted only if it passes `eval/sim_parity.py` | Proposed | 2026-10-22 | Newest stable Isaac Lab line; 3.0 is beta / early access with a new physics backend. Move to 3.x only after its GA and only in its own PR that passes parity again |
