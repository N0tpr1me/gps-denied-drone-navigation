# GPS-Denied Drone Navigation

Software that lets a small quadrotor fly reliably without GPS (indoors, in tunnels, under
bridges, in confined spaces) on the position estimate the drone already has, for example
camera + IMU visual-inertial odometry (VIO). It is built to stay safe when that estimate is
noisy, late, jumpy, frozen or missing for a while. We do not build the estimate itself.

Focus: civilian inspection. Customers: small and mid-sized drone makers who fly PX4.

**Software only.** There is no drone, VIO computer or test rig of our own. The benchmark
drone is the PX4 Gazebo x500, and real flights are planned on a design partner's drone.
**Nothing has flown on real hardware yet:** every result below is from simulation (our
training simulator, PX4 SIH and PX4 + Gazebo software-in-the-loop) and unit tests.

## The product: Structure Hold

One onboard PX4 flight mode that holds a drone steady 1 to 3 m from a structure (a wall, a
pier, the underside of a bridge deck) in gusts, without GPS.

- A small learned controller (a neural network trained with reinforcement learning, PPO)
  sends body rates and thrust to PX4's own rate controller. It is compiled into the
  firmware as plain C++: no ML runtime, no heap.
- A classical watchdog with no learning in it checks every tick and hands control back to
  stock PX4 the moment anything looks wrong (estimate invalid, missing for too long or
  frozen, too much tilt, too far off the hold point).
- A forward rangefinder, when fitted, keeps the hold point off the wall (the wall guard).
- Every result is measured against a properly tuned stock PX4.

### Where it stands (Nov 2026)

| What | Status |
| --- | --- |
| Released policy | **v22** (since 2026-10-31). v19 is the fallback release. |
| Rangefinder candidate | **v24** reads the forward rangefinder's wall distance as an input. Re-flown in Gazebo with the rangefinder on 2026-11-09 it held 5 to 12 % tighter than v22 (0.18 against 0.20 m) and came 18 to 38 % less toward the wall, with 0 crashes. Not released: the release is Zain's decision (`docs/wall-range.md`). |
| Gate 1 (simulator) | **Passed** on the x500 through an EKF2-like estimator, normal and stress test sets, against a classical PD controller (`docs/gate1.md`). Customer interviews, the other half of Gate 1, are still open. |
| Gate 2 (PX4 Gazebo SITL vs tuned PX4) | On barometer height (`EKF2_HGT_REF` 0, the set-up the safety manual requires), 8 seeds, 3 wind cases, flown 2026-11-09: v22 holds to about **0.20 m** (mean 95th-percentile hold error 0.198 / 0.195 / 0.195 m at calm / 4 / 8 m/s) against tuned PX4's 0.327 / 0.325 / 0.329 m, **about 40 % tighter** (39 to 41 %), with 0 crashes (`docs/baseline.md`, "Scored like for like"). Since 2026-11-10 a Gazebo hold is scored like for like: the height offset EKF2 has on the ground is taken out, and both controllers are measured against a fixed hold point. Scored the old way the same flights read 0.29 to 0.33 m against 0.49 to 0.53 m (34 to 46 %). Figures quoted before 2026-11-09 (43 to 45 %, 54 to 55 %) were flown on older set-ups and scored the old way. |
| Field failures | The failures integrators actually report (VIO reset to zero, VIO restart, link loss, a frozen estimate, slow drift) are scripted and flown in Gazebo against tuned PX4 (`docs/field-faults.md`). |
| Evaluation kit | `./evaluate.sh` flies all five field failures with Structure Hold and with tuned PX4 in one command and prints the comparison table; a partner's real flight session becomes a one-page benchmark report (`docs/eval-kit.md`). |
| Wall-distance hover | `SH_HB_RANGE`, **on by default**: if the position estimate is lost next to a measured wall, the drone keeps hovering at its distance from the wall on the rangefinder alone instead of drifting. |
| Frozen-fix trip | `SH_WD_FRZ_T` (1 s), **on by default**: the watchdog trips when the VIO keeps sending a position that has stopped moving. |
| Pilot warnings | Every trip, refused engage and wall-guard action is sent to the ground station (QGroundControl) as a plain-English message, with the reason and the parameter behind it. |
| Module safety | The module clamps its own parameters, refuses to engage on the ground, stops on disarm, and follows a fixed, tested handback order (next to a wall, a guarded hover first, then stock PX4 modes). Its C++ tests run in CI. |

Gate dates: Gate 1 31 Jan 2027, Gate 2 30 Apr 2027, Gate 3 (real flight on a partner's
drone) 31 Jul 2027.

### Tools for a drone maker

- **Wind fit** (`eval/fit_wind.py`, `docs/wind-fit.md`): fits the simulator's drag, wind,
  turbulence and near-wall shelter from flight logs. Checked on Gazebo holds with known
  wind: within about 3 %.
- **Airframe fit** (`eval/fit_airframe.py`, `docs/airframe-fit.md`): turns one PX4
  autotune flight into a simulator model of the customer's drone (thrust, motor lag,
  torque, rate gains), so the policy can be fine-tuned on it. Checked on the Gazebo x500:
  within 1 to 3 %.
- **VIO error fit** (`eval/vio_fit.py`, `docs/vio-fit.md`): fits the simulator's estimate
  errors to real VIO. First fits done on the public EuRoC drone dataset, including our own
  Basalt VIO run.
- **Integration guide** (`docs/integration-guide.md`): the interface spec (inputs,
  outputs, modes, trips, parameters, messages, resource budget). Confidential, for drone
  makers under agreement.
- **Safety manual and failure table (FMEA)** (`docs/safety-manual.md`): operating limits,
  do-not-use cases, safety layers, pilot messages, and each failure mode with the test that
  covers it. A draft, not a certified document.

## Second capability: Route Follow

Fly an operator-given route (PX4 mission waypoints) through tunnels and corridors on the
same position estimate, with optional 1D rangefinders to keep off the walls
(`docs/route-follow.md`). Started 2026-11-01, simulator and CPU only so far. The first
policy follows the route 22 to 38 % tighter than the PD yardsticks across corridor and
tunnel routes. It is not a PX4 mode yet. Structure Hold stays the main product.

## Not in scope

VIO or SLAM, mapping, path planning, obstacle avoidance, a ground station, and vision in
the policy (`docs/decisions.md`). The module uses the drone's existing position estimate.

## Layout

| Folder | What lives there |
| --- | --- |
| `envs/` | Wind and near-wall model, estimate errors and an EKF2-like estimator, simulator backends, the Structure Hold and Route Follow tasks |
| `training/` | `launch.py` (runs everything, enforces the rules), training scripts, configs, released policies |
| `eval/` | Scoring (on true position), Gate 1, the tuned-PX4 baseline and SITL benchmark, the fit tools |
| `sim2real/` | PX4 SITL Docker image, Gazebo world and wind plugin, the estimate bridge and field faults, the PX4 module |
| `experiments/LOG.md` | One line per real run, failures included |
| `docs/` | Design notes, results and decisions (`decisions.md` is the record of what is settled) |

## Quick start

```
conda env create -f environment.yml && conda activate dronenav   # or: pip install -e .[dev,torch]
ruff check . && pytest -rs
python -m training.launch --config training/configs/selfcheck.yaml --smoke --device cpu
```

`launch.py` refuses a busy GPU, more than 1 GPU, or a run expected to take over 1 hour
unless `--approved` is passed. It writes each run to its own folder under `runs/` and logs
real (non-smoke) runs in `experiments/LOG.md`. More commands, including SITL and the PX4
module's tests, are in `CLAUDE.md` and `sim2real/px4/README.md`.

## Compute

Training so far runs on CPU in our own batched PyTorch simulator. A GPU server (multiple
RTX 4500 Ada) is being set up for Isaac Lab and parallel Gazebo flights; everything it
needs is prepared (`bash training/server_setup.sh`, `docs/gpu-server.md`, `docs/isaac-lab.md`).

## License

Proprietary. Copyright (c) 2026 Zain Ali, all rights reserved. No use, copying or
distribution without written permission; see `LICENSE`.
