# From v10 to a real flight: what is left

Policy v10 holds 1 m from the wall in Gazebo with a p95 of 0.37 to 0.39 m against tuned
PX4's 0.57 m, on three seeds per wind (`docs/baseline.md`). It is the first policy to beat
tuned PX4 in every wind. Every number so far comes from two simulators, the mock
(`envs/`) and PX4 SITL in Gazebo (`sim2real/`), and the two share more than it seems. This
page lists what both assume that a real drone will not give us, so the next training target
comes from the real-flight gaps rather than from more mock polishing (2026-10-23).

## What Gazebo did and did not test

Gazebo replaced the mock's physics, PX4's controllers and EKF2 with the real code: the
rotor model, PX4 v1.17's rate loop and filters, EKF2, the module's C++ and its timing.
v10 passing there means the policy survives PX4 as it runs on the drone.

Gazebo did **not** replace three models, because the benchmark feeds it ours:

- **The estimate.** The bridge runs Gazebo's true pose through `envs/estimate`, the same
  corruption the policy trained on (`sim2real/estimate_bridge/README.md`).
- **The wind.** The wind plugin is a C++ copy of `envs/disturbance`, held equal by a parity
  test (`docs/decisions.md`).
- **The structure.** Gazebo's wall is the same flat, infinite plane the mock uses.

So Gazebo has never tested the policy against a real estimate, real wind near a real
structure, or a real airframe. Those are the gaps below.

## The gaps, largest first

| # | What both simulators assume | What a real drone has | Risk | How to close it |
| --- | --- | --- | --- | --- |
| 1 | **Estimate**: a made-up VIO. The delay is 50 to 150 ms, with noise, a bias walk, jumps and dropouts. Rates and sizes come from papers (ORB-SLAM3, the qVIO class), not from a flight (`docs/decisions.md`, estimate defaults). | The drone's own VIO, looking at a wall 1 to 3 m away. How good it is depends on the wall's texture and the light. It has its own reset behaviour and reports its own covariance. | **High.** The whole robustness claim is measured against this model. | Software only (2026-10-30): fit `envs/estimate` to real VIO error from public datasets with truth (`eval/vio_fit.py`, `docs/vio-fit.md`). First fit, 12 EuRoC flights: the drift when hovering matches the default, noise and jumps are milder, the 60 s drift reaches 0.4 m against the 0.3 m cap. Delay and tracking loss still need an own VIO run; a partner flight in front of a wall is the final check. |
| 2 | **Wind near a structure**: even turbulence (a seeded sum of sinusoids), steady shelter near a flat wall fitted to one published data point, and no up- or downdrafts. Not modelled: wall-induced yaw torque (training adds a random one, up to 0.02 N m), ground effect, an overhead slab (`envs/disturbance/README.md`). | Swirling air on the wall's lee side, vortex shedding off edges, updrafts along a face, and gusts that change with height. | **High** for the gust claim, which is the product. | The Phase 3 fan rig, plus outdoor logs (estimate and IMU) near a real structure. Until then, randomise the shelter and add vertical gusts. |
| 3 | **Heading**: the module takes the heading at engage as the wall normal (`StructureHold.cpp`: structure x = the heading at engage). Heading comes from the magnetometer, which is clean in Gazebo. Every policy holds about 11 deg off that heading (`docs/gazebo_check.md`). | Magnetometer errors near steel, such as pylons, bridges and turbines. The pilot engages at whatever angle they happen to face. | **Medium to high.** A wrong wall normal turns the hold into a slow slide along or into the wall. | Fuse VIO yaw in EKF2 instead of the magnetometer (`EKF2_EV_CTRL` yaw bit). Train with an error between the engage heading and the true wall normal, and with heading drift. Neither option exists yet. |
| 4 | **Airframe**: the Gazebo x500 (2.06 kg, SDF rotor constants). Training randomises mass (0.85 to 1.2), inertia, thrust-to-weight, motor lag and PX4's hover-thrust error (`training/configs/structure_hold_ppo_v10.yaml`). | The test drone, Starling 2 (specs not entered yet) or the fallback X500 V2. Each has its own props, motors and battery sag over a flight. | **Medium** on an X500 V2. **High** on a Starling 2, a different size class v10 has never flown. | Measure hover thrust and the thrust curve, from a stand or a hover log. Add the drone as an airframe in `envs/structure_hold/airframes.py`, then fine-tune on it. |
| 5 | **Rate loop and tick**: the rate gains autotuned on the Gazebo x500 (`eval/baselines/px4_tuned_v1.params`). The module ticks every 12 ms (83 Hz), because Gazebo's gyro arrives every 4 ms. v10 trained at 12 ms only, with 0 to 30 ms of extra rate lag. | The real drone's own tuned `MC_*RATE_*` gains. A faster gyro, so the module ticks close to every 10 ms. | **Medium**, and cheap to check. | Fly v10 at a 10 ms tick on `gz_x500_sitl` in the mock (one check run). Randomise the tick (10 to 12 ms) and the rate gains in training. |
| 6 | **Sensors**: the policy reads the true attitude and a filtered true gyro. There is no vibration and no accelerometer bias beyond the fusion's small IMU noise. | Vibration, gyro noise and attitude-estimate error. | **Medium.** In the mock, noisy rate or velocity inputs lift the hold point (`SensorNoise` docstring). | `SensorNoise` exists, off by default (`envs/structure_hold/config.py`). v7 trained with it and sat 0.08 m low on a clean estimate, so it was not released. Set the noise levels from a real hover log, not by guess. |
| 7 | **EKF2 tuning**: the mock's fusion is a simplified EKF2 matched to Gazebo. In Gazebo, EKF2 fused vision 154 ms behind, against `EKF2_EV_DELAY` 100 ms (`docs/baseline.md`). | Real EKF2 on the real VIO's latency and covariance. | **Medium.** v10 still beats v9 in the check with the fix 150 to 200 ms late (run `20261023-084937`). | Measure the VIO latency and set `EKF2_EV_DELAY` to it. Keep randomising the mismatch in training. |
| 8 | **Where the fix is**: Gazebo sends the model frame, 0.24 m below the IMU, with no lever-arm correction. | The camera's known offset, which `EKF2_EV_POS_*` corrects. | **Low.** The real case is the easier one. | Set the offset on the drone. Keep the lever arm in the mock as a stress. |
| 9 | **Safety layer**: the watchdog, the fallback and the hand-back to Position mode are tested in SITL only. In v10's Gazebo flights there were no trips, and handback took 0.016 to 0.02 s. | RC override, PX4 failsafes, a low battery, a lost companion link, and a VIO restart in flight. | Needed before any flight. **Not a training target.** | A props-off bench test on the flight controller (tick time, inference time, parameters, trips by `structure_hold trip`). Then a tethered hover, then a mocap room, flying tuned PX4 first. |
| 10 | **Truth for scoring**: the simulators score the true position (`docs/metrics.md`). | Truth has to come from mocap indoors or RTK outdoors, logged but never fused. | Needed for every real number. | Book mocap, or carry RTK, from the first real flight. |

Two product gaps sit outside the table because no amount of training closes them. The hold
point is the engage point, so standoff from the wall is the pilot's job. Gazebo showed what
that costs (PR #49, `docs/gazebo_check.md` "v11 in Gazebo"): a 1.4 m estimate jump toward
the wall just before a switch to stock Position mode, which moves its setpoint with EKF2's
reset, put the next engage's hold point on the wall face, and the props struck it as the
estimate recovered. Stock PX4 walks into the wall the same way, so neither the policy nor an
engage guard on the estimate alone prevents it; a measured distance to the structure (a
rangefinder feeding PX4's collision prevention, or the camera's depth to the wall) as a
floor under the hold would. The module half of that floor exists since 2026-10-27
(`SH_WALL_MIN`, the wall guard in `sim2real/px4/README.md`): with a forward rangefinder the
hold point stays at least 0.7 m off the measured wall. The stock Position mode half is PX4
Collision Prevention on the same sensor, so the test drone needs a forward rangefinder
fitted. And the policy
assumes a flat wall, while real structures include lattice towers and blades (both in
`sim2real/px4/README.md`, "Not here yet"; mapping is out of scope in `docs/decisions.md`).

## What Isaac Lab would change, and what it would not

As built (`envs/sims/isaac_quad.py`, `docs/isaac-lab.md`), Isaac Lab moves one PhysX rigid
body per drone. Its mass and inertia, motor lag, clipping and wind all come from the mock's
code, with no drag or damping of its own. So it brings GPU throughput, not fidelity:

- Throughput buys bigger batches, wider randomisation (rows 2 to 7 together), longer
  episodes, and maybe a recurrent policy.
- It closes none of rows 1 to 4 by itself. Those need models or data we do not have yet:
  real VIO logs, a fitted wind field, the real airframe's numbers.

Isaac matters once wide randomisation gets too slow on 4 CPU cores. It is not on the path to
the first real flight.

## What only hardware gives

Rows 1, 3, 4, 6, 9 and 10 need the drone, or at least its VIO and IMU in hand. The cheapest
order:

1. **A logging flight with no policy flying.** Fly by hand, or in Position mode on RTK, in
   front of a wall. Log VIO, IMU, magnetometer and truth. That one log fits rows 1, 3, 6 and
   7, and gives the hover thrust for row 4. Checklist and logger settings: `docs/logging-flight.md`.
2. **A props-off bench test** of the module on the flight controller (row 9, and row 5's
   tick). Procedure and script: `docs/bench-test.md`.
3. **A tethered or mocap hover**: tuned PX4 first, then the released policy, with the
   watchdog armed.

The decision record says to order hardware by March 2027. The logging flight is the step
that most changes what we train next. If that order came earlier, the time would go to
training against real data, not guessed data.

## Next training target

Until real logs exist, train for what is known to differ, not for a better Gazebo score:

- **v11 = v10 plus the cheap gaps.** Randomise the tick (10 to 12 ms) and the rate gains
  (row 5). Add an error between the engage heading and the wall normal, with a slow heading
  drift (row 3). Add mild sensor noise below v7's levels (row 6). Use wider shelter and
  vertical gusts (row 2). Judge it by `eval/configs/gazebo_check.yaml`, where it must not
  lose to v10, plus a stress set of those same conditions where it must beat v10.
- **Then the real estimate.** Once a logging flight exists, replace or refit `envs/estimate`
  from it (row 1). That retrain matters more than any of the above.

Do not spend more runs closing the last 0.03 m between the mock and Gazebo. That gap is the
truth point, not the policy (`docs/gazebo_check.md`), and it says nothing about the real
drone.
