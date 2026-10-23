# The Gazebo-like check: run it before any Gazebo re-fly

Two policies in a row (v7's path and v8) beat v6 on the Gate 1 test set and then flew worse
than v6 in Gazebo. The mock had none of what made them fail there. This page says what that
was, how the mock now models it, and the check a new policy must pass in the mock before it
goes to the Mac for a Gazebo re-fly (`docs/decisions.md`, 2026-10-23).

    python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu   # ~15 min

To judge a new policy, add it under `params.controllers` in that config (keep v6, the
reference) and run it. The verdict is in the run's `report.md` and `summary.json`.

## What Gazebo showed

The module thread exported every tick of 20 Gazebo flights (v6 and v8: one clean estimate,
nine `vio_default` flights each; `sim2real/px4/tools/export_traces.py`). Replaying them
through the policy offline reproduces the logged actions (mean error 0.005), so the traces
are what the policy saw.

- **The lift is a wobble.** Each flight is either quiet (the estimate 0.03 m above the hold
  point) or locked in a 1.6 Hz roll and yaw oscillation that never stops once it starts. It
  starts at an estimate jump, at a dropout and the EKF reset after it, or right from engage.
  Its oscillating rate and last-action inputs push the policy's mean thrust up by about
  0.05, which holds the drone 0.2 to 0.5 m high. v6 wobbled in 6 of 9 rough flights, v8 in
  9 of 9; neither on the clean estimate.
- **It runs through yaw.** PX4 v1.17 low-passes the yaw torque at `MC_YAW_TQ_CUTOFF` = 2 Hz.
  A yaw-rate command answers with gain 0.27 and 130 degrees of lag at 1.6 Hz in Gazebo; the
  mock gave 0.32 and 80 degrees.
- **The module ticks at 83 Hz**, not 100: it runs on gyro samples once 10 ms have passed,
  and Gazebo's gyro comes at 250 Hz.
- **The fix is not the centre of mass.** The bridge sends the pose of the x500 model frame,
  0.24 m below the IMU, and EKF2 has no lever-arm correction. When the drone rolls, the fix
  swings sideways and its velocity carries the roll rate.
- **The module reads EKF2, not the fix.** EKF2 fuses the fix at its assumed 100 ms delay
  with the accelerometer and smooths what the module sees.

## What the mock now has

All off by default, so every earlier result (Gate 1, training runs) reproduces unchanged.
The airframe `gz_x500_sitl` (`envs/structure_hold/airframes.py`) is `gz_x500` with all of
them on:

| piece | where | setting |
| --- | --- | --- |
| yaw torque low-pass | `envs/sims/rate_pid.py` `yaw_torque_cutoff_hz` | 2 Hz (`MC_YAW_TQ_CUTOFF`) |
| gyro low-pass, seen by the rate loop and the policy | `envs/sims/rate_pid.py` `gyro_cutoff_hz` | 40 Hz (`IMU_GYRO_CUTOFF`) |
| module tick | config `dt` | 0.012 s |
| fix below the CG | config `vision_offset_down_m` | 0.24 m |
| EKF2-like fusion | `envs/estimate/ekf.py`, config `fusion` | the bridge's `ekf2_ev.params` |
| the policy's rate input | `envs/structure_hold/env.py` `_observe` | the filtered gyro, one tick old |

The last row is a calibration. The mock has no actuator or transport latency, which SITL
has; giving the policy the filtered gyro as the rate loop last read it (12 ms old) stands in
for it. With the current sample instead, lift and wobble came out about 40 % below Gazebo's
(v6 +0.15 m, v8 +0.26 m; runs in `experiments/LOG.md`, 2026-10-23 04:28 and 04:40).

Each piece alone does little; together they reproduce Gazebo. Without the PX4 filters
nothing wobbled; at a 100 Hz tick v6 wobbled in 9 % of episodes instead of about 65 %;
without the lever arm everything wobbled, clean estimate included; without the fusion (the
raw fix and the lever arm) the policies crashed or wobbled throughout.

## The check

`eval/gazebo_check.py` flies each controller on `gz_x500_sitl` the way the benchmark flies
in Gazebo: 60 s, 1 m from the wall, the hold point 2.4 m up, the engage at the hold point.
Four cells: calm air on a clean estimate, and calm, 4 m/s and 8 m/s on `vio_default` with the
`wall1m_*` scenarios' gusts and turbulence (the mock draws the wind direction; Gazebo blows
into the wall). Seeds 7001 and 7002, 64 episodes each, so 128 per cell and controller.

Per episode, over the last 40 s (the first 20 s hold the engage transient):

- **lift**: the mean height of the estimate above the hold point, Gazebo's "offset up";
- **wobble**: the roll action's 1.2 to 2.2 Hz band has an rms above 0.1. In Gazebo the
  wobbling flights sit at 0.22 to 0.35, the quiet ones at 0.003 to 0.06;
- **p95** over the whole episode on the true centre of mass (`eval/metrics.py`), and crashes.

A policy **passes** (beats v6) when, pooled over the three rough cells:

1. its mean p95 is lower than v6's (paired bootstrap 95 % interval of the ratio below 1);
2. its lift is no higher than v6's;
3. its wobble rate is no higher than v6's;
4. its crash rate is at most 1 point above v6's;

and on the clean cell its p95 is within 0.02 m of v6's. Only a policy that passes goes to
the Mac. The report prints Gazebo's own numbers for v6 and v8 beside the mock's, so the
check's fidelity stays in view; when a new policy flies in Gazebo, add its numbers under
`gazebo` in the config.

## How well it predicts Gazebo

Run `20261023-044747_gazebo_check_s7001_1a8b463` (12 min on 4 CPU cores), against what v6
and v8 did in Gazebo (PR #35, PR #37; Gazebo's lift and wobble re-measured from the traces on
this check's definitions):

| cell | controller | mock p95 (m) | mock lift (m) | mock wobble | Gazebo p95 / lift / wobble |
| --- | --- | --- | --- | --- | --- |
| calm, clean | v6 | 0.08 | +0.02 | 0 % | 0.11 / +0.04 / 0 of 1 |
| calm, clean | v8 | 0.19 | +0.09 | 20 % | 0.11 / -0.01 / 0 of 1 |
| calm | v6 | 0.49 | +0.17 | 63 % | 0.62 / +0.21 / 2 of 3 |
| calm | v8 | 0.68 | +0.49 | 95 % | 0.67 / +0.39 / 3 of 3 |
| 4 m/s | v6 | 0.50 | +0.17 | 61 % | 0.64 / +0.23 / 2 of 3 |
| 4 m/s | v8 | 0.68 | +0.48 | 94 % | 0.68 / +0.41 / 3 of 3 |
| 8 m/s | v6 | 0.60 | +0.19 | 70 % | 0.68 / +0.32 / 2 of 3 |
| 8 m/s | v8 | 0.69 | +0.44 | 91 % | 0.79 / +0.48 / 3 of 3 |

Pooled over the rough cells the mock has v6 at +0.18 m and 65 % wobbling, v8 at +0.47 m and
93 %, v8's p95 29 % worse than v6's: v8 does not beat v6, which is what Gazebo said (+0.26 m
and 6 of 9, +0.43 m and 9 of 9, p95 10 % worse). On the old mock (Gate 1) v8 beat v6.

Where it is off:

- The mock exaggerates v8's p95 penalty (29 % against 10 %) and under-states v6's lift at
  8 m/s (+0.19 m against +0.32 m). It ranks them right; the margins are approximate.
- The mock's truth is the centre of mass; Gazebo's is the model frame 0.18 to 0.24 m below
  it, which swings in a wobble: probably one reason Gazebo's p95 runs 0.03 to 0.15 m above
  the mock's.
- The mock draws the wind direction; Gazebo blows into the wall. v6 crashed in 4.7 % of the
  8 m/s episodes (none in Gazebo's three); the check judges crashes only against v6.
- v8 wobbled in 20 % of the clean episodes; its one clean Gazebo flight did not. One flight
  cannot tell those apart.

Each new Gazebo flight is a test of the check: put its numbers under `gazebo` in the config
and see whether the mock still ranks the policies the way Gazebo does.
