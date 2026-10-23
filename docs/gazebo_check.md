# The Gazebo-like check: run it before any Gazebo re-fly

Two policies in a row (v7's path and v8) beat v6 on the Gate 1 test set and then flew worse
than v6 in Gazebo. The mock had none of what made them fail there. This page says what that
was, how the mock now models it, and the check a new policy must pass in the mock before it
goes to the Mac for a Gazebo re-fly (`docs/decisions.md`, 2026-10-23).

    python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu   # ~25 min

To judge a new policy, add it under `params.controllers` in that config (keep v10, the
reference: the released policy) and run it. The verdict is in the run's `report.md` and
`summary.json`.

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
| EKF2's resets to the fix | `envs/estimate/ekf.py` `FusionConfig.resets` | from 2026-10-23 06:30 |

EKF2 v1.17 does not only gate a fix that jumps: a horizontal fix rejected for more than 1 s,
a height rejected for more than 5 s, or the first fix after a gap of more than 0.4 s
(fusion stopped) each reset the estimate onto the fix in one step. The fusion first had only
the gate, so a jump in the mock was held off for good while in Gazebo it arrived a second
later as a step. With the resets the check's p95 sits closer to Gazebo's (next section).

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
Five cells: calm air on a clean estimate, and calm, 4 m/s and 8 m/s on `vio_default` with the
`wall1m_*` scenarios' gusts and turbulence, the wind from any direction, plus 8 m/s with the
mean wind blowing into the wall as Gazebo blows it (`wind8_wall`, from 2026-10-23 07:00;
see "The 8 m/s drift" below). Seeds 7001 and 7002, 64 episodes each, so 128 per cell and
controller.

Per episode, over the last 40 s (the first 20 s hold the engage transient):

- **lift**: the mean height of the estimate above the hold point, Gazebo's "offset up";
- **wobble**: the roll action's 1.2 to 2.2 Hz band has an rms above 0.1. In Gazebo the
  wobbling flights sit at 0.22 to 0.35, the quiet ones at 0.003 to 0.06;
- **p95** over the whole episode on the true centre of mass (`eval/metrics.py`), and crashes.

A policy **passes** (beats the reference, the released policy: v6 until v9's Gazebo flights,
then v9, and v10 since its Gazebo flights, PR #44) when, pooled over the rough cells:

1. its mean p95 is lower than the reference's (paired bootstrap 95 % interval of the ratio
   below 1);
2. its lift is no higher than the reference's;
3. its wobble rate is no higher than the reference's;
4. its crash rate is at most 1 point above the reference's;

and on the clean cell its p95 is within 0.02 m of the reference's. Only a policy that passes
goes to the Mac. The report prints Gazebo's own numbers for v6, v8, v9 and v10 beside the mock's,
so the check's fidelity stays in view; when a new policy flies in Gazebo, add its numbers
under `gazebo` in the config.

## How well it predicts Gazebo

Run `20261023-063008_gazebo_check_resets` (the four cells before `wind8_wall`, with EKF2's
resets; 26 min on 4 CPU cores), against what v6, v8 and v9 did in Gazebo (PR #35, PR #37,
PR #42; Gazebo's lift and wobble re-measured from the traces on this check's definitions).
The mock's 4 and 8 m/s winds here come from any direction; Gazebo's blow into the wall.

| cell | controller | mock p95 (m) | mock lift (m) | mock wobble | Gazebo p95 / lift / wobble |
| --- | --- | --- | --- | --- | --- |
| calm, clean | v6 | 0.08 | +0.02 | 0 % | 0.11 / +0.04 / 0 of 1 |
| calm, clean | v8 | 0.19 | +0.09 | 20 % | 0.11 / -0.01 / 0 of 1 |
| calm, clean | v9 | 0.08 | +0.02 | 0 % | 0.08 / +0.04 / 0 of 1 |
| calm | v6 | 0.58 | +0.18 | 65 % | 0.62 / +0.21 / 2 of 3 |
| calm | v8 | 0.71 | +0.49 | 95 % | 0.67 / +0.39 / 3 of 3 |
| calm | v9 | 0.42 | +0.05 | 0 % | 0.44 / +0.06 / 0 of 3 |
| 4 m/s | v6 | 0.60 | +0.18 | 59 % | 0.64 / +0.23 / 2 of 3 |
| 4 m/s | v8 | 0.73 | +0.47 | 93 % | 0.68 / +0.41 / 3 of 3 |
| 4 m/s | v9 | 0.44 | +0.05 | 0 % | 0.49 / +0.07 / 0 of 3 |
| 8 m/s | v6 | 0.64 | +0.20 | 69 % | 0.68 / +0.32 / 2 of 3 |
| 8 m/s | v8 | 0.72 | +0.45 | 94 % | 0.79 / +0.48 / 3 of 3 |
| 8 m/s | v9 | 0.46 | +0.05 | 0 % | 0.64 / +0.06 / 0 of 3 |

Pooled over the rough cells:

| controller | mock p95 / lift / wobble | Gazebo p95 / lift / wobble |
| --- | --- | --- |
| v6 | 0.61 m / +0.19 m / 64 % | 0.65 m / +0.26 m / 6 of 9 |
| v8 | 0.72 m / +0.47 m / 94 % | 0.71 m / +0.43 m / 9 of 9 |
| v9 | 0.44 m / +0.05 m / 0 % | 0.52 m / +0.06 m / 0 of 9 |

The mock ranks the three as Gazebo does (v9, then v6, then v8) and gets v9's lift and the
absence of any wobble right. Before the resets (run `20261023-044747`, v6 and v8 only) the
p95s were 0.53 and 0.68 and v8's penalty over v6 29 %; with them 0.61 and 0.72 and 19 %,
against Gazebo's 10 %.

Where it is off:

- v9's 8 m/s: 0.46 m against 0.64. One of v9's three 8 m/s Gazebo flights drifted 1 m off
  the hold point; with the wind from any direction the mock rarely shows that. With the wind
  into the wall it does (next section), hence the `wind8_wall` cell.
- The mock's truth is the centre of mass; Gazebo's is the model frame 0.18 to 0.24 m below
  it, which swings in a wobble: one reason Gazebo's p95 runs a few centimetres above the
  mock's.
- Crashes: v6 crashed in 4.2 % of the mock's rough episodes, v9 in 1.3 %; none of the 27
  Gazebo flights did. The check judges crashes only against the reference.
- v8 wobbled in 20 % of the clean episodes; its one clean Gazebo flight did not. One flight
  cannot tell those apart.

Each new Gazebo flight is a test of the check: put its numbers under `gazebo` in the config
and see whether the mock still ranks the policies the way Gazebo does.

### v10: the first forward test

v10 was the first policy released on this check as it stands (EKF2's resets, `wind8_wall`),
so its Gazebo flights (PR #44, 2026-10-23) test the check's prediction rather than fit it.
Mock: run `20261023-074020`. Gazebo: the module thread's clean flight and nine `vio_default`
flights, re-measured from the traces on this check's definitions.

| cell | v10 mock p95 / lift / wobble | v10 Gazebo p95 / lift / wobble | v9 mock p95 | v9 Gazebo p95 |
| --- | --- | --- | --- | --- |
| calm, clean | 0.04 / +0.03 / 0 % | 0.04 / +0.03 / 0 of 1 | 0.08 | 0.08 |
| calm | 0.34 / +0.04 / 0 % | 0.38 / +0.04 / 0 of 3 | 0.42 | 0.44 |
| 4 m/s | 0.34 / +0.04 / 0 % | 0.38 / +0.05 / 0 of 3 | 0.44 | 0.49 |
| 8 m/s | 0.36 / +0.03 / 0 % | 0.39 / +0.05 / 0 of 3 | 0.46 | 0.64 |
| 8 m/s into the wall | 0.36 / +0.05 / 0 % | (Gazebo's 8 m/s flights) | 0.57 | 0.64 |

Pooled over the rough cells, v10's p95 over v9's was 0.73 in the mock and 0.73 in Gazebo
(0.38 m against 0.52 m over the nine flights). The mock called the size of the improvement,
the absence of any wobble, the lift to within 0.01 m, and the fix for v9's drift (next
section). Its p95 still sits 0.03 to 0.04 m under Gazebo's in each rough cell, the few
centimetres the truth point accounts for (above). The benchmark's own scoring (`docs/baseline.md`) puts v10 at 0.37 / 0.38 / 0.39 m against
tuned PX4's 0.57 m.

## The 8 m/s drift (v9)

In one of v9's three 8 m/s Gazebo flights (`wall1m_wind_8ms`, seed 2) the drone slid away
from the wall and along it, from about 18 s into the hover to 1.0 m off the hold point at
44 s, at a few centimetres per second, then came back within 6 s after a 2 s dropout. The
estimate followed the truth all along (position within 0.05 m, velocity within 0.01 m/s)
and the module's position error showed the whole offset: the policy saw it and did not
correct it. Nothing tripped, nothing saturated, the lean held steady at about 2.5 degrees
against the wind.

The mock does the same, on a clean estimate and without turbulence or gusts, as soon as the mean
wind blows into the wall (60 s, 8 episodes per row, `gz_x500_sitl`; a probe outside the
check, the turbulence of the 8 m/s scenario does not change it: 1.06 m and 0.45 m):

| policy | wind | largest offset (m) | mean offset, last 10 s (m) |
| --- | --- | --- | --- |
| v9 | 8 m/s into the wall | 1.08 | 0.58 |
| v9 | 8 m/s along the wall, either way | 0.08 to 0.14 | 0.02 to 0.11 |
| v9 | 8 m/s away from the wall | 0.06 | 0.02 |
| v9 | 4, 6, 10, 12 m/s into the wall | 0.85, 1.10, 1.03, 0.84 | 0.25, 0.75, 0.39, 0.23 |
| v6 | 8, 10, 12 m/s into the wall | 0.15, 0.18, 0.25 | 0.03, 0.10, 0.15 |

The mock's lean at 8 m/s is 2.3 degrees, Gazebo's 2.2 to 3.7, so the wind force matches. v9's
training could not see this: 20 s episodes, the drift takes 20 to 40 s to grow, and the wind
blows into the wall in only a fraction of episodes. Gazebo's other two 8 m/s flights and its
4 m/s flights did not drift within 60 s; the mock's drift is stronger than Gazebo's.

v10 (`training/configs/structure_hold_ppo_v10.yaml`) is v9 trained 1000 iterations more with
60 s episodes and the mean wind always blowing with a component into the wall. In the same
probe it holds within 0.09 m at 8 m/s into the wall and 0.19 m at 12 m/s, and does not drift
in any other direction. In Gazebo none of its three 8 m/s flights drifted; seed 2, v9's
drift flight, held 0.35 m p95 with a horizontal offset of 0.04 m (v9 1.01 m and 0.46 m;
`docs/baseline.md`).

Every policy so far also holds its heading about 11 degrees (0.2 rad) off the wall normal,
in Gazebo and in the mock alike: a trait of the policy, not a gap in the mock. The heading
penalty, `heading` times 1 - cos(error), is nearly flat that close to zero.

## Policies judged

| policy | check run | pooled rough p95 / lift / wobble | verdict | Gazebo |
| --- | --- | --- | --- | --- |
| v6 | reference | 0.53 m / +0.18 m / 65 % | reference | 0.65 m / +0.26 m / 6 of 9 |
| v8 | `20261023-044747` | 0.68 m / +0.47 m / 93 % | does not beat v6 | 0.71 m / +0.43 m / 9 of 9 |
| v9 | `20261023-053223` | 0.40 m / +0.05 m / 0 % | beats v6 (p95 ratio 0.75) | 0.52 m / +0.06 m / 0 of 9; 8 m/s drift in 1 of 3 (PR #42) |
| v10 | `20261023-074020` (five cells, with resets) | 0.35 m / +0.04 m / 0 % (v9 0.47 m / +0.05 m / 0 %) | beats v9 (p95 ratio 0.73) | 0.38 m / +0.05 m / 0 of 9, no drift (PR #44); the reference since |
