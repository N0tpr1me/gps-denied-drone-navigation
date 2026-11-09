# The Gazebo-like check: run it before any Gazebo re-fly

Two policies in a row (v7's path and v8) beat v6 on the Gate 1 test set and then flew worse
than v6 in Gazebo. The mock had none of what made them fail there. This page says what that
was, how the mock now models it, and the check a new policy must pass in the mock before it
goes to the Mac for a Gazebo re-fly (`docs/decisions.md`, 2026-10-23).

    python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu   # ~50 min

That config flies v19 (the reference: the released policy), v10 and v18. To judge a new
policy, copy it with v19 and the candidate under `params.controllers` (about 35 min; as
`gazebo_check_v19.yaml` did with v10 and v19) and run it. The verdict is in the run's `report.md` and
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
then v9, v10 from its Gazebo flights (PR #44), and v19 since its Gazebo flights, PR #61) when, pooled over the rough cells:

1. its mean p95 is lower than the reference's (paired bootstrap 95 % interval of the ratio
   below 1);
2. its lift is no higher than the reference's, or under 0.05 m above the hold point
   (`lift_max_m`; a soft rule since 2026-11-03: the rule was written against v5's 0.3 to
   0.45 m Gazebo lift, and v24 failed it alone by sitting 1.5 cm above its hold point where
   v22 sits 1.7 cm below, `docs/wall-range.md`);
3. its wobble rate is no higher than the reference's;
4. its crash rate is at most 1 point above the reference's;
5. its **ring** is no more than 5 % above the reference's: the median, over episodes, of the
   roll action's 1.2 to 2.2 Hz rms (from 2026-10-23 14:00, after v11's Gazebo flights; see
   "v11 in Gazebo" below);

and on the clean cell its p95 is within 0.02 m of the reference's. Two more cells fly the
benchmark's post-hold sequence (`fly_reengage`, after `sim2real/px4/sitl_test.py`): hold
for 20 s, switch to stock Position mode (the PD, holding the estimate where the switch found
it and moving that setpoint with each EKF2 reset, as PX4's flight tasks do) for 5.4 s, then
re-engage, which captures the hold point from the estimate as `StructureHold::engage` does,
and fly 14.6 s more, at 4 m/s. In `reengage` (`vio_default`) a policy must have no
**re-engage failure** (a crash after a re-engage whose hold point is at least 0.6 m off the
wall) and crash at most 1 point more than the reference during the hold. Crashes under stock
PX4 and after a re-engage at the wall are printed, not scored: the sequence causes them, for
every controller. `reengage_jump` adds Gazebo seed
3's 1.4 m vision jump toward the wall 0.5 to 3 s before the switch; it is printed, not
scored. From 2026-10-23 the check scores wall contact at the x500's 0.40 m centre distance
(rotors 0.24 to 0.26 m out, 0.14 m props), not the scorer's 5-inch 0.25 m.

**Replay cells** (`events` in the config, from 2026-10-23 18:00, after v12's Gazebo flights)
replay a Gazebo flight's estimate faults. Scripted jumps and dropouts start at the same
moment in every episode, and the random ones are off. `seed1_jump` is Gazebo seed 1's 1.8 m
jump along the wall (see "v12 in Gazebo" below). A replay cell is scored on its own, not in
the pooled p95:
- the mean **worst toward the wall** (the largest true displacement along the wall normal)
  may be at most 0.05 m above the reference's;
- crashes may be at most 1 point above the reference's.

Every cell now prints the worst toward the wall.

Only a policy that passes goes to the Mac. The report prints Gazebo's own numbers beside the
mock's, so the check's fidelity stays in view:
- v10 and v12 on the matched fault timeline (PR #52), with tuned PX4's on a line under the
  table;
- v6 to v9 on the older start-keyed one.

When a new policy flies in Gazebo, add its numbers under `gazebo` in the config.

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

## v11 in Gazebo: the seed 3 wall strike and the 1.7 Hz ring

v11 held tighter than v10 in every Gazebo hold (PR #49: mean p95 0.33 / 0.32 / 0.33 m
against 0.37 / 0.38 / 0.39 m), as the check predicted, but lost control after the
scripted re-engage in all four seed 3 flights, and rang at 1.7 Hz on 2 of 9. v10 stayed
the module's policy.

**Seed 3 is a prop strike, set up by the sequence, not by the policy.** In each of the four
ulogs (calm, 4 m/s twice, 8 m/s):

| step | v11, seed 3 (all four) | v10, seed 3, 4 m/s (PR #44) |
| --- | --- | --- |
| vision jump, about 1.4 m toward the wall | 91.8 to 92.5 s | 91.3 s |
| EKF2 horizontal reset onto it (+1.12 to +1.15 m) | 92.8 to 93.6 s | 92.3 s |
| scripted switch to stock Position mode | from 0.4 s after the reset to 0.04 s before it | 1.3 s after it |
| hold point the re-engage captured (E, the wall at 0) | +0.03 to +0.10 m | −0.42 m |
| true centre distance to the wall when the spin starts | 0.35 to 0.40 m (the props reach up to 0.40 m) | none; closest 0.54 m |

PX4's Position mode moves its setpoint with an EKF2 reset, so it holds the jumped estimate,
and as the jump decays it walks the drone toward the wall (1.16 m off it to 0.55 m in 5.4 s).
The re-engage then captures its hold point from that estimate, on the wall face, and any
controller that holds it closes on the wall as the estimate recovers. In the calm flight the
spin began at 101.6 s, just after the test-fault handback, with stock PX4 in charge. v10
missed it on timing: its reset landed inside its own hold, which keeps the hold point it
engaged with, so it backed away before the switch. The policy sees only the estimate minus
the hold point, so no training can prevent this.

`reengage_jump` reproduces it (seed 7001, 64 episodes): v10 crashes in 91 %, v11 in 94 %, the
PD flying every phase in 62 %. Holding the re-engage off for 10 s after a reset leaves v10
at 92 %, the crashes moving into the stock phase. The policies do worse than the PD because
they answer a sudden offset slowly (v10 moves 0.25 m of a 1.1 m reset step in 1.7 s), which
is right inside a hold and wrong when a switch captures the estimate. What would prevent
the strike is a measured distance to the structure (`docs/real-flight-gap.md`, the product
gaps under the table).

**The ring is v11's own.** The thresholded wobble rate scored both at 0 %, but v11's median
1.2 to 2.2 Hz roll rms was above v10's in every rough cell (pooled 0.0150 against 0.0128,
+17 %; run `20261023-122516`), the same direction as Gazebo's ring (roll-rate std 0.10
against 0.08 rad/s). Hence rule 5. v11's steeper position penalty raised its gain, and so
its ring for the lag it flies with. Not v11's own: the ~10 degree heading offset (v10
+10.7, v11 +10.2 degrees in `wind8`) and a full yaw command for the first 50 ms of every
engage (both).

## v12 in Gazebo: the drift toward the wall on seed 1

v12 flew against v10 and tuned PX4 on the matched fault timeline (`vio_at_engage` for the
hold runner, `vio_at_hover` for stock PX4; PR #52, `docs/baseline.md`). Its hold p95 was
looser than v10's in 8 of 9 flights (mean 0.35 against 0.33 m) while it beat v10 on rms, on
the ring and on the offset up. Seed 1 is where it loses. The bridge's draws for seed 1
(replayed offline from `sim2real/estimate_bridge`) put a 1.80 m vision jump 35.14 s into the
hold, (−0.28, +1.78, −0.11) m ENU: nearly all of it along the wall, 0.28 m of it away from
the wall, decaying with the 5 s time constant. A 0.66 s dead-reckoned dropout follows 4.8 s
later. The module traces (calm air; the winds look the same):

| after the jump | v10 | v12 |
| --- | --- | --- |
| along the wall, true | follows the jumped estimate about 0.4 m | stays within 0.1 m |
| heading | +6 degrees, back within 4 s | +17 degrees over 4 s, back over the next 9 s |
| toward the wall, true | at most 0.04 m | 0.27 m 5 s after the jump, back after 13 s |

v12 does not chase the jump along the wall. That keeps its p95 low in the mock. But it yaws
17 degrees and drifts toward the wall, then leaves 0.15 to 0.2 m of error toward the wall
standing for about 6 s while 0.3 to 0.5 m of the jump along the wall is still in the
estimate. v10 chases the jump along the wall and keeps its distance from it.

**The mock reproduces it.** The same jump and dropout replayed on `gz_x500_sitl`, with the
random jumps and dropouts off, 64 episodes per run:

| replay | v10 | v11 | v12 |
| --- | --- | --- | --- |
| jump and dropout, calm / 4 m/s / 8 m/s into the wall | 0.08 / 0.07 / 0.05 m | 0.06 / 0.05 / 0.06 m | 0.26 / 0.26 / 0.25 m |
| jump only, calm | 0.06 m | | 0.26 m |
| dropout only, calm | 0.05 m | | 0.05 m |
| the jump's along-wall part only (1.8 m), calm | 0.03 m | | 0.22 m |
| Gazebo, seed 1, 0 / 4 / 8 m/s | 0.04 / 0.03 / 0.07 m | not flown | 0.27 / 0.28 / 0.31 m |

The numbers are the mean worst toward the wall. The jump does it, the dropout does not, and
the along-wall part alone still draws v12 toward the wall. The mock's p95 on the replay is
the wrong way round, though: v12 0.29 m and v10 0.49 m, against Gazebo's 0.42 and 0.43 m on
seed 1. The mock's v10 chases the jump further than Gazebo's. So the check scores the drift
itself.

The new cell `seed1_jump` replays the jump and dropout at 4 m/s. Its rule: the mean worst
toward the wall may be at most 0.05 m above the reference's, and crashes at most 1 point
above. The cell stays out of the pooled p95.

On the full check (run `20261023-170223`, 128 episodes per cell), `seed1_jump` gives v12
0.26 m toward the wall against v10's 0.08 m, so v12 no longer passes. It fails that rule
only. The random-fault cells would not have caught it: pooled over them, v12 is 0.10 m
toward the wall and v10 0.11 m. A jump of nearly 2 m mostly along the wall is rare in their
draws. In Gazebo's three seeds, seed 1 is that jump.

## Policies judged

| policy | check run | pooled rough p95 / lift / wobble | verdict | Gazebo |
| --- | --- | --- | --- | --- |
| v6 | reference | 0.53 m / +0.18 m / 65 % | reference | 0.65 m / +0.26 m / 6 of 9 |
| v8 | `20261023-044747` | 0.68 m / +0.47 m / 93 % | does not beat v6 | 0.71 m / +0.43 m / 9 of 9 |
| v9 | `20261023-053223` | 0.40 m / +0.05 m / 0 % | beats v6 (p95 ratio 0.75) | 0.52 m / +0.06 m / 0 of 9; 8 m/s drift in 1 of 3 (PR #42) |
| v10 | `20261023-074020` (five cells, with resets) | 0.35 m / +0.04 m / 0 % (v9 0.47 m / +0.05 m / 0 %) | beats v9 (p95 ratio 0.73) | 0.38 m / +0.05 m / 0 of 9, no drift (PR #44); the reference until v19 |
| v11 | `20261023-122516` (`gazebo_check_v11.yaml`) | 0.28 m / −0.01 m / 0 % (v10 0.35 m / +0.04 m / 0 %) | beats v10 (p95 ratio 0.82, interval 0.80 to 0.84); one crash in 512 rough episodes (8 m/s into the wall), within the +1 % rule | hold 0.33 m / at the hold point / 1.7 Hz ring in 2 of 9; seed 3 lost after the re-engage 4 of 4 (PR #49, see above) |
| v11 | `20261023-144745` (extended check, `gazebo_check_v12.yaml`) | 0.28 m / −0.01 m / ring 0.0154 (v10 0.34 m / +0.04 m / ring 0.0124) | does not beat v10: fails the ring rule (+24 %); no re-engage failures | as above |
| v12 | `20261023-144745` (extended check) | 0.27 m / −0.03 m / ring 0.0090; no crash in 512 rough episodes | beats v10 (p95 ratio 0.81, interval 0.78 to 0.83; failed: none); no re-engage failures; calm clean 0.056 m against v10's 0.038 m, inside the 0.02 m slack | matched timeline (PR #52): hold 0.35 / 0.35 / 0.35 m against v10's 0.33 / 0.33 / 0.35 (lower in 1 of 9), less ring, −0.02 m offset; seed 1 drift toward the wall 0.27 to 0.31 m (v10 0.03 to 0.07); v10 stays |
| v12 | `20261023-170223` (with `seed1_jump`, `gazebo_check.yaml`) | 0.27 m / −0.03 m / ring 0.0090; toward the wall 0.10 m (v10 0.34 m / +0.04 m / 0.0124 / 0.11 m) | does not beat v10: fails `seed1_jump_toward_wall_ok`, 0.26 m against v10's 0.08 m (Gazebo 0.29 and 0.05); every other rule passes | as above |
| v13 | `20261023-182106` (`gazebo_check_v13.yaml`) | 0.26 m / −0.04 m / ring 0.0073; toward the wall 0.08 m; no crash in 512 rough episodes (v10 0.34 m / +0.04 m / 0.0124 / 0.11 m) | does not beat v10 (p95 ratio 0.79, interval 0.76 to 0.82): fails `calm_clean_p95_ok`, 0.066 m against v10's 0.038 m (slack 0.02 m), from a 0.05 m sag below the hold point; passes `seed1_jump` (0.09 m against v10's 0.08 m, v12 0.26 m) and the re-engage rules | not flown |
| v15 | `20261023-200916` (`gazebo_check_v15.yaml`) | 0.30 m / +0.00 m / ring 0.0135; toward the wall 0.11 m; no crash in 512 rough episodes (v10 0.34 m / +0.04 m / 0.0124 / 0.11 m) | does not beat v10 (p95 ratio 0.88, interval 0.85 to 0.91): fails `ring_not_higher` (+9 %) and `seed1_jump_toward_wall_ok`, 0.25 m against v10's 0.08 m; calm clean 0.021 m against 0.038 m, no sag | not flown |
| v17 | `20261023-215357` (`gazebo_check_v17.yaml`) | 0.28 m / +0.01 m / ring 0.0140; toward the wall 0.05 m; no crash in 512 rough episodes (v10 0.34 m / +0.04 m / 0.0124 / 0.11 m) | does not beat v10 (p95 ratio 0.85, interval 0.82 to 0.87): fails `ring_not_higher` (+13 %) and `calm_clean_p95_ok`, 0.062 m against 0.058 m, from standing 0.05 m back from the wall; passes `seed1_jump` (0.04 m against v10's 0.08 m) and the re-engage rules | not flown |
| v18 | `20261024-000518` (`gazebo_check_v18.yaml`) | 0.26 m / −0.01 m / ring 0.0074; toward the wall 0.07 m; no crash in 512 rough episodes (v10 0.34 m / +0.04 m / 0.0124 / 0.11 m) | beats v10 (p95 ratio 0.78, interval 0.76 to 0.81; failed: none): calm clean 0.041 m against 0.038 m, `seed1_jump` 0.06 m toward the wall against v10's 0.08 m, ring 40 % below v10's, no re-engage failures; tighter than v10 in every rough cell | matched timeline (PR #58): hold 0.29 / 0.30 / 0.30 m against v10's 0.33 / 0.33 / 0.35, but lower on seed 1 only (3 of 9); a commanded 3.3 Hz ring in 9 of 9 (2.2 to 5 Hz rate setpoint 0.118 rad/s against 0.023); toward the wall 0.08 / 0.09 / 0.13 against 0.08 / 0.07 / 0.08; one post-hold strike; v10 stays |
| v19 | `20261024-024323` (`gazebo_check_v19.yaml`, with the fast-ring rule and seeds 2 and 3) | 0.24 m / −0.00 m / ring 0.0097, fast ring 0.017 rad/s; toward the wall 0.07 m; no crash in 512 rough episodes (v10 0.34 m / +0.04 m / 0.0124, 0.026 rad/s / 0.11 m) | beats v10 (p95 ratio 0.72, interval 0.70 to 0.75; failed: none); calm clean 0.045 m; replays, p95 against v10: seed 1 0.22 against 0.46 m, seed 2 0.27 against 0.26 m (ratio 1.01, interval 1.00 to 1.02), seed 3 0.31 against 0.36 m | matched timeline (PR #61): hold 0.28 / 0.29 / 0.30 m against v10's 0.33 / 0.33 / 0.35, lower in 6 of 9 (the other 3 within +2 to +5 %); no ring (roll-rate peak 1.2 to 2.3 Hz); toward the wall 0.07 / 0.07 / 0.09 against 0.08 / 0.07 / 0.08, seed 3 0.21 against 0.17 m; no crash or post-hold failure; released, the reference since |
| v12 | `20261024-032135` (`gazebo_check.yaml`, v10 the reference) | 0.27 m / −0.03 m / ring 0.0090, fast ring 0.037 rad/s; toward the wall 0.10 m (v10 0.34 m / +0.04 m / 0.0124, 0.026 rad/s / 0.11 m) | does not beat v10 (p95 ratio 0.81): fails `fast_ring_not_higher` (+42 %), `seed1_jump_toward_wall_ok`, and the seed 2 and seed 3 replays' p95 (0.28 against 0.26 m, 0.42 against 0.36 m) | as above: lower than v10 in 1 of 9 |
| v18 | `20261024-032135` (`gazebo_check.yaml`, v10 the reference) | 0.26 m / −0.01 m / ring 0.0074, fast ring 0.119 rad/s; toward the wall 0.07 m | does not beat v10 (p95 ratio 0.78): fails `fast_ring_not_higher` (4.6 times v10's) and the seed 2 replay's p95 (0.28 against 0.26 m); seed 3 a tie (0.36 against 0.36 m) | as above: ring in 9 of 9, lower than v10 in 3 of 9 |
| v10, v18 | `20261024-061132` (`gazebo_check.yaml`, v19 the reference, seed replays on the bridge stream) | v19 0.24 m / −0.00 m / ring 0.0097, fast ring 0.017 rad/s / toward the wall 0.07 m; v10 0.34 m; v18 0.26 m | neither beats v19: v10 p95 ratio 1.38 (fails p95, lift, both ring rules, seed 1's drift and the seed 1 and 3 replays' p95), v18 ratio 1.08 (fails p95, the fast ring, seed 1's drift and all three replays' p95) | v19 beat both in Gazebo (PR #61) |
| v20 | `20261024-120832` (`gazebo_check_v20.yaml`, v19 the reference, replays engaged at 28 s) | 0.23 m / −0.00 m / ring 0.0096, fast ring 0.021 rad/s; toward the wall 0.07 m; no crash in 512 rough episodes (v19 0.24 m / −0.00 m / 0.0097, 0.017 rad/s / 0.07 m) | does not beat v19 (p95 ratio 0.96, interval 0.95 to 0.98): fails `fast_ring_not_higher` (+25.4 %) and `reengage_no_reengage_fail` (1 in 64); lower in all seven hold cells; replays seed 1 0.25 against 0.26 m, seed 2 0.33 against 0.34, seed 3 0.21 against 0.23 and 0.16 m toward the wall against 0.18; calm clean 0.037 m against 0.045 | not flown |
| v10, v18 | `20261103-100625` (`gazebo_check.yaml`, v19 the reference; the first run with the fusion on barometer height, Decided 2026-11-03) | v19 0.21 m / −0.01 m / ring 0.0097, fast ring 0.017 rad/s / toward the wall 0.07 m, no crash in 512 rough episodes; v10 0.28 m / +0.04 m / 0.0127, 0.027 rad/s / 0.11 m (0.8 % crash); v18 0.23 m / −0.02 m / 0.0074, 0.119 rad/s / 0.07 m | neither beats v19 (v10 p95 ratio 1.35, v18 1.07, the same failed rules as on vision height). The barometer takes 0.02 to 0.03 m off every policy's rough p95 (v19 0.24 → 0.21 m) and 0.04 m off v19's seed 1 replay (0.22 → 0.18 m; seed 2 0.20, seed 3 0.23 m with 0.25 m toward the wall), leaves the ring and fast ring where they were and moves no verdict. These are the reference numbers for candidates from here on; v22 on the barometer is in `docs/baseline.md` (0.176 m against v19's 0.210) | as above |

v11 was the second forward test. If the check holds as it did for v10 (ratio 0.73 in the mock and in Gazebo), v11 should fly about 0.31 m p95 in Gazebo against v10's 0.38 m, sit at the hold point rather than 0.04 m above it, and show no wobble.
Gazebo held part of it: v11's hold p95 was 0.33 m (the ratio 0.87, not 0.82) and it sat at the hold point, but it rang at 1.7 Hz, which the thresholded wobble rate missed. The ring rule and the re-engage cells came out of that flight.

v12 is the third forward test, the first on the extended check. It is v11 trained further against up to 60 ms of extra rate-loop lag (the ring's cause in the mock) with the x500's 0.40 m contact radius. On the check's ratio it should fly about 0.31 m hold p95 in Gazebo against v10's 0.38 m, ring less than v10, and sit 0.03 to 0.04 m below the hold point (the mock's offset; v10 sits 0.04 m above). Fly it with `-- --profile vio_at_engage` for both controllers. The re-engage after seed 3's jump is not a policy ranking (v10, v11 and v12 crash on it 90 to 94 % in the mock), so it is read from the `post_hold_fail` column, not the hold score.

v13 is v12 trained further with the toward-the-wall penalty at 2.0 per metre (0.5 before). It fixes seed 1's drift in the mock and keeps v12's gains elsewhere, but on a clean estimate it sits 0.05 m below the hold point (v12 0.04 m below, v10 0.03 m above), so the calm clean cell fails its slack. The sag is the policy's own: at zero error its thrust is a little under hover, with the lever arm (`vision_offset_down_m`) or without, so it settles where the height error makes up the difference. A plain MLP has no integral action and training draws PX4's hover-thrust estimate up to 20 % off, so some steady offset is expected; its size and sign have moved with each continuation run. Not flown.

v14 is v13 trained with a re-engage about every 20 s (`reengage_rate_hz`: the hold point moves to a fresh 0.2 m scatter, the policy's history is refilled), for Gate 1's start transient; it was not run through the full check (seed 1 replay 0.06 m toward the wall, sag 0.04 m). v15 is v14 with the height-only reward term (`reward.height` 0.5 over 0.1 m): the sag is gone, but seed 1's drift came back. In the replay v15 chases the jump's 0.28 m away from the wall and moves 0.24 m toward it while it leaves the 1.78 m along the wall alone; v14 holds x. Which way a continuation run settles on that jump has flipped from run to run (v12 0.26 m, v13 0.09, v14 0.06, v15 0.25), which is why the check scores it. v16 doubled the toward-the-wall penalty (4.0) and still drifted 0.19 m in the replay. v17 is v16 with training jumps and dropouts at 0.1 Hz, as v10 had them (0.04 since v11): the drift is gone (0.04 m), but on a clean estimate it now stands 0.05 m back from the wall, the price of a one-sided penalty at 4.0 with five times the eval rate of jumps, and it rings more. The ring came with the re-engages: on a scratch probe (4 m/s, seed 7003) v13 0.0055, v14 0.0155, v15 0.0100, v17 0.0144, v10 0.0091.

v18 is v17 trained further with the toward-the-wall penalty back at 2.0 and the body-rate penalty doubled (`ang_vel` 0.2), for v17's two failed rules. It is the first candidate since the check was extended to pass every rule: on a clean estimate it stands 0.03 m back from the wall and 0.01 m low (calm clean 0.041 m), it rings least of any policy so far (median 0.0074 against v10's 0.0124), and on seed 1's jump it moves 0.06 m toward the wall (v10 0.08 m). It keeps v17's Gate 1 start: on the headline cell through the fusion (seed 8001, 256 episodes, scratch) p95 0.222 m against the PD's 0.346 m (−36 %, v17 −37 %). It is the fourth forward test. On the check's ratio it would fly about 0.26 m hold p95 on the matched timeline against v10's 0.33 m, but v11 (ratio 0.82 here, 0.87 in Gazebo) and v12 (0.81 here, 1.04 in Gazebo) kept less than half of their gain, so anywhere from 0.26 to 0.33 m is in line with past transfer. Fly it as v12 was flown: `-- --profile vio_at_engage` for the hold runner, `wall1m_*`, seeds 1 to 3, v10 alongside on the same timeline.

Gazebo (PR #58) kept part of v18's gain and showed what the check missed. Hold p95 0.29 / 0.30 / 0.30 m against v10's 0.33 / 0.33 / 0.35, but all of it on seed 1 (−37 to −42 %); on seeds 2 and 3 v18 was 1 to 25 % looser. And it rang at 3.3 to 3.4 Hz in all 9 flights, commanded: the 2.2 to 5 Hz rms of the roll and pitch rate setpoints was 0.118 and 0.113 rad/s against v10's 0.020 and 0.023, just above the 1.2 to 2.2 Hz band the ring rule scores (where v18 was quieter than v10). The mock has the same ring. On the 4 m/s cell (seed 7003, a scratch probe) v18's 2.2 to 5 Hz roll and pitch rate commands are 0.113 and 0.124 rad/s with the peak at 3.33 Hz, against v10's 0.013 and 0.016 and v12's 0.019 and 0.036 (Gazebo 0.023 and 0.036). It came with v16's toward-the-wall penalty at 4.0 (roll / pitch: v13 0.013 / 0.017, v14 0.032 / 0.041, v15 0.021 / 0.030, v16 0.105 / 0.114, v17 0.070 / 0.198), and v18's body-rate penalty at 0.2 kept it. The check now scores it: `fast_ring_rms` is the larger of the roll and pitch rate commands' 2.2 to 6 Hz rms (rad/s), and its median over the rough cells may be at most 25 % above the reference's (`fast_ring_not_higher`); v10 and v12 are 1.6 times apart in Gazebo on it, v10 and v18 five times. The other lesson, that v18's mean win came from one seed, the check cannot score directly: its rough cells draw random fault timelines, so only a replay cell is per seed. Gazebo seeds 2 and 3's faults in the hold are now replayed too (`seed2_events`, `seed3_events`, extracted from the bridge's corruptor as seed 1's was), and every replay cell must have a p95 no worse than the reference's (`p95_not_worse`: the paired ratio's interval reaches 1 or below). On them the mock ranks v18 against v10 mostly as Gazebo did: seed 1 0.35 against 0.47 m (Gazebo 0.26 against 0.41 at 4 m/s), seed 2 8 % looser (Gazebo 10 %), seed 3 a tie (Gazebo 11 % looser at 4 m/s, 1 to 25 % over the three winds); v18 fails the seed 2 rule (scratch probe, seed 7003).

v19 is v18 trained with a penalty on the 1 to 6 Hz part of its rate commands (`reward.action_band` 5.0), the band the ring sat in. The ring is gone in the mock: 2.2 to 5 Hz roll and pitch rate commands 0.015 and 0.012 rad/s on the 4 m/s probe (v18 0.113 and 0.124, v10 0.013 and 0.016), and the check's fast ring 0.017 rad/s against v10's 0.026. It kept v18's hold and more: rough p95 ratio 0.72 against v10 (v18 0.78), it sits at the hold point in height, and on the replays it is tighter than v10 on seeds 1 and 3 and level on seed 2. It is the fifth forward test. On the check's ratio it would fly about 0.24 m hold p95 against v10's 0.33 m; v11, v12 and v18 kept from none to half of their check gain in Gazebo, so 0.24 to 0.33 m is in line with past transfer, and seed 2 is where it could still lose. Fly it as v18 was flown.

With the fast-ring rule and the seed 2 and 3 replays, the check now fails both policies Gazebo held back after flying them, v12 and v18 (run `20261024-032135`, with v10 as the reference), and on the rules Gazebo would have failed them on: v18 on the fast ring and seed 2, v12 on seed 1's drift, the seeds 2 and 3 hold and a fast ring 1.4 times v10's (Gazebo 1.3 to 1.6 times, by script).

Gazebo (PR #61) released v19. Hold p95 0.28 / 0.29 / 0.30 m against v10's 0.33 / 0.33 / 0.35 (−16 / −12 / −14 %), lower in 6 of 9 flights and within +2 to +5 % in the other three, where v10 flown twice differs by 2 to 10 %; no ring (the roll rate's 1 to 6 Hz peak at 1.2 to 2.3 Hz, nothing near v18's 3.4 Hz); no crash, contact or post-hold failure. The check called it: the ratio was 0.72 here and 0.86 in Gazebo (v11 0.82 and 0.87, v12 0.81 and 1.04, v18 0.78 and 0.89), and the replays' ranking held per seed (seed 1 much tighter, seed 2 level, seed 3 mostly tighter). v19 is the reference from here on (`gazebo_check.yaml`: v19, v10, v18; the Gazebo columns hold all three).

One thing the check gets the wrong way round: seed 3's drift toward the wall. In Gazebo v19 comes 0.21 m toward the wall on seed 3 against v10's 0.17 m (as v12 and v18 did; 0.79 m clearance at a 1 m hold); in the mock's `seed3_events` replay v19 comes 0.24 m against v10's 0.27 m. The mock replay is also looser than Gazebo on that seed for v10 (p95 0.36 against 0.26 m). The watch item for the next candidate is that drift; the check scores it only against the reference, so a candidate that drifts like v19 on seed 3 still passes.

From 2026-10-24 the seed replay cells fly the bridge's whole fault stream (`bridge_stream`, cell key `bridge`) instead of its jumps and dropouts alone. The bridge's corruptor (`sim2real/estimate_bridge`) is seeded and its output does not depend on where the drone flies, so running it offline with the flight's seed and profile, engaged at 25 s of sim time as the benchmark is, gives the position offset Gazebo's EKF2 was fed at every step of the hold: its slow bias walk as well as its jumps, with its dropouts and its delay. The env's own bias walk, jumps and dropouts are off in those cells; noise stays. The jump-only replays left out seed 3's bias, 0.05 to 0.10 m toward the wall for the whole hold (the estimate says the drone is nearer the wall than it is, so every controller stands back), and seed 2's, about 0.05 m. Hold p95 at 4 m/s, run `20261024-061132` against Gazebo (seed 1 is the mean of its three winds, as the Gazebo column has it):

| replayed seed | v10 mock / Gazebo (m) | v19 mock / Gazebo | v18 mock / Gazebo | jump-only replay, v10 / v19 |
| --- | --- | --- | --- | --- |
| 1 | 0.45 / 0.43 | 0.26 / 0.30 | 0.31 / 0.26 | 0.46 / 0.22 |
| 2 | 0.31 / 0.31 | 0.32 / 0.32 | 0.35 / 0.34 | 0.26 / 0.27 |
| 3 | 0.25 / 0.26 | 0.22 / 0.24 | 0.27 / 0.29 | 0.36 / 0.31 |

Seeds 2 and 3 now sit within 0.03 m of Gazebo for all three policies and rank them as Gazebo does. Seed 1 still ranks v18 against v19 the other way (Gazebo has v18 0.04 m tighter, the mock 0.05 m looser). Toward the wall on seed 3, v10 / v19 / v18 come 0.19 / 0.17 / 0.18 m here against 0.17 / 0.21 / 0.23 in Gazebo. v19's Gazebo traces (exported for this) show where that is: the mock's fused estimate follows EKF2's within 0.03 m through all three jumps, both stand 0.10 m back before them, and in both v19 follows the 48 s jump further than v10 (Gazebo +0.10 m against +0.05). What the mock gets wrong is v10's swing after the 40 s jump, 0.18 m against Gazebo's 0.11, which makes v10's mock worst 0.03 m too high. The rule allows 0.05 m above the reference, so the misranking does not change a verdict. Run `20261024-051703` is void: its replays held the offset at engage for the whole episode (fixed, with a test).

The replays now engage the stream 28 s after the bridge's first sample (run `20261024-120832` on), where the hold runner's anchor fell in all 24 v19 flights of the 8-seed set (PR #62, measured from the bridge logs by the module thread; fitting the stream to Gazebo's vision error is flat from 27.5 to 29.5 s, so it cannot pin it). At 25 s seed 2's bias came out at 0.04 m per axis against Gazebo's 0.10. The time matters because the bridge's bias walk ran from its first sample, so the bias at the anchor grew with the takeoff. The same effect made tuned PX4 look better than v19 on seeds 2 and 8: the stock runner anchors at altitude, 3 to 4 s earlier, and met a smaller bias, and on those quiet seeds each controller sits on its own estimate's error (`docs/baseline.md`). PR #65's `bias_from_anchor` starts the walk at the anchor for flights from then on. The replays still build the bridge without it so that they match the flights already flown; they switch when the 8-seed set is re-flown on the new stream. v19 at 28 s, 4 m/s: seed 1 0.26 m, seed 2 0.34, seed 3 0.23 and 0.18 m toward the wall (Gazebo 0.30 / 0.32 / 0.24 and 0.21).

v20 is v19 trained further with a charge on its true speed toward the wall (`reward.toward_wall_vel` 2.0 above 0.05 m/s, clipped at 1 m/s), aimed at seed 3's drift: charging the approach rather than the position leaves the standoff alone, where a larger `toward_wall` moved it (v16, v17). It holds a little tighter everywhere (pooled p95 ratio 0.96 against v19) and comes 0.02 m less toward the wall on the seed 3 replay, but it fails the fast-ring rule by 0.4 % of the reference and has one re-engage failure in 64. Checked gains of 18 to 22 % kept none to half of themselves in Gazebo (v11, v12, v18), so 4 % would not show there, and v20 was not flown. It was the one bounded attempt after v19 at v19's weak seeds; the other targets were not the controller's (seeds 2, 4 and 8, above), so it stopped after one round. v19 stays the reference.

From PR #65's re-fly on (v19 and tuned PX4, 8 seeds, PR #67) the replays fly the corrected stream: `bridge_stream` takes the profile's `bias_from_anchor`, which `vio_at_engage` now sets, so the bias walk starts at the engage and the hold meets the same bias whenever the anchor fires. `bridge: {bias_from_anchor: false}` replays a flight from before it (`gazebo_check_v20.yaml` keeps it, to reproduce run `20261024-120832`). On the corrected stream seed 3 has no standing bias toward the wall any more, and v19 comes nearer the wall there in Gazebo (0.28 to 0.29 m against 0.21). Hold p95 at 4 m/s, mock replay (scratch, 64 episodes) against v19's corrected flights (batch `20261024-111238`):

| replayed seed | v19 mock / Gazebo (m) | v19 toward the wall, mock / Gazebo | v20 mock | v10 mock |
| --- | --- | --- | --- | --- |
| 1 | 0.19 / 0.23 | 0.00 / - | 0.17 | 0.41 |
| 2 | 0.30 / 0.28 | 0.01 / - | 0.29 | 0.28 |
| 3 | 0.26 / 0.26 | 0.24 / 0.28 | 0.25 (0.22 toward) | 0.30 (0.26 toward) |

Seeds 2 and 3 sit within 0.02 m of Gazebo; seed 1 is 0.04 m tighter in the mock. On seed 3's drift the mock is 0.04 to 0.05 m short of Gazebo (v10 has not flown the corrected stream, so the v19 against v10 ranking there is unchecked); v20's edge over v19 there stays at 0.02 m. The Gazebo columns of the check config now give v19's corrected numbers; the seed rows of v10, v12 and v18 are from the old stream.

## Seed repeatability (2026-11-09)

Week 2's item 3 (`docs/gpu-server-plan.md`): v22's recipe (`training/configs/structure_hold_ppo_v22.yaml`, v19's actor grown to 21 inputs, a 200-iteration critic warm-up and 1000 iterations) trained again on seeds 1, 2 and 3, then the three beside the released v22 in the check (`eval/configs/gazebo_check_seeds.yaml`, v22 the reference, 128 episodes per cell and controller). It ran in the cloud container on 4 CPU cores at commit `63a05e0`: `20261109-185248_structure_hold_ppo_v22_s1_63a05e0` (27 min, iteration 950 selected, selection p95 0.180 m), `20261109-191954_structure_hold_ppo_v22_s2_63a05e0` (26 min, iteration 950, 0.200 m), `20261109-194610_structure_hold_ppo_v22_s3_63a05e0` (27 min, iteration 1199, 0.150 m), and the check `20261109-201321_gazebo_check_seeds_s7001_63a05e0`. The check took 61 min against its 45 min budget because another thread's test suite shared the four cores for part of it; it is a scoring run and nothing in it depends on wall time. Against the plan's wording this is the mock half only; the Isaac half waits on Isaac parity.

One thing to hold in mind before the numbers: v22 itself (seed 0, 2026-10-30) trained with the fusion on vision height, and `gz_x500_sitl`, which the recipe and the check both fly, has been on barometer height since 2026-11-03 (Decided). The three re-trainings therefore trained on the fusion the check now flies and v22 did not, so the gap between the three and v22 mixes the height source with the seed. The spread between the three is the seed-luck number.

Rough-estimate cells pooled (512 episodes each; no crash anywhere, wobble 0 % for all four):

| controller | p95 (m) | lift (m) | ring (median rms) | fast ring (median, rad/s) | toward the wall (m) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| v22 (seed 0, vision-height training) | 0.176 | −0.013 | 0.0034 | 0.017 | 0.057 | reference |
| v22 recipe, seed 1 | 0.158 | −0.000 | 0.0033 | 0.018 | 0.070 | beats v22 (p95 ratio 0.90, interval 0.87 to 0.92; failed: none) |
| v22 recipe, seed 2 | 0.166 | +0.004 | 0.0034 | 0.013 | 0.063 | does not beat v22 (ratio 0.94, interval 0.91 to 0.98): fails `seed1_jump_p95_not_worse` only, 0.269 against 0.183 m |
| v22 recipe, seed 3 | 0.161 | −0.008 | 0.0044 | 0.015 | 0.091 | does not beat v22 (ratio 0.91, interval 0.89 to 0.93): fails `ring_not_higher` (+29 %) and `seed3_events_p95_not_worse`, 0.234 against 0.216 m |

Hold p95 in the replay cells, v22 / seed 1 / seed 2 / seed 3: Gazebo seed 1's faults 0.183 / 0.170 / 0.269 / 0.116 m; seed 2's 0.163 / 0.140 / 0.141 / 0.161; seed 3's 0.216 / 0.207 / 0.204 / 0.234, with 0.201 / 0.187 / 0.186 / 0.216 m toward the wall. Calm clean estimate 0.055 / 0.049 / 0.051 / 0.047 m. The re-engage cells are the same to within 2 % for all four (3.1 % crash on the hold / stock / re-engage cell, 94.5 to 96.9 % on the 1.4 m jump cell, no re-engage failure), as they should be: those cells are the fallback's and the fusion's, not the policy's.

What it says:

- Seed luck in the pooled score is about 5 %: 0.158 to 0.166 m end to end, p95 ratio intervals that overlap (0.87 to 0.92, 0.91 to 0.98, 0.89 to 0.93). Lift sits within 0.012 m, ring within 30 % (0.0033 to 0.0044), toward the wall 0.063 to 0.091 m.
- Seed luck in the verdict is larger than that. Two of the three fail, each on a cell the other two pass: seed 2 on the seed 1 replay alone (0.269 m where the other seeds hold 0.116 to 0.170; one jump at 35 s decides it), seed 3 on the ring rule and the seed 3 replay. The 128-episode replay cells and the ring rule move between seeds of the same recipe by more than a candidate usually differs from the reference on them.
- For judging candidates: a pooled p95 gain inside about 10 % of the reference, or a pass or fail that rests on one replay cell or on the ring rule by less than about 30 %, is within one seed's luck and says nothing about the recipe. The releases so far were outside it (v9, v10, v11, v19 at ratios 0.72 to 0.82; v24 against v22 at 0.77 on the server the same day); v20's 0.96 was inside it. A candidate whose case rests on a margin inside the band should be trained on three seeds and judged on all three before anything is said about it, and a replay-cell failure by one seed is a watch item, not a verdict, until a second seed repeats it.
- All three re-trainings hold 6 to 10 % tighter than v22 in the pooled cells and sit at the hold point where v22 sits 0.013 m below it. The likeliest reason is the height source, not the seed (above): the recipe trained on the barometer fusion lands where the check flies. That is a reason to retrain the release on the barometer fusion before the next Gazebo re-fly, and the mock has not predicted Gazebo on its own before, so it is a Proposed step, not a release.
