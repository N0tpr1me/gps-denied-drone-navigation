# The wall-range input: the policy reads the forward rangefinder

Until v22 the policy saw only the position estimate minus the hold point. When the estimate
jumps, drifts or dead-reckons through a dropout, the policy follows it, and along the wall
normal that is the direction that hurts. A forward rangefinder (already read by the module's
wall guard, `WallGuard.hpp`) measures the distance to the wall directly, independent of the
estimate. This page is what the policy now reads from it, how the mock and the module make
that number, and what it bought.

## The input

Two values per step after v22's 21 (`envs/structure_hold/env.py`, `OBS`):

| index | what |
| --- | --- |
| 21 | wall range error: the hold point's wall distance minus the measured perpendicular distance, clipped at 1 m (`range.obs_clip_m`). Positive toward the wall, like `pos_err` x, but measured on the wall |
| 22 | usable flag: 1 while a fresh, accepted reading backs it, else 0 (and the error is 0) |

"Measured perpendicular distance" is the newest reading times the cosine between the body x
axis and the wall normal, exactly the wall guard's `wall_m`. "The hold point's wall distance"
is a reference latched at engage:

- In the module (`RangeReference`, `WallGuard.hpp`): the wall guard's clearance at the first
  usable reading after engage. The hold point is the estimate at engage, so that is the
  vehicle's own measured distance; from then on an estimate jump or drift does not move it.
  Each wall-guard shift of the hold point moves it back by the same amount.
- In the mock: the hold point's true distance from the wall (`range_ref`), set at each reset
  and re-engage. `eval/gazebo_check.py`'s scripted re-engage latches it the module's way (the
  vehicle's distance at the switch).

A policy without the input reads the first 20 or 21 values as before (`policy.json`
`input.num_obs`), and the module reads it only when its header says `HAS_WALL_RANGE`.

## The rangefinder model (`envs/structure_hold/config.py`, `RangeSensor`)

Off by default (`present_prob` 0: the inputs stay 0). Per episode the vehicle has one with
probability `present_prob`; it samples at 20 Hz (the Gazebo bridge's rate) and the input
holds the newest sample. Each reading is the beam's distance to the wall plane with white
noise (1 cm + 0.5 % of range), a per-episode scale error (up to 2 % in training), outliers
(a reading drawn anywhere in 0.1 to 12 m, 1 % of readings in training) and dropouts. It
counts while it is at most 0.2 s old, inside the sensor's range and the body x axis is within
35 deg of the wall normal: the wall guard's own tests. Its own random generators, so turning
it on leaves every other draw of an episode as it was (v22's rows are the same with it on).

**Spike gate** (`spike_gate_m`, 0.25 m, in the mock, in `RangeReference` and, since 2026-11-02, in the wall guard itself, `WallGuard::acceptWall`): a reading more
than the gate from the last accepted one is not used, and the input goes unusable until the
next accepted reading, unless the reading before it (also rejected) agrees with it within
the gate, so a real step is accepted at its second reading. Without it, single outliers
kicked v23 for a whole 50 ms sample: on the Gazebo-like check with a rangefinder its 2.2 to
6 Hz rate content doubled (0.032 against v22's 0.017 rad/s). Scratch runs on the 4 m/s cell
(48 episodes, seed 7101): no outliers 0.013, outliers and no gate 0.031, outliers and the
gate 0.014 rad/s, p95 0.141 / 0.142 / 0.142 m. v23 was trained before the gate existed
(its configs say `spike_gate_m: 0`); v24 and v25 train with it.

## Candidates

All continue v22. The new first-layer columns start at zero, so each starts out flying
exactly as v22; the wall-range error's normaliser is v22's for `pos_err` x.

| | Training | Rangefinder in training |
| --- | --- | --- |
| v23 | v22's ONNX + 200 critic warm-up + 1000 iterations (run `20261031-233843`), then 300 more (`20261101-005736`), iteration 1450 selected | 70 %, no spike gate |
| v24 | v23 + 500 iterations (`20261101-033118`), last checkpoint | 50 %, spike gate on |
| v25 | v24 + 300 iterations with twice the height-error weight (`20261101-052113`), against v24's lift | 50 %, spike gate on |

The first v23 run's selection (256 episodes, rangefinder in 70 % of them) kept iteration 200,
v22 itself: middle checkpoints had crashed 1 to 3 % and the last one once. Re-scored on 512
episodes the last checkpoints were crash-free and tighter (`experiments/LOG.md`), so the
second chunk continued from the last and selected on 1024 episodes.

## Gazebo-like check (`eval/configs/gazebo_check_v2{3,4}{,_range}.yaml`), against v22

Rough-estimate cells pooled, 128 episodes per cell; the `_range` files put a rangefinder on
every flight. v22 does not read it, so its rows are the same in both.

| | p95 (m) | lift (m) | fast ring (rad/s) | toward the wall (m) | seed 3 replay: p95 / toward the wall (m) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| v22 | 0.196 | -0.017 | 0.017 | 0.055 | 0.220 / 0.188 | reference |
| v23, rangefinder, no gate (`20261101-012507`) | 0.161 (ratio 0.82) | +0.010 | 0.032 | 0.051 | 0.112 / 0.073 | fails lift, ring, fast ring |
| v23, no rangefinder (`20261101-020619`) | 0.216 (ratio 1.10) | +0.023 | 0.013 | 0.047 | - | fails p95, lift, seed 2 |
| v24, rangefinder (`20261101-043726`) | 0.151 (ratio 0.77, 0.75 to 0.79) | +0.012 | 0.017 | 0.036 | 0.117 / 0.075 | fails lift only |
| v24, no rangefinder (`20261101-035616`) | 0.187 (ratio 0.95, 0.92 to 0.99) | +0.016 | 0.016 | 0.069 | - | fails lift only |
| v25, no rangefinder (`20261101-054339`) | 0.197 (ratio 1.00) | +0.008 | - | - | - | fails p95, lift, seed 1 |
| v24, rangefinder, barometer height (GPU server CPUs, `20261109-161451`, v22 0.176 there) | 0.136 (ratio 0.77) | +0.004 | - | - | 0.106 / 0.076 (v22 0.217 / 0.201) | passes (lift rule soft since 2026-11-03) |

With a rangefinder the seed 3 replay (Gazebo's estimate jumps of 0.96, 0.30 and 0.36 m,
the flight that came closest to the wall) halves: v24 goes 0.075 m toward the wall where v22
goes 0.19. The scripted re-engage crashes less (0.8 % against 2.3 %), and after a 1.4 m jump
toward the wall right before the re-engage (not scored: no estimate-holder can avoid it)
77 % crash against v22's 93 %, because the range reference is latched on the wall, not on
the jumped estimate.

Re-run on 2026-11-09 on the GPU server's CPUs after the check moved to barometer height
(2026-11-03): the same ratio, 0.77, and a smaller lift, so v24 is still cleared for the
Gazebo re-fly. The scripted re-engage after a 1.4 m jump toward the wall crashed 80 % against
v22's 95 % there.

The lift rule (then `lift_not_higher`: mean height offset on the estimate not above the
reference's; since 2026-11-03 `lift_ok`, which also passes a lift under 0.05 m, so v24
passes it on these numbers) was the only rule v24 failed, both ways. v24 sits 1.2 to 1.6 cm above its hold
point where v22 sits 1.7 cm below, about the same size the other way. The rule was written
against v5's 0.3 to 0.45 m Gazebo lift, but it is the rule. v25 doubled the height-error
weight to fix it: the lift halved (+0.008 m) but the hold loosened back to v22's level
without a rangefinder (ratio 1.00), so it still fails the lift rule and now fails p95 too. It
was not kept, and its check with a rangefinder was stopped.

## Gate 1 (held-out seeds 9001 to 9003, x500 through the EKF2-like fusion, vs the cascaded PD)

| | Pooled p95, policy vs PD | Headline 8 m/s 1 m | Crash | Verdict |
| --- | --- | --- | --- | --- |
| v22 (run `20261030-230516`) | 0.190 vs 0.358 m | -47 % | 0 vs 1.4 % | PASS 6/6 |
| v23, no rangefinder (`20261101-025656`) | 0.210 vs 0.358 m | -41 % | 0 vs 1.4 % | PASS 6/6 |
| v24, rangefinder on every episode (`20261101-062513`) | 0.167 vs 0.358 m | -53 % | 0.02 vs 1.4 % | PASS 6/6 |
| v24, no rangefinder (`20261101-070002`) | 0.195 vs 0.358 m | -45 % | 0 vs 1.4 % | PASS 6/6 |

## Gate 1 stress set with a rangefinder (`eval/configs/gate1_x500_ekf_stress_range_v24_p{1,2,3}.yaml`)

The whole set takes about 100 min on 4 cores, so it runs as three parts under the 1 h limit
(runs `20261101-082618`, `20261101-090418`, `20261101-093743`); pooled here by cell.

| | Envelope cells | Pooled p95, policy vs PD | Crash, policy vs PD |
| --- | --- | --- | --- |
| v22, no rangefinder (run `20261030-232614`) | 13/13 | 0.209 vs 0.380 m | 0.05 vs 0.65 % |
| v24, rangefinder on every episode | 13/13 | 0.181 vs 0.380 m | 0.02 vs 0.65 % |

Outside the envelope v24 passes too: motor lag 80 ms (0.202 vs 0.356 m), estimate delay 250
to 350 ms (0.190 vs 0.379 m), 3 to 5 s estimate dropouts (0.745 vs 0.792 m, crash 6.1 vs
9.0 %), the full wall yaw moment, and the two cells that attack the rangefinder: 10 %
outliers and 1 to 5 s sensor dropouts at 0.2 Hz both hold 0.164 m, the same as a clean one
(0.162), so the spike gate and the usable flag do their job. The 2 to 3 s dropout cell is
where the rangefinder helps least (0.388 vs 0.484 m): with no estimate the error along the
wall drifts and the range only pins the normal axis.

## Readings that are too long (2026-11-26)

On glass, a mirror-like or wet face, or over water, the beam goes through or bounces away
and the sensor reads whatever is behind: too far by half a metre to several metres, reading
after reading. The spike gate took that for a real step at its second reading. The input
then said the wall had moved away (clipped at 1 m) and v24 flew at the wall to close the
gap. The environmental-factors work found it in the mock before v24 flew anywhere
(`docs/environment-factors.md`, cell `range_glass`: every episode's rangefinder reads 0.5 to
3 m too far for 20 to 80 % of the time, in stretches of half a second on average): v24
struck the wall in 58.7 % of 768 episodes. v22, which does not read the range, in none.

**The long-step check** (`RangeReference::update` in `WallGuard.hpp`; the env's
`range.long_step_check`, on by default in both):

- A step to a **longer** distance is accepted, at its second reading as before, only when
  the position estimate moved away from the wall by about as much since the last accepted
  reading (within the gate, 0.25 m). Otherwise the input stays unusable, and the policy
  flies on the estimate as it does without a rangefinder, until readings come back to
  within the gate of the last accepted distance.
- A step to a **shorter** distance is still taken on the sensor's word. It moves the vehicle
  away from the wall, and it is the case the input exists for: the estimate jumped and the
  vehicle followed it in.
- Such a shorter step may also be dust the beam stopped on. So the distance it replaced is
  kept, with where the estimate was, and a longer step that lands back on it is accepted.
  After 40 readings in a row at the new distance (2 s at 20 Hz) the old one is forgotten.
- A reading within the gate of the last accepted one is used whatever the estimate did, so
  an estimate jump or drift with a steady range is still measured against the wall.
- Without a finite estimate no longer step is accepted.

The estimate's position along the wall normal is what the module already has (`pos . n`,
the heading of the engage). The wall guard's own gate is unchanged (Decided 2026-11-02): the
guard only ever moves the hold point away from the wall, so a long reading costs it its
protection (safety manual F15) but cannot pull the point in.

What it gives up: a real longer step that the estimate did not see leaves the input off
until the readings return or the estimate's own motion explains them. That is the vehicle
pushed out during a range dropout while the estimate jumped, and a recess in the face on a
surface pass (a part that stands proud is a shorter step and is followed out and back).
The policy then flies like v22, which is the price of not trusting a longer reading alone.

**Mock** (`eval/configs/environment_factors.yaml`, the three seeds x 256 episodes of that
doc's table, v24, mean p95 on true position and crashes; runs `20261126-151425` (with the check) and `-151445` (without), the reference
latched from the reading at the engage as the module does, `range.latch_reference`):

| Cell | 2026-11-10 (reference = true standoff, no check) | Latched, no check | Latched, with the check | v22 (does not read the range) |
| --- | --- | --- | --- | --- |
| nominal | 0.153 m, 0.1 % | 0.152 m, 0.0 % | 0.152 m, 0.0 % | 0.175 m, 0.0 % |
| glass or water (0.5 to 3 m too far, 20 to 80 % of the time) | 0.816 m, 58.7 % | 1.167 m, 26.0 % (112 of 768 touched the wall) | 0.871 m, 2.9 % (none touched the wall) | 0.175 m, 0.0 % |
| losing the wall (dropouts 0.5 Hz, 5 % stray returns) | 0.155 m, 0.1 % | | 0.154 m, 0.0 % | 0.175 m, 0.0 % |
| rain 10 to 50 mm/h | 0.182 m, 0.1 % | | 0.180 m, 0.0 % | 0.212 m, 0.0 % |
| a field day (everything at once, moderate, glass in 20 % of episodes) | 0.354 m, 11.2 % | | 0.378 m, 1.0 % | 0.270 m, 0.0 % |

With the check v24 no longer flies at glass: no episode touched the wall, against 112 of 768
without it. What is left is the other direction. In 21 of the 22 glass episodes counted as
crashes the drone was more than 3 m from the hold point because it had backed away (one
touched the ground); in the field day all 8 were that. These are the episodes where the
beam was already through the glass at the engage: the reference is latched too far, the
real wall then reads as too close, and the policy backs off by that much. That is also why
v24's p95 on glass stays at 0.87 m and why it is looser than v22 on the field day. In the
module the hold-error trip (`SH_WD_POS_ERR`, 2 m) hands back before 3 m (the mock has no watchdog). Without
the latch, the mock's reference was the true standoff and this case did not show; the
2026-11-10 column is that. Where nothing is glass the check changes nothing (nominal,
losing the wall, rain). Near glass or water v22 is still the better policy, and the face
stays outside the envelope.

**Flown holds.** The 24 Gazebo holds v24's release rests on (calm, 4 and 8 m/s, seeds 1 to
8, `docs/baseline.md`) were replayed from their logs through the gate before and after:
of 126,054 logged wall distances neither gate held one back, so the check would not have
changed a tick of those flights and the benchmark numbers stand. Gazebo's rangefinder has
no long readings; the check has not flown.

**Not covered.** A beam already through the glass at the engage is in the mock now (the
latched reference, above) but not in the module's own tests beyond `test_wall_guard.cpp`;
nothing stops the latch from taking a long reading, so the hold backs away until the wall
shows, by up to the extra distance. The wall guard, Structure Hover and
Structure Land on the same sensor (F15). Both stay in the safety manual's do-not-use list:
glass, wet or mirror-like faces and water are outside the envelope with or without the
check. The glass numbers are assumed, not measured on a sensor.

## Where it stands

v24 beats v22 everywhere it was measured in the mock, most with a rangefinder (Gazebo-like
check p95 ratio 0.77, seed 3 toward the wall 0.075 against 0.188 m, Gate 1 headline -53 %
against -47 %, stress 13/13) and a little without one (check ratio 0.95; Gate 1 pooled
0.195 against v22's 0.190 m, a tie). Its one miss was the check's lift rule: it sits about
1.5 cm above the hold point where v22 sits 1.7 cm below; since 2026-11-03 lift is a soft
rule (under 0.05 m passes, `docs/gazebo_check.md`), so it passes the check both ways on the
recorded numbers. At that point it was not released: the module header stayed v22's
(`HAS_WALL_RANGE` false) until a Gazebo re-fly with `--rangefinder` beat v22 (it did, and
v24 was released on 2026-11-10; the end of this section). The module reads the range on
its own whenever its header is generated from a wall-range policy (`sim2real/px4/README.md`).
The numbers above are on vision height, which `gz_x500_sitl` fused until 2026-11-03; the
check now runs on barometer height (`docs/gazebo_check.md` "Policies judged"), where the
server's `v24_check_range` job re-scores v24 against v22 before the Gazebo re-fly.

**The Gazebo re-fly (2026-11-09, flown in the cloud, one flight at a time).** v24 compiled
into its own image and v22, both with `--rangefinder`, 8 seeds on calm, 4 and 8 m/s
(`docs/baseline.md`, "v24 with the rangefinder against v22"):

| | v22 | v24 | change |
| --- | --- | --- | --- |
| mean p95, calm / 4 / 8 m/s (m), against the setpoint as logged | 0.340 / 0.314 / 0.377 | 0.274 / 0.273 / 0.257 | −19 / −13 / −32 % |
| the same with the pre-take-off height offset out (how a Gazebo hold is scored since 2026-11-10) | 0.204 / 0.202 / 0.196 | 0.179 / 0.181 / 0.185 | −12 / −11 / −5 % |
| worst toward the wall, mean (m) | 0.060 / 0.066 / 0.091 | 0.049 / 0.047 / 0.056 | −18 / −29 / −38 % |
| closest to the wall (m) | 0.83 to 0.85 | 0.89 to 0.91 | further off |
| mean lift, offset out (m) | −0.016 / −0.016 / −0.014 | +0.003 / +0.011 / +0.016 | inside the soft rule |
| rate roughness | about 2.2 | 1.35 / 1.59 / 1.97 | smoother |
| crashes, wall contacts | 0 / 24 | 0 / 24 | |
| against tuned PX4, p95 as scored | −31 / −39 / −29 % | −44 / −47 / −51 % | |
| against tuned PX4, p95 like for like (the scoring since 2026-11-10) | −38 / −38 / −40 % | −45 / −44 / −44 % | |

Gazebo confirms the direction the mock gave: v24 is tighter than v22 on mean p95 and on the
move toward the wall in all three scenarios, with no crash, so it meets the plan's release
check. The size is smaller than the mock's ratio of 0.77: with the height offset out it is
0.88 to 0.94 in p95, close to what two flights of the same seed can differ by; the gain
toward the wall and the smoother commands are the clearer part. v24 needs a forward
rangefinder to be worth it (without one it ties v22 in the mock).

**Released.** Zain chose "Release v24" on a decision card on 2026-11-10 (decision row
2026-11-10). The module took it on 2026-11-26, after the long-step check for glass and water
(above, "Readings that are too long"), which the release waited for. v24 is the released
policy for a drone with a forward rangefinder: the module's `PolicyWeights.hpp` and the test
vectors are generated from it and the parts list names it; the evaluation kit is flown again
with it next (`docs/eval-kit.md`). v22 stays in `training/policies/` as the fallback for a
drone without a forward rangefinder: export the header from `structure_hold_v22.onnx` for
that build (`sim2real/px4/README.md`). What is known and accepted with the release is in the
decision row: the p95 gain over v22 is small (5 to 12 %, scored like for like), the gain
toward the wall (18 to 38 % less) and the smoother commands are the clear part, and seed 6
is v24's loosest seed. Surface passes:
v24 was scored on them in the mock on 2026-11-26 (`eval/configs/gate1_x500_ekf_pass.yaml`,
rangefinder on, reference latched at the engage, held-out seeds 9001 to 9003, run
`20261126-200259`): PASS 7/7 against the PD, pooled p95 0.15 against 0.38 m, crash 0.0 %
against 3.7 %, and 12 to 21 % tighter than v22 in every cell (0.13 to 0.18 m against 0.16
to 0.21 m). It has not flown a pass in Gazebo.
