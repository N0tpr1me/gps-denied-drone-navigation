# Gate 1: the policy holds steady in sim on a corrupted estimate

Gate 1 (build plan, 31 Jan 2027) asks whether a learned Structure Hold policy holds a
quadrotor near a wall in the simulator, on a GPS-free estimate that is noisy, late, jumpy and
drops out, better than a classical controller does. This page fixes how that is measured,
the pass rules, and the current numbers. The protocol is `eval/gate1.py`; the configs are
`eval/configs/gate1.yaml` (nominal), `gate1_stress.yaml` (stress) and `gate1_x500.yaml` (the
nominal test set on the PX4 Gazebo x500).

It is a mock-simulator gate. The yardstick here is the cascaded PD in
`envs/structure_hold/baseline.py`, not tuned PX4: the comparison with tuned PX4 is Gate 2,
in SITL (`docs/baseline.md`).

## The test set

- **Held-out seeds.** Seeds 9001, 9002 and 9003 build every episode: hold point, start pose,
  wind, gust schedule, estimate corruption and airframe draws. Seeds 9000 to 9999 are
  reserved: `training/train_ppo.py` refuses to train, quick-evaluate (seed + 1000) or select
  checkpoints (seed + 2000) on them. No policy has flown these episodes.
- **Fixed cells.** Mean wind exactly 0, 4 or 8 m/s (direction drawn per episode) times the
  hold point exactly 1 m or 2 m from the wall: six cells. Training draws the mean wind from
  0 to 12 m/s and the distance from 1 to 3 m.
- **Gusts in every episode.** Two gusts of 2 to 5 m/s lasting 0.5 to 3 s, both starting
  inside the 20 s episode. Training draws gust starts up to 50 s, so most training episodes
  have none.
- **Standard corruption.** Delay 50 to 150 ms, noise, bias walk, jumps of 0.1 to 2 m and
  dropouts of 0.2 to 3 s at 0.02 Hz each, dead-reckoning during dropouts. The config states
  every number, so a later change of a default cannot move the test set.
- **One paired episode per env.** Each env flies only its first episode. The policy and the
  PD fly the same episodes, so differences are measured episode by episode. 3 seeds x 1024
  episodes = 3072 per cell and controller.

## What is scored

Every episode is scored with `eval/metrics.py` on the true trajectory, until it ends.

- **p95**: the 95th percentile of the 3-D position error in an episode, averaged over the
  episodes of a cell. Interval: bootstrap over episodes, 2000 resamples.
- **Crash rate**: share of episodes ending before 20 s by touching the wall, hitting the
  ground, tilting past 80 degrees or leaving the 3 m box. Interval: Wilson.
- **Policy against PD**: the change in mean p95 (policy / PD - 1) and the difference in
  crash rate, each with a paired bootstrap interval.
- **Command roughness**, reported and not judged: the root mean square of the time
  derivative of the rate setpoints (rad/s2) and of the normalised thrust command (1/s),
  `eval/metrics.py`. A thrust or roll limit cycle around the hold point moves the drone only
  centimetres, so p95 misses it; this column does not. Added after v3's 2 Hz cycle in Gazebo.

All intervals are 95 %.

## Pass rules

Set on 2026-10-22, before the first scored run.

1. **Hold**, every cell: the policy's p95 is significantly lower than the PD's (the whole
   interval of the p95 change is below 0).
2. **Safety**, every cell: the policy does not crash significantly more than the PD (the
   interval of the crash difference reaches 0 or goes below it).
3. **Headline**, 8 m/s at 1 m: the policy's p95 is at least 30 % below the PD's. This is
   Gate 2's target, measured here against the PD.
4. **Crash ceiling**: the policy's crash rate pooled over the nominal cells is at most 5 %.
5. **Stress**: rules 1 and 2 hold in every stress cell inside the envelope below, and the
   pooled crash rate over those cells is at most 10 %.

Gate 1 passes when all five hold.

**Open question on rule 5.** Dropouts of 3 to 5 s crash the PD (49 %) about as often as v3
(46 %): position is simply unknown for that long, and on PX4 EKF2 declares it invalid after
5 s and the watchdog hands back. With that cell inside the envelope the pooled 10 % ceiling
cannot be met by any controller. Proposed: judge that cell on rules 1 and 2 only and leave
it out of the pooled ceiling, or move it outside the envelope. The rule is unchanged until
that is decided (`docs/decisions.md`).

## The stress set

One thing that will differ on the real drone at a time, on the same held-out seeds. Base
condition: mean wind drawn 0 to 8 m/s, hold point 1 to 2 m, the gust schedule and corruption
above. 3 x 512 paired episodes per cell. The knobs are in `envs/structure_hold/config.py`
(`airframe_rand`, `wall_yaw_torque`) and are off by default.

| Cell | What changes | In the envelope |
| --- | --- | --- |
| mass x0.7, x1.5 | Airframe mass. The module scales thrust through PX4's hover thrust, so this mostly changes how hard the wind pushes | yes |
| hover thrust error +/-15 % | PX4's hover-thrust estimate is off, so every thrust command is off by that much | yes |
| thrust/weight x0.75 | Max thrust 1.5 times the weight instead of 2 | yes |
| inertia x0.5, x2 | Airframe inertia with PX4's default rate gains unchanged (an untuned rate loop) | yes |
| motor lag 50 ms | Thrust and torque lag (nominal 20 ms) | yes |
| motor lag 80 ms | | no |
| estimate delay 150 to 250 ms | Longer than the 50 to 150 ms v3 trained on | yes |
| estimate delay 250 to 350 ms | | no |
| dropouts 3 to 5 s, 0.05 Hz | Longer and more frequent dropouts | yes |
| wall yaw moment 25 %, 50 % | Placeholder near-wall yaw moment, as a share of the 0.05 N m yaw authority at the wall in a 10 m/s wind | yes |
| wall yaw moment 100 % | | no |
| combined | Mass 0.8 to 1.3, inertia 0.7 to 1.5, hover error +/-10 %, motor lag 20 to 50 ms, delay 50 to 200 ms, yaw moment up to 25 % | yes |

The stress scales are on the 0.8 kg placeholder airframe. The wall yaw moment is a
placeholder model (one published data point exists); the fan rig will fit it.

## The x500 test set

`eval/configs/gate1_x500.yaml` is the nominal test set with one change: the airframe is the
PX4 Gazebo x500 (`airframe: gz_x500`, `envs/structure_hold/airframes.py`), the vehicle the
module is benchmarked on against tuned PX4 (`docs/baseline.md`). The preset carries the
x500's mass (2.06 kg), inertia, rotor constants and ESC curve, PX4's control allocation and
its saturation, and PX4's rate PID at the tuned-v1 gains in the loop. A policy trained on
another airframe flies through the module's thrust mapping for its own mass and max thrust
(`eval.gate1.ModuleThrustScale`), exactly as the module would fly it. Same seeds, cells and
rules; the PD is the same cascaded PD on the x500.

`eval/configs/gate1_x500_ekf_v10.yaml` is the same set judged through the EKF2-like fusion
(`gz_x500_sitl`'s, with EKF2's resets): the policy and the PD read the fused estimate, as the
module reads EKF2, instead of the raw fix 50 to 150 ms late. Nothing else changes (airframe
`gz_x500`, 10 ms tick, no PX4 filters, no lever arm). Gate 1 is judged this way since
2026-10-24 (`docs/decisions.md`, Decided); the raw-fix set stays as a stress result, and both
results are below.

## Current numbers

### v3 (the placeholder release), nominal: run `20261022-154625`

| Cell | p95, v3 vs PD | Crash, v3 vs PD | Verdict |
| --- | --- | --- | --- |
| 0 m/s, 1 m | 0.47 vs 0.57 m (-18 %) | 6.6 % vs 17.0 % | pass |
| 0 m/s, 2 m | 0.49 vs 0.67 m (-27 %) | 2.1 % vs 2.7 % | pass |
| 4 m/s, 1 m | 0.47 vs 0.59 m (-21 %) | 5.8 % vs 17.1 % | pass |
| 4 m/s, 2 m | 0.50 vs 0.73 m (-31 %) | 1.5 % vs 2.6 % | pass |
| 8 m/s, 1 m (headline) | 0.52 vs 0.73 m (-29.6 %) | 7.0 % vs 21.3 % | pass |
| 8 m/s, 2 m | 0.62 vs 0.97 m (-36 %) | 2.9 % vs 3.8 % | pass |

Pooled: p95 0.51 vs 0.71 m, crashes 4.3 % vs 10.8 %. Rules 1, 2 and 4 hold; rule 3 misses
by 0.4 points. At 1 m most crashes are wall contacts after an estimate jump.

### v3, stress: run `20261022-161518`

11 of 13 envelope cells pass; the pooled crash rate is 17.7 % (PD 19.1 %), over the 10 %
ceiling. Where v3 breaks:

| Cell | v3 p95 / crash | PD p95 / crash |
| --- | --- | --- |
| estimate delay 150 to 250 ms | 1.24 m / 67.8 % | 0.94 m / 19.7 % |
| combined | 0.86 m / 30.8 % | 0.90 m / 18.4 % |
| motor lag 80 ms (outside) | 0.65 m / 15.0 % | 0.68 m / 8.1 % |

Everything else passes with margin. Inertia x2 hurts both (v3 49 % crashes, PD 94 %), the
placeholder's wall yaw moment is too small to matter even at 100 %.

### v4 candidate, nominal: run `20261022-185035`

v3's recipe with delay up to 250 ms, a 25-step history and the stress envelope as domain
randomisation (`training/configs/structure_hold_ppo_v4.yaml`, run `20261022-171612`).
Pooled p95 0.66 m (v3 0.51, PD 0.71), crashes 2.7 % (v3 4.3 %, PD 10.8 %); 3 of 6 cells
pass, the headline is -5 %. It halves v3's crashes at 1 m but is about 20 % looser in every
cell, so it does not keep v3's nominal numbers and is not released (the stress set was not
run on it).

### On the x500: v5 (the release), v3 and the PD, run `20261022-214627`

v5 is v3's recipe trained on the x500 (`training/configs/structure_hold_ppo_v5.yaml`, run
`20261022-210442`). v3 flies through the module's thrust mapping for its own 0.8 kg.

| Cell | p95: v5 / v3 / PD (m) | Crash: v5 / v3 / PD | v5 verdict |
| --- | --- | --- | --- |
| 0 m/s, 1 m | 0.53 / 0.48 / 0.48 | 2.0 / 4.5 / 9.4 % | fail (+12 %) |
| 0 m/s, 2 m | 0.53 / 0.48 / 0.52 | 0.9 / 1.5 / 2.4 % | fail (+2 %) |
| 4 m/s, 1 m | 0.52 / 0.47 / 0.49 | 2.1 / 4.8 / 8.7 % | fail (+8 %) |
| 4 m/s, 2 m | 0.54 / 0.48 / 0.54 | 1.0 / 1.4 / 2.4 % | fail (+1 %) |
| 8 m/s, 1 m (headline) | 0.56 / 0.48 / 0.51 | 3.1 / 4.7 / 8.9 % | fail (+8 %) |
| 8 m/s, 2 m | 0.61 / 0.50 / 0.60 | 1.6 / 1.7 / 2.1 % | fail (+1 %) |

Pooled: p95 0.55 / 0.48 / 0.52 m, crashes 1.8 / 3.1 / 5.7 %. Thrust roughness 0.9 / 4.0 /
1.1 per second, rate roughness 25 / 28 / 11 rad/s2.

On the x500 the PD is a much stronger yardstick than on the placeholder (pooled p95 0.52 m,
not 0.71 m): the x500's rate loop answers a roll step in about 60 ms instead of 190 to
270 ms. v5 crashes least in every cell and never cycles, but holds 1 to 12 % looser than the
PD, so it fails rule 1 everywhere. v3 holds tightest here, but only with the 2 to 3 Hz thrust
and roll cycle it flew in Gazebo (4 times the thrust roughness) and only with PX4's hover
thrust exactly right: the test set does not vary it, and with the estimate 18 % low v3
crashes, 15 % high it sits 0.7 m off (v5: 0.08 m and 0.20 m). The Gazebo benchmark against
tuned PX4 (`docs/baseline.md`) flies v5 next; that is the number that counts for Gate 2.

The first two v5 runs show why the reward changed. Run `20261022-193416` (v3's reward) held
within 0.07 m in calm air but drove thrust 20 +/- 6 N in a 2.8 Hz cycle that runs through its
own previous-action input: freezing that input stops it, and the cycle costs almost nothing
in p95. Raising the action-change penalty to 1.0 (run `20261022-201654`) collapsed the
exploration noise and left the cycle. Charging the thrust swing and the body rates instead
(run `20261022-210442`) removed it: thrust within +/-0.2 N of the weight.

### On the x500: v6, run `20261022-231539`

v6 is v5 trained 1000 iterations longer (`training/configs/structure_hold_ppo_v5_resume.yaml`,
run `20261022-223115`, checkpoint 1500), scored on the same episodes as v5 above.

| Cell | p95: v6 / v5 / PD (m) | Crash: v6 / v5 / PD | v6 verdict |
| --- | --- | --- | --- |
| 0 m/s, 1 m | 0.47 / 0.53 / 0.48 | 1.5 / 2.0 / 9.4 % | fail (-1 %) |
| 0 m/s, 2 m | 0.48 / 0.53 / 0.52 | 0.8 / 0.9 / 2.4 % | pass (-8 %) |
| 4 m/s, 1 m | 0.50 / 0.52 / 0.49 | 1.6 / 2.1 / 8.7 % | fail (+2 %) |
| 4 m/s, 2 m | 0.49 / 0.54 / 0.54 | 0.8 / 1.0 / 2.4 % | pass (-8 %) |
| 8 m/s, 1 m (headline) | 0.52 / 0.56 / 0.51 | 2.1 / 3.1 / 8.9 % | fail (+1 %) |
| 8 m/s, 2 m | 0.57 / 0.61 / 0.60 | 1.6 / 1.6 / 2.1 % | pass (-6 %) |

Pooled: p95 0.50 / 0.55 / 0.52 m, crashes 1.4 / 1.8 / 5.7 %; thrust roughness 1.1 per second,
the same as the PD's. v6 is better than v5 in every cell, tighter than the PD at 2 m and level
with it at 1 m, with a quarter of its crashes. It is the release for the Gazebo re-fly.

In Gazebo, v5 held 0.3 to 0.45 m above its hold point and wobbled at 1.6 Hz in roll
(rate setpoint std 0.42 rad/s, against 0.04 in the mock), with PX4's hover thrust and the
estimate both correct. The mock does not show that wobble under more estimate delay (to the
trained 150 ms), rate-loop lag (to 100 ms), stale gyro and attitude inputs (to 50 ms) or an
estimate that leads the truth. Noisy rate or velocity inputs lift the hold in the mock (white
noise 0.15 on either: +0.1 m for v5 and v6), so the wobble probably causes the lift. v6 starts
lower: it sits 0.01 m high in calm air, where v5 sat 0.09 m high.

### On the x500: v7 and v8, runs `20261023-004222` and `20261023-020243`

v6 in Gazebo (PR #35) trails tuned PX4 by 9 to 19 %, almost all of it a hold 0.1 to 0.4 m
above the hold point, larger on the seeds where the vision estimate is rough (0.2 m
innovation, jumps, dropouts). Two candidates, both v6 continued for 1000 iterations with one
change, scored on the same episodes as v6 above.

- v7 (`structure_hold_ppo_v7.yaml`) trains with noise on the gyro, attitude and velocity
  inputs (`cfg.sensor_noise`). Under noisy inputs it lifts half as much as v6, but on clean
  inputs it sits 0.08 m low: it learned an offset rather than ignoring the noise.
- v8 (`structure_hold_ppo_v8.yaml`) trains with the hold point 3.5 m up instead of 2 m. In the
  mock v6 climbs and backs off the wall when its wall-axis velocity or yaw-rate input is noisy
  or the estimate jumps. The policy cannot see its height, and at 2 m the ground (a crash at
  0.3 m) is 1.7 m below the hold point while the box edge is 3 m away, so a climb when the
  estimate looks rough pays as insurance against a jump. At 3.5 m the ground is beyond the box
  edge. The test set keeps its 2 m hold point.

| Cell | p95: v8 / v7 / v6 / PD (m) | Crash: v8 / v7 / v6 / PD | v8 verdict |
| --- | --- | --- | --- |
| 0 m/s, 1 m | 0.45 / 0.50 / 0.47 / 0.48 | 1.0 / 1.0 / 1.5 / 9.4 % | pass (-6 %) |
| 0 m/s, 2 m | 0.47 / 0.50 / 0.48 / 0.52 | 0.8 / 1.0 / 0.8 / 2.4 % | pass (-10 %) |
| 4 m/s, 1 m | 0.47 / 0.50 / 0.50 / 0.49 | 1.3 / 0.7 / 1.6 / 8.7 % | fail (-4 %, CI to +1 %) |
| 4 m/s, 2 m | 0.48 / 0.51 / 0.49 / 0.54 | 1.2 / 1.0 / 0.8 / 2.4 % | pass (-10 %) |
| 8 m/s, 1 m (headline) | 0.49 / 0.52 / 0.52 / 0.51 | 1.6 / 1.6 / 2.1 / 8.9 % | fail (-4 %) |
| 8 m/s, 2 m | 0.54 / 0.56 / 0.57 / 0.60 | 1.4 / 1.3 / 1.6 / 2.1 % | pass (-11 %) |

Pooled: p95 0.48 / 0.51 / 0.50 / 0.52 m, crashes 1.2 / 1.1 / 1.4 / 5.7 %. v8 is tighter than
v6 in every cell with the same crash rate and passes 4 of 6 cells; rate roughness 32 rad/s2
(v6 25), thrust 0.9 per second (v6 1.1). In the mock at a 2 m hold point its lift under 0.15
white noise on the velocity input is +0.08 to +0.10 m (v6 +0.13 to +0.15), on the yaw-rate
input +0.06 m (v6 +0.11), under 0.1 Hz estimate jumps +0.17 m (v6 +0.24). The cost: in calm
air on a clean estimate it drifts along the wall about +/-0.1 m on a 10 s cycle, where v6
holds within 0.02 m. v8 replaces v6 for the next Gazebo re-fly; v7 is not released.

In Gazebo (PR #37) v8 went the other way: best yet on the clean estimate (p95 0.11 m, no
vertical offset), but on the corrupted-estimate benchmark it sat 0.42 m above its hold point
(v6 0.23 m), wobbled at 1.6 Hz in all 9 flights (v6 6 of 9) and scored p95 0.67 / 0.68 /
0.79 m against v6's 0.62 / 0.64 / 0.68. The module stays on v6. This is the second time a
mock gain did not carry over, so a mock score is no longer enough to release: the mock first
has to reproduce the rough-estimate failure (`docs/decisions.md`, 2026-10-23).

### On the x500: v9, run `20261023-060333`

v9 is v6 continued for 1000 iterations on `gz_x500_sitl`, the mock that reproduces Gazebo
(`docs/gazebo_check.md`); it beat v6 there and in Gazebo (PR #42: p95 0.44 / 0.49 / 0.65 m
against tuned PX4's 0.57). On this test set, scored on the same episodes as v6 above, it
fails:

| Cell | p95: v9 / v6 / PD (m) | Crash: v9 / v6 / PD | v9 verdict |
| --- | --- | --- | --- |
| 0 m/s, 1 m | 0.56 / 0.47 / 0.48 | 17.5 / 1.5 / 9.4 % | fail |
| 0 m/s, 2 m | 0.64 / 0.48 / 0.52 | 12.8 / 0.8 / 2.4 % | fail |
| 4 m/s, 1 m | 0.56 / 0.50 / 0.49 | 15.4 / 1.6 / 8.7 % | fail |
| 4 m/s, 2 m | 0.63 / 0.49 / 0.54 | 12.2 / 0.8 / 2.4 % | fail |
| 8 m/s, 1 m (headline) | 0.56 / 0.52 / 0.51 | 15.9 / 2.1 / 8.9 % | fail |
| 8 m/s, 2 m | 0.70 / 0.57 / 0.60 | 12.7 / 1.6 / 2.1 % | fail |

Pooled: p95 0.61 / 0.50 / 0.52 m, crashes 14.4 / 1.4 / 5.7 %. Three quarters of the crashes
are wall contacts. They come from the estimate's delay. This test set's x500 has no EKF, so
the fix reaches the policy 50 to 150 ms late, uncompensated. v9 trained (and Gazebo flies)
behind EKF2, which fuses the fix at its delay and predicts it forward with the IMU, so v9
never had to fly on a late estimate. Taking the corruption apart (256 episodes, calm, 1 m,
a scratch probe): with only the delay v9 crashes in 14 % of episodes; with only the noise,
the bias walk or the jumps in none; with only the dropouts in 1 %. On a clean estimate, none.
The test set is still the one Gate 1 is judged on, so v9 does not pass Gate 1; on the PX4
stack, where the estimate comes through EKF2, it is the better policy.

### On the x500: v10, run `20261023-081233`

v10 is v9 continued for 1000 iterations with 60 s episodes, the wind into the wall and
EKF2's resets (`training/configs/structure_hold_ppo_v10.yaml`); it beats v9 on the
Gazebo-like check in every cell. Here it fails harder than v9:

| Cell | p95: v10 / v9 / PD (m) | Crash: v10 / v9 / PD | v10 verdict |
| --- | --- | --- | --- |
| 0 m/s, 1 m | 0.68 / 0.56 / 0.48 | 40.2 / 17.5 / 9.4 % | fail |
| 0 m/s, 2 m | 0.87 / 0.64 / 0.52 | 35.6 / 12.8 / 2.4 % | fail |
| 4 m/s, 1 m | 0.67 / 0.56 / 0.49 | 39.2 / 15.4 / 8.7 % | fail |
| 4 m/s, 2 m | 0.86 / 0.63 / 0.54 | 33.6 / 12.2 / 2.4 % | fail |
| 8 m/s, 1 m (headline) | 0.66 / 0.56 / 0.51 | 37.8 / 15.9 / 8.9 % | fail |
| 8 m/s, 2 m | 0.90 / 0.70 / 0.60 | 34.8 / 12.7 / 2.1 % | fail |

Pooled: p95 0.77 / 0.61 / 0.52 m, crashes 36.9 / 14.4 / 5.7 %. The cause is the same as
v9's: the 50 to 150 ms late fix with nothing to compensate it (in the same scratch probe,
the delay alone: v10 32 %, v9 14 %; the rest of the corruption alone: 0 to 1 %; the run took
61 min because it shared the CPU with other runs). v10 corrects position errors harder than
v9, which is what stopped the drift and also what makes a late estimate worse. Behind the
EKF-like fusion, with the fix 150 to 200 ms late (later than the 100 ms EKF2 assumes), it
still beats v9 (run `20261023-084937`). Every policy trained on the PX4-like stack fails
this test set, because the module never sees a raw fix: whether Gate 1 should judge the x500
through EKF2 is an open question (`docs/decisions.md`).

### On the x500, both ways: v10 raw and through the EKF2-like fusion, runs `20261023-081233` and `20261023-100656`

The same held-out episodes (3 seeds x 1024 per cell), once on the raw fix as the test set is
written and once through the EKF2-like fusion. The PD flies both. "Change" is v10's p95
against the PD's on the same episodes; the headline cell needs -30 %.

| Cell | Raw: p95 v10 / PD (m) | Raw: crash v10 / PD | Raw verdict | Fused: p95 v10 / PD (m) | Fused: change | Fused: crash v10 / PD | Fused verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.68 / 0.48 | 40.2 / 9.4 % | fail | 0.30 / 0.32 | -7 % | 0.0 / 2.8 % | pass |
| 0 m/s, 2 m | 0.87 / 0.52 | 35.6 / 2.4 % | fail | 0.30 / 0.34 | -13 % | 0.0 / 0.0 % | pass |
| 4 m/s, 1 m | 0.67 / 0.49 | 39.2 / 8.7 % | fail | 0.30 / 0.32 | -8 % | 0.0 / 3.0 % | pass |
| 4 m/s, 2 m | 0.86 / 0.54 | 33.6 / 2.4 % | fail | 0.30 / 0.36 | -16 % | 0.0 / 0.0 % | pass |
| 8 m/s, 1 m (headline) | 0.66 / 0.51 | 37.8 / 8.9 % | fail | 0.31 / 0.37 | -15 % | 0.0 / 2.9 % | fail (headline missed) |
| 8 m/s, 2 m | 0.90 / 0.60 | 34.8 / 2.1 % | fail | 0.34 / 0.44 | -25 % | 0.0 / 0.0 % | pass |
| **Pooled** | **0.77 / 0.52** | **36.9 / 5.7 %** | **FAIL, 0 of 6** | **0.31 / 0.36** | | **0.0 / 1.4 %** | **FAIL, 5 of 6** |

Through the fusion, v10 holds tighter than the PD in every cell (every interval below 0)
and does not crash once in 18,432 episodes. It fails Gate 1 on one rule only: at 8 m/s and
1 m it is 15 % below the PD (interval -17 to -12 %), not 30 %. The PD gains from the fusion
as much as v10 does: its pooled p95 drops from 0.52 to 0.36 m and its crashes from 5.7 to
1.4 %. So judging through EKF2 removes the delay failure, but the PD gets the same help and
the headline stays out of reach: v10 needs about 15 more points there. In Gazebo, where the yardstick is tuned PX4 rather than this
PD, v10 is 31 % below at 8 m/s (`docs/baseline.md`). The fused run took 29 min on 4 idle
cores.

### On the x500, both ways: v17, runs `20261023-222618` (fused) and `20261023-225211` (raw)

v17 is v13 (v10's line, `training/configs/structure_hold_ppo_v13.yaml`) continued four
times: a fresh 0.2 m start scatter about every 20 s of training with the policy's history
refilled (v14, `reengage_rate_hz`; Gate 1 opens every 20 s episode with that transient), a
reward on the height error alone (v15), the toward-the-wall penalty at 4.0 (v16) and training
jumps and dropouts at 0.1 Hz (v17, `training/configs/structure_hold_ppo_v17.yaml`, run
`20261023-211755`, iteration 11500). It is a Gate 1 candidate, not the release: it fails two
of the Gazebo-like check's re-fly rules (`docs/gazebo_check.md`), so v10 still flies.
Configs `eval/configs/gate1_x500_ekf_v17.yaml` and `gate1_x500_v17.yaml`; the same
held-out episodes as v10's.

| Cell | Raw: p95 v17 / PD (m) | Raw: crash v17 / PD | Raw verdict | Fused: p95 v17 / PD (m) | Fused: change | Fused: crash v17 / PD | Fused verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.82 / 0.48 | 58.5 / 9.4 % | fail | 0.22 / 0.32 | -30 % | 0.0 / 2.8 % | pass |
| 0 m/s, 2 m | 0.96 / 0.52 | 57.3 / 2.4 % | fail | 0.22 / 0.34 | -34 % | 0.0 / 0.0 % | pass |
| 4 m/s, 1 m | 0.81 / 0.49 | 57.2 / 8.7 % | fail | 0.22 / 0.32 | -31 % | 0.0 / 3.0 % | pass |
| 4 m/s, 2 m | 0.94 / 0.54 | 56.5 / 2.4 % | fail | 0.23 / 0.36 | -37 % | 0.0 / 0.0 % | pass |
| 8 m/s, 1 m (headline) | 0.79 / 0.51 | 56.0 / 8.9 % | fail | 0.23 / 0.37 | -36 % | 0.0 / 2.9 % | pass (headline ok) |
| 8 m/s, 2 m | 0.95 / 0.60 | 56.3 / 2.1 % | fail | 0.25 / 0.44 | -44 % | 0.0 / 0.0 % | pass |
| **Pooled** | **0.88 / 0.52** | **57.0 / 5.7 %** | **FAIL, 0 of 6** | **0.23 / 0.36** | | **0.0 / 1.4 %** | **PASS, 6 of 6** |

Through the fusion v17 passes Gate 1: every cell's interval is below 0, it does not crash
once in 18,432 episodes, and the headline is -36 % (interval -38 to -35 %) against the
-30 % rule (v10 -15 %). What moved the headline was the start: on a scratch seed (8001) v13
still had 0.22 m of its start offset left 1 s in and v14 0.15 m, against the PD's 0.19 m.
On the raw fix, 50 to 150 ms late, it fails as v10 did and worse (crash 57 % against
v10's 37 %): it is trained behind the fusion, which compensates the delay. Which scoring
counts is Zain's call (the Proposed row in `docs/decisions.md`). The fused run took 26 min
and the raw one 23 min on 4 idle cores.

### On the x500, both ways: v18, runs `20261024-003735` (fused) and `20261024-010343` (raw)

v18 is v17 continued with the toward-the-wall penalty back at 2.0 and the body-rate penalty
at 0.2 (`training/configs/structure_hold_ppo_v18.yaml`, run `20261023-232144`, iteration
12500), picked on the Gazebo-like check (`docs/gazebo_check.md`), where it was the first
policy to pass every rule. Run once on the held-out seeds as the candidate for both Gate 1
and the Gazebo re-fly. Configs `eval/configs/gate1_x500_ekf_v18.yaml` and
`gate1_x500_v18.yaml`; the same held-out episodes as v10's and v17's.

| Cell | Raw: p95 v18 / PD (m) | Raw: crash v18 / PD | Raw verdict | Fused: p95 v18 / PD (m) | Fused: change | Fused: crash v18 / PD | Fused verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.82 / 0.48 | 74.4 / 9.4 % | fail | 0.23 / 0.32 | -27 % | 0.0 / 2.8 % | pass |
| 0 m/s, 2 m | 1.07 / 0.52 | 73.3 / 2.4 % | fail | 0.23 / 0.34 | -32 % | 0.0 / 0.0 % | pass |
| 4 m/s, 1 m | 0.81 / 0.49 | 72.8 / 8.7 % | fail | 0.23 / 0.32 | -28 % | 0.0 / 3.0 % | pass |
| 4 m/s, 2 m | 1.07 / 0.54 | 72.5 / 2.4 % | fail | 0.23 / 0.36 | -35 % | 0.0 / 0.0 % | pass |
| 8 m/s, 1 m (headline) | 0.81 / 0.51 | 72.3 / 8.9 % | fail | 0.24 / 0.37 | -35 % | 0.0 / 2.9 % | pass (headline ok) |
| 8 m/s, 2 m | 1.07 / 0.60 | 71.2 / 2.1 % | fail | 0.25 / 0.44 | -44 % | 0.0 / 0.0 % | pass |
| **Pooled** | **0.94 / 0.52** | **72.7 / 5.7 %** | **FAIL, 0 of 6** | **0.24 / 0.36** | | **0.0 / 1.4 %** | **PASS, 6 of 6** |

Through the fusion v18 passes as v17 did: headline -35 % (interval -37 to -33 %), no crash in
18,432 episodes. On the raw fix it fails worse than v17 (crash 73 % against 57 %; v10 37 %).
In Gazebo (PR #58) it held 9 to 12 % tighter than v10 on the mean but lower on one seed of
three, with a 3.3 Hz ring, so v10 stays the release and v18 is not the Gate 1 policy of
record either; v17 and v18 show that the fused Gate 1 is passed by this line of training.
The fused run took 26 min and the raw one 43 min (it shared the 4 cores for 15 min).

### On the x500, both ways: v19 (the release), runs `20261024-041240` (fused) and `20261024-044004` (raw)

v19 is v18 continued with a penalty on the 1 to 6 Hz part of its rate commands
(`reward.action_band` 5.0; `training/configs/structure_hold_ppo_v19.yaml`, run
`20261024-020006`, iteration 13500), which removed v18's 3.3 Hz ring. Gazebo released it
(PR #61): hold p95 12 to 16 % below v10's and 44 to 47 % below tuned PX4's on the matched
timeline. Run once on the held-out seeds as the released policy. Configs
`eval/configs/gate1_x500_ekf_v19.yaml` and `gate1_x500_v19.yaml`; the same held-out
episodes as v10's, v17's and v18's.

| Cell | Raw: p95 v19 / PD (m) | Raw: crash v19 / PD | Raw verdict | Fused: p95 v19 / PD (m) | Fused: change | Fused: crash v19 / PD | Fused verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 1.21 / 0.48 | 63.2 / 9.4 % | fail | 0.24 / 0.32 | -26 % | 0.0 / 2.8 % | pass |
| 0 m/s, 2 m | 1.27 / 0.52 | 62.6 / 2.4 % | fail | 0.24 / 0.34 | -31 % | 0.0 / 0.0 % | pass |
| 4 m/s, 1 m | 1.20 / 0.49 | 62.3 / 8.7 % | fail | 0.24 / 0.32 | -27 % | 0.0 / 3.0 % | pass |
| 4 m/s, 2 m | 1.25 / 0.54 | 61.3 / 2.4 % | fail | 0.24 / 0.36 | -34 % | 0.0 / 0.0 % | pass |
| 8 m/s, 1 m (headline) | 1.18 / 0.51 | 61.3 / 8.9 % | fail | 0.24 / 0.37 | -34 % | 0.0 / 2.9 % | pass (headline ok) |
| 8 m/s, 2 m | 1.26 / 0.60 | 62.2 / 2.1 % | fail | 0.26 / 0.44 | -42 % | 0.1 / 0.0 % | pass |
| **Pooled** | **1.23 / 0.52** | **62.2 / 5.7 %** | **FAIL, 0 of 6** | **0.24 / 0.36** | | **0.0 / 1.4 %** | **PASS, 6 of 6** |

Through the fusion the released policy passes Gate 1: every cell, headline -34 % (interval
-36 to -32 %), 7 crashes in 18,432 episodes (0.04 %) against the PD's 1.4 %. On the raw fix 50 to
150 ms late it fails like every policy since v10 (crash 62 %; v18 73 %, v17 57 %, v10 37 %):
the policies are trained and flown behind EKF2, which takes out that delay on the drone, and
none of them was trained on the raw fix. Gate 1 is judged through the fusion (Decided
2026-10-24, `docs/decisions.md`), so v19 passes Gate 1's sim test; the raw-fix result stays
as the delay stress result. The fused run took 27 min and the raw one 37 min.

## Rule 5's dropout cell: what a 3 s cap would show (2026-10-24)

Zain asked what happens if the long-dropout stress cell stopped at 3 s instead of running 3 to
5 s. `eval/configs/gate1_x500_ekf_dropout_v19.yaml` flies v19 and the PD on the x500 through
the EKF2-like fusion, under the stress set's base condition (mean wind 0 to 8 m/s, hold 1 to
2 m), on diagnostic seeds 8001 and 8002 (2048 paired episodes per cell). It informs the rule;
it is not a Gate 1 verdict, which stays on the held-out seeds.

The first run (`20261024-132813`) passed every cell, 3 to 5 s included, with v19 at 0.24 m
p95 and no crash. That says more about the mock than the policy: the fusion dead-reckons on
the true acceleration, so a dropout costs it almost nothing. On the vehicle EKF2 leaves some
accelerometer and tilt error unestimated, hidden while the fix is fused and drifting the
estimate by half the error times the dropout's length squared once it stops.
`fusion.imu_bias_mps2` (off by default) adds that error per vehicle while the fix is stale.
Run `20261024-134117`, p95 v19 / PD and crash v19 / PD:

| Dropouts (0.05 Hz unless marked) | No IMU error | IMU error 0.1 m/s^2 (about 0.5 deg) | IMU error 0.2 m/s^2 (about 1 deg) |
| --- | --- | --- | --- |
| 0.2 to 3 s, 0.02 Hz (nominal) | 0.24 / 0.35 m, 0.0 / 1.1 %: pass | 0.27 / 0.38 m, 0.1 / 1.2 %: pass | 0.34 / 0.42 m, 1.2 / 2.3 %: pass |
| 2 to 3 s (capped at 3 s) | 0.24 / 0.35 m, 0.0 / 0.9 %: pass | 0.43 / 0.48 m, 0.5 / 1.6 %: pass (−10 %) | 0.71 / 0.68 m, 6.9 / 6.3 %: fail on hold (a tie) |
| 3 to 5 s (as written) | 0.24 / 0.34 m, 0.0 / 0.9 %: pass | 0.79 / 0.77 m, 10.7 / 10.0 %: fail on hold (a tie) | 1.04 / 1.01 m, 33 / 32 %: fail on hold (a tie) |

Once the estimate has drifted, no controller can hold the point: v19 and the PD crash alike,
and in the 3 to 5 s cell both crash in 1 flight of 10 at half a degree of error and 1 of 3 at a full degree. Capped
at 3 s, with half a degree, v19 passes and stays ahead of the PD; at a full degree even 3 s
is a tie. On PX4, EKF2 gives up on the position after 5 s without aiding and the watchdog
hands back (`SH_WD_EST_TO` 5 s), so 3 to 5 s is the stretch where the module flies on an
estimate that is drifting fastest. The rule stays as written until Zain decides
(`docs/decisions.md`, rule 5 row).
