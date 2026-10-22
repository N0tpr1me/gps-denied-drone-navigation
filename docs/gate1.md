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
