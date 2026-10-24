# Tuned-PX4 baseline v1

The classical reference every Structure Hold result is compared against:
`eval/baselines/px4_tuned_v1.params`, stock PX4 v1.17.0 on the Gazebo `gz_x500`, GPS-free on
the estimate bridge, tuned for wind 1 m from a wall. Frozen on 2026-10-22. The commits in the LOG lines (`6753b79`, `fa992fd`, `4ce5c63`) are
this work's branch before its rebase onto main. The flight code and parameter files are the ones in this PR.
Every flight below is a line in `experiments/LOG.md`.

## Headline

1 m from the wall, wind into the wall, 60 s hover, `vio_default` corrupted estimate, seeds 1–3,
scored on the TRUE position by `eval.metrics` (mean over seeds, worst in brackets):

| scenario | p95 hold error m | worst toward wall m | crashes | failsafes |
|---|---|---|---|---|
| calm 0 m/s, stock | 0.623 (1.075) | 0.217 (0.356) | 0/3 | 2/3 |
| calm 0 m/s, **tuned v1** | **0.567 (0.911)** | **0.240 (0.420)** | 0/3 | 2/3 |
| 4 m/s, stock | 0.601 (1.019) | 0.194 (0.327) | 0/3 | 2/3 |
| 4 m/s, **tuned v1** | **0.570 (0.919)** | **0.230 (0.406)** | 0/3 | 2/3 |
| 8 m/s, stock | 0.610 (1.061) | 0.225 (0.392) | 0/3 | 2/3 |
| 8 m/s, **tuned v1** | **0.570 (0.929)** | **0.214 (0.386)** | 0/3 | 2/3 |

Same scenarios on the clean estimate (seed 1), which isolates what the controller does
against wind:

| scenario | p95 stock → tuned m | worst toward wall stock → tuned m |
|---|---|---|
| calm 0 m/s | 0.009 → 0.010 | 0.003 → 0.002 |
| 4 m/s | 0.029 → 0.016 (−45 %) | 0.053 → 0.024 |
| 8 m/s | 0.102 → 0.037 (−64 %) | 0.147 → 0.077 |

**Reading it.** The tuning does what it should against wind: on a clean estimate the 8 m/s
hold error drops by about two thirds. On the corrupted estimate the wind hardly matters,
for stock or tuned: each seed scores the same in calm air as at 8 m/s, and p95 goes from 0.31 m
(seed 3) to 1.0 m (seed 1) depending only on what the estimate does. The tuned set is 5–9 %
better on mean p95. On seed 3 it goes about 0.1 m further toward the wall than stock, because
the stiffer loop follows the estimate's errors faster. Seeds 1–3 are not enough to call that
trade-off significant. No flight touched the wall: the closest was 0.54 m.

**What dominates the error: estimate jumps.** In seed 1 a 1.7 m jump arrives during the hover.
EKF2 rejects it (innovation test) for exactly 1 s, the hard-coded `no_aid_timeout_max` in
`ev_pos_control.cpp`, then resets its position to the jumped vision. The vehicle then flies
about 0.9 m to "correct" an error that is not there, and comes back as the jump decays (time
constant 5 s). There is no parameter for that 1 s, so no PX4 tuning removes it. This is
exactly the gap Structure Hold is meant to close. The failsafes are PX4's position-loss
failsafe tripping on the longer dropouts (2.5 s at takeoff in seed 1, during the hover in
seed 3): Land for about a second, then the mission resumes.

## How it was tuned

1. **Attitude and rate loops: PX4 autotune.** `python -m eval.baselines.autotune` flies
   `autotune_open_calm` (still air, 4 m from the wall, clean estimate, GPS off), sets
   `MC_AT_START 1` in Hold, and reads the gains back after disarm. Two runs
   (`MC_AT_SYSID_AMP` 0.7 and 1.5) identified roll and pitch within 1 % of each other.
   Both times yaw never passed the module's variance test before its 20 s timeout, so the
   module failed the sequence and saved nothing. The roll and pitch values in the file are
   the module's own parallel-form result from the first run, copied from the ulog's
   `autotune_attitude_control_status`. Yaw stays stock. The rate gains came out within
   about 30 % of the x500's stock values. The attitude gain went from 4.0 to 6.5.
2. **Position and velocity loops: hand-tuned for wind.** Tuned only on tuning seeds 11 and
   12 at 8 m/s, never on the evaluation seeds 1–3. Candidates (each = the autotune result plus):

   | candidate | MPC_XY_P | VEL_P_ACC | VEL_I_ACC | VEL_D_ACC | corrupted p95 s11 / s12 m | clean p95 s11 m |
   |---|---|---|---|---|---|---|
   | stock (no autotune) | 0.95 | 1.8 | 0.4 | 0.2 | 0.662 / 0.145 | 0.048 |
   | A (autotune only) | 0.95 | 1.8 | 0.4 | 0.2 | 0.673 / 0.142 | – |
   | B | 1.2 | 2.4 | 1.2 | 0.2 | 0.656 / 0.146 | 0.026 |
   | C (softer) | 0.8 | 1.8 | 0.8 | 0.2 | 0.673 / 0.152 | – |
   | **E = v1** | **1.5** | **3.0** | **2.0** | **0.3** | 0.653 / 0.147 | **0.017** |
   | D (A, vision velocity only: `EKF2_EV_CTRL 4`, baro height) | 0.95 | 1.8 | 0.4 | 0.2 | never armed | – |

   On the corrupted estimate the candidates were indistinguishable (within 0.02 m). The clean
   estimate separated them, so E was chosen on that. The higher velocity integral gain is
   what rejects the mean wind and the gusts. E stayed stable through the corrupted tuning
   flights (jumps and dropouts, no failsafe), which was the check that it is not too stiff
   for a 100 ms-late estimate.
3. **EKF2 unchanged.** The estimator settings are `sim2real/estimate_bridge/ekf2_ev.params`.
   The estimate path is part of the benchmark, not the controller under test. Candidate D
   tried the obvious estimator fix: fuse only the vision velocity, which the corruption model
   leaves unjumped. Stock PX4 then never reports a position good enough to arm. Making that
   work would also exploit a gap in the corruption model, since a real VIO's velocity
   glitches too when it relocalises.

## Reproduce

```bash
sim2real/docker/run.sh build                          # once; image structure-hold/px4-sitl:v1.17.0
pip install -e .[dev,ulog]
python -m eval.baselines.run_baseline --label stock --params sim2real/estimate_bridge/ekf2_ev.params
python -m eval.baselines.run_baseline --label tuned_v1 --params eval/baselines/px4_tuned_v1.params
python -m eval.baselines.run_baseline --compare runs/<stock>/batch.json runs/<tuned>/batch.json
python -m eval.baselines.run_baseline --label clean_tuned_v1 --params eval/baselines/px4_tuned_v1.params --seeds 1 -- --profile clean
python -m eval.baselines.autotune                     # the autotune flight
```

Each flight takes about 2 minutes on the Mac (Docker Desktop, arm64, real-time factor about
0.95). Nine flights per parameter set. The scenarios are `sim2real/gazebo/scenarios/wall1m_*`:
the T6 scenarios moved to 1 m with a 60 s hover and both gusts inside it. Seed `s` seeds both
the wind draw and the estimate corruption.

## Limits

- Three seeds per scenario. The seed-to-seed spread (0.3 to 1.0 m) is far larger than the
  stock-to-tuned difference, so the corrupted-estimate comparison above is a description, not
  a significance test. Comparisons with the policy should use the same seeds, paired.
- Yaw gains are stock (autotune could not identify yaw on the gz x500).
- One airframe (gz_x500), one wall geometry, wind straight into the wall. No near-wall
  torque, and no slab effect in the disturbance model yet (sim2real/gazebo/README).
- Hold quality only (Takeoff → Hold). The 5 m pass is not scored here.

## Structure Hold v3 vs tuned PX4

Gazebo SITL on the Mac, 2026-10-22: the Structure Hold module with policy v3 at its
defaults (yaw from the policy, fallback off), flown by
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v3 --params eval/baselines/px4_tuned_v1.params`
on the same scenarios, seeds, estimate draws, PX4 parameter set and truth scoring as the
tuned v1 rows above. The 60 s window is the time spent in Structure Hold. One LOG line per flight.

| scenario | p95 hold error m, tuned v1 → v3 | worst toward wall m, tuned v1 → v3 | crashes | failsafes |
|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) → 1.873 (1.924) | 0.240 (0.420) → 0.396 (0.548) | 0/3 → 2/3* | 2/3 → 0/3 |
| 4 m/s | 0.570 (0.919) → 1.675 (1.929) | 0.230 (0.406) → 0.423 (0.554) | 0/3 → 2/3* | 2/3 → 0/3 |
| 8 m/s | 0.570 (0.929) → 1.569 (1.925) | 0.214 (0.386) → 0.486 (0.646) | 0/3 → 1/3* | 2/3 → 0/3 |

Mean over seeds 1–3, worst in brackets. \*No flight touched the wall (0 contacts, closest
0.41 m). Each "crash" is a flight whose acceptance sequence did not finish: the module's
watchdog tripped on position error (60–81 s, seeds 1 and 2) and handed back to stock
Position mode, after which the scripted re-engage was refused, as a latched trip should be.
Fallback share and yaw-guard share were 0 in every flight; stale time peaked at 2.0 s in
the seed-2 dropouts. On the clean estimate in calm air (plumbing run) v3 scored p95 1.06 m
against tuned PX4's 0.010 m.

**Reading it.** Policy v3 does not transfer to Gazebo's x500. It is 2.7 to 3.3 times worse
than tuned PX4 on p95 in every scenario, and goes further toward the wall. The cause is the
controller, not the estimate: with an exact, fresh estimate in still air it still holds
1 m off. Its roll-rate command swings ±1.5 to 2 rad/s and its thrust 0.5 to 0.9 in a 2 Hz limit cycle,
with the error parked at 0.6 to 1 m and not converging. The mock it was trained on (a 0.8 kg
5-inch quad with a first-order rate lag) does not reproduce the x500's response, so this is
a sim-to-sim gap to close in training (the x500's mass, thrust and rate response, or
randomising them), not something to tune in the module. What did work is the safety layer:
every run was GPS-free with vision fused, the watchdog handed back to stock PX4 every time it fired
(0.02 s measured in the ulog of each flight that finished the sequence), and no flight hit PX4's own position-loss failsafe while the module
was flying. The fallback run (8 m/s seed 1, `SH_FB_EN=1`) showed no fallback episodes: the
position-error trip came before any dropout over 0.5 s, so the fallback could not be judged here.

## Structure Hold v5 vs tuned PX4

Same benchmark as the v3 section above, on policy v5 (trained on the gz x500 airframe),
module headers at `6f8b72b`, image rebuilt, 2026-10-22:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v5 --params eval/baselines/px4_tuned_v1.params`.
Mean over seeds 1–3, worst in brackets.

| scenario | p95 hold error m, tuned v1 / v3 / **v5** | worst toward wall m, tuned v1 / v3 / **v5** | crashes, tuned / v3 / v5 | failsafes, tuned / v3 / v5 |
|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) / 1.873 (1.924) / **0.717 (0.942)** | 0.240 (0.420) / 0.396 (0.548) / **0.136 (0.171)** | 0/3 / 2/3 / 0/3 | 2/3 / 0/3 / 1/3† |
| 4 m/s | 0.570 (0.919) / 1.675 (1.929) / **0.700 (0.923)** | 0.230 (0.406) / 0.423 (0.554) / **0.177 (0.223)** | 0/3 / 2/3 / 0/3 | 2/3 / 0/3 / 0/3 |
| 8 m/s | 0.570 (0.929) / 1.569 (1.925) / **0.759 (1.046)** | 0.214 (0.386) / 0.486 (0.646) / **0.252 (0.477)** | 0/3 / 1/3 / 1/3‡ | 2/3 / 0/3 / 0/3 |

† PX4's position-loss failsafe during takeoff (Land at 16.4 s, seed 3), before Structure Hold
was engaged; none in the hold. ‡ Not the module: seed 3 held the full 60 s, handed back
in the scripted test, then flipped during stock PX4's final Land in the 8 m/s wind while a
vision jump was arriving (EKF2 innovation 3 m), and never reported touchdown. No v5 flight
touched the wall (0 contacts; closest clearance 0.53 m, 8 m/s seed 1).

**Reading it.** The 2 Hz roll and thrust cycle that sank v3 is gone. On the clean calm
plumbing flight v5 passes every check, including position_held. Its thrust sits at
0.717 ± 0.03 (v3 swung 0.44–0.90), and its roll-rate command std is 0.43 rad/s (v3 1.51),
with no saturation. Its p95 there is 0.41 m against tuned PX4's 0.010 m. Almost all of that is a
steady offset: v5 holds 0.3–0.45 m above its hold point in every flight and about 0.1 m off
horizontally. That offset is what keeps v5 behind tuned PX4 on p95: 0.70–0.76 m mean
against 0.57 m (+23 to +33 %), against v3's 1.57–1.87 m. v5 stays further from the wall than tuned PX4 in calm
air and at 4 m/s (worst toward wall 0.14 and 0.18 m vs 0.24 and 0.23 m) and slightly
closer at 8 m/s (0.25 vs 0.21 m, seed 1 0.48 m). The module side was clean in all nine flights: every
hold ran its full 60 s with no watchdog trip (only the scripted test trip), fallback and
yaw-guard shares 0, stale time up to 2.25 s in the dropout seeds, handback 0.02 s, every
flight GPS-free with vision fused, no PX4 failsafe while the module flew. Next lever: the
vertical offset (hover thrust or altitude bias in the x500 mock), then the horizontal one.

## Structure Hold v6 vs tuned PX4

Same benchmark again, on policy v6 (v5's x500 recipe trained longer), module headers at
`7eee67a`. That commit also carries the age fix: the policy is fed age 0 while the estimate
is fresh (`StructureHold.cpp`: `sample.age_s = sample.stale ? _stale_s : 0`). Image
rebuilt, 2026-10-23:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v6 --params eval/baselines/px4_tuned_v1.params`.
Mean over seeds 1–3, worst in brackets.

| scenario | p95 hold error m, tuned v1 / v3 / v5 / **v6** | worst toward wall m, tuned v1 / v3 / v5 / **v6** | crashes, tuned / v3 / v5 / v6 | failsafes, tuned / v3 / v5 / v6 |
|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) / 1.873 (1.924) / 0.717 (0.942) / **0.620 (0.859)** | 0.240 (0.420) / 0.396 (0.548) / 0.136 (0.171) / **0.158 (0.221)** | 0/3 / 2/3 / 0/3 / 0/3 | 2/3 / 0/3 / 1/3† / 0/3 |
| 4 m/s | 0.570 (0.919) / 1.675 (1.929) / 0.700 (0.923) / **0.641 (0.889)** | 0.230 (0.406) / 0.423 (0.554) / 0.177 (0.223) / **0.188 (0.267)** | 0/3 / 2/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |
| 8 m/s | 0.570 (0.929) / 1.569 (1.925) / 0.759 (1.046) / **0.679 (0.934)** | 0.214 (0.386) / 0.486 (0.646) / 0.252 (0.477) / **0.303 (0.475)** | 0/3 / 1/3 / 1/3‡ / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |

†, ‡ as in the v5 section (neither was the module). v6 had no crash and no PX4 failsafe.
One check failed outside the hold: on 8 m/s seed 3 the scripted re-engage after the hold tripped on
position error before the injected test fault, so `status_trip_logged` was false. The trip
and the handback themselves worked.

| per flight, mean over the 9 batch flights | v5 | v6 |
|---|---|---|
| steady offset above the hold point m (range) | +0.39 (+0.35 to +0.46) | +0.23 (+0.10 to +0.39) |
| steady horizontal offset m (range) | 0.12 (0.10 to 0.14) | 0.09 (0.04 to 0.19) |
| roll-rate setpoint std rad/s, peak | 0.53, 1.6 Hz in 9/9 | 0.51, 1.6 Hz in 6/9, 10.9 Hz in 3/9 |
| thrust mean | 0.713 | 0.714 |

**Reading it.** v6 is the closest yet to tuned PX4: mean p95 0.62 / 0.64 / 0.68 m against
0.57 m (+9 to +19 %), 8 to 14 % better than v5. It matches or beats tuned PX4 on the worst
seed (0.86–0.93 m against 0.91–0.93 m) and on seed 2 (0.43–0.45 m against 0.44 m). It loses
on seed 3 (0.57–0.66 m against 0.34–0.35 m). It stays further from the wall than tuned PX4 in calm air and at 4 m/s
(0.16 and 0.19 m against 0.24 and 0.23 m) but comes closer at 8 m/s (0.30 against 0.21 m).
On the clean plumbing flight v6 holds 0.114 m p95 (v5 0.41 m). The vertical offset is down to
+0.04 m (v5 +0.29 m), the horizontal one to 0.045 m, and thrust is 0.701 ± 0.03. The 1.6 Hz wobble is gone there:
the roll and pitch setpoints now chatter at 10.9 Hz (std 0.49 and 0.57 rad/s), which the
rate loop mostly filters (measured roll-rate std 0.16 rad/s against v5's 0.43). On the corrupted
estimate the wobble and the offset come back on seeds 1 and 3. Those are the rough-estimate
seeds: EKF2 vision innovation 0.18–0.23 m rms with 1–4 jumps, against 0.04 m on seed 2. There
the setpoints peak at 1.6 Hz again and the vehicle sits +0.17 to +0.39 m high. On seed 2 it
behaves as in the clean flight (+0.10 to +0.13 m, 10.9 Hz). The module side was clean in all
nine flights: every hold ran its full 60 s with no watchdog trip, fallback and yaw-guard shares 0,
stale time up to 2.25 s in the dropout seeds, handback 0.016–0.024 s, GPS-free with vision
fused, 0 wall contacts (closest clearance 0.52 m, 8 m/s seed 1).

## Structure Hold v8 vs tuned PX4

Same benchmark again, on policy v8 (the x500 recipe trained with the hold point 3.5 m up),
module headers at `b3e875f`, image rebuilt, 2026-10-23:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v8 --params eval/baselines/px4_tuned_v1.params`.
Mean over seeds 1–3, worst in brackets.

| scenario | p95 hold error m, tuned v1 / v5 / v6 / **v8** | worst toward wall m, tuned v1 / v5 / v6 / **v8** | crashes, tuned / v5 / v6 / v8 | failsafes, tuned / v5 / v6 / v8 |
|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) / 0.717 (0.942) / 0.620 (0.859) / **0.671 (0.967)** | 0.240 (0.420) / 0.136 (0.171) / 0.158 (0.221) / **0.255 (0.434)** | 0/3 / 0/3 / 0/3 / 0/3 | 2/3 / 1/3† / 0/3 / 0/3 |
| 4 m/s | 0.570 (0.919) / 0.700 (0.923) / 0.641 (0.889) / **0.677 (0.971)** | 0.230 (0.406) / 0.177 (0.223) / 0.188 (0.267) / **0.252 (0.380)** | 0/3 / 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |
| 8 m/s | 0.570 (0.929) / 0.759 (1.046) / 0.679 (0.934) / **0.793 (1.018)** | 0.214 (0.386) / 0.252 (0.477) / 0.303 (0.475) / **0.351 (0.487)** | 0/3 / 1/3‡ / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |

†, ‡ as in the v5 section (neither was the module). v8 had no crash, no PX4 failsafe and
no failed check other than the script's position_held drift tolerance (6 of 9, as v6).

| | v6 | v8 |
|---|---|---|
| clean calm flight: p95, offset up, horizontal | 0.114 m, +0.04 m, 0.045 m | 0.109 m, −0.01 m, 0.064 m |
| clean calm flight: along-wall drift (truth) mean, p2p, period | −0.024 m, 0.14 m, 2.1 s | −0.031 m, 0.16 m, 2.4 s |
| clean calm flight: roll / pitch rate setpoint std, peak | 0.49 / 0.57 rad/s, 10.9 Hz | 0.39 / 0.39 rad/s, 10.7 Hz |
| clean calm flight: thrust | 0.701 ± 0.030 | 0.710 ± 0.016 |
| batch: offset up, mean (range) | +0.23 m (+0.10 to +0.39) | +0.42 m (+0.36 to +0.55) |
| batch: offset up, seeds 1 / 2 / 3 | +0.25 / +0.11 / +0.32 m | +0.40 / +0.42 / +0.45 m |
| batch: horizontal offset, mean | 0.09 m | 0.08 m |
| batch: roll-rate setpoint std, mean | 0.51 rad/s | 0.64 rad/s |
| batch: 1.6 Hz wobble | 6/9 flights (seeds 1, 3) | 9/9 flights |
| batch: along-wall drift (truth) p2p, mean | 0.55 m | 0.50 m |

The offsets are the module's position error (hold point minus estimate) over the hold.
The along-wall drift is truth against the hold point, so in the batch it also contains the
estimate's own jumps and bias walk. Its 20–30 s period on seeds 1 and 3 is that, not the
policy.

**Reading it.** On the clean flight v8 is the best policy so far: p95 0.109 m and no
vertical offset (−0.01 m against v6's +0.04 m). Its thrust is steadier (± 0.016) and its
setpoint chatter is lower (0.39 rad/s). The along-wall drift the training side expected
(about ±0.1 m on a 10 s cycle) did not show: drift along the wall is 0.16 m peak to peak at
2.4 s, about v6's. On the corrupted estimate that advantage is gone. The 1.6 Hz wobble now
shows in every flight, including seed 2, which v6 held cleanly. The vertical offset comes
back larger than v6's in every seed (+0.36 to +0.55 m, mean +0.42 against +0.23 m). Mean
p95 is 0.67 / 0.68 / 0.79 m: 6 to 17 % worse than v6 and 18 to 39 % worse than tuned PX4's
0.57 m. v8 also comes closer to the wall than v6 in every scenario (0.25 / 0.25 / 0.35 m
against 0.16 / 0.19 / 0.30 m), and closer than tuned PX4. The module side was clean: every hold ran its full 60 s with no watchdog trip,
fallback and yaw-guard shares 0, stale time up to 2.26 s in the dropout seeds, handback
0.016–0.02 s, GPS-free with vision fused, 0 wall contacts (closest clearance 0.52 m, 8 m/s
seed 1). **Verdict: v8 is worse than v6 overall.** It is better only on the clean estimate
(z offset, chatter). On the benchmark itself it loses on p95, distance toward the wall,
vertical offset and the 1.6 Hz wobble. Its along-wall drift is about the same. v6 stays the
better policy in Gazebo. The module therefore stays on v6: its headers were regenerated from
`structure_hold_v6.onnx` in the same PR, and v8 is kept only as a recorded experiment.

## Structure Hold v9 vs tuned PX4

Same benchmark again, on policy v9 (the x500 recipe trained on the Gazebo-like mock),
module headers at `44eda1d`, image rebuilt, 2026-10-23:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v9 --params eval/baselines/px4_tuned_v1.params`.
Mean over seeds 1–3, worst in brackets.

| scenario | p95 hold error m, tuned v1 / v6 / v8 / **v9** | worst toward wall m, tuned v1 / v6 / v8 / **v9** | crashes, tuned / v6 / v8 / v9 | failsafes, tuned / v6 / v8 / v9 |
|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) / 0.620 (0.859) / 0.671 (0.967) / **0.442 (0.534)** | 0.240 (0.420) / 0.158 (0.221) / 0.255 (0.434) / **0.017 (0.051)** | 0/3 / 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |
| 4 m/s | 0.570 (0.919) / 0.641 (0.889) / 0.677 (0.971) / **0.491 (0.517)** | 0.230 (0.406) / 0.188 (0.267) / 0.252 (0.380) / **0.020 (0.059)** | 0/3 / 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |
| 8 m/s | 0.570 (0.929) / 0.679 (0.934) / 0.793 (1.018) / **0.646 (1.007)** | 0.214 (0.386) / 0.303 (0.475) / 0.351 (0.487) / **0.136 (0.241)** | 0/3 / 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 / 0/3 |

v9 had no crash, no PX4 failsafe and no failed check other than the script's position_held
drift tolerance (6 of 9, as v6 and v8). Change in mean p95 against tuned PX4: −22 / −14 /
+13 %; against v6: −29 / −23 / −5 %; against v8: −34 / −27 / −19 %.

| per seed | p95 m, calm / 4 / 8 m/s | v6 p95 m | worst toward wall m | offset up, module / truth m | roll-rate setpoint peak |
|---|---|---|---|---|---|
| seed 1 | 0.534 / 0.517 / 0.591 | 0.859 / 0.889 / 0.934 | 0.051 / 0.059 / 0.156 | +0.05 / +0.21, +0.08 / +0.23, +0.09 / +0.24 | 7.7, 10.8, 10.7 Hz |
| seed 2 | 0.437 / 0.448 / 1.007 | 0.434 / 0.439 / 0.446 | 0.000 / 0.000 / 0.241 | +0.05 / −0.01, +0.06 / +0.00, +0.02 / −0.03 | 7.6, 7.6, 10.5 Hz |
| seed 3 | 0.357 / 0.509 / 0.342 | 0.568 / 0.596 / 0.657 | 0.000 / 0.000 / 0.013 | +0.03 / +0.06, +0.06 / +0.08, +0.06 / +0.09 | 7.7, 10.8, 10.7 Hz |

| | v6 | v8 | v9 |
|---|---|---|---|
| clean calm flight: p95, offset up, horizontal | 0.114 m, +0.04 m, 0.045 m | 0.109 m, −0.01 m, 0.064 m | 0.081 m, +0.04 m, 0.050 m |
| clean calm flight: along-wall drift (truth) mean, p2p | −0.024 m, 0.14 m | −0.031 m, 0.16 m | +0.021 m, 0.06 m |
| clean calm flight: roll / pitch rate setpoint std, peak | 0.49 / 0.57 rad/s, 10.9 Hz | 0.39 / 0.39 rad/s, 10.7 Hz | 0.38 / 0.30 rad/s, 10.8 Hz |
| clean calm flight: thrust | 0.701 ± 0.030 | 0.710 ± 0.016 | 0.717 ± 0.007 |
| batch: offset up (module), mean (range) | +0.23 m (+0.10 to +0.39) | +0.42 m (+0.36 to +0.55) | +0.06 m (+0.02 to +0.09) |
| batch: offset up (module), seeds 1 / 2 / 3 | +0.25 / +0.11 / +0.32 m | +0.40 / +0.42 / +0.45 m | +0.07 / +0.04 / +0.05 m |
| batch: offset up (truth), mean (range) | +0.27 m (+0.04 to +0.54) | +0.46 m (+0.33 to +0.59) | +0.10 m (−0.03 to +0.24) |
| batch: horizontal offset, mean | 0.09 m | 0.08 m | 0.15 m (0.11 without 8 m/s seed 2) |
| batch: roll-rate setpoint std, mean | 0.51 rad/s | 0.64 rad/s | 0.31 rad/s |
| batch: measured roll-rate std, mean | 0.44 rad/s | 0.69 rad/s | 0.11 rad/s |
| batch: 1.6 Hz wobble | 6/9 flights (seeds 1, 3) | 9/9 flights | 0/9 flights |
| batch: along-wall drift (truth) p2p, mean | 0.55 m | 0.50 m | 0.44 m |

Offsets and drift are measured as in the v8 section. The module offset is the hold point
minus the estimate. The truth offset is the hold point minus the true position, so it
also carries the estimate's bias (seed 1's +0.2 m is that).

**Reading it.** v9 is the first policy to beat tuned PX4 in Gazebo, and it does so in calm air and at 4 m/s:
mean p95 0.44 and 0.49 m against 0.57 m (−22 and −14 %). Its worst seed there (0.53 m) is
far under tuned PX4's (0.91–0.92 m). It barely approaches the wall: worst toward wall 0.02 m mean
in those two scenarios, against 0.23–0.24 m for tuned PX4 and 0.16–0.19 m for v6. The two
Gazebo failure modes of v6 and v8 are gone. The 1.6 Hz wobble shows in none of the nine
flights: setpoints peak at 7.6–7.7 Hz or 10.5–10.8 Hz with half v6's spread, and the
measured roll rate is a quarter of v6's. The vertical offset is +0.02 to +0.09 m against the
estimate in every seed, rough-estimate seeds included (v6 +0.10 to +0.39 m). On the clean
plumbing flight it holds 0.081 m p95 (v6 0.114 m, v8 0.109 m), with the steadiest thrust yet
(± 0.007) and a third of v8's along-wall drift. The weak spot is 8 m/s seed 2, the
clean-estimate seed (EKF2 vision innovation 0.04 m rms), at p95 1.007 m. About 20 s into the hold the
vehicle walked off the hold point slowly, along the wall and away from it (0.03 m/s), and the
estimate followed it (estimate error p95 0.35 m). The module's error grew to about 1.0 m by
+44 s, then the policy pulled back to the hold point within 8 s, with no fallback and no trip. That one
flight makes the 8 m/s mean +13 % against tuned PX4 and its worst 1.007 m against v6's
0.934 m. The other two 8 m/s seeds are the best 8 m/s flights so far (0.59 and 0.34 m), and
at 8 m/s v9 still comes less close to the wall than tuned PX4 or v6 (0.14 m against 0.21 m tuned
and 0.30 m v6). The module side was clean: every hold ran its full 60 s with no watchdog trip,
fallback and yaw-guard shares 0, no saturated action, stale time up to 2.25 s in the dropout
seeds, handback 0.016–0.02 s, GPS-free with vision fused, 0 wall contacts (closest
clearance 0.81 m, 8 m/s seed 2). **Verdict: v9 is better than v6 overall.** Its mean p95 is
lower in every wind (−29 / −23 / −5 %) and it keeps further from the wall in every wind. Its vertical offset
is about a quarter of v6's, and the wobble shows in 0 of 9 flights against 6. The one open
item is the slow 8 m/s excursion on seed 2.

## Structure Hold v10 vs tuned PX4

Same benchmark again, on policy v10 (v9's recipe, also trained on wind into the wall and
EKF2's resets to the fix), module headers at `02c7c8e`, image rebuilt, 2026-10-23:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v10 --params eval/baselines/px4_tuned_v1.params`.
Mean over seeds 1–3, worst in brackets.

| scenario | p95 hold error m, tuned v1 / v9 / **v10** | worst toward wall m, tuned v1 / v9 / **v10** | crashes, tuned / v9 / v10 | failsafes, tuned / v9 / v10 |
|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) / 0.442 (0.534) / **0.372 (0.449)** | 0.240 (0.420) / 0.017 (0.051) / **0.087 (0.188)** | 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 |
| 4 m/s | 0.570 (0.919) / 0.491 (0.517) / **0.375 (0.444)** | 0.230 (0.406) / 0.020 (0.059) / **0.075 (0.171)** | 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 |
| 8 m/s | 0.570 (0.929) / 0.646 (1.007) / **0.391 (0.472)** | 0.214 (0.386) / 0.136 (0.241) / **0.084 (0.163)** | 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 1/3 |

v10 had no crash and no failed check other than the script's position_held drift tolerance
(6 of 9, as v6, v8 and v9). The one failsafe (8 m/s seed 3) came during PX4's own
auto-takeoff, 15 s before the module engaged: 1.1 s in Land, then the takeoff resumed; none
in the hold. Change in mean p95 against tuned PX4: −34 / −34 / −31 %; against v9: −16 / −24
/ −40 %. RMS against v9: −21 / −27 / −44 %.

| per seed | p95 m, calm / 4 / 8 m/s | v9 p95 m | worst toward wall m | offset up, module / truth m | roll-rate setpoint peak, std |
|---|---|---|---|---|---|
| seed 1 | 0.449 / 0.444 / 0.472 | 0.534 / 0.517 / 0.591 | 0.072 / 0.053 / 0.090 | +0.04 / +0.19, +0.05 / +0.20, +0.06 / +0.21 | 10.8, 10.8, 10.7 Hz; 0.17, 0.18, 0.25 rad/s |
| seed 2 | 0.337 / 0.354 / 0.353 | 0.437 / 0.448 / 1.007 | 0.000 / 0.000 / 0.000 | +0.04 / −0.01, +0.04 / −0.01, +0.05 / +0.00 | 10.8, 10.8, 10.7 Hz; 0.18, 0.19, 0.23 rad/s |
| seed 3 | 0.331 / 0.328 / 0.347 | 0.357 / 0.509 / 0.342 | 0.188 / 0.171 / 0.163 | +0.03 / +0.06, +0.04 / +0.07, +0.04 / +0.07 | 10.8, 10.8, 10.7 Hz; 0.19, 0.20, 0.27 rad/s |

| | v9 | v10 |
|---|---|---|
| clean calm flight: p95, offset up, horizontal | 0.081 m, +0.04 m, 0.050 m | 0.036 m, +0.03 m, 0.014 m |
| clean calm flight: along-wall drift (truth) mean, p2p | +0.021 m, 0.06 m | −0.008 m, 0.03 m |
| clean calm flight: roll / pitch rate setpoint std, peak | 0.38 / 0.30 rad/s, 10.8 Hz | 0.19 / 0.14 rad/s, 10.8 Hz |
| clean calm flight: thrust | 0.717 ± 0.007 | 0.723 ± 0.004 |
| batch: offset up (module), mean (range) | +0.06 m (+0.02 to +0.09) | +0.04 m (+0.03 to +0.06) |
| batch: offset up (module), seeds 1 / 2 / 3 | +0.07 / +0.04 / +0.05 m | +0.05 / +0.04 / +0.04 m |
| batch: offset up (truth), mean (range) | +0.10 m (−0.03 to +0.24) | +0.09 m (−0.01 to +0.21) |
| batch: horizontal offset, mean | 0.15 m (0.11 without 8 m/s seed 2) | 0.03 m |
| batch: roll-rate setpoint std, mean | 0.31 rad/s | 0.21 rad/s |
| batch: measured roll-rate std, mean | 0.11 rad/s | 0.08 rad/s |
| batch: 1.6 Hz wobble | 0/9 flights | 0/9 flights |
| batch: along-wall drift (truth) p2p, mean | 0.44 m | 0.33 m |
| batch: mean truth offset from the wall, seeds 1 / 2 / 3 | −0.04 / −0.28 / −0.14 m | −0.01 / −0.12 / −0.04 m |

Offsets and drift are measured as in the v8 section. The last row is the hold-window mean of
the true position minus the hold point along the wall normal, negative away from the wall.

Vision delay in Gazebo: EKF2 fuses each vision sample 154 ms behind real time in all nine
benchmark flights and 158 ms in the clean flight (`estimator_aid_src_ev_pos` timestamp minus
timestamp_sample, 152–164 ms), against `EKF2_EV_DELAY` 100 ms. `vehicle_visual_odometry`
itself shows 0 ms, because PX4 stamps the sample when it arrives (no TIMESYNC). That is the
same as the v9 flights (154 ms in every flight, clean included), so the delay is not higher.

**Reading it.** v10 is the best policy in Gazebo so far, and it beats tuned PX4 in every wind:
mean p95 0.37 / 0.38 / 0.39 m against 0.57 m (−34 / −34 / −31 %), and its worst flight
(0.47 m) is half of tuned PX4's worst (0.91–0.93 m). Against v9 the mean p95 is lower in every
wind (−16 / −24 / −40 %). The slow 8 m/s excursion on seed 2 is gone. That flight now holds
0.353 m p95 (v9 1.007 m), with the module error under 0.12 m for the whole hold, 0.04 m
horizontal offset (v9 0.46 m), along-wall p2p 0.21 m (v9 1.06 m) and no approach to the
wall at all. The truth error that remains there is the estimate's own bias, which starts
0.4 m off and shrinks as EKF2 corrects. v10 is also calmer: rate-setpoint std about two thirds of
v9's in the batch and half on the clean flight, the steadiest thrust yet (± 0.004), and no
1.6 Hz wobble in any flight. The vertical offset is +0.03 to +0.06 m against the estimate
in every flight. On the clean plumbing flight it holds 0.036 m p95 (v9 0.081 m). The one thing
that got worse is how close it comes to the wall in calm air and at 4 m/s: worst toward wall
0.09 and 0.08 m mean against v9's 0.02, all from seed 3 (0.17–0.19 m; seeds 1 and 2 match v9).
That is not overcorrection. Seed 3's error spikes are no sharper than v9's on the same seed
(module error rate p99 0.32–0.33 m/s against 0.33–0.37; peak 1.13–1.14 against 0.94–1.11 m), and
its setpoint std is lower. The difference is margin. v9 sat on seed 3 with a standing 0.15 m
offset away from the wall, which soaked up the estimate's 0.2 m error toward the wall at about
+45 s. v10 sits on the hold point (mean offset −0.04 m), so it follows that estimate error
0.17–0.19 m toward the wall, then recovers. Its closest clearance was 0.79 m, and it still comes
less than half as close as tuned PX4 (0.08–0.09 m against 0.21–0.24 m). The module side was
clean: every hold ran its full 60 s with no watchdog trip, fallback and yaw-guard shares 0, no
saturated action, stale time up to 2.26 s in the dropout seeds, handback 0.016–0.02 s,
GPS-free with vision fused, 0 wall contacts. **Verdict: v10 is better than v9 overall.** Its
mean p95 is lower in every wind (−16 / −24 / −40 %) and its vertical offset is slightly smaller
(+0.04 against +0.06 m). The wobble shows in 0 of 9 flights for both, and v10's setpoints
are calmer. Worst toward wall is better at 8 m/s (0.08 against 0.14 m) but worse in calm air
and at 4 m/s (0.09 and 0.08 against 0.02 m), because v10 holds the point rather than standing
off the wall.

### Engage guard confirmation (2026-10-23, `957760d`)

The question was whether the engage hold-off (`SH_RST_HOLD` 10 s, PR #50) changes the v10 hold
numbers. The image was rebuilt at `957760d` (`PolicyWeights.hpp` `NAME` structure_hold_v10,
`SH_RST_HOLD` default 10.0 in the built module). The benchmark was flown again exactly as
`20261023-092547`, giving batch `20261023-140712`:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v10 --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3`
(`vio_default`).

| p95 m, old `092547` → new `140712` | calm 0 m/s | 4 m/s | 8 m/s |
|---|---|---|---|
| seed 1 | 0.449 → no hold | 0.444 → no hold | 0.472 → no hold |
| seed 2 | 0.337 → no hold | 0.354 → no hold | 0.353 → no hold |
| seed 3 | 0.331 → 0.324 | 0.328 → 0.313 | 0.347 → 0.312 |
| seed 3 worst toward wall m | 0.188 → 0.185 | 0.171 → 0.172 | 0.163 → 0.185 |

**Seeds 1 and 2 never engaged (6 of 9 flights). This is the hold-off doing what it was built
for, but the test script does not allow for it.** `vio_default` fires vision jumps during and just after takeoff. On seed 1,
EKF2's last xy reset came at 25.5 s (log time) and the engage request at 29.0 s, 3.4 s
later. On seed 2 the reset came at 21.5 s and the engage at 28.5 s, 6.9 s later. Both
requests fall inside the 10 s hold-off, so the mode was refused. `sitl_test.py` retries only the
re-engage. The first engage waits 5 s for the mode and then the flight aborts with
`TimeoutError: timed out after 5 s waiting for Structure Hold in heartbeat`. `run_baseline`
scores these flights as crashes, but there was no wall contact: the vehicle never left PX4's stock modes.
In `092547`, v10 engaged at these same moments (same seed draws) and held normally.
On `vio_default`, seeds 1 and 2 cannot be compared like for like until the first engage
waits out the hold-off too. When it does, those holds start later and meet different faults.

**Seed 3 (the like-for-like cells):** p95 0.324 / 0.313 / 0.312 m against 0.331 / 0.328 /
0.347 m. That is within flight-to-flight noise and slightly lower in every wind. Worst toward wall is unchanged
in calm air and at 4 m/s, and +0.02 m at 8 m/s. There were 0 wall contacts inside the holds. None of the three
seed 3 flights pass `ok`, the same as in `092547`. In calm air and at 4 m/s, a PX4
failsafe (nav_state 5, Return) took the vehicle out of the mode 0.3 s before the
scripted disengage. At 8 m/s only `position_held` failed (the drift tolerance on the
estimate). **Re-engage:** ack 0 (accepted). It was refused 3 / 3 / 5 times, with a wait of
3.9 / 3.9 / 6.0 s, because EKF2 reset xy at 87.6–88.0 s, 10 s before the accepted
re-engage at 98.1–99.1 s. **`estimate_reset` (6)** never appears in `structure_hold_status`
in any flight. Every refusal came from the arming check, so commander never let a switch
through that the module then had to hand back. The trip reasons logged are 0 and 1 (the scripted test fault) only.

**vio_at_engage (one flight, wall1m_wind_8ms seed 3):**
`... --label structure_hold_v10_at_engage --scenarios wall1m_wind_8ms --seeds 3 -- --profile vio_at_engage`.
The first attempt (`20261023-141843`) did not anchor: `bridge.log` printed "faults keyed
to the start" and `verify_trace` reported `fault_anchor start`, `anchor_t` null. The cause was
that `run_gazebo` and `run_estimate` rebuilt the profile without `fault_anchor` when `--seed` was
given, and `run_baseline` always passes it. This is fixed in `367b780`
(`EstimateProfile.with_seed`, with a test). The flight was flown again as `20261023-142203`
(image unchanged; the fix is host-side). The bridge printed
`Structure Hold engaged at t=33.54 s, fault clock anchored at 25.0 s (step 1440)`, and
`verify_trace` reported `ok`, `fault_anchor engage`, `anchor_t 33.544`, `anchor_step 1440`,
with no jumps or dropouts before the engage. **Hold p95 0.241 m** (rms 0.151, worst toward wall 0.165,
min clearance 0.84 m, 0 contacts in the window). Do not compare this with the `vio_default` numbers: the fault timeline is different.
The flight did not complete. A 1.6 m vision jump came at 95.07 s bridge time (log
≈ 90.0 s), 1.3 s after the scripted disengage. EKF2 reset onto it at log 91.54 s, and stock
Position mode walked the vehicle toward the wall as the jump decayed. Truth
clearance fell to 0.33 m, and the x500 struck the wall at about log 96.8 s and fell (truth
altitude 2.6 → 0.2 m). This is the same mechanism as v11's seed 3 strike, here in PX4's own mode.
The hold-off refused the re-engage until 10.5 s after the reset. The re-engage was then accepted at
log 102.06 s on a vehicle already tumbling, and the tilt watchdog (4) handed back after 0.2 s.
The runner's 14 s retry window ran out and the flight was scored incomplete. The module did
what it should. The loss is stock Position mode flying on a jumped estimate between the two engages.

**Outcome.** `SH_RST_HOLD` ships off (default 0, `b355f61` in PR #50). Two reasons: the mock
replay of the seed 3 sequence shows the hold-off does not prevent the strike (v11 section
above), and on the benchmark it refused the first engage on two seeds of three, because
`vio_default` resets EKF2 within 10 s of the scripted engage. With the default at 0 the
released module flies the benchmark exactly as in `092547`; the seed 3 cells above are the
confirmation that the module change itself does not touch the hold (the hold-off was on
at 10 s in that batch and the hold numbers are within noise). The param stays for
operators who want the mode unavailable for a while after a reset.
`vio_at_engage` is the profile for matched comparisons, and it now has a known property:
the seed 3 jump that `vio_default` lands at about 92 s arrives 67 s after the engage, which
is 1.3 s after the scripted 60 s hold ends, so the stock phase between the two engages
meets it on every controller. Whether to shorten or move that window in the acceptance
sequence, or to score the hold and the post-hold sequence separately, is the Gate 1
thread's call; the hold scoring itself (first 60 s) is unaffected.

## Gate 2 evidence: v10 vs tuned PX4 on 8 seeds

Seeds 4–8 flown for both controllers on 2026-10-23 at `ad907ce` (same image, module headers
from `02c7c8e`, `NAME` structure_hold_v10), pooled with seeds 1–3 (tuned `20261022-171934`,
v10 `20261023-092547`):

```bash
python -m eval.baselines.run_baseline --label tuned_v1 --params eval/baselines/px4_tuned_v1.params --seeds 4 5 6 7 8
python -m eval.baselines.run_baseline --runner hold --label structure_hold_v10 --params eval/baselines/px4_tuned_v1.params --seeds 4 5 6 7 8
python -m eval.baselines.run_baseline --pool <seeds 1-3 batch.json> <seeds 4-8 batch.json> --out runs/pooled_<label>_seeds1-8.json
python -m eval.baselines.run_baseline --compare runs/pooled_tuned_v1_seeds1-8.json runs/pooled_structure_hold_v10_seeds1-8.json
```

**Seed count: n = 8 seeds per wind, 24 flights per controller, of which 7 seeds (21 flights
per controller) produced a hold.** Seed 7 failed before any hold for both controllers. Its
estimate fault schedule fires two vision jumps (0.80 m at 14 s, 1.17 m at 17.7 s) while the
vehicle is still on the ground arming. EKF2 then reports "heading estimate not stable".
Tuned PX4 was refused arming in calm air and at 8 m/s (scored as crashes), and at 4 m/s it
armed and flew away during PX4's own takeoff (34 m horizontal, failsafe, the 14.1 m p95
below). v10 armed after about 140 refusals in all three winds, then PX4's stock takeoff went
wrong the same way (truth 8–40 m off, 0.2–0.5 m up). The runner switched to Structure Hold,
the watchdog tripped on tilt and handed back to stock PX4 within 0.2 s, and the runner timed
out, so the flight is scored as a crash. Neither result measures the hold. The 7-seed view
below drops seed 7 for both controllers. The protocol itself is unchanged.

The 12 v10 flights of seeds 4, 5, 6 and 8 flew to the end, but the flight venv lacked
pymavlink, so `run_gazebo` stopped before scoring. They were scored afterwards from their
saved ulog, estimate and wind traces with the same functions (`check_ulog`, `verify`,
`score_hold`, `row_from_summary`). The flights were not re-flown.

### Pooled compare (8 seeds, as `--compare` prints it; means skip unscored flights)

| scenario | p95 m | worst toward wall m | rms m | crashes | failsafes |
|---|---|---|---|---|---|
| wall1m_calm_0ms | 0.477 (1.099) → 0.315 (0.541), -34 % | 0.200 (0.420) → 0.086 (0.188), -57 % | 0.222 (0.449) → 0.171 (0.246), -23 % | 1/8 → 1/8 | 2/8 → 0/8 |
| wall1m_wind_4ms | 2.187 (14.141) → 0.317 (0.554), -85 % | 0.171 (0.406) → 0.082 (0.171), -52 % | 1.154 (7.676) → 0.174 (0.255), -85 % | 0/8 → 1/8 | 3/8 → 0/8 |
| wall1m_wind_8ms | 0.480 (1.097) → 0.331 (0.579), -31 % | 0.193 (0.386) → 0.097 (0.192), -50 % | 0.223 (0.449) → 0.183 (0.275), -18 % | 1/8 → 1/8 | 2/8 → 1/8 |

Tuned PX4 → v10, mean over seeds, worst in brackets. All crashes are seed 7. The 4 m/s tuned
mean is inflated by seed 7's takeoff flyaway.

### Like for like (7 seeds, seed 7 dropped for both)

| scenario | p95 m, tuned → v10 | median p95 m | worst toward wall m | min clearance m, worst | crashes | failsafes (in window) | contacts |
|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.477 (1.099) → **0.315 (0.541), −34 %** | 0.352 → 0.331 | 0.200 (0.420) → 0.086 (0.188), −57 % | 0.536 → 0.796 | 0/7 → 0/7 | 2/7 (1) → 0/7 (0) | 0 → 0 |
| 4 m/s | 0.479 (1.098) → **0.317 (0.554), −34 %** | 0.348 → 0.328 | 0.196 (0.406) → 0.082 (0.171), −58 % | 0.549 → 0.821 | 0/7 → 0/7 | 2/7 (1) → 0/7 (0) | 0 → 0 |
| 8 m/s | 0.480 (1.097) → **0.331 (0.579), −31 %** | 0.344 → 0.347 | 0.193 (0.386) → 0.097 (0.192), −50 % | 0.571 → 0.794 | 0/7 → 0/7 | 2/7 (1) → 1/7 (0) | 0 → 0 |

v10's one failsafe (8 m/s seed 3) is in PX4's own auto-takeoff, before the module engaged
(see the v10 section above).

### Per-seed p95 m (`--pool` spread)

| tuned PX4 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | mean | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.911 | 0.438 | 0.352 | 0.162 | 0.213 | 1.099 | n/a | 0.168 | 0.477 | 1.099 |
| 4 m/s | 0.919 | 0.443 | 0.348 | 0.161 | 0.213 | 1.098 | 14.141 | 0.171 | 2.187 | 14.141 |
| 8 m/s | 0.929 | 0.435 | 0.344 | 0.161 | 0.211 | 1.097 | n/a | 0.180 | 0.480 | 1.097 |

| v10 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | mean | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.449 | 0.337 | 0.331 | 0.129 | 0.230 | 0.541 | n/a | 0.186 | 0.315 | 0.541 |
| 4 m/s | 0.444 | 0.354 | 0.328 | 0.126 | 0.230 | 0.554 | n/a | 0.185 | 0.317 | 0.554 |
| 8 m/s | 0.472 | 0.353 | 0.347 | 0.123 | 0.246 | 0.579 | n/a | 0.199 | 0.331 | 0.579 |

### Per-seed worst toward wall m

| tuned PX4 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 |
|---|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.284 | 0.015 | 0.420 | 0.085 | 0.111 | 0.384 | n/a | 0.102 |
| 4 m/s | 0.269 | 0.013 | 0.406 | 0.081 | 0.116 | 0.384 | 0.000 | 0.099 |
| 8 m/s | 0.246 | 0.011 | 0.386 | 0.081 | 0.128 | 0.382 | n/a | 0.114 |

| v10 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 |
|---|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.072 | 0.000 | 0.188 | 0.062 | 0.090 | 0.073 | n/a | 0.115 |
| 4 m/s | 0.053 | 0.000 | 0.171 | 0.052 | 0.085 | 0.081 | n/a | 0.134 |
| 8 m/s | 0.090 | 0.000 | 0.163 | 0.040 | 0.077 | 0.117 | n/a | 0.192 |

### v10 against tuned PX4 per seed (p95 change; worst toward wall change m)

| | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 8 |
|---|---|---|---|---|---|---|---|
| calm 0 m/s | −51 %; −0.212 | −23 %; −0.015 | −6 %; −0.232 | −20 %; −0.023 | **+8 %**; −0.021 | −51 %; −0.311 | **+11 %; +0.013** |
| 4 m/s | −52 %; −0.216 | −20 %; −0.013 | −6 %; −0.236 | −22 %; −0.029 | **+8 %**; −0.030 | −50 %; −0.303 | **+8 %; +0.035** |
| 8 m/s | −49 %; −0.156 | −19 %; −0.011 | **+1 %**; −0.222 | −23 %; −0.040 | **+17 %**; −0.051 | −47 %; −0.266 | **+10 %; +0.078** |

Module side, v10 seeds 4, 5, 6 and 8 (12 flights): every hold ran its full 60 s (engaged
stretch 60.6–60.7 s) with no watchdog trip and no handback in the hold. Fallback, yaw-guard
and saturated-action shares were 0, and handback latency was 0.02 s. Stale time was at most
1.84 s, in the dropout seeds 6 and 8. All 12 flights were GPS-free with vision fused, had 0
wall contacts and no failsafe. Closest clearance was 0.825 m (8 m/s seed 8). The only failed
check was the script's position_held drift tolerance on seed 6 (3 of 12), the same check
that failed on seeds 1 and 3. The test-fault trip after the hold handed back as designed.

**Reading it.** On the 7 seeds that produced a hold, v10 stays at least 30 % below tuned PX4
in every wind. Mean p95 is −34 / −34 / −31 %, so 8 m/s clears the bar by one point. Its worst
flight in any wind (0.58 m) is about half of tuned PX4's (1.10 m), worst toward wall halves
(0.10 against 0.19 m at 8 m/s), and the closest clearance rises from 0.54–0.57 m to
0.79–0.82 m. v10 has no crash, no failsafe in the hold and no contact on those seeds; tuned
PX4 has one failsafe inside the hover window per wind (seed 3). The spread is what limits
the claim. The mean gain comes from tuned PX4's three bad seeds (1, 6 and 3 for worst toward
wall), where v10 cuts p95 by about half. On tuned PX4's good seeds v10 is level or slightly
worse: seed 5 by +8 / +8 / +17 %, seed 8 by +11 / +8 / +10 % (worst toward wall up to
+0.08 m at 8 m/s), and 8 m/s seed 3 by +1 %. The median p95 is a tie at 8 m/s (0.344 against
0.347 m) and 0.02 m better in calm air and at 4 m/s. The per-seed numbers barely move with
wind for either controller. The estimate's fault draw per seed sets the error far more than
the gust level does in this setup. **Verdict: at 8 m/s on 7 valid seeds (8 flown), v10's mean
p95 is 31 % below tuned PX4's (0.331 against 0.480 m), just over the 30 % bar.** The gain is
in robustness to bad seeds (worst flight −47 %, worst toward wall −50 %). On a typical seed
the two are level (median −0 %). Seed 7 is a protocol problem, not a controller result:
faults injected before takeoff break both runs, so the fault schedule should start after
takeoff before another seed batch is flown.

**Next seed batch: fly the `vio_after_takeoff` profile.** It is `vio_default` with jumps and
dropouts held back for the first 24 s of the bridge's clock (`fault_start_s` in
`envs/estimate`), so no fault lands while PX4 arms and takes off. Every random draw is kept,
so each fault from 24 s on is the same, step for step, as `vio_default` with the same seed.
In the PR #42 and #44 flights the hold window opened 24.7 to 25.5 s after the bridge's first
sample, so the faults inside the hold are unchanged and new seeds stay comparable with seeds
1–8; only the pre-takeoff faults (seed 7's) are gone. Pass it after `--` for both
controllers, for example
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v10 --params eval/baselines/px4_tuned_v1.params --seeds 9 10 -- --profile vio_after_takeoff`.
`log_lines.md` then names the profile in each row.

## Structure Hold v11 vs tuned PX4

Same benchmark, on policy v11 (v10's recipe trained longer; 18 % tighter than v10 on the
Gazebo-like check), module headers at `3f7f1d9`, image rebuilt, 2026-10-23:
`python -m eval.baselines.run_baseline --runner hold --label structure_hold_v11 --params eval/baselines/px4_tuned_v1.params`
(wall1m_*, seeds 1–3, `vio_default`, as for v10). Compared against tuned PX4 `20261022-171934`
and v10 `20261023-092547`. Mean over seeds 1–3, worst in brackets.

| scenario | p95 hold error m, tuned v1 / v10 / **v11** | worst toward wall m, tuned v1 / v10 / **v11** | rms m, tuned v1 / v10 / **v11** | crashes, tuned / v10 / v11 | failsafes, tuned / v10 / v11 |
|---|---|---|---|---|---|
| calm 0 m/s | 0.567 (0.911) / 0.372 (0.449) / **0.328 (0.359)** | 0.240 (0.420) / 0.087 (0.188) / **0.078 (0.170)** | 0.238 / 0.192 / **0.167** | 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 |
| 4 m/s | 0.570 (0.919) / 0.375 (0.444) / **0.324 (0.351)** | 0.230 (0.406) / 0.075 (0.171) / **0.079 (0.166)** | 0.238 / 0.197 / **0.167** | 0/3 / 0/3 / 0/3 | 2/3 / 0/3 / 0/3 |
| 8 m/s | 0.570 (0.929) / 0.391 (0.472) / **0.333 (0.366)** | 0.214 (0.386) / 0.084 (0.163) / **0.126 (0.188)** | 0.239 / 0.209 / **0.171** | 0/3 / 1/3 / 0/3 | 2/3 / 0/3 / 0/3 |

Change in mean p95 against tuned PX4: −42 / −43 / −42 %; against v10: −12 / −14 / −15 %. RMS
against v10: −13 / −16 / −18 %. Worst toward wall against v10: −10 / +6 / +50 %. The crash
and failsafe columns score the 60 s hold window only. They miss what happened after it (below).

| per seed | p95 m, calm / 4 / 8 m/s | v10 p95 m | worst toward wall m | offset up, module / truth m | roll-rate setpoint peak, std |
|---|---|---|---|---|---|
| seed 1 | 0.321 / 0.329 / 0.366 | 0.449 / 0.444 / 0.472 | 0.061 / 0.068 / 0.128 | −0.00 / +0.15, +0.01 / +0.16, +0.02 / +0.17 | 7.9, 7.8, **1.7** Hz; 0.18, 0.18, 0.19 rad/s |
| seed 2 | 0.359 / 0.351 / 0.346 | 0.337 / 0.354 / 0.353 | 0.004 / 0.004 / 0.062 | −0.01 / −0.06, −0.01 / −0.06, −0.00 / −0.05 | 7.9, 7.8, 7.8 Hz; 0.13, 0.13, 0.13 rad/s |
| seed 3 | 0.305 / 0.292 / 0.286 | 0.331 / 0.328 / 0.347 | 0.170 / 0.166 / 0.188 | −0.01 / +0.01, −0.01 / +0.01, +0.01 / +0.03 | 7.8, 7.8, **1.8** Hz; 0.13, 0.14, 0.14 rad/s |

| | v10 | v11 |
|---|---|---|
| clean calm flight: p95, offset up, horizontal | 0.036 m, +0.03 m, 0.014 m | 0.017 m, −0.02 m, 0.003 m |
| clean calm flight: along-wall drift (truth) mean, p2p | −0.008 m, 0.03 m | −0.003 m, 0.02 m |
| clean calm flight: roll / pitch rate setpoint std, peak | 0.19 / 0.14 rad/s, 10.8 Hz | 0.13 / 0.17 rad/s, 7.8 Hz |
| clean calm flight: thrust | 0.723 ± 0.004 | 0.725 ± 0.002 |
| batch: offset up (module), mean (range) | +0.04 m (+0.03 to +0.06) | −0.00 m (−0.01 to +0.02) |
| batch: offset up (truth), mean (range) | +0.09 m (−0.01 to +0.21) | +0.04 m (−0.06 to +0.17) |
| batch: horizontal offset, mean | 0.03 m | 0.04 m |
| batch: roll-rate setpoint std, mean | 0.21 rad/s | 0.15 rad/s |
| batch: measured roll-rate std, mean | 0.08 rad/s | 0.10 rad/s |
| batch: 1.6 Hz wobble (detector: roll-setpoint peak 1.3–2 Hz) | 0/9 flights | 2/9 flights (8 m/s seeds 1 and 3) |
| batch: 1.3–2 Hz share of roll-setpoint power, seed 1 / 2 / 3 mean | 8 / 1 / 1 % | 40 / 5 / 13 % |
| batch: along-wall drift (truth) p2p, mean | 0.33 m | 0.24 m |
| batch: mean truth offset from the wall, seeds 1 / 2 / 3 | −0.01 / −0.12 / −0.04 m | −0.00 / −0.09 / −0.01 m |
| after the hold: loss of control at the scripted re-engage | 0 of 9 (0 of 21 over seeds 1–8, seed 7's failed takeoffs aside) | **3 of 3 seed 3 flights, and again on a repeat of 4 m/s seed 3** |

Vision delay: EKF2 fuses each vision sample 154 ms behind real time (157 ms on 8 m/s seed 3)
in all ten flights, against `EKF2_EV_DELAY` 100 ms, the same as v9 and v10.

**The re-engage failures.** After its 60 s hold, the runner hands back to Position mode for
5 s, re-engages Structure Hold for about 3 s, then fires the module's test fault to time the
handback. Every v11 seed 3 flight lost control in that re-engage window, 0.5–2.3 s after
re-engaging. A repeat of 4 m/s seed 3 (`20261023-132221`) did it again. The IMU shows a yaw spin of
19–25 rad/s and roll/pitch rates of 6–13 rad/s (the hold itself never exceeds 0.7 rad/s yaw),
and the truth altitude drops 0.7–2.5 m:

| flight | spin starts | gyro yaw peak | lowest truth altitude | how it ended |
|---|---|---|---|---|
| calm seed 3 | 97.2 s, 0.15 s after the test trip handed back | 21.9 rad/s | 1.88 m | stock PX4 recovered |
| 4 m/s seed 3 | 95.5 s, 0.5 s after re-engage | 24.6 rad/s | 0 m (ground) | watchdog `pos_error` trip at 96.3 s, 0.6 m up; hit the ground at about 5.7 m/s, landed and disarmed |
| 8 m/s seed 3 | 96.1 s, 2.1 s after re-engage | 19.3 rad/s | 0.43 m | the scripted test trip handed back at 97.0 s |
| 4 m/s seed 3, repeat | 96.3 s, 2.3 s after re-engage | 22.1 rad/s | 1.67 m | the scripted test trip handed back at 97.0 s |

In the 4 m/s flight the true body rate jumps from 0.2 to 4.5 rad/s yaw in one 20 ms sample,
then to 11 rad/s, while the policy's last action was benign (0.09, 0.03, 0.20, 0.24). The
motors cannot produce that yaw acceleration, so a simulator impulse probably starts it. The
wind force is under 0.6 N all flight. What v11 does next is the problem. Its actions
saturate and flip sign every tick, with collective thrust at its 0.08 floor for 0.6 s. Tilt
stays under 22°, so the tilt trip never fires, and the position trip comes only 0.6 m above
the ground. v10 never spins: gyro yaw stays at 0.45 rad/s or less in the hold and the
re-engage of all 9 benchmark flights and all 12 seed 4–8 flights that produced a hold. The
only v10 spin is seed 7's failed stock takeoff. Seed 3 is the seed that pushes the estimate
0.2 m toward the wall at about +45 s, but why its re-engage window triggers this for v11
alone is not found. The ground-truth angular velocity also shows 300–380 rad/s spikes in
these flights. Those are wrap artefacts (2π over one sample as yaw crosses ±180°), not rates,
so the gyro is used above.

**Reading it.** Inside the 60 s hold, v11 is the best policy in Gazebo so far. Mean p95 is
0.33 / 0.32 / 0.33 m against tuned PX4's 0.57 m (−42 / −43 / −42 %) and v10's 0.37 / 0.38 /
0.39 m (−12 / −14 / −15 %). Its worst flight (0.366 m) is below v10's mean. It is lower than
v10 on every seed 1 and seed 3 flight, and within 0.02 m of it on seed 2. The vertical offset that every
policy since v6 carried is gone (module −0.00 m mean, v10 +0.04 m). The clean plumbing flight
holds 0.017 m p95 (v10 0.036 m), and rate setpoints are calmer (std 0.15 against 0.21 rad/s).
Two things are worse. First, it rings at 1.7–1.8 Hz. The detector flags 2 of 9 flights, and
the band holds 36–46 % of roll-setpoint power on seed 1. On seed 1 and on 8 m/s seed 2 that
ringing comes in bursts of about 0.4 rad/s after the 1.4 m vision jump and the 2 s dropouts,
and dies out in about 5 s. On 8 m/s seed 3 it is a sustained 0.06–0.18 rad/s through the whole hold, with no estimate event behind it
(v10 0.02–0.03). The measured roll rate is 27 % higher than v10's (std 0.10 against 0.08 rad/s). Second, wind into the wall
at 8 m/s comes closer: worst toward wall 0.126 m mean against 0.084, though the closest
clearance (0.82 m) is still better than v10's 0.79 m. The module side was clean inside every
hold: full 60 s, no watchdog trip, fallback and yaw-guard shares 0, saturated-action share at
most 0.05 %, stale time up to 2.30 s in the dropout seeds, handback 0.016–0.024 s, GPS-free
with vision fused, 0 wall contacts. Outside the hold it was not clean: the repeatable loss of
control at the re-engage on seed 3, with one ground impact, which v10 never shows.
**Verdict: v10 returns.** v11's p95 is better in every wind, but it is worse on safety. It
lost control after re-engaging in 4 of 4 seed 3 flights, against 0 for v10, and it shows a
1.7–1.8 Hz oscillation that v10 does not. The module headers are regenerated from
`training/policies/structure_hold_v10.onnx` (`PolicyWeights.hpp` is byte-identical to the v10
header; `PolicyTestVectors.hpp` restored from the v10 commit, since `make_test_vectors` needs
torch, which the Mac lacks). The module README points at v10 again. v11 stays in
`training/policies/` for the training thread. It should come back only with a fix for the
saturated, sign-flipping response to a large body-rate disturbance, and a seed 3 re-engage that
holds.

**Correction (2026-10-23 13:50, from the training thread's trace of all four ulogs).** The
seed 3 loss of control is a prop strike on the wall, not a simulator impulse, and it is set
up by where the re-engage captured its hold point. In each flight the 1.4 m vision jump
toward the wall lands at 91.8 to 92.5 s and EKF2 resets onto it (+1.12 to +1.15 m xy reset)
at 92.8 to 93.6 s, 0.1 to 0.4 s before the scripted disengage. Position mode shifts its
setpoint by the reset, so it holds the jumped estimate, and as the jump decays it walks the
drone from 1.16 m to 0.55 m off the wall. The re-engage then captures E +0.03 to +0.10, a
point on the wall face, the drone keeps closing, and the spin starts at 0.36 to 0.40 m
centre distance, which is the x500's prop reach (in the calm flight after the test-fault
handback, with stock PX4 in charge). v10's seed 3 flight had luckier timing: jump at 91.27 s,
reset at 92.33 s, 1.3 s before the disengage, re-engage at E −0.42, never closer than
0.54 m. Any controller that holds that captured point would hit the wall; the policy only
sees estimate minus hold point, so a retrain cannot fix it. What is genuinely v11's is the
1.7 to 1.8 Hz ring and the standing yaw error. The post-strike response has no v10
comparison because v10 never struck the wall; in the mock a 5 rad/s yaw kick crashes both.
PR #50 followed with two changes. The bridge profile `vio_at_engage` keys the fault
schedule to the engage, so every policy meets the same faults at the same time after
engage and none before takeoff (`sim2real/estimate_bridge/README.md`); new comparisons
use it for both controllers. The module gained an engage hold-off after an EKF xy reset
(`SH_RST_HOLD`, `sim2real/px4/README.md`), but it ships off (default 0): the training
thread replayed the seed 3 sequence in the mock (64 episodes; hold, stock Position mode
with the reset-shifted setpoint, re-engage, 1.4 m jump toward the wall 0.5 to 3 s before
the switch, contact at 0.40 m) and found v10 crashes 91 %, v11 94 %, a stock-style PD
62 %, and v10 with a 10 s hold-off still 92 %, most of them in the stock phase: whatever
captures the settling estimate walks into the wall as the jump decays, and the policies
answer a sudden offset slowly by training (v10 moves 0.25 m of the 1.1 m in 1.7 s). There
is no module-only fix with the current inputs; a safe standoff needs a direct distance to
the structure (a rangefinder with PX4 Collision Prevention, or the camera's depth to the
wall), logged in `docs/real-flight-gap.md`. The verdict stands: v10 stays, v11 comes back
with the ring and the yaw error fixed, and seed 3's post-hold sequence is compared only on
a matched timeline.

## Structure Hold v12 vs v10 vs tuned PX4 (matched fault timeline)

The first comparison on a fault schedule shared by every controller. Flown 2026-10-23 on
the Mac (Docker arm64, image rebuilt for each policy), wall1m_*, seeds 1–3, tuned_v1 params
under both policies and the stock controller:

- **A, v12**, batch `20261023-154026_px4_baseline_structure_hold_v12_d701b87`, image with
  `NAME structure_hold_v12` (checked in the built `px4` binary), `SH_RST_HOLD` default 0:
  `python -m eval.baselines.run_baseline --runner hold --label structure_hold_v12 --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3 -- --profile vio_at_engage`
- **B, v10**, batch `20261023-160114_px4_baseline_structure_hold_v10_fe4667c`: the same
  command with `--label structure_hold_v10`, on an image rebuilt with the v10 headers from
  `main` (`NAME structure_hold_v10` in the binary). `fe4667c` is the scoring fix below before
  it was rebased onto `df68ebc` (`a5d7714`); the flown code is the same.
- **C, tuned PX4**, batch `20261023-162007_px4_baseline_tuned_v1_at_hover_a5d7714`:
  `python -m eval.baselines.run_baseline --label tuned_v1_at_hover --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3 -- --profile vio_at_hover`.
  `vio_at_engage` cannot serve the stock runner: its fault clock starts at the first
  Structure Hold engage, which stock PX4 never makes, so the flight would see no jump or
  dropout at all. `vio_at_hover` (`df68ebc`) starts the same schedule at 25 s when the
  vehicle first reaches the hover altitude minus 0.3 m (2.20 m here), which is at the start
  of the scored hover window, a few seconds earlier in the flight than the policies' engage.

All 27 flights: the bridge log prints the anchor line ("fault clock anchored at 25.0 s";
for C, "height 2.20 m reached at t=29.7–30.1 s"), `estimate_verify.json` ok (fault anchor
`engage` for A and B, `altitude` for C) and `wind_verify.json` ok.

**Scoring fix (`a5d7714`).** The first tabulation of batch A scored all nine flights as
crashes with no contact and 0.69 m or more of clearance: `row_from_summary` looked for the
scored window only in `metrics.window_s`, where the stock runner's summary keeps it, while
`hold_summary.json` keeps it in `hold.window_s`, so every hold-runner flight had no window
and failed the full-hold test added in `2959ebf`. Batch A was re-tabulated from its run
folders with the fix (the old files are kept as `*.misscored.*`); B and C were scored
with it.

Mean over seeds, worst in brackets:

| scenario | p95 m, tuned / v10 / **v12** | worst toward wall m, tuned / v10 / **v12** | rms m, tuned / v10 / **v12** | crashes | post-hold fail |
|---|---|---|---|---|---|
| calm 0 m/s | 0.522 (0.913) / 0.329 (0.396) / **0.345 (0.406)** | 0.229 (0.417) / 0.075 (0.184) / **0.149 (0.272)** | 0.221 / 0.173 / **0.167** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |
| 4 m/s | 0.523 (0.914) / 0.327 (0.410) / **0.347 (0.423)** | 0.220 (0.405) / 0.069 (0.172) / **0.152 (0.283)** | 0.221 / 0.176 / **0.167** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |
| 8 m/s | 0.524 (0.918) / 0.345 (0.475) / **0.353 (0.442)** | 0.218 (0.387) / 0.080 (0.166) / **0.177 (0.312)** | 0.221 / 0.185 / **0.164** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |

Against tuned PX4, mean p95: v12 −34 / −34 / −33 %, v10 −37 / −38 / −34 %. v12 against v10:
p95 +5 / +6 / +2 %, rms −3 / −5 / −11 %, worst toward wall +98 / +122 / +121 %.

Per seed (p95 / rms / worst toward wall / min clearance, m; no flight had a contact):

| cell | tuned PX4 | v10 | **v12** |
|---|---|---|---|
| calm, seed 1 | 0.913 / 0.364 / 0.262 / 0.881 | 0.396 / 0.217 / 0.042 / 0.966 | **0.406 / 0.182 / 0.272 / 0.736** |
| calm, seed 2 | 0.301 / 0.140 / 0.008 / 0.974 | 0.307 / 0.152 / 0.000 / 0.999 | **0.334 / 0.179 / 0.000 / 1.025** |
| calm, seed 3 | 0.353 / 0.159 / 0.417 / 0.571 | 0.282 / 0.149 / 0.184 / 0.805 | **0.294 / 0.141 / 0.174 / 0.819** |
| 4 m/s, seed 1 | 0.914 / 0.366 / 0.247 / 0.851 | 0.410 / 0.221 / 0.034 / 0.972 | **0.423 / 0.189 / 0.283 / 0.726** |
| 4 m/s, seed 2 | 0.302 / 0.139 / 0.008 / 0.973 | 0.313 / 0.156 / 0.000 / 1.003 | **0.328 / 0.171 / 0.000 / 1.009** |
| 4 m/s, seed 3 | 0.353 / 0.158 / 0.405 / 0.583 | 0.257 / 0.152 / 0.172 / 0.822 | **0.289 / 0.140 / 0.175 / 0.824** |
| 8 m/s, seed 1 | 0.918 / 0.365 / 0.259 / 0.885 | 0.475 / 0.242 / 0.074 / 0.931 | **0.442 / 0.196 / 0.312 / 0.691** |
| 8 m/s, seed 2 | 0.305 / 0.139 / 0.007 / 0.973 | 0.319 / 0.160 / 0.000 / 1.002 | **0.323 / 0.162 / 0.010 / 0.996** |
| 8 m/s, seed 3 | 0.350 / 0.159 / 0.387 / 0.602 | 0.242 / 0.153 / 0.166 / 0.838 | **0.293 / 0.134 / 0.208 / 0.801** |

Offsets and ring, from the ulogs over the scored window (A and B). Offset up is the mean z
minus the hold point's z; the hold point is captured on the EKF estimate, so the estimate's
offset is the controller's own and the truth's adds the estimate's bias for the seed. Ring
is the 1.2–2.2 Hz rms of the measured roll rate (`vehicle_angular_velocity`, the band of
`eval/gazebo_check.band_rms`):

| | v10 | v12 |
|---|---|---|
| offset up on the estimate, mean (range) | +0.040 m (+0.030 to +0.050) | −0.024 m (−0.037 to +0.001) |
| offset up on the truth, mean (range) | +0.066 m (−0.033 to +0.180) | +0.001 m (−0.111 to +0.134) |
| roll-rate 1.2–2.2 Hz rms, mean (range) | 0.035 rad/s (0.018 to 0.061) | 0.030 rad/s (0.012 to 0.054), lower in 7 of 9 |
| roll-rate std, mean | 0.077 rad/s | 0.050 rad/s, lower in 9 of 9 |
| roll-rate setpoint 1.2–2.2 Hz rms, mean | 0.032 rad/s | 0.029 rad/s |

v12 sits below the hold point and v10 above it, as the Gazebo-like check said (lift −0.03 /
+0.04 m), and v12 rings less, the thing it was trained for. After the hold, every flight
of every controller ran the rest of the sequence: `post_hold_fail` 0 of 9 for both
policies, seed 3 included (on this timeline seed 3's big jump lands after the sequence's
re-engage, not between the disengage and the re-engage as on `vio_default`). Wind barely
moves any controller's numbers on these seeds; the estimate's faults set them.

Two "failsafe in hover" flags in the tables are boundary artefacts, not failsafes
(`failsafe_fraction` 0 in both): v10 8 m/s seed 1 has the commanded disengage to Position at
89.08 s, exactly the window's end; tuned 8 m/s seed 3 (and its calm and 4 m/s twins) has the
commanded Land 1.1 s before the window's end, just outside the 1.0 s allowance.

Tuned PX4's mean is set by seed 1 (0.91 m p95 in every wind); on seed 2 it holds tighter
than either policy (0.30 against 0.31–0.33 m), and on seed 3 it comes closest to the wall
(0.57 m clearance).

**Verdict proposal: v12 is not better than v10 on the hold p95.** v10's p95 is lower in 8 of
9 cells (v12 wins only 8 m/s seed 1) and 2–6 % lower on the mean; three seeds cannot tell
a difference that size from noise, but they do not show v12 ahead. v12 is better on rms,
on the ring and on the vertical offset, and worse on worst toward the wall (seed 1: 0.27–
0.31 m against 0.03–0.07 m). Both beat tuned PX4 by a third on p95 with no crash. The
module stays on v10; the local SITL image was left on v10 to match `main`.

**Decision (PR #52): v10 stays the released policy.** The release rule for a candidate is
"better than v10 on the hold p95, no ring, no contact"; v12 misses the first part, so the
module headers went back to v10 before merge and only the results, the scoring fix and
`vio_at_hover` merged. What v12 did show: the 1.7 Hz ring rule works (ring 0.030 vs 0.035
rad/s, roll-rate std 0.050 vs 0.077), the vertical offset is as predicted (−0.024 m on the
estimate against v10's +0.040 m), and the matched timeline holds up (27 flights, all
anchored, no crash, no post-hold failure). The thing to fix next is worst toward the wall:
v12 lets the hold drift 0.27 to 0.31 m toward the wall on seed 1 where v10 drifts 0.03 to
0.07 m, which is also where its p95 loses. First matched tuned-PX4 numbers:
p95 0.52 m mean, 0.91 m worst (seed 1), against v10 0.33 / 0.40 and v12 0.35 / 0.41.

## Structure Hold v18 vs v10 vs tuned PX4 (matched fault timeline)

The Gazebo re-fly of v18, the candidate that passed every rule of the Gazebo-like check
(PR #57, p95 ratio 0.78 against v10). Flown 2026-10-24 on the Mac (Docker arm64), same
scenarios, seeds, params and profile as the v12 comparison above, against the same v10
and tuned PX4 batches (nothing in `sim2real/`, the bridge, the image recipe or
`eval/baselines` changed since, so neither was re-flown):

- **v18**, batch `20261024-005145_px4_baseline_structure_hold_v18_49e4272`, image rebuilt
  on `49e4272` with `NAME structure_hold_v18` (checked in the built `px4` binary),
  `SH_RST_HOLD` default 0:
  `python -m eval.baselines.run_baseline --runner hold --label structure_hold_v18 --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3 -- --profile vio_at_engage`
- **v10**, batch `20261023-160114_px4_baseline_structure_hold_v10_fe4667c` (PR #52).
- **tuned PX4**, batch `20261023-162007_px4_baseline_tuned_v1_at_hover_a5d7714` (`vio_at_hover`).

All 9 v18 flights: bridge log "fault clock anchored at 25.0 s", `estimate_verify.json` and
`wind_verify.json` ok, 0 contacts in the scored window. (A first attempt at the batch was
stopped after two flights: it ran under a Python without `pymavlink`, so `run_gazebo`'s
post-flight scoring crashed and no `hold_summary.json` was written. Those folders were
set aside, not scored.)

Mean over seeds, worst in brackets:

| scenario | p95 m, tuned / v10 / **v18** | worst toward wall m, tuned / v10 / **v18** | rms m, tuned / v10 / **v18** | crashes | post-hold fail |
|---|---|---|---|---|---|
| calm 0 m/s | 0.522 (0.913) / 0.329 (0.396) / **0.288 (0.344)** | 0.229 (0.417) / 0.075 (0.184) / **0.077 (0.207)** | 0.221 / 0.173 / **0.157** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |
| 4 m/s | 0.523 (0.914) / 0.327 (0.410) / **0.296 (0.343)** | 0.220 (0.405) / 0.069 (0.172) / **0.091 (0.222)** | 0.221 / 0.176 / **0.156** | 0 / 0 / 0 of 3 | 0 / 0 / **1** of 3 |
| 8 m/s | 0.524 (0.918) / 0.345 (0.475) / **0.304 (0.335)** | 0.218 (0.387) / 0.080 (0.166) / **0.128 (0.245)** | 0.221 / 0.185 / **0.160** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |

Against tuned PX4, mean p95: v18 −45 / −44 / −42 %. v18 against v10: p95 −12 / −9 / −12 %,
rms −9 / −12 / −13 %, worst toward wall +3 / +33 / +60 %.

Per seed (p95 / rms / worst toward wall / min clearance, m; no flight had a contact in the
scored window):

| cell | tuned PX4 | v10 | **v18** |
|---|---|---|---|
| calm, seed 1 | 0.913 / 0.364 / 0.262 / 0.881 | 0.396 / 0.217 / 0.042 / 0.966 | **0.235 / 0.151 / 0.025 / 0.981** |
| calm, seed 2 | 0.301 / 0.140 / 0.008 / 0.974 | 0.307 / 0.152 / 0.000 / 0.999 | **0.344 / 0.180 / 0.000 / 1.030** |
| calm, seed 3 | 0.353 / 0.159 / 0.417 / 0.571 | 0.282 / 0.149 / 0.184 / 0.805 | **0.285 / 0.140 / 0.207 / 0.797** |
| 4 m/s, seed 1 | 0.914 / 0.366 / 0.247 / 0.851 | 0.410 / 0.221 / 0.034 / 0.972 | **0.258 / 0.154 / 0.051 / 0.957** |
| 4 m/s, seed 2 | 0.302 / 0.139 / 0.008 / 0.973 | 0.313 / 0.156 / 0.000 / 1.003 | **0.343 / 0.176 / 0.000 / 1.021** |
| 4 m/s, seed 3 | 0.353 / 0.158 / 0.405 / 0.583 | 0.257 / 0.152 / 0.172 / 0.822 | **0.286 / 0.137 / 0.222 / 0.779** |
| 8 m/s, seed 1 | 0.918 / 0.365 / 0.259 / 0.885 | 0.475 / 0.242 / 0.074 / 0.931 | **0.276 / 0.162 / 0.138 / 0.871** |
| 8 m/s, seed 2 | 0.305 / 0.139 / 0.007 / 0.973 | 0.319 / 0.160 / 0.000 / 1.002 | **0.335 / 0.174 / 0.000 / 1.020** |
| 8 m/s, seed 3 | 0.350 / 0.159 / 0.387 / 0.602 | 0.242 / 0.153 / 0.166 / 0.838 | **0.302 / 0.144 / 0.245 / 0.766** |

The mean p95 win comes from seed 1 alone (−41 / −37 / −42 %); v18's p95 is higher than
v10's on seed 2 (+12 / +10 / +5 %) and seed 3 (+1 / +11 / +25 %): lower in 3 of 9 cells.

**Wall drift.** Seed 1, where v12 drifted 0.27–0.31 m toward the wall: v18 0.025 / 0.051 /
0.138 m against v10's 0.042 / 0.034 / 0.074 m (mean 0.071 against 0.050; the check
predicted 0.06 against 0.08). The v12 problem is gone, but v18 is not better than v10 on
the wall: worse at 4 and 8 m/s on seed 1 and in every seed 3 cell (0.21–0.25 against
0.17–0.18 m).

**Post-hold failure (4 m/s, seed 3).** The scored hold was clean (0.779 m min clearance).
At the sequence's re-engage (95.0 s) seed 3's estimate was 0.9–1.1 m off toward the wall;
in the 3 s of re-engage the truth moved from 0.78 to 0.59 m clearance, the scheduled
`test_fault` trip handed back to Position at 98.0 s, which held the faulty estimate while
the truth crept to 0.35 m, and the commanded Land at 102.5 s struck the wall about 1.5 s into the
descent (lateral velocity −1.5 then +1.8 m/s at 1.7 m height); the vehicle fell, tumbled
along the ground and never registered touchdown (`TimeoutError ... waiting for touchdown`).
The strike happened under stock PX4, outside the scored window, so `contacts` is 0; v10 and
v12 came as close on seed 3 (0.11–0.29 m truth clearance after 88 s) and landed.

Offsets and ring, from the ulogs over the scored window, as for v12 (offset up = mean z
minus the hold point's z; ring = 1.2–2.2 Hz rms of the measured roll rate):

| | v10 | v18 |
|---|---|---|
| offset up on the estimate, mean (range) | +0.040 m (+0.030 to +0.050) | −0.021 m (−0.032 to −0.010) |
| offset up on the truth, mean (range) | +0.066 m (−0.033 to +0.180) | +0.003 m (−0.094 to +0.116) |
| roll-rate 1.2–2.2 Hz rms, mean (range) | 0.035 rad/s (0.018 to 0.061) | 0.010 rad/s (0.007 to 0.016), lower in 9 of 9 |
| roll-rate std, mean | 0.077 rad/s | **0.147 rad/s, higher in 9 of 9** |
| roll-rate setpoint 1.2–2.2 Hz rms, mean | 0.032 rad/s | 0.011 rad/s |

**v18 rings at 3.4 Hz.** The 1.2–2.2 Hz band says v18 rings less, but its roll-rate std is
twice v10's. The spectrum over the scored window puts the energy just above the band:

| band, roll-rate rms rad/s (mean of 9) | 0–0.5 | 0.5–1.2 | 1.2–2.2 | 2.2–5 | 5–10 | 10–25 Hz |
|---|---|---|---|---|---|---|
| v10 | 0.016 | 0.035 | 0.035 | 0.015 | 0.007 | 0.051 |
| v18 | 0.024 | 0.019 | 0.010 | **0.130** | 0.056 | 0.019 |

The roll-rate spectrum peaks at 3.3–3.4 Hz in all 9 v18 flights (v10's peak is the 10.8 Hz
rate-loop line, or ~1 Hz). It is commanded: the 2.2–5 Hz rms of the rate setpoint is
0.118 rad/s for roll and 0.113 for pitch (peak ~4.1 Hz), against v10's 0.020 / 0.023 and
v12's 0.023 / 0.036. The ring rule moved v18's oscillation out of the band it penalises
rather than removing it; the Gazebo-like check's `wobble` (0 % for v18) does not see it.

**Verdict proposal: v18 does not pass the release rule; v10 stays.** v18 beats v10 on the
mean hold p95 (−9 to −12 %) and rms, and both policies beat tuned PX4 by more than a third,
but (1) the ring regressed: a 3.4 Hz commanded oscillation, 5–9× v10's in that band (measured and commanded, roll and pitch) and
roll-rate std 2× v10's in every flight; (2) wall drift is not better than v10's (seed 1 at
4 and 8 m/s, all of seed 3); (3) one post-hold failure (4 m/s seed 3, a wall strike under
stock Land after the re-engage), against none for v10; and the p95 win is seed 1 only.
The local SITL image was rebuilt on v10 (headers from `main`) after this batch. For the
next candidate: widen the ring metric (in the check and in training) to the whole
1–6 Hz band, or penalise action rate, before another re-fly.

**Decision (PR #58): v10 stays the released policy.** The module headers went back to v10
before merge; only the results merged. v18 is the closest candidate yet (first mean p95
win on the matched timeline, −9 to −12 %, and v12's seed 1 drift is gone), but a win on
one seed of three with a new 3.4 Hz oscillation and a post-hold strike is not a release.
Two lessons for the check: the 1.2–2.2 Hz ring band is too narrow (v18 moved the ring to
3.4 Hz and scored 0 % wobble), and a candidate must beat v10 on the hold p95 per seed,
not only on the mean.

## Structure Hold v19 vs v10 vs tuned PX4 (matched fault timeline)

The Gazebo re-fly of v19 (PR #60: v18 plus a 1–6 Hz penalty on the rate commands; the first
candidate to pass every rule of the extended Gazebo-like check, the new `fast_ring_not_higher`
rule and the seed 2 / seed 3 replay cells included, p95 ratio 0.72 against v10). Flown
2026-10-24 on the Mac (Docker arm64), same scenarios, seeds, params and profile as the v12
and v18 comparisons above, against the same v10 and tuned PX4 batches (nothing in
`sim2real/`, the bridge, the image recipe or `eval/baselines` changed since):

- **v19**, batch `20261024-033047_px4_baseline_structure_hold_v19_752012c`, image rebuilt
  on `752012c` with `NAME structure_hold_v19` (checked in the built `px4` binary),
  `SH_RST_HOLD` default 0:
  `python -m eval.baselines.run_baseline --runner hold --label structure_hold_v19 --params eval/baselines/px4_tuned_v1.params --seeds 1 2 3 -- --profile vio_at_engage`
- **v10**, batch `20261023-160114_px4_baseline_structure_hold_v10_fe4667c` (PR #52).
- **v18**, batch `20261024-005145_px4_baseline_structure_hold_v18_49e4272` (PR #58), for the ring.
- **tuned PX4**, batch `20261023-162007_px4_baseline_tuned_v1_at_hover_a5d7714` (`vio_at_hover`).

All 9 v19 flights: bridge log "fault clock anchored at 25.0 s", `estimate_verify.json` and
`wind_verify.json` ok, 0 contacts in the scored window, no failsafe, `post_hold_fail` no in
all 9 (including 4 m/s seed 3, where v18 struck the wall under stock Land). The seed 1
flights report `position_held` false, as v10's do on the same cells: the seed 1 jump
leaves the estimate frame about 1.45 m off truth (v10 1.41–1.46 m), the scenario, not the
controller.

Mean over seeds, worst in brackets:

| scenario | p95 m, tuned / v10 / **v19** | worst toward wall m, tuned / v10 / **v19** | rms m, tuned / v10 / **v19** | crashes | post-hold fail |
|---|---|---|---|---|---|
| calm 0 m/s | 0.522 (0.913) / 0.329 (0.396) / **0.278 (0.320)** | 0.229 (0.417) / 0.075 (0.184) / **0.071 (0.209)** | 0.221 / 0.173 / **0.159** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |
| 4 m/s | 0.523 (0.914) / 0.327 (0.410) / **0.287 (0.319)** | 0.220 (0.405) / 0.069 (0.172) / **0.074 (0.211)** | 0.221 / 0.176 / **0.162** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |
| 8 m/s | 0.524 (0.918) / 0.345 (0.475) / **0.295 (0.320)** | 0.218 (0.387) / 0.080 (0.166) / **0.091 (0.215)** | 0.221 / 0.185 / **0.159** | 0 / 0 / 0 of 3 | 0 / 0 / 0 of 3 |

Against tuned PX4, mean p95: v19 −47 / −45 / −44 %. v19 against v10: p95 −16 / −12 / −14 %,
rms −8 / −8 / −14 %, worst toward wall −5 / +7 / +14 %. v19 against v18: p95 −3 / −3 / −3 %,
worst toward wall −8 / −19 / −29 %.

Per seed (p95 / rms / worst toward wall / min clearance, m; no flight had a contact in the
scored window):

| cell | tuned PX4 | v10 | **v19** | p95 vs v10 |
|---|---|---|---|---|
| calm, seed 1 | 0.913 / 0.364 / 0.262 / 0.881 | 0.396 / 0.217 / 0.042 / 0.966 | **0.286 / 0.175 / 0.004 / 1.002** | −28 % |
| calm, seed 2 | 0.301 / 0.140 / 0.008 / 0.974 | 0.307 / 0.152 / 0.000 / 0.999 | **0.320 / 0.173 / 0.000 / 1.025** | +4 % |
| calm, seed 3 | 0.353 / 0.159 / 0.417 / 0.571 | 0.282 / 0.149 / 0.184 / 0.805 | **0.227 / 0.128 / 0.209 / 0.786** | −20 % |
| 4 m/s, seed 1 | 0.914 / 0.366 / 0.247 / 0.851 | 0.410 / 0.221 / 0.034 / 0.972 | **0.305 / 0.183 / 0.011 / 0.998** | −26 % |
| 4 m/s, seed 2 | 0.302 / 0.139 / 0.008 / 0.973 | 0.313 / 0.156 / 0.000 / 1.003 | **0.319 / 0.173 / 0.000 / 1.034** | +2 % |
| 4 m/s, seed 3 | 0.353 / 0.158 / 0.405 / 0.583 | 0.257 / 0.152 / 0.172 / 0.822 | **0.237 / 0.129 / 0.211 / 0.800** | −8 % |
| 8 m/s, seed 1 | 0.918 / 0.365 / 0.259 / 0.885 | 0.475 / 0.242 / 0.074 / 0.931 | **0.320 / 0.184 / 0.058 / 0.954** | −33 % |
| 8 m/s, seed 2 | 0.305 / 0.139 / 0.007 / 0.973 | 0.319 / 0.160 / 0.000 / 1.002 | **0.311 / 0.164 / 0.000 / 1.025** | −3 % |
| 8 m/s, seed 3 | 0.350 / 0.159 / 0.387 / 0.602 | 0.242 / 0.153 / 0.166 / 0.838 | **0.255 / 0.129 / 0.215 / 0.798** | +5 % |

v19's p95 is lower than v10's in 6 of 9 cells (v18: 3 of 9) and within +2 to +5 % in the
other three (calm and 4 m/s seed 2, 8 m/s seed 3). That is inside the spread v10 shows
against itself: the same policy and seed flown twice in the engage guard confirmation
differed by 2–10 % per cell (0.324 / 0.313 / 0.312 against 0.331 / 0.328 / 0.347 m on
seed 3). The check's per-replay rule predicted exactly this (seed 1 0.22 vs 0.46, seed 2
a tie at 0.27 vs 0.26, seed 3 0.31 vs 0.36).

**Wall drift.** Seed 1 is better than v10 in every cell (0.004 / 0.011 / 0.058 against
0.042 / 0.034 / 0.074 m) and v12's and v18's seed 1 drift is gone. Seed 3 is 3–5 cm
worse than v10 (0.209 / 0.211 / 0.215 against 0.184 / 0.172 / 0.166 m; min clearance
0.786 against 0.805 m), the same as v18 and v12 on that seed: seed 3's faults pull every
policy trained so far 0.2 m toward the wall, where v10 stands off a little further. No
contact and 0.79 m of clearance at a 1 m hold.

**Ring.** All three batches scored with one script over each flight's scored window (it
reproduces the v18 figures above: 0.130 rad/s, std 0.147, setpoint 0.118 / 0.113):

| mean of 9 flights | v10 | v18 | **v19** |
|---|---|---|---|
| roll-rate spectrum peak, 1–6 Hz | 1.4–1.8 Hz | 3.35–3.40 Hz in 9 of 9 | **1.2–2.3 Hz, nothing near 3.4 Hz** |
| roll-rate 2.2–5 Hz rms | 0.018 rad/s | 0.130 | **0.0085** |
| roll-rate 1.2–2.2 Hz rms | 0.037 rad/s | 0.012 | **0.013** |
| roll-rate std | 0.077 rad/s | 0.147 | **0.030** |
| rate setpoint 2.2–5 Hz rms, roll / pitch | 0.024 / 0.028 rad/s | 0.119 / 0.113 | **0.010 / 0.008** |
| module rate roughness | | 24.8 rad/s² (calm seed 1) | **0.4–1.2 rad/s²** |

The 1–6 Hz rate-command penalty did what it was meant to: v18's 3.4 Hz commanded
oscillation is gone, and v19 rings less than v10 in both bands, with a roll-rate std 2.5×
lower than v10's and 5× lower than v18's.

**Verdict proposal: release v19.** Against the release rule set after v18: (1) hold p95
lower than v10 on the mean (−12 to −16 %) and in 6 of 9 cells, the three misses +2 to
+5 %, within v10's own run-to-run spread; (2) ring lower than v10 everywhere in 1–6 Hz;
(3) no contact, no crash, no post-hold failure, no failsafe; (4) wall drift better on
seed 1, 3–5 cm worse on seed 3 with 0.79 m of clearance. Rule (1) is not met to the
letter and (4) is a small regression on one seed, but v19 is the only candidate that is
better than v10 on the controller qualities the rule exists to protect (no new
oscillation, no strike, lower p95 on the bad-estimate seeds), and strictly better than the
released v10 on p95, rms, ring and crashes. Against tuned PX4 the headline becomes
"44–47 % lower p95, 0 crashes" (v10: 31–34 % on 8 seeds; the 8-seed confirmation of v19 is
the next Mac job). The module headers stay on v19 in this PR; v10's `.onnx` remains in
`training/policies/` and the module goes back to it with the two commands in the module
README if a later batch says otherwise.

**Decision (PR #61): v19 is the released policy.** Seed 3's 3–5 cm extra drift toward the
wall is logged as the thing to watch on the next candidate, together with the 8-seed
confirmation (seeds 4–8 on the matched timeline, hold runner `vio_at_engage`, stock runner
`vio_at_hover`) before the Gate 2 table is restated for v19.

## Gate 2 evidence: v19 vs tuned PX4 on 8 seeds (matched fault timeline)

The 8-seed confirmation of the released v19, on the matched fault timeline (hold runner
`vio_at_engage`, stock runner `vio_at_hover`), flown 2026-10-24 on the Mac at `3b081e3`
(image from `752012c`, `NAME structure_hold_v19`, not rebuilt: `3b081e3` is docs only).
Seeds 4–8 were flown in two sub-hour chunks and pooled with the seeds 1–3 batches of the
v19 section above:

```bash
python -m eval.baselines.run_baseline --runner hold --label structure_hold_v19 --params eval/baselines/px4_tuned_v1.params --seeds 4 5 6 7 8 -- --profile vio_at_engage
python -m eval.baselines.run_baseline --label tuned_v1_at_hover --params eval/baselines/px4_tuned_v1.params --seeds 4 5 6 7 8 -- --profile vio_at_hover
python -m eval.baselines.run_baseline --pool <seeds 1-3 batch.json> <seeds 4-8 batch.json> --out runs/pooled_<label>_seeds1-8.json
python -m eval.baselines.run_baseline --compare runs/pooled_tuned_v1_at_hover_seeds1-8.json runs/pooled_structure_hold_v19_seeds1-8.json
```

Batches: v19 `20261024-033047` (seeds 1–3) + `20261024-040659_px4_baseline_structure_hold_v19_3b081e3`
(seeds 4–8); tuned PX4 `20261023-162007` (seeds 1–3) + `20261024-043727_px4_baseline_tuned_v1_at_hover_3b081e3`
(seeds 4–8).

**Seed count: n = 8 seeds per wind, 24 flights per controller, all 48 flown, scored and
verified** (bridge "fault clock anchored at 25.0 s", `estimate_verify` and `wind_verify` ok
in every flight). Seed 7, which never got off the ground on the start-anchored timeline
(its two jumps landed during arming; the v10 table above drops it), flies normally on the
matched timeline for both controllers: the faults now wait for the engage or the hover.
Every flight of both controllers: 0 contacts, 0 crashes, 0 post-hold failures. Tuned PX4's
three in-hover failsafes are seed 3's (v12 section); none on seeds 4–8.

### Pooled compare (8 seeds, as `--compare` prints it)

| scenario | p95 m, tuned → v19 | worst toward wall m | rms m | crashes | failsafes |
|---|---|---|---|---|---|
| wall1m_calm_0ms | 0.433 (1.097) → **0.243 (0.320), −44 %** | 0.197 (0.417) → 0.079 (0.209), −60 % | 0.210 (0.450) → 0.158 (0.188), −25 % | 0/8 → 0/8 | 0/8 → 0/8 |
| wall1m_wind_4ms | 0.432 (1.088) → **0.249 (0.319), −42 %** | 0.194 (0.405) → 0.082 (0.211), −58 % | 0.210 (0.453) → 0.160 (0.186), −24 % | 0/8 → 0/8 | 0/8 → 0/8 |
| wall1m_wind_8ms | 0.433 (1.082) → **0.254 (0.320), −41 %** | 0.197 (0.387) → 0.097 (0.215), −51 % | 0.211 (0.454) → 0.159 (0.184), −25 % | 0/8 → 0/8 | 0/8 → 0/8 |

Mean over seeds, worst in brackets. Median p95: tuned 0.270 / 0.270 / 0.271 m, v19
0.233 / 0.239 / 0.251 m, −14 / −11 / −7 %.

### Per-seed p95 m (`--pool` spread)

| tuned PX4 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | mean | worst | median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.913 | 0.301 | 0.353 | 0.174 | 0.219 | 1.097 | 0.238 | 0.172 | 0.433 | 1.097 | 0.270 |
| 4 m/s | 0.914 | 0.302 | 0.353 | 0.176 | 0.215 | 1.088 | 0.237 | 0.171 | 0.432 | 1.088 | 0.270 |
| 8 m/s | 0.918 | 0.305 | 0.350 | 0.178 | 0.213 | 1.082 | 0.237 | 0.182 | 0.433 | 1.082 | 0.271 |

| v19 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | mean | worst | median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calm 0 m/s | 0.286 | 0.320 | 0.227 | 0.180 | 0.194 | 0.301 | 0.239 | 0.201 | 0.243 | 0.320 | 0.233 |
| 4 m/s | 0.305 | 0.319 | 0.237 | 0.184 | 0.195 | 0.303 | 0.242 | 0.208 | 0.249 | 0.319 | 0.239 |
| 8 m/s | 0.320 | 0.311 | 0.255 | 0.189 | 0.194 | 0.295 | 0.247 | 0.219 | 0.254 | 0.320 | 0.251 |

### Per seed, v19 against tuned PX4 (p95 change, calm / 4 / 8 m/s)

| seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 |
|---|---|---|---|---|---|---|---|
| −69 / −67 / −65 % | +6 / +6 / +2 % | −36 / −33 / −27 % | +3 / +5 / +6 % | −11 / −9 / −9 % | −73 / −72 / −73 % | 0 / +2 / +4 % | +17 / +22 / +20 % |

Seeds 4–8 per flight (p95 / rms / worst toward wall / min clearance, m):

| cell | tuned PX4 | **v19** |
|---|---|---|
| calm, seed 4 | 0.174 / 0.137 / 0.097 / 0.909 | **0.180 / 0.148 / 0.043 / 0.957** |
| calm, seed 5 | 0.219 / 0.143 / 0.113 / 0.888 | **0.194 / 0.140 / 0.078 / 0.919** |
| calm, seed 6 | 1.097 / 0.450 / 0.383 / 0.816 | **0.301 / 0.188 / 0.039 / 0.967** |
| calm, seed 7 | 0.238 / 0.158 / 0.187 / 0.830 | **0.239 / 0.153 / 0.152 / 0.880** |
| calm, seed 8 | 0.172 / 0.132 / 0.108 / 0.917 | **0.201 / 0.157 / 0.106 / 0.902** |
| 4 m/s, seed 4 | 0.176 / 0.137 / 0.099 / 0.906 | **0.184 / 0.151 / 0.047 / 0.946** |
| 4 m/s, seed 5 | 0.215 / 0.142 / 0.113 / 0.886 | **0.195 / 0.141 / 0.079 / 0.918** |
| 4 m/s, seed 6 | 1.088 / 0.453 / 0.381 / 0.812 | **0.303 / 0.186 / 0.040 / 0.963** |
| 4 m/s, seed 7 | 0.237 / 0.159 / 0.184 / 0.833 | **0.242 / 0.156 / 0.156 / 0.884** |
| 4 m/s, seed 8 | 0.171 / 0.130 / 0.110 / 0.920 | **0.208 / 0.161 / 0.111 / 0.902** |
| 8 m/s, seed 4 | 0.178 / 0.138 / 0.108 / 0.900 | **0.189 / 0.152 / 0.067 / 0.938** |
| 8 m/s, seed 5 | 0.213 / 0.143 / 0.127 / 0.873 | **0.194 / 0.141 / 0.082 / 0.915** |
| 8 m/s, seed 6 | 1.082 / 0.454 / 0.375 / 0.792 | **0.295 / 0.179 / 0.048 / 0.951** |
| 8 m/s, seed 7 | 0.237 / 0.158 / 0.177 / 0.839 | **0.247 / 0.159 / 0.160 / 0.883** |
| 8 m/s, seed 8 | 0.182 / 0.136 / 0.133 / 0.897 | **0.219 / 0.164 / 0.147 / 0.877** |

Wall drift: v19 lower than tuned PX4 in 13 of the 15 new cells (the two exceptions are
seed 8 at 4 and 8 m/s, by 1 and 14 mm). Ring on seeds 4–8: 2.2–5 Hz roll-rate rms
0.005–0.018 rad/s (mean 0.011; seeds 1–3 0.0085, v10 0.018, v18 0.130), roll-rate std
0.023–0.044 (mean 0.030), spectrum peak 1.5–2.2 Hz in every flight. Seed 7 rings the most
of v19's flights (0.018).

**What the 8 seeds say.** The Gate 2 target (30 % lower p95 than tuned PX4 at 8 m/s, 1 m)
is met on the mean with margin: −41 % at 8 m/s, −42 to −44 % in the other winds, with 0
crashes against 0. As with v10, the mean win comes from the seeds where tuned PX4 loses the
hold: seeds 1 and 6, where stock PX4 goes past 1 m p95 and 0.38 m toward the wall while
v19 holds 0.29–0.32 m with 0.004–0.06 m toward the wall. Where tuned PX4 copes, the two
are close: v19 is better on seeds 3 and 5 (−9 to −36 %), a tie on seed 7, and worse on
seeds 2, 4 and 8 (+2 to +22 %; seed 8 is the one clear loss). The median p95 is 7–14 %
lower than tuned PX4's (v10 on the start-anchored 8 seeds: a tie). Seed 6 replaces seed 1
as the hardest seed for the baseline and is a new candidate for a check replay cell.

**Honest claim for v19:** "41–44 % lower hold p95 than tuned PX4 over 8 seeds and three
winds with no crash, no contact and no post-hold failure; the gain is robustness on the
seeds where the estimate misleads stock PX4 (two of eight, where it drifts past 1 m); on
the other six the two are within about 20 % either way and the median is 7–14 % better."
This restates the Gate 2 table for v19; the v10 table above stays as the record of the
start-anchored protocol.
