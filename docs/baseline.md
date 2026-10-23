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
