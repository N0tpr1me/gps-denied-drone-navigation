# Field faults: the estimate failures integrators actually report

Gap 1 of `docs/market-research.md`. The random corruption we train and benchmark on
(`envs/estimate`: delay, noise, bias walk, decaying jumps, dropouts up to 3 s) is a good
model of a working VIO. The failures PX4 and ArduPilot users post about are blunter, and a
buyer's first question is what the drone does in each one. This page scripts those
failures, flies them in Gazebo SITL with Structure Hold and with tuned stock PX4, and says
plainly what happened, good and bad.

## The six faults

All six live in `sim2real/estimate_bridge/field_faults.py` and are applied by the estimate
bridge on top of the normal corruption, at a set time after the fault anchor, so every
runner meets the same failure at the same moment of its scored window.

| Fault | What the bridge does | Field report it copies |
| --- | --- | --- |
| `reset_origin` | The VIO frame snaps back to where the VIO started (the takeoff point, on x, y and z) and moves on from there; ODOMETRY `reset_counter` counts up (`signal: false` hides it) | VIO "resets to 0,0,0 in the air" (ModalAI forum 1383, 5333); PX4 #19830 |
| `restart` | Nothing is sent for `duration_s`, then the same snap | VIO process crash and restart |
| `silence` | Nothing is sent for `duration_s` (longer than the 3 s dropouts we train on) | Companion computer or link lost |
| `freeze` | The same position and zero velocity keep arriving on time, never marked stale | A stuck camera pipeline; no staleness check can see it |
| `drift` | A slow ramp of `rate_mps` along `direction`, then held | Scale drift, or a spoofed source walking off |
| `degrade` | The VIO keeps running but sees less: extra noise (`noise_m`, `vel_noise_mps`), a faster random walk (`walk_mps`, kept afterwards), tracking losses of `loss_s` about `loss_hz` times a second; the forward rangefinder gets noise (`range_noise_m`), missing returns (`range_loss_frac`) and short returns off the particles (`range_short_frac`). All of it ramps up over `ramp_s` | Dust thrown up by the rotors, fog or steam, a dark underside: the cases buyers ask about (added 2026-11-02) |

Each ships as a pair of profiles that differ only in their anchor:
`field_<fault>_at_engage` for the hold runner and `field_<fault>_at_hover` for the stock
runner (`sim2real/estimate_bridge/profiles/`). They keep `vio_default`'s delay, noise and
bias walk, turn its random jumps and dropouts off, and start the fault 20 s into the scored
window. `verify_trace` replays the faults from the profile, so a flight's trace still proves
exactly what PX4 was fed.

## Scoring

`eval/field_faults.py` reads the TRUE position from each flight's bridge trace over the
25 s after the fault: closest approach to the wall (the x500's props touch at 0.40 m), the
largest move from where the vehicle was when the fault began, the lowest height, whether it
was still airborne at the end, and the module's watchdog trip. The benchmark's p95 and its
"the hold must run to the end" rule do not fit here: when the estimate snaps by metres,
handing back is the right answer.

```
python -m eval.baselines.run_baseline --runner hold --label field_reset_v22 \
    --params eval/baselines/px4_tuned_v2.params --scenarios wall1m_wind_4ms --seeds 1 \
    -- --profile field_reset_at_engage --rangefinder
python -m eval.baselines.run_baseline --label field_reset_tuned \
    --params eval/baselines/px4_tuned_v2.params --scenarios wall1m_wind_4ms --seeds 1 \
    -- --profile field_reset_at_hover
python -m eval.field_faults runs/<hold flight> runs/<stock flight>
```

The flights on this page were flown on `px4_tuned_v1.params` (vision height) unless a row says
otherwise; since 2026-11-01 the benchmark uses `px4_tuned_v2.params` (barometer height,
the finding below), as the commands above do.

## First flights (2026-11-01, cloud)

Released policy v22 in the module with the forward rangefinder (`--rangefinder`, as the
partner pack requires) against tuned PX4 (`eval/baselines/px4_tuned_v1.params`, Position
mode, no rangefinder use: stock PX4 has none for xy). Scenario `wall1m_wind_4ms` (1 m from
the wall, 4 m/s mean wind blowing into it, gusts), seed 1 only, one flight per cell, on the
cloud-built image (`sim2real/docker/cloud_build.sh`: same PX4 v1.17.0, module and world, a
different Gazebo base). Treat these as a first look, not a benchmark claim: one seed, one
scenario.

The hold flights were flown twice. In the first set the bridge anchored the fault clock on
the engage the flight sequence tries on the ground, which the module refuses (PR #101 added
that check), so each fault hit about 4 s into the hold instead of 20 s. The bridge now counts
an engage only once the vehicle is airborne (`heartbeat.airborne_engage`), and the table is
the second set. The stock runner anchors on its height and was not affected.

| Fault | Runner | Closest to wall | Largest move | Lowest height | What happened |
| --- | --- | --- | --- | --- | --- |
| reset to origin | Structure Hold | 0.99 m | 0.09 m | 2.49 m | watchdog trip (`position_error`), handed back to Structure Hover, which held in place |
| reset to origin | tuned PX4 | 0.75 m | 0.39 m | 2.51 m | climbed 2.2 m (EKF2's height snapped to the takeoff height) and moved 0.39 m toward the wall |
| restart (3 s gap, then reset) | Structure Hold | 0.95 m | 0.08 m | 2.42 m | same trip and hover |
| restart (3 s gap, then reset) | tuned PX4 | 0.77 m | 0.37 m | 2.51 m | same climb of 2.2 m |
| drift 0.6 m away from the wall | Structure Hold | 0.70 m | 0.33 m | 2.48 m | no trip; the wall guard stopped the drift at `SH_WALL_MIN` 0.70 m |
| drift 0.6 m away from the wall | tuned PX4 | 0.42 m | 0.59 m | 2.46 m | followed the drift to 2 cm from prop contact |
| freeze 5 s | Structure Hold | 0.94 m | 0.55 m | 2.46 m | no trip: nothing in the module sees a frozen estimate |
| freeze 5 s | tuned PX4 | 0.94 m | 0.46 m | 2.51 m | about the same |
| silence 8 s | Structure Hold | 0.26 m (on the ground) | 2.77 m | 0 m | **sank to the ground** in 5.5 s, 1 m from the wall, while EKF2 still showed 2.5 m (below); the watchdog tripped (`estimate_invalid`) to Altitude mode at 0.8 m; on the ground the wind then pushed it to the wall |
| silence 8 s | tuned PX4 | **0.24 m (contact)** | 0.80 m | 0 m | held its height for 4.5 s, then PX4's failsafe descended; it touched the wall on the way down |

### Why Structure Hold sank on the link loss

Not the module's horizontal logic: EKF2's height. The benchmark's EKF2 set-up at the time
(`ekf2_ev.params`, a Decided row of 2026-10-21) made vision the height reference
(`EKF2_HGT_REF 3`; barometer since 2026-11-01, `EKF2_HGT_REF 0`), with the barometer aiding through a bias
estimate. When the vision stream
stops, EKF2 puts the barometer's change into that bias and its height stops moving: in this
flight the baro innovation grew to -2.4 m while EKF2's height stayed at 2.5 m. A 4 % dip in
the policy's thrust as the fix went stale started a 0.5 m/s descent, and nothing in the loop
could see it. Stock PX4 meets the same frozen height, but nothing pushed it during
those 4.5 s, and then its failsafe took over.

The same flight with the barometer as the height reference (`EKF2_HGT_REF 1` then, which PX4 v1.17 numbers GPS and with GPS off falls back to the barometer; 0 since 2026-11-02; a scratch copy
of the tuned parameters, nothing else changed) held its height through the silence (2.39 to
2.45 m). It then failed the way the first set suggested: the watchdog handed back to
Altitude mode, which keeps the vehicle level, and the wind blew it into the wall (props
touching at 2.4 m height, 8.9 s after the link went; closest 0.26 m) and down. Stock PX4 with baro height did what it
did before: failsafe descent, wall touched on the way down.

### The same link loss with the range hover (`SH_HB_RANGE` 1, on by default since 2026-11-02)

`SH_HB_RANGE` (on by default since 2026-11-02, `sim2real/px4/structure_hold/RangeHover.hpp`) keeps Structure
Hover flying when the horizontal estimate is gone next to a measured wall, instead of the
decided handback to Altitude mode. It sends PX4's position controller an acceleration-only
horizontal setpoint (PX4 needs no horizontal estimate for that, as in Altitude mode, and
holds the height itself): the wind lean learned while the estimate was good, plus a PID on
the forward range that holds the distance measured when it started. Nothing measures drift
along the wall, so that axis gets the wind lean alone. Once the estimate has been good for
a second it goes back to holding a point on the estimate; without a usable range for 0.5 s
it trips (`range_lost`) and the decided handback order follows.

Flown on the same flight with barometer height (image built from the module branch with
`SITL_IMAGE=structure-hold/px4-sitl:rangehover sim2real/docker/cloud_build.sh`, flown with
`--image structure-hold/px4-sitl:rangehover --param SH_HB_RANGE=1`):

| | Without (Altitude mode) | With `SH_HB_RANGE` 1 |
| --- | --- | --- |
| Closest to the wall | 0.26 m (props on the wall) | 0.94 m |
| Lowest height | 0.04 m (on the ground) | 2.45 m |
| Largest move | 8.4 m | 0.48 m (along the wall) |
| Airborne at the end | no | yes |

The watchdog tripped (`estimate_invalid`) 4.5 s into the silence as before and handed back
to Structure Hover, which said "position lost, holding 0.98 m from the wall on the
rangefinder". Through the rest of the silence the true wall distance stayed 0.95 to 0.99 m
while the vehicle slid about 0.2 m along the wall; when the VIO came back it said "position
back" and held on the estimate again. One flight, one seed: it shows the mechanism works in
Gazebo, not how well it does across winds. It does nothing for the height: with vision as
EKF2's height reference the vehicle would still sink.

### The link loss over three winds and four seeds (2026-11-02)

The same 8 s silence 20 s into the hold, now on the benchmark's barometer height
(`eval/baselines/px4_tuned_v2.params`), every scenario (calm, 4 and 8 m/s mean wind into the
wall) and seeds 1 to 4: v22 with the rangefinder and `SH_HB_RANGE` 1, the same with it off
(the decided handback), and tuned PX4. Cloud image at `a4afbd6` (module as merged in #113),
36 flights in three sub-hour batches; `eval/field_faults.py` on each batch
(`field_faults.json` in the batch folders `*_silence_v22_hb1_a4afbd6`, `*_silence_v22_hb0_a4afbd6`,
`*_silence_tuned_a4afbd6`).

| | calm | 4 m/s | 8 m/s | all 12 |
| --- | --- | --- | --- | --- |
| `SH_HB_RANGE` 1: wall contacts | 0/4 | 0/4 | 0/4 | **0/12**, all airborne, closest 0.84 m |
| `SH_HB_RANGE` 1: closest to the wall, m | 0.92 to 1.03 | 0.92 to 1.02 | 0.84 to 1.01 | |
| `SH_HB_RANGE` 1: largest move, m | 0.36 to 0.55 | 0.09 to 0.52 | 0.56 to 0.97 | |
| Off (Altitude mode): wall contacts | 0/4 | 4/4 | 4/4 | **8/12** touched or passed the wall line |
| Off: largest move, m | 0.52 to 2.08 | 2.1 to 12.1 | 8.4 to 20.6 | two 8 m/s flights sank to under 0.1 m |
| Tuned PX4: wall contacts | 0/4 | 2/4 | 4/4 | **6/12**, all 12 ended on the ground (PX4's failsafe descent) |

Every v22 flight tripped `estimate_invalid` and handed back as the decided order says. With
`SH_HB_RANGE` 1 Structure Hover held the wall distance on the range through the rest of the
silence and the vehicle was still flying, with its height kept within 0.31 m (barometer),
when the estimate came back. Without it Altitude mode kept the vehicle level and any wind
into the wall carried it there; in calm air it drifted but stayed clear. Stock PX4 held for
about 4.5 s and then its failsafe landed it, touching the wall on the way down in wind.
One flight per cell; the movement along the wall (up to 0.97 m at 8 m/s) is the axis the
range cannot see.

### The freeze with the frozen-fix trip (`SH_WD_FRZ_T`, 1 s by default since 2026-11-02)

`SH_WD_FRZ_T` (seconds, 1 s by default since 2026-11-02, 0 off; `FrozenFix` in `Watchdog.hpp`) trips Structure Hold
(`estimate_frozen`) when the VIO odometry keeps arriving but has not moved by 0.1 mm for
that long. A real VIO on a flying vehicle is never that still: its noise and the vehicle's
own motion move it every sample. The trip follows the decided order: next to a measured wall
that is Structure Hover, PX4's controller with the wall guard. It is not a trip inside
Structure Hover or Land: the first flight below had it there, and Structure Hover then
handed on to Position mode, which flew the same frozen estimate without the guard and
drifted to 0.64 m from the wall. With `SH_HB_RANGE` 1 Structure Hover counts a frozen fix
as no estimate and holds the wall distance on the rangefinder until the fix moves again.

Same scenario and seed as the table above, benchmark parameters (vision height), the 5 s
freeze, `SH_WD_FRZ_T` 1 s (image `structure-hold/px4-sitl:frozen` from the module branch):

| | No trip (table above) | Trip, first version (handed on to Position mode) | Trip | Trip + `SH_HB_RANGE` 1 |
| --- | --- | --- | --- | --- |
| Closest to the wall | 0.94 m | 0.64 m | 0.97 m | 0.97 m |
| Largest move | 0.55 m | 0.40 m | 0.21 m | 0.19 m |
| Lowest height | 2.46 m | 2.47 m | 2.47 m | 2.47 m |

The trip fired 1 s into the freeze each time and told the ground station why. Staying in
Structure Hover on the frozen estimate held about as well as the range hover here: in 4 s
the wind moved the vehicle 0.2 m, and the wall guard kept it off the wall. A longer freeze
is where the range hover should matter, as on the link loss. One flight per cell, one seed.

### Dust, fog and low light: the `degrade` fault (2026-11-02)

Buyers ask what happens in rotor-blown dust, fog or steam, and on a dark underside. None
of these stops a VIO outright: it runs on fewer and worse features, gets noisier, walks off
faster and loses tracking for fractions of a second, and a laser rangefinder in the same
air gets noisy, misses returns and sees short ones off the particles. `degrade` scripts
that (`field_degrade_at_engage` / `_at_hover`): over 20 s, ramping up over the first 5 s,
the VIO gains 5 cm of position noise, 0.2 m/s of velocity noise, a random walk of
0.1 m/sqrt(s) that is kept afterwards, and 0.5 s tracking losses about once every 4 s; the
rangefinder gains 5 cm of noise, loses a fifth of its returns and a tenth come back short.
The walk and the losses come from the profile's seed, so both runners meet the same ones
(seed 1 walks about 0.55 m away from the wall and 0.6 m along it; seed 2 about 0.45 m,
toward it); the rangefinder's draws are their own stream, so `verify_trace` still holds.

v22 with the rangefinder (`--rangefinder`) against tuned PX4, barometer height
(`px4_tuned_v2.params`), calm, 4 and 8 m/s into the wall, seeds 1 and 2, cloud image
(`eval/field_faults.py` over the 25 s after the fault, `field_faults.json` in the batch
folders `*_degrade_v22_e185a80_e185a80` and `*_degrade_tuned_def668f_def668f`):

| | calm | 4 m/s | 8 m/s | all 6 |
| --- | --- | --- | --- | --- |
| v22: wall contacts | 0/2 | 0/2 | 0/2 | **0/6**, no trip, all airborne, closest 0.85 m |
| v22: closest to the wall, m | 1.02 / 0.91 | 1.01 / 1.01 | 0.99 / 0.85 | |
| v22: largest move, m | 1.01 / 0.44 | 0.57 / 0.68 | 1.03 / 0.53 | up to 1.0 m, seed 1 away from the wall |
| v22: height, m | within 0.26 | within 0.46 | within 0.31 | |
| Tuned PX4: wall contacts | 1/2 | 1/2 | 1/2 | **3/6** (seed 1 every time, 0.37 to 0.39 m) |
| Tuned PX4: closest to the wall, m | 0.39 / 0.55 | 0.37 / 0.84 | 0.37 / 0.66 | |
| Tuned PX4: largest move, m | 0.85 / 0.49 | 0.74 / 0.29 | 0.75 / 0.37 | |
| Tuned PX4: height, m | within 0.40 | within 0.46 | within 0.24 | |

Seed 1 / seed 2 in each cell. What happened:

- **Nothing trips, on purpose.** 5 cm of noise is well under the 2 m hold-error trip and
  0.5 s losses are under the 5 s stale trip; EKF2 rode through both. v22 held through the
  whole 20 s on the degraded fix, as it was trained to (Gate 1 stress covers noise and
  dropouts to 3 s).
- **Both follow the walk.** Neither controller can tell a VIO walking off from the vehicle
  moving, so both move with it: seed 1's 0.55 m walk away from the wall took stock PX4
  0.55 m toward it, to 0.37 to 0.39 m, prop contact, in all three winds. The policy would
  have done the same; the rangefinder is what kept v22 off the wall (as on the slow
  drift, F7 in the safety manual).
- **Short returns off the dust push the hold point back, and it stays back.** The wall
  guard treats a short reading as the wall being close and moves the hold point away; it
  only ever moves away (the ratchet that stops a lost sensor walking the vehicle in). In
  the dust the readings went as short as 0.10 to 0.13 m and the guard moved the hold
  point 0.6 to 0.95 m back over the 20 s (`hold.json` `wall_guard.shift_m`, less the
  engage-time shift), on top of the walk: seed 1 ended 1.9 to 2.1 m from the wall. That is
  the safe side, but it is an unplanned move of up to a metre away from the structure, into
  whatever is behind, and in a longer cloud it would keep going. The wall-range policy input
  (`RangeReference`, v23 on) had a spike gate; the guard had none. Fixed the same day,
  below.
- **Height moved 0.2 to 0.5 m on both**, from the velocity noise (0.2 m/s) that EKF2
  fuses: the barometer height reference bounds it. Stock PX4 dipped 0.46 m, v22 0.26 m.

**With the spike gate on the guard (2026-11-02, `WallGuard::acceptWall`, decided row).**
The guard now holds a reading more than 0.25 m from the last accepted one until the next
reading agrees with it, the rule `RangeReference` and the mock already used, and a held
reading counts as no range for the range hover too. The same six flights again on an image
with the gate (`*_degrade_v22_gate_ae14cbd`), the hold point's shift during the hold from
`hold.json`:

| | calm | 4 m/s | 8 m/s |
| --- | --- | --- | --- |
| Shift before the gate, m (seed 1 / 2) | 0.95 / 0.59 | 0.88 / 0.69 | 0.75 / 0.65 |
| Shift with the gate, m (seed 1 / 2) | 0.61 / 0.00 | 0.57 / 0.02 | 0.58 / 0.00 |
| Closest to the wall with the gate, m | 0.91 / 0.85 | 0.82 / 0.92 | 0.89 / 0.82 |
| Where it ended, m from the wall | 1.48 / 1.19 | 1.39 / 1.19 | 1.37 / 1.22 |

Seed 2's VIO walks toward the wall (the drone drifts away), so there is nothing for the
guard to do, and with the gate it did nothing: 0.00 to 0.02 m against 0.59 to 0.69 m of dust
before. Seed 1's VIO walks away from the wall, which carried the drone 0.5 m toward it (the
case that put stock PX4 on the wall), and the guard's 0.6 m is it catching that, as on the
slow drift; the 0.3 m of dust on top is gone. 0 of 6 touched, closest 0.82 m, no trips, and
the flights ended 1.2 to 1.5 m from the wall instead of 1.2 to 2.0 m. What is left: two
consecutive short returns that agree within 0.25 m still pass, by design (a real step
must), so a dense cloud can still move the point; a longer agreement window or a rate limit
on the shift would be the next step if a real sensor log shows that.

One flight per cell, two seeds: it shows the mechanisms, not a rate. The numbers are what
this one `degrade` setting does; a real dust cloud or fog bank may be milder or far worse,
which is why the safety manual's envelope now names visibility (F36, F37), and tells the
pilot to stop the hold when the VIO degrades rather than fly it out.

What this says, in plain words:

- **A VIO reset or restart is where the module earns its keep.** Stock PX4 climbed 2.2 m
  and slid 0.4 m toward the wall; the module tripped, handed back to its guarded hover and
  moved less than 10 cm. This is the most reported failure in the research.
- **Slow drift is caught by the rangefinder, not by the estimate.** The wall guard held
  0.70 m while stock PX4 went to 0.42 m. Stock PX4 could only match this with a wall-aware
  hold of its own.
- **A frozen estimate is invisible to both, until the module looks for it.** No trip fired;
  both wandered about half a metre and stayed 0.94 m from the wall. The frozen-fix trip
  (`SH_WD_FRZ_T`, above) caught it in 1 s and the guarded hover then moved 0.2 m; it is on
  by default since 2026-11-02 (Zain).
- **Dust, fog and low light: the module holds, but walks with the VIO.** Neither v22 nor
  stock PX4 can see a degraded VIO walking off; stock PX4 touched the wall on 3 of 6
  flights, v22 on none, because the rangefinder guard caught the walk toward the wall. The
  guard also read the dust as a close wall and backed the hold point off by up to a metre;
  the spike gate on the guard (same day) stopped that: 0 to 2 cm of dust shift on the seeds
  with nothing real to catch.
- **A long link loss is not survived by either, for two reasons.** With vision as EKF2's
  height reference the height estimate freezes and the vehicle can sink unseen: partners
  should fly the barometer (or a downward rangefinder) as the height reference, and the
  benchmark now does too (Zain's call, 2026-11-01: `px4_tuned_v2.params`). With
  the height fixed, the decided handback to Altitude mode lets the wind push the vehicle
  into the wall; the range hover (`SH_HB_RANGE`, above) kept it 0.94 m off instead, and 0 of
  12 touched over three winds. It is on by default since 2026-11-02 (Zain).

## GPS on: RTK float, a jump, a glitch, a loss and the hand-over (2026-11-02, cloud)

Gap 3 of the "Features that sell" research: the item with the most buyers behind it was what
the mode does with a GPS or RTK receiver on as well (Zain, 2026-11-02: do it; the envelope
said GPS off). The benchmark stays GPS-free. These flights put a scripted receiver beside
the vision stream: the bridge sends Gazebo's truth as HIL_GPS with seeded noise and the
faults below (`sim2real/estimate_bridge/gps.py`), PX4 takes it as its GPS (`MAV_USEHILGPS`
1), Gazebo's own receiver reports no fix (`SIM_GPS_USED` 0) and EKF2 fuses GPS position,
height and velocity beside the vision stream (`gps_on.params`, laid over
`px4_tuned_v2.params` by the runners whenever the profile has a `gps` block). The height
reference is set to the barometer explicitly (`EKF2_HGT_REF` 0: in PX4 v1.17 the value 1 the
benchmark set carried until 2026-11-02 means GPS height, and only flew on the barometer because
EKF2 falls back to it with no GPS; the benchmark set says 0 now too). RTK fixed, 10 Hz, 1 cm noise.

| Profile | What the receiver does | Field report it copies |
| --- | --- | --- |
| `gps_on` | Healthy for the whole flight (the `vio_at_*` corruption underneath) | Does a good GPS beside the VIO change the hold? |
| `field_rtk_float` | Drops to RTK float for 15 s: accuracy 0.5 m, 0.3 m noise, solution 0.8 m off | Fix lost under a structure |
| `field_gps_jump` | Solution steps 1.5 m and stays there, inside EKF2's innovation gate | Multipath under a deck, base-station change |
| `field_gps_glitch` | 4 m off for five samples, then back | A glitch |
| `field_gps_loss` | Quiet for 10 s, back 1 m off | Receiver lost under the structure, hand-over on return |

Every offset points away from the wall (`direction` -x), so EKF2's estimate says the
vehicle is further from the wall than it is and the controller moves it toward the wall:
the dangerous way. (A first round with the offsets pointing the other way moved both
controllers 0.05 to 0.8 m away from the wall and touched nothing.) Scenario
`wall1m_wind_4ms`, released policy v22 with the forward rangefinder (`--rangefinder`) against
tuned PX4, seed 1 unless a row says otherwise, cloud image at `e185a80`
(`eval/field_faults.py` tables in `runs/gps_faults_round{2,3}.json`).

| Fault | Runner | Closest to wall | Largest move | Lowest height | What happened |
| --- | --- | --- | --- | --- | --- |
| RTK float, 0.8 m off, 15 s | Structure Hold | 0.72 m | 0.31 m | 2.43 m | EKF2 followed about 0.3 m of the float solution; the wall guard stopped the hold point at `SH_WALL_MIN` 0.70 m; no trip |
| RTK float | tuned PX4 | 0.66 m | 0.35 m | 2.47 m | followed the same 0.3 m, nothing stopped it |
| GPS jump, 1.5 m, held | Structure Hold | 0.71 m | 0.33 m | 2.42 m | EKF2 followed the jump fully (estimate 1.1 m off at p95); the wall guard held 0.70 m; no trip |
| GPS jump | Structure Hold, seeds 2 and 3 | 0.70 / 0.71 m | 0.32 / 0.35 m | 2.36 / 2.42 m | the same |
| GPS jump | Structure Hold **without the rangefinder** | **0.30 m (contact)** | 0.73 m | 2.42 m | followed the estimate into the wall; `position_error` trip at the wall |
| GPS jump | tuned PX4 | **0.26 m (contact)** | 2.46 m | 0.00 m (ground) | followed the estimate into the wall and fell to the ground |
| GPS jump | tuned PX4, seeds 2 and 3 | **0.29 / 0.27 m (contact)** | 2.12 / 4.94 m | 0.03 / 0.08 m | the same, 3 of 3 seeds |
| GPS glitch, 4 m for 0.5 s | Structure Hold | 1.00 m | 0.05 m | 2.44 m | EKF2 rejected it (outside the gate); nothing happened |
| GPS glitch | tuned PX4 | 0.96 m | 0.06 m | 2.47 m | the same |
| GPS loss 10 s, back 1 m off | Structure Hold | 0.73 m | 0.31 m | 2.43 m | EKF2 reset once to vision during the loss (7 cm) and, once the receiver was back, walked 0.3 m toward its new solution over 15 s; the wall guard held 0.70 m; no trip |
| GPS loss | tuned PX4 | 0.69 m | 0.33 m | 2.47 m | the same walk, nothing stopped it |
| GPS healthy (`gps_on`) | Structure Hold | 0.97 m | hold p95 0.15 m | | against 0.19 m GPS-off on the same seed (`vio_at_engage`, flown the same day): a good RTK beside the VIO tightens the hold a little |
| GPS healthy | tuned PX4 | 0.95 m | hold p95 0.18 m | | against 0.89 m GPS-off: stock PX4 gains far more, since the vision jumps no longer move its estimate |

What this says, in plain words:

- **With GPS on, the rangefinder is what keeps the drone off the wall, for Structure Hold and
  for nothing else.** EKF2 believes a GPS solution that moves within its gate (the RTK float,
  the 1.5 m jump, the solution the receiver comes back with): the estimate walks, and every
  controller flying it walks with it. Tuned PX4 went into the wall and down on the 1.5 m jump
  on all three seeds; Structure Hold without a rangefinder did the same; with the rangefinder
  the wall guard stopped the hold point at 0.70 m every time. So the envelope with GPS on is:
  a forward rangefinder and `SH_WALL_REQ` 1, and the safety manual says so (1.2).
- **A short glitch is harmless**: EKF2's innovation gate throws it out. A healthy RTK beside
  the VIO makes the hold slightly tighter (v22 0.15 m against 0.19 m on the same seed; tuned
  PX4 0.18 against 0.89 m, since its estimate no longer follows the vision jumps).
- **Nothing catches a walk along the wall.** The guard measures the wall axis only; a GPS
  solution that walks the estimate sideways moves the drone sideways, with either controller.
  The module could spot the estimate leaving the VIO's own track while GPS is fused and tell
  the pilot (parked in `docs/decisions.md`); it would not hold the position any better than
  the guarded hover already does, and the flight-controller flash is nearly full, so no module
  code was added for this.
- The hand-over does not arrive as an EKF2 reset. With the vision stream still fusing, EKF2
  never resets to the returning GPS; its position bias estimator on the vision stream re-aligns
  to the GPS over about 15 s instead. A hold point that follows EKF2 resets (as PX4's modes do)
  would not have helped here.

## Next

- More seeds and the other wall scenarios, on the real image (GPU server or Mac), before any
  number here goes into a pitch.
- The frozen-fix trip over more seeds and a longer freeze, with and without `SH_HB_RANGE`.
- The range hover (`SH_HB_RANGE`, now on by default) over seeds 5 to 8 and longer silences.
- `field_degrade` over more seeds and a harsher setting (longer losses, a faster walk, a
  denser cloud with agreeing short returns) to find where the module's own trips take over
  and whether the guard's two-reading gate needs a longer window.
- The GPS faults over the other winds and more seeds, and the same faults with a jump along
  the wall (`direction` [0, 1, 0]), which no guard catches.
