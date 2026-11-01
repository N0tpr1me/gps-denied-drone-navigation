# Field faults: the estimate failures integrators actually report

Gap 1 of `docs/market-research.md`. The random corruption we train and benchmark on
(`envs/estimate`: delay, noise, bias walk, decaying jumps, dropouts up to 3 s) is a good
model of a working VIO. The failures PX4 and ArduPilot users post about are blunter, and a
buyer's first question is what the drone does in each one. This page scripts those
failures, flies them in Gazebo SITL with Structure Hold and with tuned stock PX4, and says
plainly what happened, good and bad.

## The five faults

All five live in `sim2real/estimate_bridge/field_faults.py` and are applied by the estimate
bridge on top of the normal corruption, at a set time after the fault anchor, so every
runner meets the same failure at the same moment of its scored window.

| Fault | What the bridge does | Field report it copies |
| --- | --- | --- |
| `reset_origin` | The VIO frame snaps back to where the VIO started (the takeoff point, on x, y and z) and moves on from there; ODOMETRY `reset_counter` counts up (`signal: false` hides it) | VIO "resets to 0,0,0 in the air" (ModalAI forum 1383, 5333); PX4 #19830 |
| `restart` | Nothing is sent for `duration_s`, then the same snap | VIO process crash and restart |
| `silence` | Nothing is sent for `duration_s` (longer than the 3 s dropouts we train on) | Companion computer or link lost |
| `freeze` | The same position and zero velocity keep arriving on time, never marked stale | A stuck camera pipeline; no staleness check can see it |
| `drift` | A slow ramp of `rate_mps` along `direction`, then held | Scale drift, or a spoofed source walking off |

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
(`EKF2_HGT_REF 3`; barometer since 2026-11-01), with the barometer aiding through a bias
estimate. When the vision stream
stops, EKF2 puts the barometer's change into that bias and its height stops moving: in this
flight the baro innovation grew to -2.4 m while EKF2's height stayed at 2.5 m. A 4 % dip in
the policy's thrust as the fix went stale started a 0.5 m/s descent, and nothing in the loop
could see it. Stock PX4 meets the same frozen height, but nothing pushed it during
those 4.5 s, and then its failsafe took over.

The same flight with the barometer as the height reference (`EKF2_HGT_REF 1`, a scratch copy
of the tuned parameters, nothing else changed) held its height through the silence (2.39 to
2.45 m). It then failed the way the first set suggested: the watchdog handed back to
Altitude mode, which keeps the vehicle level, and the wind blew it into the wall (props
touching at 2.4 m height, 8.9 s after the link went; closest 0.26 m) and down. Stock PX4 with baro height did what it
did before: failsafe descent, wall touched on the way down.

### The same link loss with the range hover (`SH_HB_RANGE` 1, Proposed)

`SH_HB_RANGE` (off by default, `sim2real/px4/structure_hold/RangeHover.hpp`) keeps Structure
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

### The freeze with the frozen-fix trip (`SH_WD_FRZ_T`, Proposed)

`SH_WD_FRZ_T` (seconds, off by default; `FrozenFix` in `Watchdog.hpp`) trips Structure Hold
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

What this says, in plain words:

- **A VIO reset or restart is where the module earns its keep.** Stock PX4 climbed 2.2 m
  and slid 0.4 m toward the wall; the module tripped, handed back to its guarded hover and
  moved less than 10 cm. This is the most reported failure in the research.
- **Slow drift is caught by the rangefinder, not by the estimate.** The wall guard held
  0.70 m while stock PX4 went to 0.42 m. Stock PX4 could only match this with a wall-aware
  hold of its own.
- **A frozen estimate is invisible to both, until the module looks for it.** No trip fired;
  both wandered about half a metre and stayed 0.94 m from the wall. The frozen-fix trip
  (`SH_WD_FRZ_T`, above, off by default) caught it in 1 s and the guarded hover then moved
  0.2 m; turning it on is Zain's call.
- **A long link loss is not survived by either, for two reasons.** With vision as EKF2's
  height reference the height estimate freezes and the vehicle can sink unseen: partners
  should fly the barometer (or a downward rangefinder) as the height reference, and the
  benchmark now does too (Zain's call, 2026-11-01: `px4_tuned_v2.params`). With
  the height fixed, the decided handback to Altitude mode lets the wind push the vehicle
  into the wall; the range hover (`SH_HB_RANGE`, above) kept it 0.94 m off instead. Turning
  it on changes the decided handback order, so it ships off by default (Zain's call).

## Next

- More seeds and the other wall scenarios, on the real image (GPU server or Mac), before any
  number here goes into a pitch.
- The frozen-fix trip over more seeds and a longer freeze, with and without `SH_HB_RANGE`.
- The link-loss pair again on `px4_tuned_v2.params` (Zain chose barometer height for the
  benchmark on 2026-11-01), and the range hover (`SH_HB_RANGE`) over more seeds, winds and
  longer silences.
- The GPS-on hand-over fault (GPS returning and the estimate jumping to it) waits for Zain:
  should partners ever fly with GPS on?
