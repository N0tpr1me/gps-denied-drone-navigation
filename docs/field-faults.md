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
    --params eval/baselines/px4_tuned_v1.params --scenarios wall1m_wind_4ms --seeds 1 \
    -- --profile field_reset_at_engage --rangefinder
python -m eval.baselines.run_baseline --label field_reset_tuned \
    --params eval/baselines/px4_tuned_v1.params --scenarios wall1m_wind_4ms --seeds 1 \
    -- --profile field_reset_at_hover
python -m eval.field_faults runs/<hold flight> runs/<stock flight>
```

## First flights (2026-11-01, cloud)

Released policy v22 in the module with the forward rangefinder (`--rangefinder`, as the
partner pack requires) against tuned PX4 (`eval/baselines/px4_tuned_v1.params`, Position
mode, no rangefinder use: stock PX4 has none for xy). Scenario `wall1m_wind_4ms` (1 m from
the wall, 4 m/s mean wind blowing into it, gusts), seed 1 only, one flight per cell, on the
cloud-built image (`sim2real/docker/cloud_build.sh`: same PX4 v1.17.0, module and world, a
different Gazebo base). Treat these as a first look, not a benchmark claim: one seed, one
scenario.

| Fault | Runner | Closest to wall | Largest move | Lowest height | What happened |
| --- | --- | --- | --- | --- | --- |
| reset to origin | Structure Hold | 0.99 m | 0.08 m | 2.50 m | watchdog trip (`position_error`), handed back to Structure Hover, which held in place |
| reset to origin | tuned PX4 | 0.75 m | 0.39 m | 2.51 m | climbed 2.2 m (EKF2's height snapped to the takeoff height) and moved 0.39 m toward the wall |
| restart (3 s gap, then reset) | Structure Hold | 0.97 m | 0.10 m | 2.46 m | same trip and hover |
| restart (3 s gap, then reset) | tuned PX4 | 0.77 m | 0.37 m | 2.51 m | same climb of 2.2 m |
| drift 0.6 m away from the wall | Structure Hold | 0.70 m | 0.33 m | 2.50 m | no trip; the wall guard stopped the drift at `SH_WALL_MIN` 0.70 m |
| drift 0.6 m away from the wall | tuned PX4 | 0.42 m | 0.59 m | 2.46 m | followed the drift to 2 cm from prop contact |
| freeze 5 s | Structure Hold | 0.82 m | 0.78 m | 2.51 m | no trip: nothing in the module sees a frozen estimate |
| freeze 5 s | tuned PX4 | 0.94 m | 0.46 m | 2.51 m | wandered less than the module |
| silence 8 s | Structure Hold | **0.29 m (contact)** | 10.96 m | 2.04 m | EKF2 marked the position invalid, the watchdog tripped (`estimate_invalid`) and handed back to Altitude mode; the wind pushed it into the wall and along it |
| silence 8 s | tuned PX4 | **0.24 m (contact)** | 0.80 m | 0 m (landed) | PX4's failsafe descended; it touched the wall on the way down |

What this says, in plain words:

- **A VIO reset or restart is where the module earns its keep.** Stock PX4 climbed 2.2 m
  and slid 0.4 m toward the wall; the module tripped, handed back to its guarded hover and
  moved less than 10 cm. This is the most reported failure in the research.
- **Slow drift is caught by the rangefinder, not by the estimate.** The wall guard held
  0.70 m while stock PX4 went to 0.42 m. Stock PX4 could only match this with a wall-aware
  hold of its own.
- **A frozen estimate is invisible to both.** No trip fired. The module wandered a bit more
  than stock PX4 (0.78 m against 0.46 m) and stayed further than 0.8 m from the wall. A
  classical frozen-estimate check (the position not changing while the IMU says the vehicle
  moves) is the obvious next watchdog trip.
- **An 8 s link loss next to a wall in wind is not survived by either.** Once EKF2 has no
  horizontal position, the module hands back to Altitude mode by the decided handback order,
  and the 4 m/s wind blows the vehicle into the wall. Stock PX4 lands and touches the wall
  on the way down. A range-guarded altitude hold (push off the wall on the rangefinder alone
  when the position is gone) would be the fix; it changes the decided handback order, so it
  is Zain's call (question in the PR that added this page).

## Next

- More seeds and the other wall scenarios, on the real image (GPU server or Mac), before any
  number here goes into a pitch.
- A frozen-estimate trip in the watchdog (classical, cheap), flown on `field_freeze_*`.
- The GPS-on hand-over fault (GPS returning and the estimate jumping to it) waits for Zain:
  should partners ever fly with GPS on?
