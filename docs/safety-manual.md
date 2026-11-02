# Structure Hold: safety manual and failure table (FMEA)

Draft for a drone maker's safety review. Gap 4 of `docs/market-research.md`. Owner: Zain;
it is not a certified document and nobody has signed it. The interfaces it refers to (inputs,
modes, trips, parameters, messages) are specified in `docs/integration-guide.md`.

**Status, in one line: every result here is from simulation (the mock, PX4 SIH and Gazebo
SITL) and unit tests. Structure Hold has not flown on a real drone.** Each row of the failure
table names the test behind it, so a reviewer can rerun it, and says plainly where there is
no test yet.

## 1. What Structure Hold is, and is not

Structure Hold is one onboard PX4 flight mode (a plain C++ module in PX4 v1.17.0, registered
as an external mode, `sim2real/px4/structure_hold/`). It holds a multicopter still 1 to 3 m
from a flat structure (a wall, a pier, a deck underside) in gusts, without GPS, on the
position estimate the drone already has (VIO into EKF2). A small learned controller (a
policy, an 8k-parameter neural network compiled into the firmware, no runtime and no heap)
commands body rates and collective thrust to PX4's own rate controller. A classical
watchdog with no learning in it watches every tick and hands control back to stock PX4
modes when anything is wrong.

It is not: a navigation system, VIO or SLAM, obstacle avoidance, a planner, a landing
system for unknown ground, or a replacement for the pilot. It does not make a bad position
estimate good; it is built to stay safe when the estimate is bad for a while, and to hand
back when it is bad for too long (`docs/decisions.md`, scope rows).

## 2. Operating envelope

Flown and tested in simulation inside this envelope. Outside it, nothing is claimed.

| Item | Envelope | Where it comes from |
| --- | --- | --- |
| Airframe | A quadrotor like the PX4 Gazebo x500 (2 kg, 0.36 m prop reach). Another airframe needs its own fit and test | `envs/structure_hold/airframes.py`, `docs/partner-pack.md` step 0 |
| Flight stack | PX4 v1.17.0 exactly, multicopter, the module built in (`boards/structurehold.px4board` on 2 MB flight controllers) | `docs/decisions.md` (PX4 pin) |
| Position source | VIO (or another external position) into EKF2 as `ODOMETRY` VISION/VIO, 20 Hz or faster, latency set in `EKF2_EV_DELAY` | `docs/partner-pack.md` step 2 |
| Height source | Barometer (`EKF2_HGT_REF` 1) or a downward rangefinder. **Not vision** (see F5) | `eval/baselines/px4_tuned_v2.params`, `docs/field-faults.md` |
| GPS | Off (`EKF2_GPS_CTRL` 0). Flying with GPS on is untested (F23) | `docs/field-faults.md`, Next |
| Forward rangefinder | Required for the wall guard and the guarded modes: forward-facing `distance_sensor`, reads 0.3 to 5 m on the wall | `docs/partner-pack.md` step 2 |
| Distance from the structure | Engage 1 to 3 m from it, facing it (body x within 35 deg of the wall normal) | Gate 1 and Gate 2 scenarios |
| Wind | Mean wind up to 8 m/s with gusts, from any direction including straight into the wall | `docs/gate1.md`, `docs/baseline.md` |
| Estimate faults trained on | Delay 50 to 150 ms, noise, bias walk, jumps, dropouts up to 3 s | `envs/estimate`, `docs/decisions.md` |
| Height above ground | Tested 2 to 3.5 m. Engaging on the ground is refused | `sitl_test.py` ground-engage check |

### Do not use Structure Hold

- Without a pilot holding the sticks and the mode switch, able to take over in Position mode.
- With vision as EKF2's height reference (`EKF2_HGT_REF` 3).
- Without a working forward rangefinder pointing at the structure, unless the reviewer
  accepts losing every guard that uses it (wall guard, Structure Hover, Structure Land's
  guard, `SH_HB_RANGE`).
- Next to a structure the rangefinder cannot see reliably: thin poles, mesh, glass, water
  spray, or a face more than 35 deg off the heading.
- In mean wind above 8 m/s, in rain or icing, or on an airframe it has not been fitted to.
- With a VIO whose latency is unknown or above about 150 ms.
- With people closer than 5 m, or between the drone and the structure.
- Near magnetic disturbance (steel) on a magnetometer heading, unless the VIO's yaw is fused.

## 3. Safety design

Each layer works without the one above it. Only the first contains learning.

| Layer | What it does | Learned? |
| --- | --- | --- |
| Policy | Holds the point in wind and through short estimate faults | Yes |
| Last-step check | A non-finite rate becomes 0, a non-finite thrust the hover thrust; rates clamped to the policy's scale | No |
| Brake-and-hover fallback | After `SH_FB_T` of stale fix, brakes to a level hover (`docs/fallback.md`). Off by default for v22, which flies dropouts itself; `SH_FB_EN` 1 turns it on | No |
| Watchdog | Trips on: test fault, invalid estimate (EKF2 flags or NaN), stale estimate (`SH_WD_EST_TO`), frozen fix (`SH_WD_FRZ_T`, off by default), tilt (`SH_WD_TILT`), hold error (`SH_WD_POS_ERR`). A trip latches until disarm | No |
| Handback order | Decided rows, pinned by `test_handback.cpp`: next to a measured wall, Structure Hover; else Position (Altitude if no horizontal estimate), then Hold, then Land, Land repeated until it takes. A trip in Structure Land goes to stock Land first | No |
| Wall guard | The hold point never comes closer than `SH_WALL_MIN` (0.7 m) to the wall the rangefinder measures, whatever the estimate says | No |
| Guarded modes | Structure Hover and Structure Land: PX4's own position controller, with the wall guard on the setpoint | No |
| Entry checks | Refuses to engage on the ground, while a trip is latched, (optionally) right after an EKF reset (`SH_RST_HOLD`) or without a usable range (`SH_WALL_REQ`) | No |
| Parameter clamps | Every `SH_` parameter clamped to its range by the module (`ParamLimits.hpp`); PX4 does not enforce limits | No |
| Ground station | Every safety action is sent to QGroundControl as STATUSTEXT (section 4) | No |
| PX4 itself | Stock failsafes (battery, RC loss, geofence) and the pilot's mode switch override the module at any time | No |

## 4. What the pilot sees, and what to do

| Message in QGroundControl | Severity | Meaning | Pilot action |
| --- | --- | --- | --- |
| `Structure Hold: watchdog trip: <code> (<reason>), handing back` | critical | The watchdog stopped the policy. The reason names the limit, e.g. `tilt (tilted past SH_WD_TILT)` | Let the handback finish, or take Position mode. Back off and land |
| `Structure Hold: locked out until disarm` | info | After a trip the mode cannot be re-engaged in this flight | Land and disarm before trying again; write the trip in the log |
| `Structure Hold: handing back to <mode>` | info | Which stock or guarded mode was asked for. Repeated `(repeated until it takes)` on the last one (Land) | If it is Land and you do not want to land, take Position mode |
| `Structure Hold: engage refused: <code> (<reason>)` | critical | The switch into the mode was refused, e.g. `landed (on the ground)` | Fix the cause; stay in your stock mode |
| `Structure Hold: engage refused, EKF reset its position N s ago` | critical | `SH_RST_HOLD` is running after an EKF reset | Wait, then engage again |
| `Structure Hold: wall guard moved the hold point X m back` | warning | The estimate put the hold point too close to the measured wall; it was moved back | Check the VIO; expect the drone to step back |
| `Structure Hold: policy produced a non-finite action` | critical | The last-step check replaced the output | Take Position mode and land; send the log |
| `Structure Hover: position lost, holding X m from the wall on the rangefinder` | critical | `SH_HB_RANGE` 1: the horizontal estimate is gone and Structure Hover holds the distance on the range alone. It slides along the wall with the wind | Take over and land soon; the hold along the wall is not controlled |
| `Structure Hover: position back, holding on the estimate` | info | The estimate came back | Continue or land |

`sitl_test.py` checks the trip and wall-guard messages arrive in every SIH run.

## 5. Failure table (FMEA)

Severity of the effect if nothing caught it: **4** wall contact or crash, **3** large
uncommanded move or loss of height, **2** a degraded hold that stays clear, **1** nuisance.
"Residual" is the severity left with the default parameters and the envelope above.
Evidence: **U** unit test (`sim2real/px4/tests/`), **S** PX4 SIH check (`sitl_test.py`),
**G** Gazebo SITL flight, **M** mock simulator, **none** not tested yet. Gazebo field-fault
results are one seed, one scenario unless stated (`docs/field-faults.md`).

### Position estimate

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Estimate snaps back to the VIO origin in flight | VIO reset | Stock PX4 climbed 2.2 m and slid 0.39 m toward the wall | 3 | Hold error over `SH_WD_POS_ERR` | Trip, Structure Hover: moved 0.09 m, 0.99 m from the wall | 1 | G (`field_reset`) |
| F2 | VIO restarts: a gap, then the same snap | VIO process crash | As F1 | 3 | As F1 | As F1: moved 0.08 m, 0.95 m from the wall | 1 | G (`field_restart`) |
| F3 | Short dropout, under 3 s | Feature loss, link hiccup | Drift while EKF2 dead-reckons | 3 | Trained on (stale flag in the observation) | Policy holds through it | 2 | M (Gate 1 stress, capped 2 to 3 s cell passes) |
| F4 | Long silence, over 5 s | Companion computer or link lost | Drift into the wall | 4 | Stale or invalid estimate trip | Default: Altitude mode; any wind into the wall carries it there (8 of 12 flights touched, every one at 4 and 8 m/s). With `SH_HB_RANGE` 1: 0 of 12, closest 0.84 m, all airborne. Tuned PX4: 6 of 12 touched, all 12 landed by failsafe | **4 by default, 2 with `SH_HB_RANGE` 1** | G (`field_silence`, barometer height, 3 winds x 4 seeds) |
| F5 | Height estimate freezes during a silence | `EKF2_HGT_REF` 3 (vision height): the baro change goes into its bias | Sank to the ground unseen in 5.5 s | 4 | Nothing in the loop can see it | Prevented by the envelope: barometer height held 2.39 to 2.45 m | 1 with baro height | G (`field_silence`, vision vs baro) |
| F6 | Fix keeps arriving but stops moving | Stuck camera pipeline | Wandered 0.55 m (stock PX4 0.46 m) | 3 | `SH_WD_FRZ_T` (off by default) | Default: nothing. With 1 s: trip, Structure Hover, moved 0.21 m, 0.97 m from the wall | **3 by default, 2 with `SH_WD_FRZ_T`** | U, G (`field_freeze`) |
| F7 | Slow drift toward or away from the wall | VIO scale drift, a source walking off | Stock PX4 followed it to 0.42 m (props at 0.40 m) | 4 | Rangefinder (wall guard) | Stopped at `SH_WALL_MIN` 0.70 m. Drift along the wall or away from it is not caught | 2 | G (`field_drift`) |
| F8 | Random jumps, noise, bias walk | A working but imperfect VIO | A looser hold | 2 | Trained on | Gate 2: v22 p95 54 to 55 % below tuned PX4 (8 seeds) | 1 | M, G (`docs/baseline.md`) |
| F9 | Hold point captured on a jumped estimate at engage | Engage right after an EKF reset | Hold point on the wall face, contact | 4 | Wall guard; `SH_RST_HOLD` (off by default) | The guard moves the point back to `SH_WALL_MIN` | 2 | U, G (`docs/baseline.md`, wall guard on seed 3) |
| F10 | Latency above the trained range | Wrong `EKF2_EV_DELAY`, slow VIO | Oscillation, crash (an early policy crashed 68 % at 150 to 250 ms in the mock) | 4 | Tilt or hold-error trip, late | Handback | 3: keep inside the envelope | M (`docs/gate1.md` stress) |
| F11 | EKF2 flags the estimate invalid, or a NaN | Estimator fault | Policy flies garbage | 4 | Invalid-estimate trip, same tick | Handback | 1 | U |
| F12 | Heading at engage is wrong | Magnetometer near steel | Wall normal wrong: guard and hold axes off | 3 | Rangefinder gate (35 deg) only | The guard stops using ranges beyond 35 deg; the hold keeps the wrong axes | 3: fuse VIO yaw | none |

### Rangefinder

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F13 | No range, or broken readings (0, below minimum, beyond maximum, flagged invalid) | Sensor fault, bad surface | A false push or a missing guard | 3 | Samples gated: age under 0.2 s, inside the sensor's range, not flagged, angle under 35 deg | The guard does nothing on a bad sample; `SH_WALL_REQ` 1 refuses to engage without a usable range | 2 | U, S (`--range-faults`) |
| F14 | Range lost while Structure Hover holds on it alone | Sensor fault during `SH_HB_RANGE` | Uncontrolled drift | 4 | No usable range for 0.5 s | `range_lost` trip, then Altitude mode, Hold, Land | 3 | U |
| F15 | Range sees something other than the wall | Thin pole, mesh, glass, a person | Guard holds off the wrong thing, or not at all | 3 | Nothing | As the range says | 3: keep inside the envelope | none |

### Controller and module

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 | Policy commands a large tilt or oscillates | Outside its training: airframe, wind, latency | Crash | 4 | Tilt trip (`SH_WD_TILT` 45 deg), hold-error trip (2 m) | Handback | 2 | U, S (an early policy on the SIH quad tripped `tilt` in 1.5 s, LOG.md 2026-10-22) |
| F17 | Non-finite policy output | Numerical fault | Motors commanded with NaN | 4 | Last-step check | Rate 0, hover thrust, critical message | 1 | U |
| F18 | Out-of-range or NaN parameter | Operator error | A negative gain flies | 4 | `ParamLimits.hpp` clamps every `SH_` value | Clamped or default, with a warning | 1 | U (`test_params.cpp`) |
| F19 | Engage on the ground | Pilot error | Takes off toward a hold point on the ground | 3 | Land detector | Refused (`landed`), idle thrust, not latched | 1 | S (every run) |
| F20 | Engage while moving | Re-engage in a gust | Overshoot | 2 | Policy | Stops and holds without a trip | 1 | S (`--moving-engage`) |
| F21 | Disarm or kill switch while engaged | Pilot action | Controller keeps running on the ground | 3 | Arming state | Stops, clears the latch | 1 | U, S |
| F22 | Commander refuses a handback target | No RC for Position, no position for Hold | Stuck in the mode | 4 | Target not taken after a short wait | Next target; Land asked for again until it takes | 1 | U (`test_handback.cpp`) |
| F23 | Trip during Structure Land | Estimate fault in the landing | Hovers in Position mode near the ground | 2 | Watchdog | Stock Land first (decided 2026-10-28) | 1 | U, G |
| F24 | Control step overruns | Slow flight controller | Late commands | 3 | Loop time in `structure_hold_status`; worst step in `structure_hold status` | Nothing automatic | 2 | Bench test (`docs/bench-test.md`) |
| F25 | Firmware policy differs from the released one | Build error | Untested behaviour | 4 | `test_policy_header.py`, `test_policy.cpp` vectors in CI | Build fails CI | 1 | CI |

### Outside the module

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F26 | Low battery, RC loss, geofence | PX4 failsafes | — | — | PX4 commander | PX4's failsafe overrides the external mode; the module stops when commander leaves it | as stock PX4 | PX4 |
| F27 | GPS returns and the estimate jumps to it | GPS on, hand-over | A jump of metres | 3 | As F1 | Untested | 3: GPS off (envelope) | none |
| F28 | Air pushed back off the wall (near-wall flow) | Prop wash on the face | Pulled toward the wall | 3 | Hold-error trip, wall guard | Model exists, off by default; v24 candidate tests it | 3 | M (`docs/near-wall-flow.md`) |

### Fitting the simulator to a new drone (offline)

These faults happen on the ground, not in flight. The danger is a simulator that does not
match the drone, and a policy trained on it being released for that drone.

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F29 | Wrong drag coefficient from the wind fit | Wrong mass entered: drag scales with it | Policy trained on the wrong wind response | 3 | Release rule: beat the released policy on `gazebo_check`, then a Gazebo re-fly; partner bench and tethered steps before any scored flight | 2 | `eval/fit_wind.py`, `docs/wind-fit.md` |
| F30 | Wind heading and crossflow turned | Heading error in the logged attitude (compass near steel) | As F29 | 3 | As F29 | 2 | as F29 |
| F31 | Turbulence strength and time constant off by 10 to 30 % | A 60 s hold shows only part of the spectrum | Too mild or too harsh training wind | 2 | Fit several holds; as F29 | 2 | as F29 |
| F32 | Max thrust wrong | Thrust curve fitted over the narrow band an autotune flies, then extrapolated | Policy expects authority the drone lacks in strong gusts | 3 | As F29; tilt and hold-error trips in flight | 2 | `eval/fit_airframe.py`, `docs/airframe-fit.md` |
| F33 | Axis cross-coupling missing | Measured but not modelled in the env (the x500 shows about 10 % yaw into roll) | Hold looser than the mock predicts | 2 | As F29 | 2 | as F32 |
| F34 | Wrong rate gains in the env | A parameter file that is not the one flown | Policy tuned to a different rate loop | 3 | As F29 | 2 | as F32 |
| F35 | Log without the needed topics | `SDLOG_PROFILE` not 147 | No fit | 1 | The fit tool refuses the log | 1 | as F32 |

## 6. What a reviewer should set or decide

| Setting | Default | Recommendation | Why |
| --- | --- | --- | --- |
| `EKF2_HGT_REF` | PX4's | 1 (baro) or a downward rangefinder | F5 |
| `SH_WALL_MIN` | 0.7 m | Prop reach plus at least 0.3 m | F7, F9 |
| `SH_WALL_REQ` | 0 | 1 | F13 |
| `SH_HB_RANGE` | 0 (Proposed) | 1 if the rangefinder is reliable on your structure | F4 drops from 4 to 2 |
| `SH_WD_FRZ_T` | 0 (Proposed) | 1 s | F6 |
| `SH_WD_EST_TO`, `SH_WD_POS_ERR`, `SH_WD_TILT` | 5 s, 2 m, 45 deg | 1 m and 30 deg for first flights | Earlier trips while building trust |
| `SH_RST_HOLD` | 0 | Operator choice | F9 (stock PX4 has the same settling problem) |

`SH_HB_RANGE` and `SH_WD_FRZ_T` change the decided handback behaviour, so they ship off until
Zain turns them on (`docs/decisions.md`, Proposed rows of 2026-11-01).

## 7. Open items

- **No real flight.** Gate 3 is a partner's drone (`docs/partner-pack.md`). Every residual
  above is a simulation number.
- Field faults are one seed and one scenario each; F4 and F6 need more seeds and winds
  (queued).
- No test yet for F12 (wrong heading), F15 (wrong surface), F27 (GPS hand-over).
- The rest of the assurance pack in gap 4: an ASTM F3201 matrix, the argument for the
  learned part, an SBOM and a statement of origin.

## Change control

Change a row when its evidence changes, in the same PR as the code or result, and keep the
test reference current. A change to the handback order or a watchdog default is a decision
row first (`docs/decisions.md`).
