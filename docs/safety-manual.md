# Structure Hold: safety manual and failure table (FMEA)

| | |
| --- | --- |
| Document | Structure Hold safety manual, with the failure table (FMEA) |
| Version | 1.1, ready for sign-off (section 8) |
| Date | 2026-11-02 |
| Checked against | `main` at `868aeb4`: module parameters, trips, handback order and pilot messages as in `sim2real/px4/structure_hold/`; results as in `docs/baseline.md` and `docs/field-faults.md` |
| Owner | Zain |
| Companion | `docs/integration-guide.md` (the interface: inputs, modes, trips, parameters, messages) |

This is the safety case a drone maker's reviewer reads before flying Structure Hold (gap 4
of `docs/market-research.md`). It says where the mode may be used, how it is built to fail
safely, what the pilot sees, and every failure we know of with the test behind it. It is
our own engineering document, not a certification.

**Status, in one line: every result here is from simulation (the mock, PX4 SIH and Gazebo
SITL) and unit tests. Structure Hold has not flown on a real drone.** Each row of the failure
table names the test behind it, so a reviewer can rerun it, and says plainly where there is
no test yet.

### Words used

- **VIO**: visual-inertial odometry, the camera-based position source the drone already has.
- **EKF2**: PX4's state estimator; it fuses the VIO with the IMU and the barometer.
- **Estimate**: EKF2's position, the only position the module sees.
- **Policy**: the small learned controller inside the module.
- **Trip**: the watchdog stopping the policy and handing control back.
- **Handback**: the switch from Structure Hold to a stock or guarded mode after a trip.
- **Structure Hover, Structure Land**: the module's two guarded modes. PX4's own position
  controller flies them; the module only keeps their target point off the wall.
- **p95**: the hold error that 95 % of the flight stays inside.

## 1. What Structure Hold is, and is not

Structure Hold is one onboard PX4 flight mode (a plain C++ module in PX4 v1.17.0, registered
as an external mode, `sim2real/px4/structure_hold/`). It holds a multicopter still 1 to 3 m
from a flat structure (a wall, a pier, a deck underside) in gusts, without GPS, on the
position estimate the drone already has (VIO into EKF2). The policy is a small neural
network (about 25,000 numbers, compiled into the firmware: no runtime, no memory allocated
in flight). It commands body rates and total thrust to PX4's own rate controller. A
classical watchdog, with no learning in it, checks every control step and hands control back
to stock PX4 modes when anything is wrong.

It is not: a navigation system, VIO or SLAM, obstacle avoidance, a planner, a landing
system for unknown ground, or a replacement for the pilot. It cannot make a bad position
estimate good. It is built to stay safe while the estimate is bad for a short time, and to
hand back when it is bad for too long (`docs/decisions.md`, scope rows).

## 2. Operating envelope

Tested in simulation inside this envelope. Outside it, nothing is claimed.

| Item | Envelope | Where it comes from |
| --- | --- | --- |
| Airframe | A quadrotor like the PX4 Gazebo x500 (2 kg, 0.36 m prop reach). Another airframe needs its own fit and test | `envs/structure_hold/airframes.py`, `docs/airframe-fit.md`, `docs/partner-pack.md` step 0 |
| Flight stack | PX4 v1.17.0 exactly, multicopter, with the module built in (`boards/structurehold.px4board` on 2 MB flight controllers) | `docs/decisions.md` (PX4 pin) |
| Position source | VIO (or another external position) into EKF2 as `ODOMETRY` VISION/VIO, 20 Hz or faster, its latency set in `EKF2_EV_DELAY` | `docs/partner-pack.md` step 2 |
| Height source | Barometer (`EKF2_HGT_REF` 1) or a downward rangefinder. **Not vision** (see F5) | `eval/baselines/px4_tuned_v2.params`, `docs/field-faults.md` |
| GPS | Off (`EKF2_GPS_CTRL` 0). Flying with GPS on is untested (F27) | `docs/field-faults.md`, Next |
| Forward rangefinder | Required for the wall guard and the guarded modes: a forward-facing `distance_sensor` that reads 0.3 to 5 m on the wall | `docs/partner-pack.md` step 2 |
| Obstacle sensors | Optional: PX4 collision-prevention data (`obstacle_distance` from a 360 deg lidar or a companion, side-facing `distance_sensor`s). With them the keep-out holds the hold point `SH_KO_DIST` (default: PX4's `CP_DIST`) from obstacles other than the structure. Tested in SIH with one simulated lidar and one pole | `docs/decisions.md` (keep-out row), `sitl_test.py --keep-out` |
| Distance from the structure | Engage 1 to 3 m from it, facing it (nose within 35 deg of square to the wall) | Gate 1 and Gate 2 scenarios |
| Wind | Mean wind up to 8 m/s with gusts. The mock trains every direction; Gazebo flew 0, 4 and 8 m/s straight into the wall, the hardest case | `docs/gate1.md`, `docs/baseline.md` |
| Estimate faults trained on | Delay 50 to 150 ms, noise, slow bias drift, jumps, dropouts up to 3 s | `envs/estimate`, `docs/decisions.md` |
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
| Last-step check | A non-finite rate becomes 0 and a non-finite thrust becomes hover thrust; rates are clamped to the policy's range | No |
| Brake-and-hover fallback | After `SH_FB_T` (0.5 s) without a fresh fix, brakes to a level hover (`docs/fallback.md`). Off for v22, which flies dropouts itself (`SH_FB_EN` -1 follows the policy's own setting); `SH_FB_EN` 1 turns it on | No |
| Watchdog | In Structure Hold it trips on: a test fault, an invalid estimate (EKF2 flags or a NaN), a stale estimate (`SH_WD_EST_TO`, 5 s), a frozen fix (`SH_WD_FRZ_T`, 1 s), tilt (`SH_WD_TILT`, 45 deg), hold error (`SH_WD_POS_ERR`, 2 m). In Structure Hover on the rangefinder alone it trips when the range is lost for 0.5 s. A trip latches until disarm | No |
| Handback order | Decided rows, pinned by `test_handback.cpp`. Next to a measured wall: Structure Hover (with `SH_HB_RANGE` 1 also when the position estimate is gone). Otherwise Position mode (Altitude mode if there is no horizontal estimate), then Hold, then Land, with Land asked for again until it takes. A trip in Structure Land goes to stock Land first | No |
| Wall guard | The hold point never comes closer than `SH_WALL_MIN` (0.7 m) to the wall the rangefinder measures, whatever the estimate says | No |
| Keep-out | With collision-prevention sensors, the hold point (in all three modes) never comes closer than the keep-out to an obstacle they see; the structure ahead keeps `SH_WALL_MIN` instead. It only moves the point away, it does not avoid or plan | No |
| Guarded modes | Structure Hover and Structure Land: PX4's own position controller, with the wall guard on its target point. With `SH_HB_RANGE` 1, Structure Hover holds the wall distance on the rangefinder alone while the position estimate is gone | No |
| Entry checks | Refuses to engage on the ground, while a trip is latched, on an invalid or frozen estimate, and (optionally) right after an EKF reset (`SH_RST_HOLD`) or without a usable range (`SH_WALL_REQ`) | No |
| Parameter clamps | The module clamps every `SH_` parameter to its range (`ParamLimits.hpp`); PX4 itself does not enforce limits | No |
| Ground station | Every safety action is sent to QGroundControl as a text message (section 4) | No |
| PX4 itself | Stock failsafes (battery, RC loss, geofence) and the pilot's mode switch override the module at any time | No |

## 4. What the pilot sees, and what to do

| Message in QGroundControl | Severity | Meaning | Pilot action |
| --- | --- | --- | --- |
| `Structure Hold: watchdog trip: <code> (<reason>), handing back` | critical | The watchdog stopped the policy. The reason names the limit, e.g. `tilt (tilted past SH_WD_TILT)` | Let the handback finish, or take Position mode. Back off and land |
| `Structure Hold: locked out until disarm` | info | After a trip the mode cannot be engaged again in this flight | Land and disarm before trying again; note the trip in the flight log |
| `Structure Hold: handing back to <mode>` | info | Which stock or guarded mode was asked for. The last one (Land) shows `(repeated until it takes)` | If it is Land and you do not want to land, take Position mode |
| `Structure Hold: engage refused: <code> (<reason>)` | critical | The switch into the mode was refused, e.g. `landed (on the ground)` | Fix the cause; stay in your stock mode |
| `Structure Hold: engage refused, EKF reset its position N s ago (SH_RST_HOLD N s)` | critical | `SH_RST_HOLD` is waiting out an EKF reset | Wait, then engage again |
| `Structure Hold: wall guard moved the hold point X m back, to Y m from the wall` | warning | The estimate put the hold point too close to the measured wall; it was moved back | Check the VIO; expect the drone to step back |
| `Structure Hold: keep-out moved the hold point X m, to Y m from the nearest obstacle` | warning | An obstacle seen by the collision-prevention sensors was inside the keep-out; the hold point moved away from it | Check what it is; expect the drone to step away |
| `Structure Hold: keep-out has no room, holding between obstacles` | critical | Obstacles on both sides are closer together than twice the keep-out | Take Position mode and fly out |
| `Structure Hold: keep-out X m is on but no obstacle data is coming in` | warning | At engage: the keep-out is set but no lidar scan or side range is arriving | Check the obstacle sensors, or set `SH_KO_DIST` 0 if the drone has none |
| `Structure Hold: fallback, braking on a stale estimate` / `fresh fix, policy flies again` | warning | The brake-and-hover fallback started or ended (only if `SH_FB_EN` turns it on) | Watch the drift; take over if the fix does not come back |
| `Structure Hold: policy produced a non-finite action` | critical | The last-step check replaced the output. It should never happen | Take Position mode and land; send the log |
| `Structure Hold: engaged on the ground (SH_BENCH 1), props off only` | warning | Bench-test setting left on | Props off. Set `SH_BENCH` 0 before any flight |
| `Structure Hover: position lost, holding X m from the wall on the rangefinder` | critical | `SH_HB_RANGE` 1: the horizontal estimate is gone and Structure Hover holds the distance on the range alone. It slides along the wall with the wind | Take over and land soon; nothing controls the position along the wall |
| `Structure Hover: position back, holding on the estimate` | info | The estimate came back | Continue or land |

`sitl_test.py` checks that the trip and wall-guard messages arrive in every SIH run, and
`sim2real/px4/tests/test_safety_manual.py` checks every message above against the module's
code.

## 5. Failure table (FMEA)

How to read it. **Sev** is how bad the effect would be if nothing caught the failure:
**4** wall contact or crash, **3** a large unplanned move or loss of height, **2** a looser
hold that stays clear of the wall, **1** a nuisance. **Residual** is how bad it still is
with the default parameters, inside the envelope above. **Evidence** is the test behind the
row: **U** unit test (`sim2real/px4/tests/`), **S** PX4 SIH check (`sitl_test.py`), **G**
Gazebo SITL flight, **M** mock simulator, **none** not tested yet.

Gazebo field faults (`docs/field-faults.md`) are one flight on one seed and one scenario (1 m
from the wall, 4 m/s wind blowing into it) unless a row says otherwise. F1, F2, F6 and F7
were flown on the earlier vision-height set-up; F4 was flown on barometer height, as the
envelope requires, over three winds and four seeds.

### Position estimate

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Estimate snaps back to the VIO origin in flight | VIO reset | Stock PX4 climbed 2.2 m (EKF2's vision height snapped too) and slid 0.39 m toward the wall | 3 | Hold error over `SH_WD_POS_ERR` | Trip, Structure Hover: moved 0.09 m, 0.99 m from the wall | 1 | G (`field_reset`) |
| F2 | VIO restarts: a gap, then the same snap | VIO process crash | As F1 | 3 | As F1 | As F1: moved 0.08 m, 0.95 m from the wall | 1 | G (`field_restart`) |
| F3 | Short dropout, under 3 s | Feature loss, link hiccup | Drift while EKF2 dead-reckons | 3 | Trained on (the policy sees how old the fix is) | Policy holds through it | 2 | M (Gate 1 stress: the 2 to 3 s cell passes) |
| F4 | Long silence, over 5 s | Companion computer or link lost | Drift into the wall | 4 | Invalid or stale estimate trip | Default (`SH_HB_RANGE` 1): Structure Hover holds the wall distance on the range; 0 of 12 flights touched, closest 0.84 m, all still flying. With `SH_HB_RANGE` 0: Altitude mode, and any wind into the wall carries it there (8 of 12 touched, every one at 4 and 8 m/s). Tuned PX4: 6 of 12 touched, all 12 landed by its failsafe | **2 by default, 4 with `SH_HB_RANGE` 0** | G (`field_silence`, barometer height, 3 winds x 4 seeds) |
| F5 | Height estimate freezes during a silence | `EKF2_HGT_REF` 3 (vision height): EKF2 puts the barometer's change into a bias | Sank to the ground unseen in 5.5 s | 4 | Nothing in the loop can see it | Prevented by the envelope: on barometer height it held 2.39 to 2.45 m | 1 with barometer height | G (`field_silence`, vision vs barometer) |
| F6 | Fix keeps arriving but stops moving | Stuck camera pipeline | Wandered 0.55 m (stock PX4 0.46 m) | 3 | Frozen-fix trip, `SH_WD_FRZ_T` (1 s) | Default: trip after 1 s, Structure Hover, moved 0.21 m, 0.97 m from the wall. With `SH_WD_FRZ_T` 0: nothing catches it | **2 by default, 3 with `SH_WD_FRZ_T` 0** | U, G (`field_freeze`) |
| F7 | Slow drift toward or away from the wall | VIO scale drift, a source walking off | Stock PX4 followed it to 0.42 m (props at 0.40 m) | 4 | Rangefinder (wall guard) | Stopped at `SH_WALL_MIN` 0.70 m. Drift along the wall, or away from it, is not caught | 2 | G (`field_drift`) |
| F8 | Random jumps, noise, slow bias drift | A working but imperfect VIO | A looser hold | 2 | Trained on | Gate 2 on barometer height: v22 p95 43 to 45 % below tuned PX4, 8 seeds, 0 crashes (54 to 55 % on the earlier vision-height set-up) | 1 | M, G (`docs/baseline.md`) |
| F9 | Hold point taken from a jumped estimate at engage | Engage right after an EKF reset | Hold point on the wall face, contact | 4 | Wall guard; `SH_RST_HOLD` (off by default) | The guard moves the point back to `SH_WALL_MIN` | 2 | U, G (`docs/baseline.md`, wall guard on seed 3) |
| F10 | Latency above the trained range | Wrong `EKF2_EV_DELAY`, slow VIO | Oscillation, crash (an early policy crashed 68 % of runs at 150 to 250 ms in the mock) | 4 | Tilt or hold-error trip, late | Handback | 3: keep inside the envelope | M (`docs/gate1.md` stress) |
| F11 | EKF2 flags the estimate invalid, or a NaN | Estimator fault | Policy flies on garbage | 4 | Invalid-estimate trip, same step | Handback | 1 | U |
| F12 | Heading at engage is wrong | Magnetometer near steel | Wall direction wrong: guard and hold axes off | 3 | Only the rangefinder's 35 deg limit | The guard stops using ranges beyond 35 deg; the hold keeps the wrong axes | 3: fuse VIO yaw | none |

### Rangefinder

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F13 | No range, or broken readings (0, below minimum, beyond maximum, flagged invalid) | Sensor fault, bad surface | A false push or a missing guard | 3 | Each reading is checked: under 0.2 s old, inside the sensor's range, not flagged, angle under 35 deg | The guard does nothing on a bad reading; `SH_WALL_REQ` 1 refuses to engage without a usable range | 2 | U, S (`--range-faults`) |
| F14 | Range lost while Structure Hover holds on it alone | Sensor fault during `SH_HB_RANGE` | Uncontrolled drift | 4 | No usable range for 0.5 s | `range_lost` trip, then Altitude mode (Position mode if the estimate is back), Hold, Land | 3 | U |
| F15 | Range sees something other than the wall | Thin pole, mesh, glass, a person | Guard holds off the wrong thing, or not at all | 3 | Nothing | Follows what the range says | 3: keep inside the envelope | none |

### Obstacle sensors (keep-out)

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F36 | An obstacle beside or behind the hold point, inside the keep-out | Engaged next to a pole, pipe or side wall; the estimate drifts toward it | Contact with something other than the wall | 4 | Keep-out on the collision-prevention sensors (`SH_KO_DIST`, or `CP_DIST` by default) | Moves the hold point straight away until it is the keep-out from the obstacle; warning. Without obstacle sensors nothing catches it | **2 with obstacle sensors, 4 without** | U, S (`--keep-out`: pole at 0.5 m, keep-out 1 m) |
| F37 | Obstacles on two sides closer together than twice the keep-out | Narrow gap, corridor | Contact on either side | 4 | Keep-out has no room | Holds between them; critical message | 3: pilot flies out | U |
| F38 | An obstacle keeps coming (a person walking up) | Moving obstacle | The hold point is pushed away with no limit, possibly toward something no sensor sees | 3 | Nothing beyond the first warning; the hold-error trip moves with the point | Keeps stepping away | 3: no people within 5 m (envelope) | U (ratchet) |
| F39 | False obstacle points | Dust, spray, the drone's own props or legs in the lidar's view | A false push away from nothing | 2 | Bins at or below the sensor's minimum or at its maximum are ignored; scans over 0.5 s old are ignored | Moves away from the false point only | 2 | U |
| F40 | Keep-out set but no obstacle data | Sensor unplugged, wrong MAVLink routing | No keep-out | 3 | Warning at engage | Holds as without the keep-out | As F36 without sensors | U |

### Controller and module

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 | Policy commands a large tilt or oscillates | Outside its training: airframe, wind, latency | Crash | 4 | Tilt trip (`SH_WD_TILT` 45 deg), hold-error trip (2 m) | Handback | 2 | U, S (an early policy on the SIH quad tripped `tilt` in 1.5 s, LOG.md 2026-10-22) |
| F17 | Non-finite policy output | Numerical fault | Motors commanded with NaN | 4 | Last-step check | Rate 0, hover thrust, critical message | 1 | U |
| F18 | Out-of-range or NaN parameter | Operator error | A negative gain flies | 4 | `ParamLimits.hpp` clamps every `SH_` value | Clamped or default, with a warning | 1 | U (`test_params.cpp`) |
| F19 | Engage on the ground | Pilot error | Takes off toward a hold point on the ground | 3 | Land detector | Refused (`landed`), idle thrust, not latched | 1 | S (every run) |
| F20 | Engage while moving | Re-engage in a gust | Overshoot | 2 | Policy | Stops and holds without a trip | 1 | S (`--moving-engage`) |
| F21 | Disarm or kill switch while engaged | Pilot action | Controller keeps running on the ground | 3 | Arming state | Stops, clears the latch | 1 | U, S |
| F22 | PX4 refuses a handback mode | No RC for Position, no position for Hold | Stuck in the mode | 4 | Mode not taken after a short wait | Next mode in the order; Land asked for again until it takes | 1 | U (`test_handback.cpp`) |
| F23 | Trip during Structure Land | Estimate fault in the landing | Hovers in Position mode near the ground | 2 | Watchdog | Stock Land first (decided 2026-10-28) | 1 | U, G |
| F24 | Control step overruns | Slow flight controller | Late commands | 3 | Loop time in `structure_hold_status`; worst step in `structure_hold status` | Nothing automatic | 2 | Bench test (`docs/bench-test.md`) |
| F25 | Firmware policy differs from the released one | Build error | Untested behaviour | 4 | `test_policy_header.py` and the `test_policy.cpp` test vectors in CI | CI fails the build | 1 | CI |

### Outside the module

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | What the drone does | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F26 | Low battery, RC loss, geofence | PX4 failsafes | — | — | PX4 commander | PX4's failsafe overrides the external mode; the module stops when PX4 leaves it | As stock PX4 | PX4 |
| F27 | GPS returns and the estimate jumps to it | GPS on, hand-over | A jump of metres | 3 | As F1 | Untested | 3: GPS off (envelope) | none |
| F28 | Air pushed back off the wall (near-wall flow) | Prop wash on the face | Pulled toward the wall | 3 | Hold-error trip, wall guard | The model exists but is off by default; the v24 candidate tests it | 3 | M (`docs/near-wall-flow.md`) |

### Fitting the simulator to a new drone (offline)

These faults happen on the ground, not in flight. The danger is a simulator that does not
match the drone, and a policy trained on it being released for that drone.

| # | Failure | Cause | Effect if not caught | Sev | How it is caught | Residual | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F29 | Wrong drag coefficient from the wind fit | Wrong mass entered: drag scales with it | Policy trained on the wrong wind response | 3 | Release rule: beat the released policy on `gazebo_check`, then a Gazebo re-fly; partner bench and tethered steps before any scored flight | 2 | `eval/fit_wind.py`, `docs/wind-fit.md` |
| F30 | Wind heading and crossflow turned | Heading error in the logged attitude (compass near steel) | As F29 | 3 | As F29 | 2 | As F29 |
| F31 | Turbulence strength and time constant off by 10 to 30 % | A 60 s hold shows only part of the gust spectrum | Training wind too mild or too harsh | 2 | Fit several holds; as F29 | 2 | As F29 |
| F32 | Max thrust wrong | Thrust curve fitted over the narrow band an autotune flies, then extrapolated | Policy expects power the drone lacks in strong gusts | 3 | As F29; tilt and hold-error trips in flight | 2 | `eval/fit_airframe.py`, `docs/airframe-fit.md` |
| F33 | Axis cross-coupling missing | Measured but not modelled in the env (the x500 shows about 10 % yaw into roll) | Hold looser than the mock predicts | 2 | As F29 | 2 | As F32 |
| F34 | Wrong rate gains in the env | A parameter file that is not the one flown | Policy tuned to a different rate loop | 3 | As F29 | 2 | As F32 |
| F35 | Log without the needed topics | `SDLOG_PROFILE` not 147 | No fit | 1 | The fit tool refuses the log | 1 | As F32 |

## 6. What a reviewer should set or decide

| Setting | Default | Recommendation | Why |
| --- | --- | --- | --- |
| `EKF2_HGT_REF` | PX4's | 1 (barometer) or a downward rangefinder | F5 |
| `SH_WALL_MIN` | 0.7 m | Prop reach plus at least 0.3 m | F7, F9 |
| `SH_WALL_REQ` | 0 | 1 | F13 |
| `SH_HB_RANGE` | 1 | Keep 1 if the rangefinder is reliable on your structure | F4 is 2 with it, 4 without |
| `SH_WD_FRZ_T` | 1 s | Keep 1 s | F6 |
| `SH_WD_EST_TO` | 5 s | Keep | F4 |
| `SH_WD_POS_ERR` | 2 m | 1 m for first flights | Earlier trips while building trust |
| `SH_WD_TILT` | 45 deg | 30 deg for first flights | Earlier trips while building trust |
| `SH_KO_DIST` | -1 | Keep -1 and set PX4's `CP_DIST` if the drone has collision-prevention sensors | F36 |
| `SH_RST_HOLD` | 0 s | Operator choice | F9 (stock PX4 has the same settling problem) |
| `SH_BENCH` | 0 | Always 0 in flight | Bench test only (`docs/bench-test.md`) |

`SH_HB_RANGE` and `SH_WD_FRZ_T` are on by default since 2026-11-02 (`docs/decisions.md`);
setting either to 0 goes back to the earlier behaviour in F4 and F6. The defaults in this
table are checked against the module by `sim2real/px4/tests/test_safety_manual.py`.

## 7. Open items

- **No real flight.** Gate 3 is on a partner's drone (`docs/partner-pack.md`). Every
  residual above is a simulation number.
- F1, F2, F6 and F7 are one flight each on the earlier vision-height set-up; they should be
  re-flown on barometer height over more seeds and winds. F4 has seeds 1 to 4; seeds 5 to 8
  and longer silences are next (`docs/field-faults.md`, Next).
- No test yet for F12 (wrong heading), F15 (wrong surface) or F27 (GPS hand-over). The
  keep-out (F36 to F40) is tested in unit tests and one SIH lidar case, not in Gazebo or
  with a real lidar's noise.
- The rest of the assurance pack in gap 4: an ASTM F3201 matrix, the argument for the
  learned part, a software bill of materials and a statement of origin.

## 8. Sign-off

Signing means: I have read this manual, the envelope and the failure table match what was
built and tested at the commit named at the top, and the open items in section 7 are known
and accepted for the stated use (simulation results and a partner's supervised test flights
under `docs/partner-pack.md`). It does not certify the mode for unsupervised or commercial
operation.

| Role | Name | Signature | Date |
| --- | --- | --- | --- |
| Prepared by | | | |
| Approved by (owner) | Zain | | |
| Partner reviewer (optional) | | | |

Any change to sections 2 to 6 after signing makes a new version (1.1, 1.2, ...) that is
signed again. Record each one below.

| Version | Date | Commit | What changed | Signed by |
| --- | --- | --- | --- | --- |
| 1.0 | 2026-11-02 | `868aeb4` | First signed version: checked against the module after `SH_HB_RANGE` and `SH_WD_FRZ_T` went on by default | |
| 1.1 | 2026-11-02 | this PR | The keep-out from PX4's collision-prevention data (`SH_KO_DIST`): envelope row, safety layer, three pilot messages, F36 to F40, section 6 row | |

## Change control

Change a row when its evidence changes, in the same PR as the code or result, keep the test
reference current, and add a line to the version table in section 8. A change to the
handback order or a watchdog default is a decision row first (`docs/decisions.md`).
