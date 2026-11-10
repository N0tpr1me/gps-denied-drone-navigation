# Structure Hold: integration guide and interface spec

> **Confidential. Copyright (c) 2026 Zain Ali. All rights reserved.** For a drone maker
> who has signed an evaluation or licence agreement with us, and nobody else.

This is the contract between the Structure Hold module and the PX4 drone it runs on: what
the module needs from the drone, what it gives back, its modes, its parameters, what it
tells the pilot, and what it costs the flight controller. It is written for the drone
maker's firmware and flight-test engineers. The step-by-step bring-up (build, bench,
tethered checks, scored flights) is `docs/partner-pack.md`; the hazards, their causes and
what covers each one are in the safety manual, `docs/safety-manual.md`. Read both before
the first flight.

Everything here is about module behaviour as of policy v22 (`training/policies/README.md`)
on PX4 v1.17.0. The code is the authority: the parameter table below is checked against
`sim2real/px4/structure_hold/structure_hold_params.c` by
`sim2real/px4/tests/test_integration_guide.py`, and the handback order against
`Handback.hpp` by `tests/test_handback.cpp`.

## In one picture

```
 your VIO ──ODOMETRY──▶ EKF2 ──vehicle_local_position──┐
 forward rangefinder ──distance_sensor─────────────────┤
 obstacle sensors (optional) ──obstacle_distance───────┤
 gyro, attitude, land detector, hover thrust ──────────┤
                                                       ▼
                                  Structure Hold module (one work-queue item, 100 Hz)
          pass path ─ policy (MLP) ─ fallback ─ watchdog ─ wall guard ─ keep-out
                                                       │
         Structure Hold (ext1):  vehicle_rates_setpoint ──▶ PX4 rate controller ──▶ mixer
         Structure Land/Hover (ext2/ext3): trajectory_setpoint ──▶ PX4 position controller
         on a fault: vehicle_command (mode change) ──▶ commander ──▶ stock PX4 mode
                                                       │
                       structure_hold_status (logged) + STATUSTEXT to the ground station
```

The module never replaces PX4's estimator, rate controller, mixer, failsafes or arming
checks. It is a flight mode like Position mode, registered through PX4's external mode
interface, and commander can take the vehicle out of it at any time. Through the same
interface it adds one arming check of its own (`SH_WD_GND_V`, 0 turns it off): while the
vehicle sits disarmed and at rest and the VIO reports that it is moving, commander refuses
to arm, in every mode.

## What the drone must provide

| Need | Requirement | Why |
| --- | --- | --- |
| Firmware | PX4 v1.17.0, or a fork of exactly that release, for everything in this guide and every hold number. The same module source also builds and passes its simulator acceptance flights on v1.16.2 and v1.18.0-rc1 (`docs/px4-versions.md`) | The interface below is the same on the three releases (the module adapts to v1.18's external-mode and parameter changes at build time), but the benchmark, the bench test and the safety manual are on v1.17.0 only; a real flight on another release is a re-check on our side first |
| Airframe | A multicopter that flies well in stock Position mode on its own VIO with GPS off | Structure Hold flies on top of your rate loop and your estimate; it does not fix either. GPS or RTK may be fused as well, with the forward rangefinder and `SH_WALL_REQ` 1 (`docs/field-faults.md`, "GPS on") |
| Flight controller | About 125 KB of free flash and a single-precision FPU | The policy's weights are compiled in. A 2 MB board needs the multicopter-only build (`sim2real/px4/boards/structurehold.px4board`) |
| Position estimate | VIO into EKF2 as external vision (`ODOMETRY`, `estimator_type` VISION or VIO), logged as `vehicle_visual_odometry` | The hold point and the policy's position error come from EKF2's `vehicle_local_position`; the VIO samples tell a fresh fix from a stale one |
| Heading | A heading that points the same way relative to the wall for the whole hold | The heading at engage is taken as the wall's normal (VIO yaw is safer than a compass near steel) |
| Forward rangefinder | `distance_sensor` with orientation forward (`ROTATION_NONE`), reading 0.3 m to at least 5 m on your structure | The wall guard, Structure Land and Structure Hover use it to keep off the wall. Optional to the code, required for flights near a structure (`SH_WALL_REQ` 1) |
| Obstacle sensors (optional) | PX4 collision-prevention data: `obstacle_distance` (a 360 deg lidar, or `OBSTACLE_DISTANCE` from a companion) and side-facing `distance_sensor`s, with `CP_DIST` set | The keep-out holds the hold point that far from any obstacle they see (`SH_KO_DIST`). Without them nothing changes |
| Pilot | An RC mode switch with Position mode on it, and a kill switch | Position mode is the pilot's way out of anything |

Height: fly the barometer (`EKF2_HGT_REF` 0) or a downward rangefinder (2) as EKF2's height
reference, not GPS (1, PX4's default) and not vision (3): with vision height, EKF2's height freezes when the VIO goes quiet and
the vehicle can sink unseen (`docs/field-faults.md`). The EKF2 settings to check are listed
in `docs/partner-pack.md`, step 2.

## Inputs (uORB topics the module reads)

| Topic | Used for | Freshness rule |
| --- | --- | --- |
| `vehicle_angular_velocity` | Wakes the module (at most every 2 ms) and the policy's body rates | The 10 ms control tick follows it |
| `vehicle_local_position` | Position, velocity, EKF2's validity flags and reset counters | Invalid xy or z, or a NaN, trips `estimate_invalid` at once. A heading reset (`heading_reset_counter`, `delta_heading`) turns the module's wall normal and yaw setpoint with it, so the vehicle does not turn; an xy reset moves Hover's and Land's setpoint, and is what `SH_RST_HOLD` waits out |
| `vehicle_attitude` | Attitude for the observation, the tilt trip and the heading at engage | A NaN heading refuses the engage |
| `vehicle_visual_odometry` | Fix freshness (`SH_FIX_SRC`), the frozen-fix trip | No new sample for `SH_FIX_GAP` marks the fix stale for the policy; for `SH_FB_T` starts the fallback; for `SH_WD_EST_TO` trips |
| `distance_sensor` (any instance) | The forward range for the wall guard, the range input of rangefinder policies, Structure Hover's range hold | A sample counts if under 0.2 s old, inside the sensor's own limits, not flagged bad, and the body x axis is within 35 deg of the wall normal |
| `obstacle_distance` | The keep-out (`SH_KO_DIST`, `KeepOut.hpp`): every bin above the sensor's minimum and below its maximum is an obstacle point; north-aligned (`MAV_FRAME_GLOBAL`, `LOCAL_NED`) or body-aligned (`MAV_FRAME_BODY_FRD`) | A scan counts if under 0.5 s old (PX4 collision prevention's own timeout) |
| `distance_sensor` facing 45 to 315 deg (yaw orientations 1 to 7) | Keep-out obstacle points, one per instance, on the sensor's bearing | Under 0.5 s old, inside the sensor's limits, not flagged bad |
| `hover_thrust_estimate` | Scales the policy's thrust to your drone | Falls back to `MPC_THR_HOVER` when invalid; clamped to 0.1 to 0.9 |
| `vehicle_land_detected` | Refuses an engage on the ground | |
| `vehicle_status` | Armed state and the active mode | A disarm ends the hold and any handback |
| `register_ext_component_reply`, `arming_check_request` | The external-mode handshake with commander | |

With no VIO odometry at all (`SH_FIX_SRC` 0 and no `vehicle_visual_odometry`), local
position updates count as fixes, which is weaker: EKF2 keeps publishing while it dead
reckons.

## Outputs (what the module publishes)

| Topic | When | Content |
| --- | --- | --- |
| `vehicle_rates_setpoint` | Every tick in Structure Hold | FRD body rates (rad/s, clamped by `SH_RATE_MAX` or the policy's own limit) and normalised collective thrust. PX4's rate controller and allocation fly them |
| `trajectory_setpoint` | Every tick in Structure Land and Structure Hover | xy position on the guarded point, yaw, descent rate (Land) or held height (Hover); acceleration-only xy in the range hold (`SH_HB_RANGE`) |
| `vehicle_command` | On a trip or refusal | `DO_SET_MODE` for each handback target in turn |
| `register_ext_component_request`, `arming_check_reply`, `config_control_setpoints` | At start and on every commander poll | Mode registration, availability, and which PX4 controllers commander must enable |
| `structure_hold_status` | Every tick while engaged, at a low rate otherwise; logged by default | State, trip reason, setpoint, error, commands, fallback, wall distance and guard shift, keep-out distance, nearest obstacle and shift, the surface pass's state and how far it has moved the hold point (`pass_state`, `pass_offset_ned`), loop time (`sim2real/px4/msg/StructureHoldStatus.msg`) |
| STATUSTEXT (via `mavlink_log`) | Whenever a safety function acts, at each engage, near a watchdog limit, and every `SH_GCS_PER` s while a mode flies | Below |
| `debug_array` (MAVLink `DEBUG_FLOAT_ARRAY` "SH_STATUS") | 5 Hz; PX4 streams it at 1 Hz on a ground-station link, 10 Hz onboard | The live numbers: state, hold error, wall distance, clearance, guard shift, fix age, time left before the no-fix trip, closest watchdog limit and how much of it is used, trip, tilt (`docs/pilot-status.md`) |

In Structure Hold the attitude and position controllers are idle: commander enables only
the rate loop and allocation for it. In Structure Land and Structure Hover commander runs
PX4's whole multicopter cascade and the module only moves the setpoint.

## Modes

| Mode in QGC | External mode | MAVLink `custom_mode` (main / sub) | `nav_state` | Selectable when | Who flies |
| --- | --- | --- | --- | --- | --- |
| Structure Hold | ext1 | AUTO (4) / 11 | 23 | Armed, in the air, no latched trip, not within `SH_RST_HOLD` of an EKF2 xy reset, a usable forward range if `SH_WALL_REQ` 1, local position valid | The policy, through PX4's rate loop |
| Structure Land | ext2 (`SH_LAND_EN` 1) | AUTO (4) / 12 | 24 | Armed and in the air, also after a latched trip | PX4's position controller on the guarded xy point |
| Structure Hover | ext3 (`SH_HB_GUARD` 1) | AUTO (4) / 13 | 25 | Armed and in the air, also after a latched trip; without a local position only with `SH_HB_RANGE` 1 | PX4's position controller on the guarded point |

Mode numbers are assigned by commander in registration order; with `SH_LAND_EN` 0 Structure
Hover moves up to ext2. On the PX4 shell: `commander mode ext1` / `ext2` / `ext3`.

**Engage.** The hold point and heading are where the vehicle is when the mode is switched
in. Fly to the standoff in Position mode, face the structure, switch in. A switch while
the vehicle still moves is fine: it stops and holds (checked at about 2 m/s in SITL,
`sitl_test.py --moving-engage`), but the hold point is where the switch happened.

**Surface pass** (`SH_PASS_LEN`, 0 = off by default; `docs/surface-passes.md`). With a
length set, each engage flies one pass: the hold point stays at the engage point for
`SH_PASS_T0` seconds, then moves `SH_PASS_LEN` metres in a straight line at right angles to
the heading at engage, level (positive to the right of the nose, negative to the left) or,
with `SH_PASS_DIR` 1, straight up (positive) or down. It speeds up at 0.5 m/s^2 to
`SH_PASS_SPD`, slows down the same way, and holds at the end. The nose stays on the engage
heading and the standoff is the one at engage, so square up to the face first: each 5 deg
off square changes the standoff by 0.09 m per metre flown. The four values are read at the
engage; changing them mid-pass changes nothing until the next one. The same policy flies
it: its velocity input is taken relative to the moving point, so nothing else changes.

- The wall guard and the keep-out act on the moving point, as on a still one.
- A short dropout of the position fix does not hold the pass back: EKF2 dead-reckons and
  the vehicle follows the point as before. Once the fix has been stale for 2 s the point
  brakes and waits, and goes on when the fix is fresh, so a pass through a long dropout
  takes that much longer. With the brake-and-hover fallback on (`SH_FB_EN` 1) it waits from
  `SH_FB_T` instead, because the vehicle brakes then.
- The pass ends early, braking to a stop where the point then stays for the rest of the
  hold, when an obstacle seen by the collision-prevention sensors is within the keep-out of
  where the point would come to rest (0.25 m ahead at 0.5 m/s, plus 0.1 m), when the wall
  guard has used up `SH_SHIFT_MAX`, or when a pass down comes within 1 m of the ground
  (PX4's `dist_bottom`, when valid). The pilot is told why.
- To stop a pass by hand, switch to Position mode. To fly the next one, engage again; set
  `SH_PASS_LEN` 0 for a plain hold. There is no raster or route: one straight line per
  engage, and whatever starts the next one (the pilot, a companion computer setting the
  parameter and the mode) is the integrator's.

**Disengage.** Switch to any other mode. Position mode is stock PX4 and always works.

**State** (`structure_hold_status.state`): 0 unregistered, 1 standby, 2 engaged,
3 handback, 4 landing, 5 hover.

## Watchdog trips and the handback

A trip hands control back to stock PX4 and latches until disarm: Structure Hold cannot be
switched in again on that flight. Checked every tick in this order:

| `trip_reason` | Name | Trips when | Latched |
| --- | --- | --- | --- |
| 1 | `test_fault` | `SH_TEST_FAULT` set (or `structure_hold trip`) | Yes |
| 2 | `estimate_invalid` | EKF2 flags xy or z invalid, or a NaN in position or velocity | Yes |
| 3 | `estimate_stale` | No fresh fix for `SH_WD_EST_TO` | Yes |
| 4 | `tilt` | Tilt over `SH_WD_TILT` | Yes |
| 5 | `position_error` | More than `SH_WD_POS_ERR` from the hold point (estimate) | Yes |
| 6 | `estimate_reset` | Engage refused: EKF2 reset its xy position less than `SH_RST_HOLD` ago | No |
| 7 | `landed` | Engage refused: the land detector says on the ground | No |
| 8 | `range_lost` | Structure Hover on the rangefinder alone lost its range for 0.5 s | Yes |
| 9 | `estimate_frozen` | The VIO fix arrives on time but has not moved for `SH_WD_FRZ_T` | Yes |
| 10 | `estimate_runaway` | The VIO's velocity changed more than the IMU's (EKF2's acceleration) over 3 s, by over `SH_WD_VIO_DV`, for 0.5 s: a VIO running away. Also trips Structure Hover unless it holds on the rangefinder (`SH_HB_RANGE` 1); the handback then treats the horizontal estimate as lost. Clears only when the VIO's ODOMETRY `reset_counter` changes (a VIO restart) or on disarm | Yes |

The handback order (decided rows in `docs/decisions.md`; `Handback.hpp`):

1. A trip in Structure Hold with a usable forward range seeing the wall within 5 m and a
   valid xy estimate: **Structure Hover** first. With `SH_HB_RANGE` 1 also without the xy
   estimate.
2. A trip in Structure Land: **stock Land** first.
3. Then, and for every other trip or refusal: **Position mode** (**Altitude mode** if the
   horizontal estimate is invalid), then **Hold**, then **Land**, each tried after the
   previous one did not take. Land is asked for again every 0.3 s until commander takes it
   or the pilot picks a mode. In between, the module holds the vehicle level at hover
   thrust.

The brake-and-hover fallback is not a handback: after `SH_FB_T` without a fresh fix the
module brakes on the estimated velocity and holds level, and the policy takes over again
on the first fresh fix (`docs/fallback.md`). It is on or off per policy (`SH_FB_EN` -1 uses
the policy's own setting; v22's is off).

PX4's own failsafes (RC loss, battery, geofence, position loss for `COM_POS_FS_DELAY`)
act on the external modes exactly as on stock modes, and they win.

## Parameters

All under `SH_`, in QGC's "Structure Hold" group. The module clamps every value to the
range below itself (PX4 does not), and a NaN or out-of-range choice takes the default with
a warning (`ParamLimits.hpp`). "Reboot" means the change applies after a reboot.

| Parameter | Default | Range | Unit | What it does |
| --- | --- | --- | --- | --- |
| `SH_EN` | 1 | 0 or 1 | | Start the module at boot (reboot) |
| `SH_WALL_MIN` | 0.7 | 0 to 5 | m | The hold point never comes closer than this to the measured wall; set your prop reach plus at least 0.3 m. 0 turns the guard off |
| `SH_KO_DIST` | -1.0 | -1 to 5 | m | Keep-out from obstacles PX4's collision-prevention sensors see (not the structure ahead, which keeps `SH_WALL_MIN`). -1 follows PX4's `CP_DIST` (off unless you set it); 0 off |
| `SH_WALL_REQ` | 0 | 0 or 1 | | 1: the mode is unavailable without a usable forward range. Set 1 on any drone with one |
| `SH_SHIFT_MAX` | 3.0 | 0 to 10 | m | Retreat cap: the wall guard and the keep-out each move the hold point at most this far in one hold, then stop and tell the pilot. Keep it above the biggest estimate jump your drone can meet (1.5 m with GPS on). 0 removes the cap |
| `SH_PASS_LEN` | 0.0 | -10 to 10 | m | Surface pass: after the engage the hold point moves this far along the wall at the same standoff, then holds there. Positive to the right of the nose, negative to the left (up and down with `SH_PASS_DIR` 1). 0 no pass: a plain hold. Read at each engage |
| `SH_PASS_DIR` | 0 | 0 or 1 | | Pass direction: 0 along the wall, 1 up or down it |
| `SH_PASS_SPD` | 0.5 | 0.1 to 1 | m/s | Pass cruise speed; it speeds up and slows down at 0.5 m/s^2. Tested at 0.5 m/s |
| `SH_PASS_T0` | 3.0 | 0 to 60 | s | Hold this long after the engage before the pass starts |
| `SH_LAND_EN` | 1 | 0 or 1 | | Register Structure Land, the guarded landing (reboot) |
| `SH_HB_GUARD` | 1 | 0 or 1 | | Register Structure Hover and hand back into it first next to a measured wall (reboot) |
| `SH_HB_RANGE` | 1 | 0 or 1 | | Structure Hover holds the wall distance on the rangefinder alone when the position estimate is gone |
| `SH_WD_EST_TO` | 5.0 | 0.05 to 10 | s | Watchdog: no fresh fix for this long trips |
| `SH_WD_POS_ERR` | 2.0 | 0.2 to 10 | m | Watchdog: this far from the hold point trips |
| `SH_WD_TILT` | 45.0 | 10 to 80 | deg | Watchdog: this much tilt trips |
| `SH_WD_FRZ_T` | 1.0 | 0 to 10 | s | A VIO fix that has not moved for this long trips. 0 off |
| `SH_WD_VIO_DV` | 0.8 | 0 to 10 | m/s | VIO runaway: the VIO's velocity change over 3 s may differ from the IMU's by this much. Checks only an ODOMETRY velocity in `MAV_FRAME_LOCAL_NED` or `MAV_FRAME_BODY_FRD` (`LOCAL_FRD` is not checked). Replay your own healthy logs first (`python -m eval.vio_runaway replay <ulog>`) and keep it at least twice their largest value. 0 off |
| `SH_WD_EV_CUT` | 1 | 0 or 1 | | On a VIO runaway caught while a Structure mode flies or hands back, clear `EKF2_EV_CTRL`'s position and velocity bits (yaw kept) so EKF2 stops fusing the VIO, which it otherwise follows with its attitude; the old value goes back when the VIO restarts (`reset_counter`) or on disarm. The change is a normal parameter write, so PX4 saves it; if the module stops mid-flight it puts it back. 0 leaves EKF2 alone |
| `SH_WD_GND_V` | 0.3 | 0 to 5 | m/s | Pre-arm VIO check: with the vehicle disarmed and at rest (the land detector's `at_rest`, IMU only), a VIO whose velocity averages more than this over 1 s, 2 s in a row, blocks arming in every mode through an arming check the module registers with commander (no mode attached, name "Structure Hold VIO"), with a critical message repeated every 15 s. Any velocity frame is checked (a speed is the same in all). Arming is allowed again after 10 s under the limit at rest, or 5 s after the VIO goes silent; a runaway caught by `SH_WD_VIO_DV` while disarmed blocks and clears the same way. Never acts once armed. 0 = off |
| `SH_RST_HOLD` | 0.0 | 0 to 60 | s | Refuse an engage this long after an EKF2 xy reset. 0 off |
| `SH_GCS_PER` | 10.0 | 0 to 60 | s | Period of the one-line status sent to the ground station while a mode flies (`docs/pilot-status.md`). 0 off; the engage line and near-limit warnings are sent regardless |
| `SH_FIX_SRC` | 0 | 0 to 2 | | What counts as a fix: 0 auto (VIO odometry if present, else local position), 1 VIO odometry, 2 local position |
| `SH_FIX_GAP` | 0.1 | 0.01 to 1 | s | A gap this long between fixes tells the policy its fix is stale; about twice your VIO's sample period |
| `SH_CTRL` | 1 | 0 or 1 | | 1 the policy; 0 the placeholder cascade (test only) |
| `SH_FB_EN` | -1 | -1 to 2 | | Brake-and-hover fallback: -1 the policy's default, 0 off, 1 on, 2 forced (test) |
| `SH_FB_T` | 0.5 | 0.05 to 5 | s | Stale time before the fallback brakes |
| `SH_FB_K` | 1.5 | 0.2 to 5 | 1/s | Fallback braking gain on the estimated velocity |
| `SH_FB_TILT` | 4.0 | 1 to 12 | 1/s | Fallback attitude gain |
| `SH_YAW_GUARD` | -1 | -1 to 1 | | Module holds the heading instead of the policy: -1 the policy's default, 0 off, 1 on |
| `SH_YAW_P` | 3.0 | 0.5 to 10 | 1/s | Heading gain of the yaw guard |
| `SH_RATE_MAX` | 0.0 | 0 to 6 | rad/s | Body rate clamp; 0 uses the policy's own action scale |
| `SH_TEST_FAULT` | 0 | 0 or 1 | | Test hook: trip the watchdog on the next tick |
| `SH_BENCH` | 0 | 0 or 1 | | Bench test only: allow engaging on the ground, props off |
| `SH_P_POS` | 0.95 | 0.1 to 3 | 1/s | Placeholder cascade (`SH_CTRL` 0) position gain |
| `SH_P_VEL` | 1.8 | 0.1 to 6 | 1/s | Placeholder cascade velocity gain |
| `SH_P_ATT` | 6.5 | 1 to 12 | 1/s | Placeholder cascade attitude gain |
| `SH_V_MAX` | 1.0 | 0.2 to 5 | m/s | Placeholder cascade speed limit |
| `SH_VEL_SRC` | 0 | 0 or 1 | | Placeholder cascade velocity: 0 estimator, 1 differentiated position |

For scored flights load `sim2real/partner/structure_hold_flight.params`, which sets every
`SH_` value the benchmark flew plus the logger (`docs/partner-pack.md`, step 2). The
rangefinder hover (`SH_HB_RANGE`) and the frozen-fix trip (`SH_WD_FRZ_T`) are on by default
since 2026-11-02 (`docs/decisions.md`); setting them to 0 goes back to Altitude mode after a
lost estimate and to no frozen-fix trip (`docs/field-faults.md`).

## What the pilot sees

QGroundControl shows these as STATUSTEXT messages, and they are in the log. The first
table is what happens when a safety function acts; the second is the live status before
that (`docs/pilot-status.md`, which also lists the live number array for the MAVLink
Inspector).

| Severity | Message | When |
| --- | --- | --- |
| Critical | `Structure Hold: watchdog trip: <name> (<reason in plain words, with its limit parameter>), handing back` | Any trip (names as in the trip table) |
| Critical | `Structure Hold: engage refused: <name> (<reason in plain words>)` | An engage refused (on the ground, frozen fix, a NaN estimate) |
| Critical | `Structure Hold: engage refused, EKF reset its position <t> s ago (SH_RST_HOLD <t> s)` | Within the reset hold-off |
| Critical | `Structure Hold: policy produced a non-finite action` | Should never happen; the setpoint is replaced by a safe one |
| Critical | `Structure Hold: keep-out has no room, holding between obstacles` | Obstacles closer together than twice the keep-out; the hold point sits between them |
| Critical | `Structure Hold: wall guard has moved the hold point <s> m back this hold (SH_SHIFT_MAX), not moving it further` | The retreat cap: the guard stops moving the point; the pilot takes over |
| Critical | `Structure Hold: keep-out has moved the hold point <s> m this hold (SH_SHIFT_MAX), not moving it further` | The same cap on the keep-out |
| Critical | `Structure Hover: position lost, holding <d> m from the wall on the rangefinder` | `SH_HB_RANGE` 1 only |
| Critical | `Structure Hold: VIO velocity ran <v> m/s from the IMU's (SH_WD_VIO_DV), VIO not trusted until it restarts` | The VIO runaway check, in any mode and on the ground |
| Critical | `Structure Hold: EKF2 stops fusing the VIO (EKF2_EV_CTRL <a> to <b>) until it restarts or disarm` | `SH_WD_EV_CUT` 1, a runaway caught while a Structure mode flies or hands back |
| Critical | `Structure Hold: arming blocked, the VIO reports motion with the vehicle on the ground (SH_WD_GND_V)` | The pre-arm VIO check, disarmed only; repeated every 15 s while it lasts |
| Warning | `Structure Hold: wall guard moved the hold point <s> m back, to <d> m from the wall` | The guard pushed the hold point |
| Warning | `Structure Hold: keep-out moved the hold point <s> m, to <d> m from the nearest obstacle` | The keep-out pushed the hold point (first push of each engage) |
| Warning | `Structure Hold: keep-out <d> m is on but no obstacle data is coming in` | At engage, with the keep-out on and no fresh `obstacle_distance` or side range |
| Warning | `Structure Hold: pass stopped after <s> m: <reason>` | A surface pass ended early (`keep-out`, `wall guard cap` or `ground`); the hold point stays where it stopped |
| Warning | `Structure Hold: fallback, braking on a stale estimate` / `fresh fix, policy flies again` | The fallback starts and ends |
| Warning | `Structure Hold: engaged on the ground (SH_BENCH 1), props off only` | Bench test |
| Info | `Structure Hold: pass <l> m at <v> m/s` / `pass done` | A surface pass starts moving (`SH_PASS_LEN`, `SH_PASS_SPD`) and reaches its end |
| Info | `Structure Hold: locked out until disarm` | After a latched trip |
| Info | `Structure Hold: handing back to <mode>` | Each handback target in turn |
| Info | `Structure Hover: position back, holding on the estimate` | `SH_HB_RANGE` 1 only |
| Info | `Structure Hold: EKF2 fuses the VIO again (EKF2_EV_CTRL <a>)` | The VIO restarted or the vehicle disarmed after the cut |
| Info | `Structure Hold: arming allowed again, the VIO is still or silent` | The pre-arm block lifted |

| Severity | Message | When |
| --- | --- | --- |
| Warning | `<mode>: no position fix for <t> s, watchdog hands back at <limit> s` | Half of `SH_WD_EST_TO` without a fix |
| Warning | `<mode>: position fix has not moved for <t> s, watchdog trips at <limit> s` | Half of `SH_WD_FRZ_T` (Structure Hold only) |
| Warning | `<mode>: tilted <a> deg, watchdog trips at <limit> deg` | Three quarters of `SH_WD_TILT` |
| Warning | `<mode>: <e> m off the hold point, watchdog trips at <limit> m` | Half of `SH_WD_POS_ERR` |
| Info | `<mode>: back inside the watchdog limits` | Every limit under 40 % again after a warning |
| Info | `<mode>: engaged, <d> m from the wall` (or `no wall range`, `wall guard off`) | Each mode as it starts |
| Info | `<mode>: <e> m off the hold point, <d> m from the wall, fix <t> s old` | Every `SH_GCS_PER` s while a mode flies |
| Info | `Structure Hover: on the rangefinder, <d> m from the wall, holding <r> m` | The same, on the rangefinder alone |

`<mode>` is Structure Hold, Structure Hover or Structure Land. Each limit warns once per
crossing and at most every 10 s, so a gusty tilt does not flood the link.

`structure_hold status` on the PX4 shell prints the state, the policy and its defaults,
what is flying, the worst control step since boot and the tick interval.

## Resource budget

| Resource | Cost | Where measured |
| --- | --- | --- |
| Flash | About 125 KB (mostly the policy's weights, `static constexpr`) | `sim2real/px4/README.md`, real flight controllers |
| RAM | The observation history and two MLP scratch rows (about 4 KB), no heap | `docs/bench-test.md`, CPU budget |
| Stack | A few locals on the work queue's stack; nothing large | Same |
| CPU | One policy step per 10 ms tick, estimated 150 to 300 us on an STM32H7 (1.5 to 3 % of a core), measured on your board by the bench test | `docs/bench-test.md` |
| Wake-ups | At most one per 2 ms whatever the gyro rate | Same |
| Log | `structure_hold_status` at the tick rate while engaged | `install.sh` adds it to the logger |
| Link | One `DEBUG_FLOAT_ARRAY` (about 250 bytes) per second on a ground-station link, plus a status line every `SH_GCS_PER` s | `docs/pilot-status.md` |

The bench test (`sim2real/px4/bench_test.py`) times the policy on your flight controller
and flags anything over 1 ms.

## Fitting it to your drone

The policy was trained on the PX4 Gazebo x500 with mass, motors, rate loop and wind
randomised around it. A drone outside that range may need a fine-tune. Two tools turn your
logs into simulator numbers, so we check and fine-tune on your drone before it flies:

- **Airframe fit** (`docs/airframe-fit.md`): one PX4 autotune flight with logging profile
  147, the parameter file and the weighed mass give your drone's thrust curve, motor lag,
  torque per axis and rate gains as an env airframe.
- **Wind fit** (`docs/wind-fit.md`): holds in free air and at several distances from a wall
  in a steady wind give the drag, turbulence and near-wall shelter your drone meets.

A fine-tuned policy ships as a new `PolicyWeights.hpp` in the module, released only after
it beats the released policy in our simulator checks and in Gazebo
(`training/policies/README.md`). The module reads what the header declares
(`HAS_ENGAGE_TIME`, `HAS_WALL_RANGE`), so the same module code flies every release.

## What it does not do

- It makes no position estimate, map or plan, and does no obstacle avoidance. It keeps the
  hold point off the measured wall (`SH_WALL_MIN`) and, with your collision-prevention
  sensors, at least `SH_KO_DIST` from any obstacle they see, by moving the point away from
  what is too close; it never steers round anything (`docs/decisions.md`, keep-out row).
  An obstacle that keeps coming keeps pushing the hold point away.
- A surface pass is one straight line along a flat face. It does not follow a curved or
  stepped surface, plan a raster, or look for obstacles beyond what PX4's
  collision-prevention sensors report; without those sensors nothing on board sees what is
  on the line (`docs/safety-manual.md`, F48).
- It does not choose the standoff: the hold point is where the vehicle was at engage,
  pushed back by the wall guard if too close.
- It does not cover stock modes: in Position mode, Land and failsafes PX4 flies as stock,
  without the wall guard (Structure Land and Structure Hover are the guarded versions).
- It does not detect a VIO that drifts smoothly; the frozen-fix trip catches only a fix
  that stops moving (`docs/field-faults.md`).

The full hazard list, with each one's cause, effect, what covers it and what the operator
must do, is the safety manual, `docs/safety-manual.md`.

## Integration checklist

1. Send us the airframe sheet, parameter dump and one logging flight (`docs/partner-pack.md`,
   step 0). We check your drone against the training range, with the airframe fit.
2. Build the module into your firmware (step 1); `structure_hold status` says `standby`.
3. Check EKF2, the forward rangefinder and the `SH_` parameters (step 2). Run
   `python -m eval.log_health <flight.ulg>` on one flight log: it says in plain English whether
   EKF2 really fuses your VIO, in the right frame, with the right heading source and
   `EKF2_EV_DELAY` (`docs/log-health.md`).
4. Bench test, props off (step 3, `docs/bench-test.md`).
5. Tethered checks with tight watchdog limits (step 4).
6. Scored sessions against your own tuned stock PX4 (steps 5 and 6).
