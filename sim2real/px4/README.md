# Structure Hold PX4 module

One onboard PX4 flight mode, built into `px4_sitl` first and later into the target
airframe. Since 2026-10-24 it flies the released policy (v19, `training/policies/`, trained on the Gazebo x500; v10 from 2026-10-22 to 2026-10-24) with the
brake-and-hover fallback and a yaw guard available as options; the v0 placeholder cascade
stays selectable (`SH_CTRL 0`).

What it does today:

- **Registers "Structure Hold" as a named flight mode** with commander through PX4's
  external-component handshake (`register_ext_component_request` / `_reply`,
  `arming_check_request` / `_reply`, `config_control_setpoints`), driven from inside the
  firmware the way `mc_nn_control` does. On the pinned release (v1.17.0) the mode gets
  `nav_state` 23 (EXTERNAL1), shows up in QGC and on an RC mode switch, and PX4's
  failsafes and arming checks apply to it. No commander patch needed.
- **Holds the position and yaw it had when engaged** with the policy
  (`PolicyController.hpp`, `SH_CTRL 1`, default) or the placeholder P cascade
  (`HoldController.hpp`, `SH_CTRL 0`). Both output FRD body rates and a normalised
  collective thrust on `vehicle_rates_setpoint`. PX4's stock rate controller and control
  allocation stay in charge; the attitude and position controllers are idle while the mode
  is active.
- **Policy inference** is a compiled-in MLP (`MlpPolicy.hpp`, no heap, no PX4 headers, no
  third-party library; docs/decisions.md). `tools/export_policy_header.py` turns the
  released `training/policies/structure_hold_v19.onnx` + `.json` into
  `structure_hold/PolicyWeights.hpp`: weights, the history length (15, from
  `policy.json input.history`, never hardcoded), the observation layout, the action scaling,
  the mock airframe the policy was trained on, and the policy's own defaults for the two
  options below (`output.yaw`, `output.fallback`, `output.max_rate_rad_s`). The module builds the env's observation
  (structure frame = heading at engage, ENU-style, FLU body) from PX4's NED / FRD state
  (the age input is the env's: 0 while the fix is fresh, the time since the last fix while
  stale; the sample latency is reported in the status message only),
  keeps the rolling history, and maps the action back (FLU → FRD rates, thrust in newtons for
  the mock airframe scaled through the hover thrust). `tests/test_policy_header.py` fails
  when the header no longer matches the released policy.
- **Per-policy defaults from `policy.json`, overridable in the field.** `SH_YAW_GUARD`
  and `SH_FB_EN` default to -1 (the policy's own default), `SH_RATE_MAX` to 0 (the
  policy's action scale); 0 / 1 (and `SH_FB_EN 2`) override them without a rebuild. v3
  declares `output.yaw = "policy"` (it holds the heading itself; the guard drives it into
  the wall, 71 % crashes in the mock), `output.fallback = "off"` (it flies through dropouts
  itself; the rule costs it 0.06 m p95 and 1.1 points of crash rate) and 2 rad/s actions.
  For v2, whose `policy.json` has neither key, the guard is on, the fallback on, 6 rad/s.
- **Yaw guard** (`SH_YAW_GUARD`, `SH_YAW_P`): the module holds the wall-facing heading
  captured at engage with its own P loop and ignores the policy's yaw output. Needed by v2
  (no heading term), harmful to v3. The body-rate setpoints are clamped by `SH_RATE_MAX`
  (v2 flew SIH only at 1 to 2 rad/s, see below).
- **Brake-and-hover fallback** (docs/fallback.md) once the position fix has been stale for
  `SH_FB_T` (0.5 s): acceleration setpoint `-SH_FB_K × v_est` (1.5, retuned from 2.5 for the
  lagged rate loop) clamped at 6 m/s², no position term, P attitude loop (`SH_FB_TILT` 4),
  heading held on the structure; hands back on the first fresh fix. The policy keeps running
  meanwhile so its history stays current. A fix is a new `vehicle_visual_odometry` sample
  (VIO / the estimate bridge) or, with no odometry source, a local position update
  (`SH_FIX_SRC`). `SH_FB_EN 2` forces the fallback every tick for tests.
- **Watchdog** (`Watchdog.hpp`, plain C++, no PX4 headers, unit-tested): trips on a
  test fault, an invalid or stale estimate (no fix for `SH_WD_EST_TO`, 5 s so the fallback
  can fly through dropouts; EKF2-invalid flags trip at once), too much tilt or too much
  hold error, in that priority order. A trip latches until disarm. On a trip the module hands back:
  Position mode (Altitude mode if the horizontal estimate is gone), then Hold, then Land
  if the previous request was refused, holding level at hover thrust in between; Land is
  asked for again every 0.3 s until commander takes it (`Handback.hpp`, unit-tested step by
  step against the decided rows).
- **Hardening (Phase 2, 2026-11)**, each with a unit test or an SIH check:
  - every `SH_` parameter is clamped to its documented range by the module itself
    (`ParamLimits.hpp`; PX4 does not enforce `@min`/`@max`), a NaN takes the default and an
    out-of-range enum the default, with a warning; a negative `SH_YAW_P` or `SH_FB_K`, a NaN
    `SH_FIX_GAP` or an `SH_CTRL` of 5 can no longer fly. The hover thrust (estimate or
    `MPC_THR_HOVER`) is clamped to 0.1 to 0.9, a NaN estimate falls back to the parameter;
  - the mode cannot be armed into, and a switch into it on the ground (land detector) is
    refused and handed back at idle thrust (`trip_reason` `landed`, 7, not latched): the
    hold point would be the ground and the policy never learned a take-off;
  - a disarm while engaged (kill switch) or during a handback ends it; before, the
    controller kept running on the ground and a trip there stayed latched into the next flight;
  - a NaN in the position or velocity estimate counts as invalid (`estimate_invalid`), a
    NaN heading or estimate at the switch refuses the engage, and every rate setpoint goes
    through one last check (a non-finite rate becomes 0, a non-finite thrust the hover thrust);
  - Structure Hover and Structure Land move their setpoint with EKF2's height and heading
    resets as well as xy ones, as PX4's own modes do (Hover held the old height before);
  - `Run()` wakes at most every 2 ms (a flight controller can publish the gyro at up to
    8 kHz; the tick stays 10 ms), and `structure_hold status` prints the worst control step
    since boot and the tick interval (`docs/bench-test.md`, CPU budget).
- **Engage hold-off after an EKF reset** (`SH_RST_HOLD`, seconds; default 0 = off): for
  that long after EKF2 reset its horizontal position (`vehicle_local_position.xy_reset_counter`
  changed) the mode's arming check reports it unavailable, so commander keeps the current
  mode, and if the switch slipped through anyway (commander polls every 300 ms) the module
  hands straight back with `trip_reason` `estimate_reset` (6), nothing latched. A hold point
  captured while the estimate settles after a reset is not where the vehicle is: v11
  re-engaged 0.1 to 0.4 s after a 1.1 m vision reset, captured a point on the wall face and
  struck the wall (`docs/baseline.md`, v11 section). It ships off because stock PX4
  captures the same settling estimate: in the mock replay of that sequence a 10 s hold-off
  left v10 at 92 % crashes against 91 % without it, most of them moved into the stock
  phase. It is an operator choice, not a fix; a fix needs a direct distance to the
  structure (`docs/real-flight-gap.md`). A reset during a hold never trips the watchdog;
  the policy is trained on them. `structure_hold_status.reset_age_s` says how long ago the
  last reset was (−1 if none). `sitl_test.py` retries its scripted re-engage once a second
  for up to 14 s, and records the refusals and the wait in `summary.json`.
- **Wall guard: a measured floor under the hold point** (`WallGuard.hpp`, plain C++,
  unit-tested; `SH_WALL_MIN`, default 0.7 m, 0 = off; `SH_WALL_REQ`). With a forward-facing
  rangefinder on `distance_sensor` (any instance, orientation forward; a downward one for
  EKF2's height is skipped), each tick turns the range into the hold point's distance to
  the wall along the engage heading, `range × cos(body x, normal) − (hold − pos)·normal`,
  and moves the hold point straight back when that is under `SH_WALL_MIN`. The estimate's
  own offset cancels in that sum, so it is the case the hold-off above cannot fix: a hold
  point captured on a jumped estimate that lies on the wall face (Gazebo seed 3,
  `docs/baseline.md` v11 section). The point only ever moves away from the wall during a
  hold. Samples count only when under 0.2 s old, inside the sensor's own range, not flagged
  bad, and with the body x axis within 35 deg of the wall normal; without one the guard
  does nothing, so the Gazebo x500 (no rangefinder) flies exactly as before. The policy sees
  the move as a step in its position error, like an estimate jump. `SH_WALL_REQ 1` makes
  the mode unavailable while no usable forward range comes in. Stock Position mode
  between two engages is not covered, and PX4's Collision Prevention (`CP_DIST`) does not
  cover it either: in v1.17 it only scales a deflected stick's acceleration
  (`CollisionPrevention::_calculateConstrainedSetpoint`), so with centred sticks the
  position lock follows a recovering estimate into the wall. `structure_hold_status` gains
  `wall_distance_m`, `hold_clearance_m` and `standoff_shift_m`.
- **Keep-out from PX4's collision-prevention data** (`KeepOut.hpp`, plain C++,
  unit-tested; `SH_KO_DIST`, default -1 = PX4's `CP_DIST`, 0 = off; decided 2026-11-02).
  PX4's own collision prevention only bends the sticks in Position mode, so inside the
  module's modes a drone maker's obstacle sensors did nothing. Each tick every bin of a
  fresh `obstacle_distance` scan (a 360 deg lidar, or `OBSTACLE_DISTANCE` from a companion;
  north- or body-aligned) and every side-facing `distance_sensor` (yaw orientations 1 to 7)
  becomes an obstacle point, and the hold point of Structure Hold, Hover and Land moves
  straight away from any point closer than the keep-out, never back (a ratchet). Points on
  the structure ahead (within 0.3 m of the wall plane the forward rangefinder or the lidar's
  forward 10 deg measures) are not obstacles; they keep `SH_WALL_MIN` instead, so a drone
  with a lidar and no forward rangefinder gets a wall floor too. Obstacles on both sides
  are solved together (averaged projections), so a narrow gap leaves the point between them
  with a "no room" message rather than flicking it side to side. Not used by Structure Hover
  on the range alone. `structure_hold_status` gains `keep_out_m`, `obstacle_clearance_m`,
  `keep_out_shift_m`, `keep_out_points` and `keep_out_no_room`.
- **Structure Land: a landing that keeps the wall guard** (`SH_LAND_EN`, default on; a
  second registered external mode, ext2, named "Structure Land"). After a handback on a
  jumped estimate, stock Land holds xy on the estimate, and as the estimate recovers the
  vehicle walks toward the wall: on Gazebo seed 3 every flight came closest to the wall
  under stock Land, 2 of 3 inside the 0.25 m contact radius (`docs/baseline.md`, wall
  guard section). Structure Land hands the flying to PX4's own position controller
  (commander enables the whole cascade for it) and publishes only a `trajectory_setpoint`:
  xy on the point captured at entry, descent at `MPC_LAND_SPEED`, the yaw at entry. The
  wall guard runs on that xy point every tick as in the hold, so the forward range keeps it
  `SH_WALL_MIN` from the wall down to touchdown. PX4's land detector and touchdown work as
  in stock Land (they see the climb-rate flag and the commanded descent). An EKF2 xy
  reset moves the point with the estimate, as PX4's own modes do. Available only in
  flight, and also while a trip is latched (that is when it is needed); its watchdog trips
  (stale or invalid estimate, tilt, xy error over `SH_WD_POS_ERR`) hand back to stock Land
  first, since the landing was already asked for, then Position mode, Hold and Land as for
  the hold (Zain's call: on the near-wall-flow Gazebo batch a tilt trip at the start of the
  landing left the vehicle hovering 0.12 m up in Position mode). Without a usable range it is a plain position-holding landing.
  `structure_hold_status.state` is `landing` (4) while it flies. The guard's push rate,
  low-passed over 0.3 s, goes in as xy velocity feed-forward (position-only setpoints left
  PX4's loop trailing the pushed point: 0.37 m closest on Gazebo seed 3 at 8 m/s), and the
  descent slows from `MPC_LAND_SPEED` to 0.3 m/s while the vehicle is 0.15 to 0.4 m closer
  to the wall than the point.
- **Structure Hover: the guarded handback** (`SH_HB_GUARD`, default on; a third mode, ext3).
  The same guarded setpoint without the descent, height held. A trip in Structure Hold hands
  back into it first when a usable forward range sees the wall within 5 m, then falls
  through to Position mode (then Hold, then Land) if commander refuses it. Stock Position
  mode after a trip on a jumped estimate creeps toward the wall as the estimate recovers
  (Gazebo seed 3: 0.59 to 0.31 m before the landing); here the range keeps the point off
  the wall. The pilot leaves it for Position mode as usual. State `hover` (5). Chosen by
  Zain over handing straight into Structure Land, or no change (2026-10-28).
- **Structure Hover on the rangefinder alone** (`SH_HB_RANGE`, default on since
  2026-11-02; Decided, `docs/decisions.md`). With the horizontal estimate gone the decided handback
  goes to Altitude mode, which keeps the vehicle level while a wind into the wall pushes it
  onto the face (`docs/field-faults.md`, the link-loss fault). With 1, a trip next to a
  measured wall goes to Structure Hover even without a horizontal estimate, and Structure
  Hover, whenever the estimate is invalid or more than 1 s stale, holds the measured wall
  distance on the forward range alone (never under `SH_WALL_MIN`), leaning into the wind it
  held against before the estimate went, with the height on PX4's own altitude control
  (`RangeHover.hpp`: an acceleration-only xy `trajectory_setpoint`, as Altitude mode flies).
  The estimate back and fresh for 1 s: it holds a point on the estimate again. No usable
  range for 0.5 s trips it (`range_lost`) to Altitude mode, Hold, Land. Structure Hover then
  no longer needs a local position to be selected.
- **Frozen-fix trip** (`SH_WD_FRZ_T`, seconds, default 1.0 since 2026-11-02,
  0 = off; Decided, `docs/decisions.md`). A stuck VIO pipeline can keep sending its last pose on
  time: nothing goes stale and EKF2 keeps it valid (`docs/field-faults.md`, the freeze
  fault). With a limit set, a VIO odometry fix that has not moved by 0.1 mm for that long
  (timed by its own sample times; `FrozenFix` in `Watchdog.hpp`) trips Structure Hold
  (`estimate_frozen`, trip 9) and refuses an engage. Not a trip in Structure Hover or
  Land, where Position mode would only fly the same frozen estimate without the wall guard;
  with `SH_HB_RANGE` 1 Structure Hover counts a frozen fix as no estimate and holds the wall
  distance on the rangefinder until the fix moves again.
- **Publishes `structure_hold_status`** every control tick while engaged (state, trip
  reason, setpoint, error, commanded rates and thrust, controller, fallback active, yaw
  guarded, stale time, action flown, loop time), logged by default. `structure_hold status`
  on the PX4 shell prints the policy's defaults and what is actually flying.
- **Tells the ground station** (MAVLink STATUSTEXT through PX4's `mavlink_log`, shown by
  QGroundControl) whenever a safety function acts: a watchdog trip or refusal with its
  reason code, the same reason in plain words with the parameter that sets the limit
  (`Watchdog::reason`, e.g. "watchdog trip: tilt (tilted past SH_WD_TILT), handing back")
  and an engage refusal (critical), that Structure Hold stays locked out until disarm (info), a wall-guard push and the fallback braking
  (warning), and each handback target (info). Regulators ask for exactly this: a failure of
  a system the flight relies on must be shown to the pilot (`docs/market-research.md`,
  gap 2). `sitl_test.py` checks the trip and wall-guard messages arrive.
- **Parameters** under `SH_`: `SH_EN`, `SH_CTRL`, yaw guard `SH_YAW_GUARD` / `SH_YAW_P`,
  rate clamp `SH_RATE_MAX`, fallback `SH_FB_EN` / `SH_FB_T` / `SH_FB_K` / `SH_FB_TILT`, fix
  source `SH_FIX_SRC` / `SH_FIX_GAP`, watchdog thresholds `SH_WD_*` (the frozen-fix trip `SH_WD_FRZ_T` among them), the engage hold-off
  after an EKF reset `SH_RST_HOLD`, the wall guard `SH_WALL_MIN` / `SH_WALL_REQ`, Structure Land `SH_LAND_EN`, Structure Hover `SH_HB_GUARD` and its range-only option `SH_HB_RANGE`, the test hook `SH_TEST_FAULT`, the bench-test-only ground engage `SH_BENCH`, placeholder gains
  `SH_P_*`, `SH_V_MAX`, `SH_VEL_SRC`.

Not here yet: choosing the standoff (the hold point is the engage point, kept outside
`SH_WALL_MIN` by the wall guard), passes along the surface, the fault injector that corrupts
the estimate the policy sees.

## Layout

```
sim2real/px4/
  structure_hold/        the module: CMakeLists.txt, Kconfig, StructureHold.{hpp,cpp},
                         Watchdog.hpp, WallGuard.hpp, Handback.hpp, RangeHover.hpp, ParamLimits.hpp,
                         PolicyController.hpp, MlpPolicy.hpp,
                         PolicyWeights.hpp (generated), HoldController.hpp,
                         structure_hold_params.c
  msg/StructureHoldStatus.msg
  tools/                 export_policy_header.py (onnx -> PolicyWeights.hpp),
                         make_test_vectors.py (Python env -> tests/PolicyTestVectors.hpp)
  install.sh             wires the module into a PX4-Autopilot tree (idempotent)
  boards/                structurehold.px4board: multicopter-only variant that fits a 2 MB board
  sitl_test.sh / .py     acceptance flight in PX4 SITL (SIH quadrotor, no Gazebo)
  bench_test.py          props-off bench checks on a flight controller (docs/bench-test.md)
  run_gazebo.py          the same flight in Gazebo with wind and the corrupted estimate
                         (Docker image, sim2real/estimate_bridge), scored on the truth
  tests/                 test_watchdog.cpp, test_wall_guard.cpp, test_keep_out.cpp, test_policy.cpp (+ PolicyTestVectors.hpp),
                         test_params.cpp, test_handback.cpp, test_range_hover.cpp, test_policy_header.py,
                         test_param_limits.py
```

After a policy release, regenerate both headers and commit them:

```bash
python -m sim2real.px4.tools.export_policy_header training/policies/structure_hold_v22.onnx
python -m sim2real.px4.tools.make_test_vectors training/policies/structure_hold_v22.onnx
```

A policy with the engage-time input (v22 on, 21 inputs per step) gets `HAS_ENGAGE_TIME`
in its header and reads `VehicleSample::engage_s`, the seconds since `StructureHold::engage`
(a wall-guard move of the hold point does not restart it, as in the env).

A policy with the wall-range input (v23 candidate on, 23 inputs per step) gets
`HAS_WALL_RANGE` and reads `VehicleSample::range_ok` / `range_err_m` from the forward
rangefinder the wall guard already reads. `RangeReference` (`WallGuard.hpp`, unit-tested in
`test_wall_guard.cpp`) latches the hold point's measured wall distance at the first usable
range after an engage (the vehicle's own distance, since the hold point is where it engaged)
and moves it back with each wall-guard shift; the input is that reference minus
`range × cos(body x, normal)`, positive toward the wall, clipped at `RANGE_OBS_CLIP_M`, and
zero with the flag off whenever the guard has no usable sample. The guard does not need to
be on (`SH_WALL_MIN 0` still measures). See `docs/wall-range.md`.

`RATE_HZ` in the generated header is the tick the policy trained at (v9, v10, v19 and v22: 83 Hz, the
12 ms tick the module's 10 ms `CONTROL_INTERVAL` gives on Gazebo's 4 ms gyro). It is
informational: the module keeps `CONTROL_INTERVAL` at 10 ms and never derives it from
`RATE_HZ`.

## Build

```bash
git clone --depth 1 --branch v1.17.0 --recurse-submodules https://github.com/PX4/PX4-Autopilot.git
sim2real/px4/install.sh /path/to/PX4-Autopilot          # --link to symlink while developing
cd /path/to/PX4-Autopilot && make px4_sitl_default
```

`install.sh` copies (or links) `structure_hold/` to `src/modules/structure_hold`, adds the
message to `msg/CMakeLists.txt`, enables `CONFIG_MODULES_STRUCTURE_HOLD` in the SITL board
config (`--board` for another), adds `structure_hold start` to `rc.mc_apps` behind
`SH_EN`, and makes the logger record `structure_hold_status`. Run it again after editing
the module when you copied rather than linked.

PX4's own build needs its Python packages (`Tools/setup/requirements.txt`; `empy<4`
does not build on Python 3.13, use 3.10 to 3.12).

### Real flight controllers

The module is about 125 KB of flash (text plus data), most of it the policy's weights, and
a stock 2 MB board does not have that spare: `px4_fmu-v6c_default` plus the module
overflowed flash by 87 KB (2026-10-31). `boards/structurehold.px4board` is a board variant
that drops the fixed-wing, VTOL and airspeed modules (as PX4's own
`fmu-v6x/multicopter.px4board` does), SIH, the MSP OSD driver, the precision-landing
target estimator and the Septentrio GNSS driver (u-blox and the other GPS drivers stay),
none of which a GPS-denied multicopter next to a wall uses, and turns the module on. With only the fixed-wing, VTOL and airspeed modules dropped, the fmu-v6x build
still overflowed by 2 KB:

```bash
cp sim2real/px4/boards/structurehold.px4board /path/to/PX4-Autopilot/boards/px4/fmu-v6c/
sim2real/px4/install.sh /path/to/PX4-Autopilot --board boards/px4/fmu-v6c/structurehold.px4board
cd /path/to/PX4-Autopilot && make px4_fmu-v6c_structurehold
```

Checked in the cloud container with Ubuntu 24.04's `gcc-arm-none-eabi` 13.2.1 (`apt-get
install gcc-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib`; PX4's
venv also needs `setuptools<70` for the UAVCAN generator): `px4_fmu-v6c_structurehold`
uses 96.0 % of flash, `px4_fmu-v6x_structurehold` 99.7 % (about 6 KB spare, so a partner
adding drivers to a 6X drops something else they do not fly). By 2026-11-02 the module's
growth (field-fault handling, range hover) had taken the 6X to 99.98 % (400 bytes spare);
the keep-out adds 3.7 KB, and dropping the Septentrio driver gives back 14 KB: with the
keep-out, `px4_fmu-v6c_structurehold` uses 95.8 % (1,883,848 B) and
`px4_fmu-v6x_structurehold` 99.5 % (1,955,400 B, about 10 KB spare). The first such build also
caught a format string the SITL build never flagged (`uint32_t` is `long unsigned` on Arm). The partner build steps are in
`docs/partner-pack.md`.

## Fly it

```bash
pip install pymavlink pyulog
sim2real/px4/sitl_test.sh /path/to/PX4-Autopilot          # writes runs/<time>_sitl_structure_hold_<commit>/
```

The flight: on the ground, check the mode cannot be armed into and that a switch into it
after arming is refused without lifting off (`trip_reason` `landed` in the ulog); take off
to 2.5 m in stock PX4, engage Structure Hold over MAVLink (DO_SET_MODE,
auto / external 1), hold 20 s, disengage to Position mode, re-engage, trip the watchdog
with `SH_TEST_FAULT`, confirm the handback and that a further engage is refused, land
through Structure Land (`--land stock` for stock Land).
`summary.json` has one boolean per check plus the numbers (drift while holding, handback
latency from the ulog, status samples logged). Exit 0 means every check passed.

`SITL_EXTRA_ARGS=--wall` adds the wall guard check: a forward rangefinder simulated over
MAVLink (`DISTANCE_SENSOR` at 20 Hz) against a virtual wall 1 m ahead of the engage point;
after the hold the wall jumps to 0.3 m ahead of the hold point, inside `SH_WALL_MIN` 0.7 m,
and the vehicle must back off to at least 0.55 m and log a nonzero `standoff_shift_m`.
After the trip the handback must go to Structure Hover, and the wall is put 0.8 m ahead and
creeps 0.6 m in at 0.1 m/s (the estimate recovering after a jump, seen from the estimate's
side); the hover must stay at least 0.55 m from it. The landing gets the same creep, but
its clearance is reported, not checked: SIH's EKF velocity is unusable (below), so PX4's
own xy loop, which flies Structure Land, tracks too poorly here to judge it (0.26 to 0.67 m
over runs; stock Land 0.15 m). Gazebo judges the landing.

`--range-faults` (with `--wall`) streams 6 s of broken ranges before the wall jump (0 cm,
5 cm, beyond the maximum, 65535 cm, a 50 cm reading flagged invalid); the vehicle must not
back off. `--moving-engage` pushes the vehicle sideways in Position mode (about 2 m/s in
SIH) and engages while it moves, the re-engage mid-gust case; it must stop and hold with no
trip. `--keep-out` simulates a 360 deg lidar (`OBSTACLE_DISTANCE`, 72 bins, body frame)
that sees one pole to the right of the hold point, sets the keep-out to 1 m through PX4's
`CP_DIST` (`--keep-out-param sh` for `SH_KO_DIST`), puts the pole 1.6 m away (nothing may
move), then 0.5 m away: the vehicle must settle at least 0.85 m from it, still in Structure
Hold, with a "keep-out" warning. First runs (2026-11-02): settled 1.00 and 1.02 m away.

From the PX4 shell: `commander mode ext1` engages, `commander mode ext2` lands through
Structure Land, `commander mode ext3` hovers in Structure Hover, `commander mode posctl` disengages,
`structure_hold status` prints the module state, `structure_hold trip` trips the watchdog. `structure_hold bench [n]` times n policy
steps on synthetic samples (refused while engaged); `bench_test.py` runs it and the other
props-off checks on a flight controller over MAVLink (`docs/bench-test.md`).

### SIH and the velocity estimate

The cloud container has no Docker daemon and no Gazebo, so the test flies PX4's built-in
SIH quadrotor (`PX4_SIM_MODEL=sihsim_quadx`, lockstep, real time). In that setup on
v1.17.0 the EKF's horizontal velocity is unusable: correlation with ground truth about
0.05 (position is fine, 0.96). Stock Position mode wanders about 0.6 m there too. The test
therefore runs the placeholder controller with `SH_VEL_SRC=1` (velocity differentiated
from the estimated position inside the module, low-pass filtered), which holds within
0.3 m. The default stays 0 (estimator velocity) for Gazebo and real flights. The Gazebo
image in `sim2real/docker/` is the simulator the baseline and benchmark use.

Position mode needs a manual-control source, so the test streams a centred joystick
(`MANUAL_CONTROL` at 20 Hz). With `--no-joystick` the stock mode is Hold instead, which
exercises the handback escalation (Position refused → Hold).

## Unit tests

```bash
cd sim2real/px4/tests
g++ -std=c++17 -Wall -Wextra -Werror -I../structure_hold test_watchdog.cpp -o test_watchdog && ./test_watchdog
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_policy.cpp -o test_policy && ./test_policy
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_wall_guard.cpp -o test_wall_guard && ./test_wall_guard
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_keep_out.cpp -o test_keep_out && ./test_keep_out
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_params.cpp -o test_params && ./test_params
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_handback.cpp -o test_handback && ./test_handback
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_range_hover.cpp -o test_range_hover && ./test_range_hover
python -m pytest sim2real/px4/tests
```

CI runs all of them; they need no PX4 tree. `test_policy.cpp` checks the compiled-in MLP
against onnxruntime (within 1e-6), and the observation and fallback action against values
recorded from the Python env (within 3e-7), plus the hand-back, the yaw guard, the
action-to-setpoint mapping and the parameter resolution against the policy's defaults, a
re-engage mid-flight (bit for bit a fresh controller), NaN and inf in every input (always a
finite setpoint) and the policy step time on the host. `test_wall_guard.cpp` covers the
broken readings real drivers and MAVLink send (inf, 0, negative, 65535 cm, unset or NaN
limits, bad quality, the age and range edges). `test_handback.cpp` walks every handback
sequence of the decided rows (including the `SH_HB_RANGE` one); `test_range_hover.cpp`
flies the range hover on a point mass with a tilt lag (a 4 m/s wind into the wall, with and
without the wind feed-forward, a drifting entry, an entry inside the standoff);
`test_watchdog.cpp` has one test per watchdog rule, the frozen fix included;
`test_params.cpp` and `test_param_limits.py` keep the
module's limits equal to `structure_hold_params.c`. The c++14 flags mirror PX4's own build.

## SIH flights (2026-10-22, cloud)

`sitl_test.sh` with `SH_CTRL=0` (cascade) and with `SH_FB_EN=2` (fallback rule alone) pass
every check (experiments/LOG.md): the frame conversions and the attitude loop hold on PX4
dynamics. Policy v2 with the policy's own yaw (`SH_CTRL=1`, guard off, before the guard
existed) tripped the position-error watchdog within 3.4 s: a saturated yaw rate from the
first tick, which is what the same observation gives in onnxruntime and what v2 does in the
mock too (the reward has no heading term). With the yaw guard the heading holds but the
roll/pitch bang-bang at 6 rad/s tips the SIH quad over in 1.5 s: PX4's rate loop reaches the
setpoint 100 to 200 ms late, the mock's does not. With the guard and `SH_RATE_MAX` 1 to
2 rad/s, v2 flies the whole acceptance sequence with no trip (drift 0.58 m at 1 rad/s,
stock Position mode wanders 0.8 m in the same run); 3 rad/s trips. SIH is not a
hold-quality test (see below); the numbers say the integration is sound and that the
policy's rate commands are the sim-to-PX4 gap to close in training (docs/decisions.md).

**Policy v3 (same day, later):** trained with a heading term and a 100 to 200 ms rate-loop
lag in the mock, 2 rad/s actions. With its own defaults (guard off, fallback off, 2 rad/s)
it flies the whole sequence with no trip and no guard: heading within 8 deg of the wall
(as in the mock), altitude within 0.1 m (v2: 0.3 to 0.8 m), horizontal drift growing
slowly to 0.84 m over the 20 s hold (stock Position mode: 0.6 m in the same run), rate
commands still at the 2 rad/s limit much of the time. The forced fallback (`SH_FB_EN=2`)
with the retuned gains 1.5 / 4 is calm (rates under 0.3 rad/s, heading within 1 deg) but,
being a pure velocity brake with no position term, follows the bias of SIH's differentiated
velocity: 1 m after 10 s, position-error trip at 2 m after 19 s (with the old 2.5 / 8 gains
it reached 0.99 m in 20 s). That is a 20 s forced test of a rule meant for 0.5 to 3 s
dropouts, not a fault; the Gazebo flight with real dropouts is where the gains get judged.
`SH_YAW_GUARD=1` on v3 is the field override only (the mock says it hurts).

**Next step:** the Gazebo flight below on the Mac, then the benchmark against tuned PX4
(M1).

## Gazebo flight (needs Docker: the Mac)

`run_gazebo.py` flies the acceptance sequence in the wall-and-slab world with wind and the
VIO-like estimate (dropouts, jumps, bias) through EKF2, GPS off, and scores the TRUE
position over the hold against the hold point the module captured. It is the first flight
that judges the policy rather than the plumbing. The image must carry the current module:

```bash
sim2real/docker/run.sh build                                   # after any change under sim2real/px4 (~5 min)
python -m sim2real.px4.run_gazebo calm_0ms --profile clean     # 1. plumbing: no wind, exact estimate
python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default                # 2. v3 defaults
python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default --param SH_FB_EN=1   # 3. fallback on
python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default --param SH_CTRL=0    # 4. cascade, regression
```

Each writes `runs/<time>_sitl_hold_<scenario>_<profile>_<commit>/` with `summary.json`
(the sequence's checks), `hold.json` (truth metrics, fallback / guard shares, stale time),
`ekf.json`, the verified wind and estimate traces, and `hold_summary.json` for the LOG
entry. About 3 minutes per run (60 s hold, two settles, the trip and the landing).

Expected results, in order; stop and look at the log the first time one is missed:

1. **calm, clean** passes every check (`"ok": true`): engage accepted, 60 s hold with
   `hold.json` p95 well under 0.5 m and `estimate_error` about zero, disengage, re-engage,
   handback under 1 s, latched refusal, landing; `fallback_share` 0, `yaw_guard_share` 0,
   `stale_max_s` about 0.02 (50 Hz odometry), `gps_free` and `vision_fused` true. If this
   one fails, the problem is plumbing (image not rebuilt, port 14540 busy, the bridge not
   started: see `bridge.log`), not the policy.
2. **wind 8 m/s, vio_default** is the policy's number: compare `hold.json metrics.p95_m`
   with the tuned-PX4 baseline (`docs/baseline.md`) and the mock (v3: 0.57 m). Expect
   `stale_max_s` up to 3 s (the profile's longest dropout), `fallback_share` 0 (off for v3),
   and `ekf.json aiding` showing dead-reckoning during dropouts. Known stock behaviour: a
   dropout longer than `COM_POS_FS_DELAY` (1 s) can make commander run its position-loss
   failsafe and leave the mode on its own; the summary then shows an unexpected mode before
   the trip step. That is PX4, not the module; note it and rerun with another `--seed`.
3. **fallback on** should show `fallback_episodes` > 0 and `fallback_share` roughly the
   share of the hold spent more than 0.5 s into a dropout, with the policy back in charge
   on each fresh fix; the hold number tells whether the rule helps or costs on PX4 as the
   mock says it costs v3.
4. **cascade** (`SH_CTRL=0`) must still pass as it did in PR #9's SIH runs; it is the
   regression check that the module's plumbing did not move while the policy went in.

Log every run in `experiments/LOG.md` against the tuned-PX4 baseline.

For the benchmark itself, fly the baseline's scenarios and seeds with the baseline's own
batch runner so the rows compare like for like (same wind and estimate draws per seed,
same PX4 parameter set underneath, same truth scoring over a 60 s window):

```bash
python -m eval.baselines.run_baseline --runner hold --label structure_hold_v19 \
    --params eval/baselines/px4_tuned_v1.params
python -m eval.baselines.run_baseline --compare runs/<tuned_v1>/batch.json runs/<structure_hold_v19>/batch.json
```

Result on 8 seeds (Gate 2): `docs/baseline.md`, "Gate 2 evidence: v19 vs tuned PX4 on 8 seeds, corrected estimate stream" (v19: p95 −47 to −48 % on the mean, −13 to −15 % on the median, lower on 6 of 8 seeds, 0 crashes; the earlier v19 and v10 tables are kept above it as the record of the protocol).

From PR #50 on, new comparisons should pass `-- --profile vio_at_engage` to
`run_baseline --runner hold` for every policy, and `-- --profile vio_at_hover` to the stock
runner for tuned PX4 (which never engages the mode, so the same schedule is keyed to
reaching the hover altitude instead): the estimate's jumps and dropouts are then keyed to
the moment the mode engages (fault clock 25 s at engage, where the seed 1 to 8 holds
opened), so every policy meets the same faults at the same time after engage and none
during takeoff (`sim2real/estimate_bridge/README.md`). The seed 1 to 8 numbers above were
flown on `vio_default` (clock from the bridge's first sample) and compare only with each other.
Known property: on seed 3 the big jump lands 67 s after the engage, 1.3 s after the 60 s
scripted hold, so stock PX4 meets it between the two engages (`docs/baseline.md`, engage
guard confirmation).

**Wall guard in Gazebo.** The x500 model has no rangefinder, so the guard does nothing in
the benchmark above. `--rangefinder` (after `--` for `run_baseline`) has the estimate
bridge also send a forward DISTANCE_SENSOR, computed from the same Gazebo truth pose
against the scenario's wall (`sim2real/estimate_bridge/README.md`), so the guard acts as it
would on a drone with a forward rangefinder; no image change beyond the module itself:

```bash
python -m eval.baselines.run_baseline --runner hold --label structure_hold_v19_wallguard \
    --params eval/baselines/px4_tuned_v1.params --seeds 3 -- --profile vio_at_engage --rangefinder
```

`hold.json` then has `wall_guard` (per engaged stretch: range share, closest wall and hold
point clearance, how far the point moved back) and, for every flight, `after_hold` (the
closest the truth came to the wall after the scored hold, where seed 3's strikes happen)
and `landing` (the Structure Land stretch: its window, how far the guard moved the point
back, the closest the truth came to the wall). `--land stock` (after `--`) flies the
sequence's old stock Land ending.

## Frames

PX4 boundary: NED world, FRD body, quaternion body → world (`docs/conventions.md`).
`structure_hold_status` positions are NED metres, rates rad/s, thrust normalised 0..1.
