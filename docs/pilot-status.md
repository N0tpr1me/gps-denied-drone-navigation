# What the pilot sees while Structure Hold flies

The trip messages (`docs/safety-manual.md`, section 4) tell the pilot what a safety
function did after it acted. This page is the other half: what the pilot sees **before**
that, while the vehicle holds. It answers the feature "the pilot's view" in the "Features
that sell" research (2026-11-02): the module's state as MAVLink messages QGroundControl
shows, in words, so a pilot (and a regulator asking whether the pilot is told) can see how
close the hold is to handing back.

Everything here comes from the Structure Hold PX4 module itself (`PilotStatus.hpp` decides,
`StructureHold.cpp` sends), so it works with stock QGroundControl and any MAVLink ground
station: no plugin, no custom message set.

## In words: ground-station text

QGroundControl shows these in its vehicle message list, with the severity it gives each
one. All are STATUSTEXT, and all are in the flight log.

| When | Severity | Message |
| --- | --- | --- |
| A mode starts flying | info | `Structure Hold: engaged, 1.52 m from the wall` (or `no wall range`, or `wall guard off` when `SH_WALL_MIN` is 0); the same for Structure Hover and Structure Land |
| Every `SH_GCS_PER` seconds (10 s) while a mode flies | info | `Structure Hold: 0.08 m off the hold point, 1.49 m from the wall, fix 0.02 s old` |
| The same, in Structure Hover on the rangefinder alone | info | `Structure Hover: on the rangefinder, 1.31 m from the wall, holding 1.30 m` |
| No position fix for half of `SH_WD_EST_TO` | warning | `Structure Hold: no position fix for 2.5 s, watchdog hands back at 5.0 s` |
| The fix stopped moving for half of `SH_WD_FRZ_T` | warning | `Structure Hold: position fix has not moved for 0.5 s, watchdog trips at 1.0 s` |
| Tilt past three quarters of `SH_WD_TILT` | warning | `Structure Hold: tilted 34 deg, watchdog trips at 45 deg` |
| Hold error past half of `SH_WD_POS_ERR` | warning | `Structure Hold: 1.00 m off the hold point, watchdog trips at 2.0 m` |
| Every limit back under 40 % after a warning | info | `Structure Hold: back inside the watchdog limits` |

Why these levels: in the Gazebo benchmark the hold error's 95th percentile is about 0.25 m
of the 2 m limit and the fix is about 0.02 s old, so half a limit only happens in a real
fault. Tilt runs higher in normal flight (about half the limit in 8 m/s gusts), so it warns
at three quarters. Tilt has no time to warn before a trip in a violent upset, and a
reset-to-origin jump trips at once; the warnings cover the slow approaches (a dropout, a
drift, a frozen VIO, a wind that is too strong).

So the link is not flooded: each limit warns once when it crosses its level and again only
after it has dropped under 40 % and 10 s have passed; at most one warning goes out per
control tick; the status line goes out once per period. The warnings and the engage line
are sent whatever `SH_GCS_PER` is; `SH_GCS_PER` 0 turns off only the periodic line.

The checks a mode does not run are not warned about: the frozen fix only in Structure Hold
(Structure Hover and Land fly PX4's own controller with the wall guard), the hold error not
in Structure Hover on the rangefinder alone, where there is no hold point on the estimate.
The no-fix time is counted as the watchdog counts it (from the older of the last fix and the
last EKF2 update; on the rangefinder alone, from EKF2 only).

## In numbers: the live array

For a flight-test engineer the module also publishes its state as one MAVLink
`DEBUG_FLOAT_ARRAY` named `SH_STATUS` (array id 21320, "SH"), five times a second. PX4
streams the newest one at the link's rate for that message: 1 Hz on a normal
ground-station link, 10 Hz on an onboard link. In QGroundControl open **Analyze Tools >
MAVLink Inspector**, pick `DEBUG_FLOAT_ARRAY`, and read or chart the entries below. For a
faster readout on the ground-station link add
`mavlink stream -u 14550 -s DEBUG_FLOAT_ARRAY -r 5` (SITL) or the same with `-d <device>`
for a radio to the SD card's `etc/extras.txt`.

| Index | Name | Unit | Meaning |
| --- | --- | --- | --- |
| 0 | `state` | | Module state: 1 standby, 2 Structure Hold, 3 handing back, 4 Structure Land, 5 Structure Hover (`StructureHoldStatus.msg`) |
| 1 | `hold_error` | m | Distance from the hold point as the watchdog measures it (xy only in Structure Land); -1 when no hold point is flown |
| 2 | `wall` | m | Forward rangefinder: perpendicular wall distance while a mode flies, the raw reading before engaging; -1 without a usable reading |
| 3 | `clearance` | m | Hold point to the measured wall (the wall guard keeps it at `SH_WALL_MIN` or more); -1 without a range |
| 4 | `guard_shift` | m | How far the wall guard has moved the hold point back since engage |
| 5 | `fix_age` | s | Time since the last position fix |
| 6 | `wd_seconds` | s | Time left before the no-fix trip (`SH_WD_EST_TO`); -1 when no mode flies |
| 7 | `wd_used` | % | How much of the closest watchdog limit is used (100 trips); -1 when no mode flies |
| 8 | `wd_closest` | | Which limit that is, as a trip code: 3 no fix, 4 tilt, 5 hold error, 9 frozen fix |
| 9 | `trip` | | Last trip or refusal code (0 none; the trip table in `docs/integration-guide.md`); cleared on disarm |
| 10 | `trip_count` | | Trips since boot |
| 11 | `tilt` | deg | Tilt from level |
| 12 | `fallback` | | 1 while the brake-and-hover fallback flies |

The rest of the 58 entries are 0. `tests/test_pilot_status_doc.py` keeps this table in
step with `LiveIndex` in `PilotStatus.hpp`. The logged `structure_hold_status` topic carries
the same margin as `watchdog_used` (a fraction, -1 when no mode flies) and
`watchdog_closest`.

## Cost

About 3.2 KB of flash on a flight controller (the strings and the code), a few hundred
bytes of RAM, and one 250-byte MAVLink message per second on a normal link. That was more
than the fmu-v6x build had spare (99.97 % full on `main`, 2026-11-02), so the
`structurehold` board variant now also drops the AUAV airspeed driver and EKF2's airspeed
and sideslip fusion, which a multicopter does not use: fmu-v6x is at 99.53 % with this
feature (9.2 KB spare); the fmu-v6c numbers are in `sim2real/px4/README.md`. PX4 builds the
`DEBUG_FLOAT_ARRAY` stream on every board without `CONSTRAINED_FLASH`, fmu-v6c and fmu-v6x
included; on a constrained board the array is simply not sent and the text still is.

## Checked

- `sim2real/px4/tests/test_pilot_status.cpp` (CI): a normal hold warns about nothing, a
  4 s dropout warns once at 2.5 s and notes the recovery, tilt warns at 75 %, a gusty tilt
  bouncing across the level warns once per 10 s, two limits at once give one event per tick
  in the watchdog's order, checks a mode does not run are skipped, a NaN counts as at the
  limit, the status line keeps its period.
- `sim2real/px4/sitl_test.py` (every PX4 SIH run): the engage line, the periodic status
  line during the 20 s hold, the live array while engaged (hold error, margin and time
  left all in range) and showing the test-fault trip afterwards. `--pilot-warning` lowers
  `SH_WD_POS_ERR` before the trip so the hold error sits at about 60 % of it, and expects
  the near-limit warning. Results of the run that added this are in the PR.
