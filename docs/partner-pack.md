# Structure Hold: flight test pack for a design partner

> **Confidential. Copyright (c) 2026 Zain Ali. All rights reserved.** For a drone maker
> who has signed an evaluation agreement with us, and nobody else. Internal draft: not
> sent to anyone yet. The open items at the end must be settled before it is.

We are software only (`docs/decisions.md`, 2026-10-30): we own no drone. Real-flight proof
(Gate 3) and the gust tests happen on a partner's PX4 drone. This pack is what the partner
follows, from building our module into their firmware to sending us flight logs that our
own scoring code reads. Every step points at the script or document it uses, so nothing in
here is a second copy of a number that lives elsewhere.

The module's interface (topics, modes, trips, every parameter, ground-station messages,
resource budget) is `docs/integration-guide.md`; the hazards and what covers each one are
in the safety manual, `docs/safety-manual.md`.

## What Structure Hold is, in one paragraph

A PX4 flight mode ("Structure Hold", external mode 1) that holds a multicopter 1 to 3 m
from a wall, pier or deck in gusts, without GPS, on the drone's own position estimate
(VIO into EKF2). A small neural network (policy v22) sends body rates and thrust to PX4's
stock rate controller. A classical watchdog hands control back to stock PX4 on any fault,
and a forward rangefinder keeps the hold point off the wall (the wall guard). The pilot
can always take over by switching to Position mode. In our PX4 SITL benchmark (Gazebo,
8 seeds, gusts up to 8 m/s, on barometer height as step 2 of this pack sets it) it held to
about 0.3 m at the 95th percentile, 34 to 46 % tighter than properly tuned stock PX4 (0.49
to 0.53 m), with no crashes (`docs/baseline.md`, re-flown 2026-11-09). On vision height, which
this pack and the safety manual do not allow in flight, it holds to about 0.2 m; the 54 to
55 % in earlier copies of this pack was that set-up. This pack is how we find out whether
the barometer-height result holds on a real drone.

## The steps, and who does what

| Step | Who | Props | Time | Output you send us |
| --- | --- | --- | --- | --- |
| 0. Airframe sheet and one logging flight | Partner flies, we check | On, by hand | Half a day | Airframe sheet, parameter dump, `.ulg` |
| 1. Build and install | Partner (we can build for you) | – | 1 hour | Build log, firmware commit |
| 2. Sensors and parameters | Partner | – | 1 to 2 hours | Parameter dump |
| 3. Bench test | Partner | **Off** | 15 minutes | `summary.json`, `.ulg` |
| 4. Tethered checks | Partner | On, tethered or netted | Half a day | `.ulg` files, notes |
| 5. Scored flight sessions | Partner | On | A day per session | Session folder (step 6) |
| 6. Send the logs; we score them | Both | – | – | We send back the report |

We say go or no-go after steps 0, 3 and 4. Do not skip ahead: each step catches problems
that make the next one unsafe.

## Step 0: tell us about your drone

The policy was trained in our simulator on the PX4 Gazebo x500 (about 2.1 kg, 0.36 m prop
reach), with the mass, motors and rate loop randomised around it. A drone far from that
may fly differently, so before anything flies our code we check your numbers in our
simulator (`docs/real-flight-gap.md`, row 4).

What we need:

- **The drone must already fly well on stock PX4 v1.17.0** (or your fork of exactly that
  release: tell us its commit) in Position mode, on its VIO, with GPS off. Structure Hold
  uses your position estimate; it does not make one.
- **Airframe sheet:** take-off mass with battery, arm length and prop size (so the prop
  reach, centre to prop tip), motors and ESCs, battery, the flight controller board, the
  VIO system (camera, rate in Hz, latency if known), the IMU and gyro rate, and the forward
  rangefinder model (step 2).
- **Parameter dump** of the drone as it flies today (QGC, Parameters, Tools, Save to file).
- **One logging flight** with no policy: `docs/logging-flight.md` is the checklist and
  `sim2real/px4/logging/logging_flight.params` the logger settings. Eight minutes by hand
  in front of a wall, with mocap truth recorded but not fused. From it we read the hover
  thrust, the VIO's real error and latency, and vibration.

We answer with go, no-go, or "go after a retrain on your airframe" (we add your drone to
`envs/structure_hold/airframes.py`, fine-tune, and re-check in our simulators first).

## Step 1: build Structure Hold into your firmware

The module is plain C++ inside PX4: no companion computer, no ONNX runtime, no heap. It
takes about 125 KB of flash, most of it the policy's weights.

That is more than a stock build has spare. On a Pixhawk 6C (`px4_fmu-v6c_default`) the
stock firmware plus the module overflows flash by 87 KB. So build a multicopter-only
variant: `sim2real/px4/boards/structurehold.px4board` drops the fixed-wing, VTOL and
airspeed modules (as PX4's own `fmu-v6x/multicopter.px4board` does), PX4's built-in
simulator, the MSP OSD driver, precision landing on a beacon and the Septentrio GPS driver,
and turns the module on. We built it on 2026-11-02, with the module as of that day (keep-out
and pilot status included), for the Pixhawk 6C (95.6 % of flash used) and the Pixhawk 6X (99.0 %, about
19 KB spare). If your drone flies GPS or RTK from a Septentrio receiver, tell us: the
driver goes back in (about 14 KB, which still fits the 6X). On another board, tell us and we check it fits.

```bash
git clone --depth 1 --branch v1.17.0 --recurse-submodules https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
cp <this repo>/sim2real/px4/boards/structurehold.px4board boards/<vendor>/<model>/
<this repo>/sim2real/px4/install.sh . --board boards/<vendor>/<model>/structurehold.px4board
make <vendor>_<model>_structurehold                  # e.g. make px4_fmu-v6c_structurehold
make <vendor>_<model>_structurehold upload           # flight controller on USB
```

`install.sh` copies the module into `src/modules/structure_hold`, adds its status message,
starts it at boot (behind `SH_EN`), and logs its status topic. PX4's build wants Python
3.10 to 3.12 for its own tools and the Arm GCC 13.2 toolchain (Ubuntu 24.04's
`gcc-arm-none-eabi` package works). If your drone needs one of the dropped modules, keep
it and drop something else you do not use; Structure Hold needs only the multicopter stack.

If your firmware is a fork, or you build with a vendor flow (for example ModalAI's), run
`install.sh` on your tree and tell us anything it could not patch. If you would rather not
build, send us the board name and your PX4 tree, and we send back a firmware file.

**Check:** on the PX4 shell (QGC, Analyze Tools, MAVLink Console), `structure_hold status`
says `state: standby` and names the policy (`structure_hold_v22`). QGC lists three new
flight modes: Structure Hold, Structure Land and Structure Hover.

## Step 2: sensors and parameters

### Position estimate (VIO into EKF2)

Keep the EKF2 setup your drone already flies on. Check these, because Structure Hold
depends on them (`sim2real/estimate_bridge/ekf2_ev.params` is what our benchmark uses):

| Parameter | Why it matters |
| --- | --- |
| `EKF2_GPS_CTRL` 0, `SYS_HAS_GPS` 0 | No GPS anywhere in the loop: the claim is GPS-denied flight. GPS or RTK may stay on (fused beside the VIO) only with the forward rangefinder and `SH_WALL_REQ` 1: when a GPS solution moves, EKF2's estimate walks with it and the rangefinder is what keeps the drone off the wall (`docs/field-faults.md`, "GPS on"; safety manual F27) |
| `EKF2_EV_CTRL` fuses VIO position (and velocity if your VIO gives a good one) | The hold is only as good as this |
| `EKF2_EV_DELAY` set to your VIO's measured latency | A wrong delay makes the estimate lag or lead in gusts (`docs/real-flight-gap.md`, row 7) |
| `EKF2_EV_POS_X/Y/Z` set to the camera's offset from the IMU | Otherwise rotations look like movement |
| `EKF2_HGT_REF` barometer (0) or a downward rangefinder (2), not vision (3) and not GPS (1, PX4's default) | With vision as the height reference, EKF2's height stops moving when the VIO goes quiet (the baro change goes into its bias), and the vehicle can sink without anything seeing it: in Gazebo an 8 s VIO silence put Structure Hold on the ground while EKF2 still showed 2.5 m; with baro height it held its height (`docs/field-faults.md`). Our benchmark uses barometer height too since 2026-11-01 (`eval/baselines/px4_tuned_v2.params`). PX4's default 1 is GPS height, which on a GPS-equipped drone moves with every GPS jump; `eval/log_health.py` warns about it |
| Heading source | The heading at engage is taken as the wall's normal. Near steel the magnetometer can be off by tens of degrees; if your VIO gives yaw, fusing it (`EKF2_EV_CTRL` yaw bit) is safer (row 3) |

The VIO must reach PX4 as `ODOMETRY` with `estimator_type` VISION or VIO (it is logged as
`vehicle_visual_odometry`, which the module also uses to tell a fresh fix from a stale one).

Before any Structure Hold flight we run the log health check on your logging flight
(`python -m eval.log_health flight.ulg`, `docs/log-health.md`). It reads the one log and says
whether EKF2 fuses the VIO, whether the VIO's frame matches the IMU and compass (ENU sent as
NED, a mirrored axis or a frame turned from north all show up), where the heading comes from,
and how late the VIO really is against `EKF2_EV_DELAY`, with the value to set. A climb of a
metre or two and a few metres side to side give it enough motion.

### Forward rangefinder (required)

The wall guard needs a rangefinder pointing straight ahead along the body x axis, seen by
PX4 as `distance_sensor` with orientation forward (`ROTATION_NONE`). Any PX4-supported
sensor works: a serial or I2C lidar, or `DISTANCE_SENSOR` over MAVLink from a companion.
It should read reliably from 0.3 m to at least 5 m on your wall. A downward rangefinder
for height is fine alongside it; the module skips it.

**Check:** `listener distance_sensor` on the shell shows a sample with `orientation: 0`
whose `current_distance` follows a board you move in front of the drone.

### Obstacle sensors (optional)

If your drone already has PX4 collision prevention set up (a 360 deg lidar such as the
LightWare SF45, `OBSTACLE_DISTANCE` from a companion, or side-facing rangefinders, with
`CP_DIST` set), Structure Hold honours it: the hold point stays at least `CP_DIST` from any
obstacle those sensors see, other than the structure itself (`SH_KO_DIST` -1, the default,
follows `CP_DIST`; set `SH_KO_DIST` to use another distance inside the hold, 0 to turn it
off). It is a keep-out, not avoidance (`docs/integration-guide.md`, inputs).

**Check:** `structure_hold status` prints `keep-out: <d> m ... obstacle data fresh`.

### Truth for scoring (mocap)

Every scored number is measured on an independent truth, never on the drone's own
estimate (`docs/metrics.md`). Indoors that is a motion-capture system. Stream it to PX4 as
`ODOMETRY` with **`estimator_type` MOCAP** (or as `ATT_POS_MOCAP`), so PX4 logs it as
`vehicle_mocap_odometry` and EKF2 never fuses it. Sent as VISION it would be fused, and the
flight would measure nothing. Check in the bench test's log that `vehicle_mocap_odometry`
is present and `vehicle_visual_odometry` comes only from the VIO.

Outdoor flights without mocap (RTK) cannot be scored yet (open items, below).

### Structure Hold parameters

Load `sim2real/partner/structure_hold_flight.params` (QGC, Parameters, Tools, Load from
file), then reboot. It sets the logger (`SDLOG_PROFILE` 147, boot to disarm) and writes out
every `SH_` value the benchmark flew, so your parameter dump records them. Then set:

| Parameter | Set it to | What it does |
| --- | --- | --- |
| `SH_WALL_MIN` | Your prop reach plus at least 0.3 m (0.7 m for a 0.36 m reach) | The hold point never comes closer to the measured wall than this |
| `SH_WALL_REQ` | 1 (in the file) | The mode refuses to engage without a usable forward range |
| `SH_FIX_GAP` | About twice your VIO's sample period (0.1 s for 20 Hz or faster) | When the policy is told its fix is stale |
| `SH_WD_EST_TO`, `SH_WD_POS_ERR`, `SH_WD_TILT` | Leave at 5 s, 2 m, 45 deg for scored runs; 1 m and 30 deg for step 4 | The watchdog: no fix for this long, this far off the hold point, or this much tilt hands back to stock PX4 |
| `SH_LAND_EN`, `SH_HB_GUARD` | 1 (in the file) | Structure Land (a landing that keeps the wall guard) and Structure Hover (where a trip next to a wall hands back to first) |
| everything else | As in the file | The policy's own defaults (`sim2real/px4/README.md` explains each) |

### RC and pilot controls

- A three-position mode switch: **Position**, **Hold**, **Structure Hold**. Position is
  the pilot's way out from anything, at any time: it is stock PX4.
- A kill switch, tested on the ground before every session.
- Structure Land and Structure Hover can stay off the switch; the watchdog uses Structure
  Hover by itself, and the pilot can pick Structure Land in QGC.

## Step 3: bench test, props off

Follow `docs/bench-test.md`. Props off for every step.

```bash
pip install pymavlink pyulog
python3 sim2real/px4/bench_test.py --device /dev/ttyACM0 --require-range \
    --expect SH_WALL_MIN=<your value> --out runs/<date>_bench/summary.json
```

It checks that the firmware is v1.17.0, the module is registered and flying the expected
policy, the policy's worst step on your CPU is under 1 ms of its 10 ms tick, every `SH_`
parameter reads back, and the forward rangefinder reaches the module. Exit code 0 means
all passed. Then the manual armed checks in the same document (mode greyed out with the
rangefinder covered, engage, the wall guard backing off from a board, a test trip, the
handback; they set `SH_BENCH` 1 so the module will engage on the ground, and it must be
back at 0 before anything flies). Send us `summary.json` and the `.ulg`; the inference time goes into our log
against your board.

## Step 4: tethered checks

The first time the policy flies the drone. Tethered to the floor (a short line that stops
it reaching the wall or the ceiling) or inside a net cage, in a room with a flat, textured
wall (brick, concrete or posters, not a plain white wall) and mocap running. Set
`SH_WD_POS_ERR` 1.0 and `SH_WD_TILT` 30 for this step.

1. **Tuned stock PX4 first:** take off in Position mode facing the wall at 2 m, hover
   60 s, switch to Hold mode 60 s, land. This is the drone's own baseline for the day and
   proves the VIO and the tether.
2. **Engage:** in Position mode at 2 m from the wall, heading square to it (within about
   5 degrees: the heading at engage is taken as the wall's normal), switch to Structure
   Hold. Hands off for 10 s. Switch back to Position mode. Do this three times.
   *Pass:* holds within about 0.3 m by eye and on mocap, no oscillation, Position mode
   takes over at once on the switch.
3. **Hold 60 s** the same way, twice.
4. **Watchdog trip:** while engaged, set `SH_TEST_FAULT` 1 from QGC. *Pass:* within a
   second the drone is in Structure Hover (the wall is in range), holding off the wall;
   a further switch to Structure Hold is refused until the drone is disarmed. Switch to
   Position mode and land.
5. **Structure Land:** from a hover at 2 m from the wall, pick Structure Land in QGC.
   *Pass:* it descends straight down at the land speed and disarms on touchdown.
6. Restore `SH_WD_POS_ERR` 2.0 and `SH_WD_TILT` 45. Pull the logs.

Stop and send us the logs at the first failed check. `structure_hold status` on the shell
and `structure_hold_status` in the log say what the module saw and why it tripped.

## Step 5: scored flight sessions

### The protocol

It mirrors our Gazebo benchmark, so the real numbers sit next to the simulated ones
(`docs/baseline.md`, `docs/fan-rig.md`).

- **Cases:** calm air first. Then 4 m/s and 8 m/s of wind into the wall, only with a fan
  rig or a measured natural wind, logged by an anemometer at the hold point
  (`docs/fan-rig.md` for how we log and calibrate wind).
- **Per case:** 8 runs of each controller, **alternating** (tuned PX4, Structure Hold,
  tuned PX4, ...), so a change in light, battery or wind hits both alike.
- **Tuned PX4** is your drone's own best stock tune: PX4 autotune for the rate and
  attitude loops, then the position loop values of `eval/baselines/px4_tuned_v1.params`
  if they fly well on your drone, saved as its own parameter file. Same EKF2 settings for
  both controllers.
- **One run:** arm and take off in Position mode at least 2 m from the wall; fly to the
  hold point, **1.5 m from the wall** for the first session and 1 m after (our benchmark
  holds at 1 m), 1.5 to 2.5 m up, heading square to the wall; switch to **Hold** (tuned
  PX4) or **Structure Hold**; hands off for 60 s; switch to Position; back off to 2 m; land.
  One run per log: disarm between runs so each `.ulg` holds one hold.
- **The take-off and the move to the hold point matter for scoring:** we line up the
  drone's estimate with mocap over them, so fly at least 1 m sideways or up on the way.

### Safety limits

| Limit | Value |
| --- | --- |
| Closest engage to the wall | 1.5 m in the first session, 1 m after; never closer than `SH_WALL_MIN` plus 0.3 m |
| Height | 1.5 to 3 m, and at least 1 m below the ceiling |
| Wind | Calm until the calm case is done with no contact; then 4 m/s; 8 m/s only after 4 m/s is clean |
| Battery | At least 40 % to engage; land at 30 % |
| People | Behind a net, or at least 5 m away and never between the drone and the wall |
| Pilot | Hands on the sticks and the mode switch for every second the drone is armed |

**Abort the run** (switch to Position mode, back off, land) on any of: oscillation or
growing wobble, drifting more than 0.5 m toward the wall, a VIO or estimator warning in
QGC, a trip you did not command, or anything that looks wrong. Write it in the run sheet
as aborted, with the reason. Aborted runs are reported, not hidden.

**Stop the session** and send us the logs before flying again after: any contact with the
wall, any crash, or two trips you did not command in one session.

## Step 6: log, send, and how we score it

### The session folder

One folder per session, named `<date>_<site>`:

```
2027-05-14_lab/
  session.yaml                 run sheet, from sim2real/partner/session_template.yaml
  01_tuned_calm_1.ulg          one log per run, numbered in the order flown
  02_hold_calm_1.ulg
  ...
  params_tuned.params          QGC parameter dumps of both set-ups
  params_structure_hold.params
  bench_summary.json           step 3, if flown that day
  vio/                         the VIO's own logs, if it writes any
  photos/                      the wall, the set-up, the drone
```

Fill in `session.yaml` during the day: the drone, the firmware commit, the VIO, the wall
measured with the mocap wand (one point on its face and the normal from the drone toward
it, in the mocap frame), your prop reach as `contact_radius_m`, and one line per run in
the order flown, aborts included. Give the logs the file names you wrote in the sheet.

Zip the folder and upload it to the link we give you. Logs are large; please do not email
them.

### What we do with it

```bash
pip install -e .[ulog]
python -m eval.partner_session <folder>/session.yaml    # writes <folder>/scored/
```

`eval/partner_session.py` scores every run with the same code as our simulators
(`eval/metrics.py`, through `eval/ulog.py`, `docs/metrics.md`):

- **The scored window** is found in the log: the first stretch in Hold mode (tuned PX4) or
  Structure Hold, minus its first 2 s, cut at 60 s, as our Gazebo benchmark does.
- **The hold point** is where the controller was told to hold at the window start. It is
  in the drone's estimate frame, so the estimate is fitted to mocap (a yaw and an offset)
  over the take-off and the move to the hold point, and the hold point carried into the
  mocap frame. The fit's residual is reported with every run.
- **The numbers:** p95 of the 3D distance from the hold point (the headline), RMS, the
  worst move toward the wall, the closest the drone came to the wall, wall contacts
  (centre closer than your prop reach), and, with gust times, the recovery time.
- **Crash column:** a wall contact, or a hold that ended before 60 s (a watchdog trip, a
  failsafe or the pilot), with the trip reason. The same rule as our benchmark: judged on
  the 60 s hold only.

`scored/report.md` has a table per controller and, per case, Structure Hold's mean p95
against tuned PX4 on your drone. `scored/benchmark_report.html` is the same result on one
printable page (headline, per-case table, every hold as a dot, every handback in plain
words, set-up and method; `eval/partner_report.py`, `docs/eval-kit.md`). We log every scored run in our experiment log, aborts and
failures included, and send you the report.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Structure Hold greyed out in QGC | No usable forward range (`SH_WALL_REQ` 1: is the wall within range and the drone within 35 deg of square to it?), the estimate is not valid, or a trip is latched (disarm to clear) |
| `structure_hold status` says `unregistered`, or the module is not running | Registration with commander failed or `SH_EN` is 0: check `SH_EN`, reboot, and send us the boot messages (`dmesg`) |
| `policy_expected` fails in the bench test | The firmware was built from another checkout; pass `--policy <name>` with the name the shell shows, and tell us |
| Hands back soon after engage with trip `estimate_stale` | The VIO stopped reaching PX4 for `SH_WD_EST_TO`; check its link and rate |
| Hands back with trip `position_error` | More than `SH_WD_POS_ERR` from the hold point: send us the log |
| Scorer says the align window "moves 0.0x m" | The take-off was straight up with no sideways move; fly 1 m sideways on the way to the hold point, or give the run an `align` window (log seconds) that covers a sideways move |

## Open items before this goes to a partner (for Zain)

1. **Evaluation agreement.** A written agreement (NDA plus evaluation licence) must be
   signed before the partner sees this pack or the code. Not drafted here.
2. **Source or binary.** Whether the partner gets the module's source (step 1 as written)
   or only a firmware file we build for their board.
3. **Gate 3 pass bar.** What result on a partner drone counts as Gate 3 passed (for
   example: lower mean p95 than tuned PX4 in every flown case, and no contacts). Agree it
   in writing before the first scored session.
4. **Outdoor truth.** `eval/ulog.py` reads mocap only; flights near a real structure
   outdoors need RTK truth read from `sensor_gps` (a small addition when a partner needs it).
5. **Upload channel** for session folders.
