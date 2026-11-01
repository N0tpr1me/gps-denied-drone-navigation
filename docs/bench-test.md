# Props-off bench test of the Structure Hold module

The first time the module runs on a real flight controller, before any hover
(`docs/real-flight-gap.md`, row 9). It answers four questions SITL cannot: does the build
run on the controller's CPU, how long does the policy take there, are the parameters what
we think, and does the forward rangefinder reach the module. Nothing here needs props, a
wall, mocap or a pilot. **Props off for every step.**

## What you need

- The flight controller with PX4 v1.17.0 built with the module, as the multicopter-only
  `structurehold` variant (a stock build plus the module does not fit in 2 MB of flash;
  `sim2real/px4/README.md`, "Real flight controllers"), on USB.
- The forward rangefinder wired and configured, pointing along the body x axis
  (`distance_sensor` orientation 0, "forward facing"). Check its driver's `SENS_*` /
  `*_ROT` parameter says forward.
- A laptop with this repo and `pip install pymavlink pyulog`.

## 1. Disarmed checks (scripted, about a minute)

```bash
python3 sim2real/px4/bench_test.py --device /dev/ttyACM0 --require-range \
    --expect SH_WALL_MIN=0.7 --out runs/$(date -u +%Y%m%d-%H%M%S)_bench/summary.json
```

On macOS the port is `/dev/tty.usbmodem*`. Hold a flat board 1 to 2 m in front of the
rangefinder while it runs. Exit 0 means every check passed. `summary.json` keeps the
shell output of each command. The checks:

| Check | Pass when |
| --- | --- |
| `firmware_answers`, `px4_v1_17_0` | the shell answers and `ver all` says Release 1.17.0 (the pinned release) |
| `module_registered` | `structure_hold status` says `state: standby` (the mode registered with commander) |
| `policy_expected` | the compiled-in policy is the released one (`--policy`, default the NAME in `structure_hold/PolicyWeights.hpp`, v22 today) |
| `inference_within_budget` | `structure_hold bench` times 2000 policy steps; the worst is under 1 ms (the tick is 10 ms). Scored only on real hardware; SITL's lockstep clock moves in 4 ms steps |
| `params_readable`, `param_*` | every `SH_` parameter reads back; each `--expect` matches |
| `range_publishing`, `module_sees_range` | `distance_sensor` has a forward sample and the module calls it fresh. Move the board: `range_m` in the summary should follow |

Record `inference_mean_us` and `inference_worst_us` in `experiments/LOG.md` with the
board name. That number is the one the module budget is written against.

## 2. Armed checks (manual, props off, about 10 minutes)

These need a position estimate, so run them with the VIO (or mocap) feeding EKF2 as it
will in flight. In QGC or on the PX4 shell (`mavlink shell` / QGC MAVLink Console):

1. `param set SH_WALL_REQ 1`. Cover the rangefinder: the mode must be unavailable
   (QGC greys it out). Uncover it: available again. Then `param set SH_WALL_REQ 0`.
2. `param set SH_BENCH 1`: since 2026-11 the module refuses to engage while the land
   detector says the vehicle is on the ground, and this lifts that for the bench only.
   Arm (props off) in Position mode, then switch to Structure Hold (`commander mode ext1`).
   `structure_hold status` says engaged and warns about `SH_BENCH`, with a `control tick`
   interval perf line at about 10 ms and no trip. With `SH_BENCH 0` the same switch must be
   refused (`engage refused: landed` in the log) and PX4 stays in Position mode.
3. With the board 1.5 m ahead, the status line `wall ... clearance ...` reads about 1.5 m.
   Bring the board to 0.4 m: `moved back` becomes nonzero and the log prints
   `wall guard: hold point ... moving it back`. (The motors will spin up and down as the
   policy reacts; that is expected with props off.)
4. `structure_hold trip`: within a second PX4 is in Position (or Altitude) mode, and a
   second `commander mode ext1` is refused (trip latched).
5. Disarm. The trip clears (`latched trip: none`). `param set SH_BENCH 0` before anything
   flies.
6. Pull the ulog: `ulog2csv -m structure_hold_status flight.ulg` (pyulog) should show
   states 1, 2 and 3, `trip_reason` 1 and a nonzero `standoff_shift_m` during step 3.

Only after both parts pass: tethered hover, tuned PX4 first, then the policy with the
watchdog armed (`docs/real-flight-gap.md`, "What only hardware gives").

## CPU budget

The module's work per 10 ms tick is one policy step (observation, history shift, the MLP)
plus the watchdog and wall guard, which are a few dozen floating-point operations. The MLP
dominates: v19 and v22 are a 315-64-64-4 ELU network, 24,512 multiply-adds and 128 `expm1f`
per step (`tests/test_policy.cpp` prints both). On an STM32H7 at 480 MHz (Pixhawk 6C) with
its single-precision FPU, at 3 to 6 cycles per multiply-add with the weights streamed from
flash, that is an estimated 150 to 300 us, 1.5 to 3 % of one core at 100 Hz (not measured yet);
the bench check above measures it and flags anything over 1 ms. On the x86 cloud container
the same step takes about 30 us (`test_policy`, informational).

Memory: the observation history (315 floats), the MLP's two scratch rows (630 floats) and
the weights (about 25k floats, 100 KB, in flash as `static constexpr`) are members or
constants, never on the stack, so the work queue's small stack only holds a few locals.

Wake-ups: `Run()` is woken by `vehicle_angular_velocity`, which PX4 publishes at
`IMU_GYRO_RATEMAX` (400 Hz by default, up to 8 kHz on some boards). Since 2026-11 the
subscription is rate-limited to one wake-up per 2 ms, so a fast gyro does not cost
thousands of empty wake-ups a second; the 10 ms tick is unchanged (10 to 12 ms with a
400 Hz gyro, 12 ms in Gazebo's 250 Hz). `structure_hold status` prints the worst control
step since boot and the tick-interval perf counter; record both with the bench numbers.

## Dry run in SITL

The disarmed script runs against SIH, which has no forward rangefinder, so it streams one
over MAVLink:

```bash
python3 sim2real/px4/bench_test.py --fake-range 1.5 --expect SH_WALL_MIN=0.7 \
    --out runs/bench_sih/summary.json      # with a px4_sitl_default SIH instance running
```

2026-10-27 (cloud, PX4 v1.17.0 SIH): every scored check passed; inference timing reported
as informational.
