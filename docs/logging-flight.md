# The first real flight: a logging flight, no policy

> On hold since 2026-10-30 (software only, `docs/decisions.md`): the estimate fit this flight
> was for comes from public datasets now (`docs/vio-fit.md`). The checklist stays for the
> first flight on a design partner's drone.

The biggest unknown between the simulators and a real drone is the estimate: every
robustness number so far is measured against a made-up VIO (`docs/real-flight-gap.md`,
row 1). One log of a real drone hovering by hand in front of a wall, with its VIO and an
independent truth both recorded, fits rows 1, 3, 6 and 7 of that table and gives the hover
thrust for row 4. No policy flies on this flight. This page is the checklist, so the flight
can happen the day a drone is available.

## What the log must contain

| Signal | PX4 topic | Why |
| --- | --- | --- |
| VIO pose and velocity, as the drone produces it | `vehicle_visual_odometry` | Fit `envs/estimate` (delay, noise, bias walk, jumps, dropouts) to the real VIO (row 1) |
| Truth: mocap indoors or RTK outdoors, **logged, never fused** | `vehicle_mocap_odometry` (mocap over MAVLink `ODOMETRY`, frame MOCAP) or the RTK receiver's `sensor_gps` | The error of the VIO is VIO minus truth; without truth there is nothing to fit (row 10) |
| IMU at full rate | `sensor_combined` | Vibration and gyro noise for `SensorNoise` (row 6); EKF2 replay |
| Magnetometer | `vehicle_magnetometer` | Heading error near the structure (row 3) |
| EKF2 output and innovations | `vehicle_local_position`, `estimator_*` | What the module would have read; replay with another `EKF2_EV_DELAY` (row 7) |
| Forward rangefinder, if fitted | `distance_sensor` | Range noise near the wall, for the wall guard (`SH_WALL_MIN`) |
| Hover thrust | `hover_thrust_estimate` | The airframe's hover point (row 4) |
| Camera timestamps vs IMU | VIO's own log | The VIO latency, for `EKF2_EV_DELAY` |

`sim2real/px4/logging/logging_flight.params` sets PX4's logger to record all of these
(`SDLOG_PROFILE` 147: default, estimator replay, high rate, computer vision) from boot to
disarm, and keeps Structure Hold off (`SH_EN` 0). It changes nothing else.

## Before the day

- [ ] Drone flies stock PX4 v1.17.0 (the pinned release) in Position mode on its VIO,
      GPS off, already test-hovered by its builder or by us in open space.
- [ ] Truth source booked: a mocap room (preferred, mm-level) or an RTK base and rover.
      Mocap streams over MAVLink `ODOMETRY` with frame `MAV_FRAME_MOCAP_NED`, so PX4 logs
      it as `vehicle_mocap_odometry`, and `EKF2_EV_CTRL` fuses only the VIO. Confirm with a
      bench check that EKF2 does not fuse the mocap stream.
- [ ] A wall: flat, textured (brick, concrete, posters; not a white wall), 3 m wide or
      more, with 4 m clear in front. Indoors for the mocap; under a bridge or by a building
      for RTK.
- [ ] `logging_flight.params` loaded (QGC, Parameters, Tools, Load from file), reboot,
      `param show SDLOG_PROFILE` says 147.
- [ ] SD card with 4 GB or more free; log one 1 minute ground test and check its size
      (this profile logs far more than the default; size the card from that test).
- [ ] Clocks: mocap and PX4 timestamps linked (PX4 logs both topics on its own clock when
      mocap arrives over MAVLink, which is enough). For RTK, the receiver feeds PX4 directly.
- [ ] Battery fully charged; a second battery for a repeat.
- [ ] Safety: a pilot who flies Position mode on VIO confidently; kill switch on the RC;
      tethered first hover if the room requires it.

## The flight (about 8 minutes of air time, one battery)

Fly in Position mode on VIO, by hand. Note the log time of each step on paper.

1. **Ground, 30 s, disarmed then armed at idle.** VIO and truth both running. (Static
   noise, vibration at idle.)
2. **Hover in open space, 2.5 m up, 60 s, sticks still.** (Hover thrust, VIO noise and
   bias drift with nothing in view but the room.)
3. **Hover facing the wall at 3 m, then 2 m, then 1.5 m, 60 s each,** heading square to
   the wall. (VIO quality as a function of distance and texture: the hold range is 1 to
   3 m.)
4. **At 1.5 m, yaw slowly 30 deg left and right of square,** twice. (Heading error,
   off-axis rangefinder, magnetometer near the structure.)
5. **At 1.5 m, slide 2 m left and right along the wall, slowly,** twice. (VIO under motion
   parallel to the surface, the motion that loses texture tracking.)
6. **Climb and descend 1 m in front of the wall,** twice.
7. **If safe, one fast stop:** at 2 m from the wall, fly 2 m along it at 1 m/s and
   release the sticks. (VIO under acceleration; the closest thing to a gust by hand.)
8. Land in open space, disarm. Wait 10 s before powering off so the log closes.

Repeat the whole sequence on the second battery if time allows. Two logs beat one.

## After the flight

- [ ] Copy the `.ulg` files and the VIO's own log into `runs/<date>_logging_flight/` (not
      git: `runs/` is ignored).
- [ ] Line in `experiments/LOG.md`: date, drone, VIO, truth source, wall, the paper notes.
- [ ] Check the log: `ulog_info` lists `vehicle_visual_odometry` and
      `vehicle_mocap_odometry` (or `sensor_gps`) at their full rates, with no gaps.
- [ ] Then the fits, in this order: VIO error vs truth into `envs/estimate`'s delay, noise,
      bias walk, jump and dropout rates (row 1); `EKF2_EV_DELAY` from the replay (row 7);
      IMU noise into `SensorNoise` (row 6); hover thrust and mass into a new airframe in
      `envs/structure_hold/airframes.py` (row 4). The retrain on the fitted estimate is the
      training run that matters most (`docs/real-flight-gap.md`, "Next training target").

## What this flight is not

Not a policy flight, not a hold-quality number, and not compared against tuned PX4: it
measures the inputs. The first hold-quality flight comes after the props-off bench test
(`docs/bench-test.md`) and a tethered hover, with tuned PX4 flown first on the same day as
the baseline.
