# Log health check: is PX4 really flying on the VIO?

Item 2 of the "Features that sell" research: the most common integration pain on the PX4 and
ModalAI forums is a VIO that is wired in but not doing what everyone thinks. EKF2 may not be
fusing it at all, it may be in the wrong frame, turned away from north, or later than
`EKF2_EV_DELAY` says, and the first sign is often a drone that drifts or toilet-bowls. This
tool reads one PX4 flight log and says, in plain English, which of those is wrong and what to
set.

```
python -m eval.log_health flight.ulg [--json report.json]    # needs pip install -e .[ulog]
```

It prints a one-line verdict and one line per check (OK, WARN, FAIL, or `--` when the log
does not hold enough to say), and exits 1 when anything failed. The JSON has every number
behind each line. It reads only the log: no parameters file, no ground station, no truth.

## What it checks

| Check | From the log | FAIL / WARN when |
| --- | --- | --- |
| Vision reaches PX4 | `vehicle_visual_odometry` rate and gaps while airborne | no vision at all (FAIL; it says when the messages went to `vehicle_mocap_odometry` because they carry `estimator_type` MOCAP, which EKF2 never fuses), or every position NaN (FAIL: PX4 leaves it empty for a `frame_id` other than `LOCAL_NED`, `LOCAL_ENU`, `LOCAL_FRD`, `LOCAL_FLU`); gaps over 0.5 s (WARN) |
| Vision fused | EKF2's aiding flags (`estimator_status_flags`, or `estimator_status.control_mode_flags`), `EKF2_EV_CTRL`, `EKF2_EV_QMIN` against the samples' quality | vision position used under 50 % of the airborne time (FAIL, saying whether `EKF2_EV_CTRL` or the quality floor is why); under 95 % (WARN) |
| Vision consistent with IMU | `estimator_aid_src_ev_pos`: rejections, test ratios, innovations | over 20 % rejected (FAIL), over 5 % (WARN) |
| GPS off | the GNSS aiding flags | any GPS fused (WARN: the log does not show GPS-free flight, and GPS hides vision mistakes) |
| Vision frame | the vision's axes against the IMU's (below), and `pose_frame` | height moving the wrong way or mirrored horizontal axes (FAIL: ENU sent as NED); a NED frame turned 20 deg or more from north (FAIL), 10 deg (WARN) |
| Heading | EKF2's heading source; with the compass, how far EKF2's heading moved from the compass once airborne; with a vision attitude, its yaw against EKF2's | no heading source (FAIL); EKF2's heading pulled 20 deg or more off the compass (FAIL), 10 deg (WARN) |
| Vision delay | the vision's real age when PX4 stamps it, measured against the IMU (below), against `EKF2_EV_DELAY` | off by 80 ms or more (FAIL), 30 ms (WARN), with the value to set; not judged when the frame check failed |
| Height reference | `EKF2_HGT_REF` and the height sources EKF2 actually fused | vision as the reference (WARN, `docs/field-faults.md`); GPS as the reference (WARN, below) |

The thresholds are first choices from the test flights below, not tuned on real flights yet:
the frame and delay bands sit at about twice the scatter the method showed on the correct
set-ups (5 deg, 15 ms).

## How frame and delay are measured

Not against EKF2's own position: EKF2 is pulled onto whatever vision it fuses, so comparing
the vision with EKF2 hides exactly these mistakes. Instead:

1. The accelerometer (`sensor_combined`) is turned into the world frame with EKF2's roll and
   pitch, gravity is added back, and it is integrated into a velocity at the IMU's own rate.
   Its heading comes from the compass when the compass is EKF2's heading source: the
   magnetometer levelled with the same roll and pitch, with the offset it had to EKF2's yaw
   on the ground in the 5 s before take-off (declination and calibration) taken out. That
   matters because EKF2 bends its own heading to make a turned vision frame fit (below).
2. The vision velocity is the central difference of its positions, by `timestamp_sample`.
3. Both are band-passed the same way: a 4 s moving average removed (the IMU's drift) and a
   0.5 s one applied (noise). The same filter on both shifts neither.
4. The lag that lines them up best, searched from -0.1 to 0.6 s, is the delay. It is read
   from the vertical motion when there is enough of it (the take-off climb is plenty), since
   the vertical accelerometer hardly depends on the attitude; otherwise from the level
   motion, allowing any turn or reflection between the axes so a wrong frame does not hide
   the delay. Vision jumps (a step faster than 5 m/s and three times the speed around it)
   are cut out with the filter's reach first.
5. At that lag, the best orthogonal 2x2 map between the level axes is fitted per 15 s
   stretch, keeping the stretches that fit well: its turn is the frame's heading offset,
   a reflection is a handedness mistake. The sign of the vertical correlation says whether
   the height is upside down.

Both need motion: a climb of a metre or two and a few metres side to side, or a gusty hover.
Below about 5 cm/s of filtered motion the checks say so and skip.

The delay is counted from the time PX4 put on the sample, which is the time EKF2 subtracts
`EKF2_EV_DELAY` from. So it is the right number for `EKF2_EV_DELAY` whether the VIO syncs
clocks with PX4 (then it is only the VIO's own processing time) or not (then it includes the
link); the check says which, from `timestamp - timestamp_sample`.

With simulator truth in the log (`vehicle_local_position_groundtruth`, SITL) the delay is
measured against truth as well, as a cross-check of the IMU method.

## Tested on Gazebo logs with mistakes made on purpose

Seven PX4 v1.17.0 SITL flights in Gazebo (x500, `wind_8ms` scenario, take-off to 2.5 m, 40 s
hover, land; the cloud image, commit e185a80 plus this change). The estimate bridge sends a
clean copy of Gazebo's truth, 100 ms late, as ODOMETRY, with one thing wrong per flight:
`health_*` profiles (`sim2real/estimate_bridge/profiles/`) and the bridge's `frame_mistake`
for the frame errors, scratch copies of `ekf2_ev.params` for the parameter errors. The last
flight is the noisy benchmark VIO (`vio_default`, seed 1: noise, bias walk, a jump, four
dropouts, delay drawn at 100 ms).

| Flight | What was wrong | Verdict | Frame | Delay (IMU / truth) |
| --- | --- | --- | --- | --- |
| `health_good` | nothing | MOSTLY OK (height reference only) | x axis -2 deg from compass north, OK | 92 / 89 ms, OK against 100 |
| `health_late` | VIO 300 ms late, `EKF2_EV_DELAY` 100 | PROBLEM: Vision delay, "set about 291 ms" | -1 deg, OK | 291 / 288 ms, FAIL |
| `health_yaw90` | VIO frame turned 90 deg, declared NED | PROBLEM: Vision frame, Heading | +88 deg from compass north, FAIL; EKF2's heading pulled -79 deg off the compass | 88 / 89 ms, not judged |
| `health_enu_as_ned` | ENU numbers in the NED fields | PROBLEM: Vision consistent with IMU (36 % rejected), Vision frame (height upside down, axes mirrored), Heading | mirrored, vertical correlation -0.92, FAIL | 144 / 88 ms, not judged |
| GPS left on | `EKF2_GPS_CTRL` 7, `SYS_HAS_GPS` 1 | MOSTLY OK: GPS off (WARN), Height reference | -5 deg, OK | 92 / 88 ms, OK |
| vision not fused | `EKF2_EV_CTRL` 0 | PROBLEM: Vision fused, "EKF2_EV_CTRL does not ask for vision position" (PX4 never armed) | skipped, no motion | skipped |
| `vio_default` | nothing (noisy VIO) | MOSTLY OK: Vision reaches PX4 (two gaps over 0.5 s, the dropouts), Height reference | -4 deg, OK | 95 / 94 ms, OK |

Every mistake was named, and nothing was flagged on the correct set-ups that was not really
there. The ENU-as-NED flight shows why the delay is not judged when the frame is wrong: EKF2
swung its heading and the vehicle around for the first 15 s, and the IMU method, which leans
on EKF2's attitude, read 144 ms against truth's 88 ms. The noisy flight moved the delay by
20 ms before vision jumps were cut out; with the cut it matches truth.

Two things the flights showed about EKF2:

- **A turned vision frame turns EKF2's heading.** With the frame 90 deg off and the compass
  as the heading source, EKF2 did not reject the vision: it turned its own heading 79 deg
  away from the compass so the vision fit, and the vehicle held position with the wrong
  heading. Against EKF2's attitude the vision then looks only 5 deg off; only the compass
  shows it. Hence the compass-based heading in step 1.
- **`EKF2_HGT_REF` 1 is GPS, not the barometer.** PX4 v1.17's values are 0 barometer,
  1 GPS, 2 rangefinder, 3 vision (`src/modules/ekf2/module.yaml`; 1 is the default). With no
  GPS EKF2 falls back to the barometer (`Ekf::checkHeightSensorRefFallback`), which is why
  every benchmark flight on `ekf2_ev.params` and `px4_tuned_v2.params` (both 1) really flew
  on barometer height, but the GPS-left-on flight above fused GPS height with that same
  value. The check warns about it. The repo's own files called 1 "barometer"; on Zain's
  call (2026-11-02, `docs/decisions.md`) every set and doc now says 0. Re-flown on 0 (`experiments/LOG.md`, 2026-11-02 18:35 to 18:41): the `health_good` flight is OK on every check, and tuned PX4 flew barometer height on both values with seed 2's p95 within its run-to-run scatter (0.17 to 0.33 m).

To fly these again (cloud container or GPU server, Docker image built):

```
python -m sim2real.estimate_bridge.run_estimate wind_8ms --profile health_yaw90 --hover-s 40
python -m eval.log_health runs/<run folder>/flight.ulg
```

## Limits

- It needs motion; a log of a still hover gets "not enough motion" for frame and delay.
- The compass-based heading assumes the compass is good in flight. Next to steel or with
  motor current interference the Heading check can blame the vision for a compass problem;
  its text says both are possible.
- A vision attitude is checked only for its yaw against EKF2's, and an FRD frame's offset is
  reported, not judged (EKF2 rotates FRD frames itself when it has the attitude).
- Camera-to-IMU offset (`EKF2_EV_POS_X/Y/Z`) is not checked yet: it shows as velocity that
  follows the yaw rate, which needs a log with yawing.
- Tested on SITL only. The first partner logging flight (`docs/partner-pack.md`) is its
  first real log.
