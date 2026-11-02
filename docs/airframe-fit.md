# Airframe fit: a customer's drone from one flight

The "airframe fitting kit" of the build plan's client package, and gap 9 of
`docs/market-research.md` ("bring their own drone without weeks of tuning"):
`eval/fit_airframe.py` turns one PX4 log, the drone's parameter file and its weighed mass
into an airframe for the env, in the shape of `gz_x500()` (`envs/structure_hold/airframes.py`).
A training config then loads it with `airframe: path/to/airframe.yaml`, and the released
policy is fine-tuned on it (`params.init_actor_onnx`, `training/train_ppo.py`). It is proven
on the Gazebo x500, whose true numbers come from its SDF.

```
python -m training.launch --config eval/configs/airframe_fit_gz_x500.yaml --smoke --device cpu  # model-made flight
python -m eval.baselines.autotune --params eval/baselines/px4_tuned_v1.params --set SDLOG_PROFILE=147
python -m training.launch --config eval/configs/airframe_fit_gz_x500.yaml --device cpu          # the Gazebo check
```

## The flight

No new flight script: the identification flight is PX4's own attitude autotune (the
Autotune button in QGroundControl), which steps the torque on roll, pitch and yaw in turn.
A partner flies it anyway, since the tuned-PX4 baseline starts with an autotune
(`docs/fan-rig.md`, `docs/partner-pack.md`). The log needs the partner pack's logging
profile, `SDLOG_PROFILE` 147: its high-rate bit logs `actuator_motors`,
`vehicle_torque_setpoint` and `sensor_combined` at the loop rate. With the default profile
those topics come at 10 Hz or not at all, and the fit refuses the log.

What the customer sends: the `.ulg`, the parameter file saved after the flight, and the
mass on a scale. The inertia is optional (below).

## What is fitted

| Env field | From | How |
| --- | --- | --- |
| `quad.max_thrust`, `actuation.esc_min_frac` | accelerometer z times mass against the motor commands | per rotor `k (e + (1 - e) u)^2` on the lagged command; `e` and the lag on a grid, `k` by least squares |
| `quad.motor_tau` | the same | the lag that fits best |
| `actuation.torque_per_unit` / `quad.inertia` | raw gyro change over 0.1 s windows against the integral of the lagged torque setpoint | per axis, against all three commands plus a trim; lag and delay on a grid |
| `quad.max_torque` | PX4's mix from `CA_ROTOR*` and the fitted hover command | where the first rotor saturates, linearised as in `gz_x500()` |
| `rate_pid` | the parameter file | `MC_*RATE_P/I/D` times `_K`, `_FF`, `MC_*R_INT_LIM`, `IMU_GYRO_CUTOFF`, `IMU_DGYRO_CUTOFF`, `MC_YAW_TQ_CUTOFF` |
| `quad.mass` | the scale | |

Only the ratio of torque to inertia reaches the rate dynamics, so without a measured
inertia the fit scales the x500's by mass and arm length and sets `torque_per_unit` to the
fitted ratio times that guess. The rate gains matter most: the policy flies on top of the
customer's own rate loop, so the env has to run theirs.

`fit.json` has every number, the replay check (the share of the gyro's motion over 0.1 s
windows the fitted model explains, R^2 per axis, with and without the other axes'
commands) and any cross-coupling between axes. `airframe.yaml` in the run folder is the
file a training config loads.

## The Gazebo check

One PX4 autotune flight of the Gazebo x500 in calm air, flown in the cloud container on
2026-11-02 (`runs/20261102-003502_px4_autotune_5220d89`: the tuned-PX4 parameter set with
`SDLOG_PROFILE` 147, a clean estimate, 41 s of roll, pitch and yaw steps at 2.5 m; the
autotune left the gains as they were, which the fit does not need). The truth is the x500's
SDF as `gz_x500()` carries it. Run `20261102-010250_airframe_fit_gz_x500_s0_204f09f`
(`experiments/LOG.md`):

| Number | SDF | Fit | Error |
| --- | --- | --- | --- |
| `max_thrust` | 34.19 N | 34.06 N | -0.4 % |
| `esc_min_frac` | 0.150 | 0.160 | +6.7 % (one grid step) |
| `motor_tau` | 18.8 ms | 18.0 ms | -4.0 % |
| Roll torque per unit / inertia | 143.9 rad/s^2 | 140.6 rad/s^2 | -2.4 % |
| Pitch | 232.5 rad/s^2 | 226.3 rad/s^2 | -2.7 % |
| Yaw | 15.53 rad/s^2 | 15.41 rad/s^2 | -0.7 % |
| `max_torque / inertia`, roll / pitch / yaw | 89.2 / 89.2 / 4.21 | 87.6 / 87.3 / 4.20 | -1.8 / -2.2 / -0.2 % |

The fitted hover command is 0.727, against PX4's own hover thrust estimate of 0.728. The
torque path fits a 14 ms lag plus a 4 ms delay (the motor lag seen through the rate
loop's filters), and the replay explains 99.7, 99.4 and 98.3 % of the gyro's motion per
0.1 s window on roll, pitch and yaw.

**Cross-coupling.** Roll with its own command alone explains only 74 % of its motion: a yaw
command moves roll by 10 % of what a roll command does, and roll moves yaw by 8 %. PX4's
mix for the x500 (`CA_ROTOR*` in the parameter file) does not match the SDF's rotor
positions exactly, so each yaw step leaks into roll. Fitting each axis on its own command
only put roll 37 % low; regressing on all three commands at once recovers it. The env
drives each axis with its own command, so this x500 leak is not in the mock; it is small
next to the training's randomisation and the benchmark already flies through it, but a
partner drone with a larger leak goes into `docs/real-flight-gap.md`.

The parameter file gives the rate loop exactly (P, I and D equal to `gz_x500()`'s, `K`
1). On model-made flights (the tests and the smoke run) the thrust and torque numbers come
back within 1 to 3 %, the motor lag within 1 ms and the coupling within 0.01.

## Limits

- **Cross-coupling is measured, not modelled.** The env drives each axis with its own
  command only. A drone whose mix leaks between axes, as the Gazebo x500 does, shows it in
  `coupling`; anything over a few percent belongs in `docs/real-flight-gap.md`.
- **The thrust curve is PX4's ESC model shape.** Rotor speed linear in the command, thrust
  quadratic in speed. A drone with `THR_MDL_FAC` set already linearises its thrust, which
  shows up as a large `esc_min_frac`; the fit stays right for the env, which uses the same
  shape.
- **Collective range.** An autotune holds altitude, so the thrust curve is fitted over a
  narrow band of collective around hover, plus the controller's corrections; max thrust is an
  extrapolation. A short climb and sink in the same flight widens the band.
- **Rotor drag and wind.** The thrust fit assumes still air. Fly it on a calm day.
- **What it does not fit:** EKF2's tuning, sensor delays and vibration, the estimate's
  error (that is `docs/vio-fit.md`), and the module's tick on the customer's flight
  controller (the bench test measures it, `docs/bench-test.md`).
