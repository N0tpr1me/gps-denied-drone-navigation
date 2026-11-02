# Wind fit: the air model from flight logs

The build plan's Phase 3 "wind model fit" (its main asset), done in software:
`eval/fit_wind.py` reads PX4 logs of holds in free air and at several distances from a wall,
in the same wind, and recovers the numbers `envs/disturbance` is built from: the drag
coefficient, the mean wind, the turbulence and the shelter near the wall. Until now the
shelter was fitted to one published data point and the training ranges were wide on
purpose (`envs/disturbance/README.md`, `docs/real-flight-gap.md` row 2). This tool is what
turns the first real wind log, from a partner's drone or a fan rig session, into a fitted
model. It is proven here on Gazebo flights where the true wind is known.

```
python -m training.launch --config eval/configs/wind_fit_gazebo.yaml --smoke --device cpu  # model-made logs
python -m training.launch --config eval/configs/wind_fit_gazebo.yaml --device cpu          # the Gazebo check
```

## How it works

The accelerometer measures the specific force, `(thrust + aero) / mass`, in the body frame.
Thrust is along body z, so body x and y times the mass are the aerodynamic force on those
axes, whatever the attitude or acceleration. With the attitude, and no vertical aero force,
that is the horizontal force in the world frame. A calm hold gives the accelerometer's own
offset, which is taken out first.

The model's force is drag on the air relative to the drone, `F = (k_lin + k_quad |v|) v`,
so each sample's relative air `v` follows from `F`, and the air is `v` plus the drone's own
velocity. From the air: the mean (speed and heading), the turbulence per horizontal axis,
and its correlation time (where the autocorrelation falls to 1/e, corrected for the hold's
own mean).

Across holds:

- **Drag.** A free-air hold with a reference wind speed (an anemometer at the hold point,
  or in Gazebo the plugin's trace) gives `k_quad`; two different speeds give `k_lin` too.
  Without a reference, the configured `k_quad` is used and every speed scales with
  `1 / sqrt(k_quad)`.
- **Shelter.** In one case (one fan setting), each wall hold's mean air along the free
  stream, over the free-stream speed, is the shelter `g` at that distance.
  `g(d) = 1 - (1 - g0) exp(-d / L)` is fitted over all holds. This needs no drag
  calibration and no mass: it is a ratio of speeds.

Reported but not fitted: the crossflow at each wall hold (a sign of the windward
deflection term in `docs/near-wall-flow.md`) and the turbulence near the wall against the
sheltered free-air value (above 1 means wake turbulence).

`fit.json` in the run folder has every hold's numbers, the fitted model, the error against
the truth when the config gives one, and a proposed `disturbance` block beside the current
default ranges, with `covered` saying whether each default range already holds the fitted
value. Changing the training ranges is a Proposed row in `docs/decisions.md` and its own PR
(`docs/fan-rig.md`, calibration step 5).

## What a log needs

| Need | Topic or field | Note |
| --- | --- | --- |
| Accelerometer | `sensor_combined` (any logging profile) | `vehicle_acceleration` only as a fallback: EKF2's bias estimate soaks up part of a steady aero force |
| Attitude | `vehicle_attitude` | Its heading error rotates the measured force (below) |
| Drone velocity | `vehicle_local_position_groundtruth` (SITL) or `vehicle_local_position` | Small during a hold; the EKF's is fine |
| Distance to the wall | position truth plus the wall, or `distance_m` per hold | A real log has no truth: give the measured standoff (rangefinder or tape) |
| Mass | `airframe` or `mass_kg` | Scales the drag, not the shelter |
| Gust times | `exclude_s` per hold | The rig logs its gust switch; Gazebo runs read them from `config.yaml` |

The procedure is the fan rig's (`docs/fan-rig.md`): a calm hold, free-air holds at two
speeds with the anemometer reading, then wall holds at 0.5 to 3 m in the same wind, each
60 s or more. On a partner's drone it is the same list flown outdoors next to a wall in a
steady wind, with a handheld anemometer for the free-air reference.

## The Gazebo check

Eight Gazebo SITL holds of stock PX4 on the x500 (`sim2real/gazebo/scenarios/windfit_*.yaml`,
60 s each, flown one at a time in the cloud container on 2026-11-02): calm, free air at 8 and
4 m/s, and 0.75, 1, 1.5, 2 and 3 m from the wall in 8 m/s blowing into it, with turbulence
and two gusts per hold. Every disturbance value is fixed in the scenarios, and none is the
model's nominal, so the fit has something to find: shelter `L` 1.4 m and `g0` 0.12 (nominal
1.06 m and 0), `k_quad` 0.032. The free-air reference speed is the plugin's trace over each
hold, as an anemometer at the hold point would read it. Run
`20261102-003514_wind_fit_gazebo_s0_503d2cd` (`experiments/LOG.md`):

| Number | Truth | Fit | Error |
| --- | --- | --- | --- |
| `k_quad` (two speeds) | 0.0320 | 0.0319 | -0.4 % |
| `k_lin` | 0 | 0.0009 | |
| Shelter `L` | 1.400 m | 1.382 m (+- 0.002) | -1.3 % |
| Shelter `g0` | 0.120 | 0.114 (+- 0.001) | -0.006 |
| Turbulence sigma, same air | 0.666 m/s | 0.648 m/s | -2.6 % |
| Turbulence tau, same air | 1.12 s | 1.18 s | +4.6 % |
| Wind heading, 8 m/s / 4 m/s | 0 deg | +1.1 / -5.6 deg | EKF2 heading |

Per hold, the shelter from the IMU and the shelter of the plugin's own air over the same
samples differ by 0.0001 to 0.0026 (0.75 m: 0.471 against 0.474). Fitting the curve to the
plugin's own air gives `L` 1.386 m and `g0` 0.120, so most of the gap to the scenario's
1.4 m is the drone moving about its hold point, not the measurement. The mean air speed
agrees within 0.01 m/s on every hold, and along the wind the air rms error is 0.03 to
0.06 m/s. Across the wind it is 0.11 to 0.45 m/s: EKF2's heading wobbles by about
2 degrees, which turns a 2 N force by 0.07 N; with Gazebo's true attitude the crosswind
force error falls from 0.079 to 0.014 N rms. Every proposed range holds the fitted values,
and so do the current defaults (`covered` true for all four), as it should be when the
scenarios were drawn from inside them.

Turbulence is scored against the plugin's own air over the same samples, not the
scenario's sigma 0.8 m/s and tau 1.5 s: a 60 s hold shows part of the model's spectrum,
and the true air itself showed sigma 0.67 m/s and tau 1.12 s. The calm hold's "speed" of
0.27 m/s rms is the drag inversion's noise near zero force (speed goes as the square root
of force); it is only used for the accelerometer offset.

On model-made logs (the tests and the smoke run) the fit returns `L` and `g0` exactly, the
drag within 3 %, and `k_lin` and `k_quad` from two speeds.

## Limits

- **Horizontal only.** Thrust hides the vertical aero force, so vertical wind (up- and
  downwash on a facade) is not measured.
- **Heading.** A heading error in the attitude estimate turns the measured force with it.
  In Gazebo, EKF2's heading wobble of about 2 degrees is most of the crosswind error;
  near steel, where the compass is off (`docs/market-research.md` gap 10), the wind heading
  and the crossflow inherit the compass error. The shelter, an along-wind ratio, does not.
- **Turbulence statistics need time.** One 60 s hold shows part of the model's spectrum
  (periods up to 300 s), so sigma and tau scatter by 10 to 30 % from one stretch of air to
  the next. More or longer holds narrow it; the fit's per-hold numbers sit beside the same
  numbers on the true air in Gazebo, so the method's own error is visible.
- **The model's shape is assumed.** The fit returns the best shelter curve of the model's
  form; the rms residual says how well that form fits. Lee-side recirculation, vortex
  shedding and yaw torque are not fitted (`docs/near-wall-flow.md`).
- **Mass and drag.** An error in the mass scales `k_quad` by the same factor; it does not
  move the shelter.
