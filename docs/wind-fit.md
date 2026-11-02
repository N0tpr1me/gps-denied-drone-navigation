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

RESULTS_PLACEHOLDER

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
