# Metrics

Every result in this project (mock simulator, Isaac Lab, PX4 SITL, real flights) is scored
by one function, `eval.metrics.compute_metrics`, so a number means the same thing wherever
it came from. This page repeats the definitions in the code's docstrings. If the two ever
disagree, the code is what produced the numbers; fix whichever is wrong in its own PR.

## Inputs

- **True position** `p(t)`, ENU, metres. Simulator state, SITL ground truth
  (`vehicle_local_position_groundtruth`) or motion capture. Never the flight controller's
  estimate: with a corrupted estimate, scoring against it hides the error we measure. The
  ulog reader refuses to fall back to the estimate unless asked (`--allow-estimate`).
- **Hold setpoint** `p_sp(t)`, ENU. A fixed hold point, or PX4's position setpoint held
  between samples.
- **Wall**: a point on the plane and its unit normal `n`, pointing from the drone toward the
  surface (the structure-frame x axis in `docs/conventions.md`).
- **Gusts**: `(start, end)` times in seconds. A step or impulse has `start == end`.
- **Commands** (optional): body-rate command (rad/s, FLU) and normalised collective thrust,
  on their own time base.
- **Window** (optional): `(t0, t1)` to skip take-off and settling.

## Definitions

Position error `e(t) = |p(t) - p_sp(t)|` (3-D, metres).

| Key | Unit | Definition |
| --- | --- | --- |
| `rms_m` | m | Root mean square of `e` over the window |
| `p95_m` | m | 95th percentile of `e` over the window. **Headline: Gate 2 needs 30% lower than tuned PX4 at 8 m/s wind, 1 m from the wall, corrupted estimate** |
| `max_m` | m | Largest `e` in the window |
| `worst_toward_surface_m` | m | `max((p - p_sp) . n)`, floored at 0. How far the drone was pushed toward the wall past its setpoint. Moving away never counts. **Headline** |
| `min_clearance_m` | m | Smallest distance from the vehicle centre to the wall plane |
| `contacts` | count | Separate episodes of touching the wall: centre within `contact_radius_m` (default 0.25 m) of the plane, or the simulator's contact flag set. Each unbroken run of contact samples counts once |
| `recovery_s` | s, per gust | Time from the gust's **end** until `e` is within 10 cm and stays there for 1 s. 0 if already inside when the gust ends. `inf` if it never settles before the next gust starts or the window ends |
| `recovery_max_s`, `recovery_mean_s` | s | Max and mean of `recovery_s` (`inf` propagates on purpose) |
| `rate_roughness_rad_s2` | rad/s² | `sqrt(mean(|dω_cmd/dt|²))` over the body-rate command vector, time-weighted |
| `thrust_roughness_1_s` | 1/s | The same for the collective thrust command |
| `duration_s` | s | Length of the scored window |

Keys that need an input which was not given (no wall, no gusts, no commands) are NaN.

## Sampling

Position metrics are computed on a uniform 100 Hz grid: position linearly interpolated,
setpoint and contact flag zero-order held. Results therefore do not depend on log rate or
gaps (tested: a 30 Hz irregular log and a 250 Hz log agree within 3%). Roughness uses the
native command samples, because interpolation would smooth the steps it measures.

## Defaults

`GRID_DT_S = 0.01`, `RECOVERY_TOL_M = 0.10`, `RECOVERY_HOLD_S = 1.0`,
`CONTACT_RADIUS_M = 0.25` in `eval/metrics.py`. Changing a default changes every result, so
it needs a row in `docs/decisions.md`. Set `contact_radius_m` to the test airframe's
centre-to-prop-tip distance once the airframe is chosen.

## Usage

```python
from eval.metrics import Surface, Trace, compute_metrics
m = compute_metrics(Trace(t, pos, hold), Surface([1, 0, 0], [1, 0, 0]), gusts=[(5.0, 6.5)])
```

```
pip install -e .[ulog]
python -m eval.metrics flight.ulg --wall-point 0,1,0 --wall-normal 0,1,0 --gust 4,6 --window 5 60
```

The ulog reader (`eval/ulog.py`) is the PX4 boundary for logs: it converts NED to ENU and
FRD body rates to FLU, and takes upward thrust as `-thrust_body[2]`.

Real flights: with no SITL ground truth the reader scores against `vehicle_mocap_odometry`
(mocap over MAVLink `ODOMETRY`, estimator type MOCAP, logged with `SDLOG_PROFILE` bit 7 at up
to ~33 Hz; never fused). Mocap is in its own frame, so give either `--hold` (the hold point
in the mocap frame) or `--align T0 T1`: the EKF estimate is fitted to mocap over that stretch
by a yaw and an offset (needs 0.5 m of horizontal motion, so use take-off and the move to the
hold point), which carries the setpoint into the mocap frame. The wall is then given in the
mocap frame too. The fit's yaw, offset and residual are printed with the metrics.
