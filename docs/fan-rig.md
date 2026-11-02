# Fan rig: gusts against a real wall

The Phase 3 rig that tests the gust claim on the real X500 V2: a wall panel, a bank of drum
fans blowing into it, fast wind sensors, and a net cage. It reproduces the Gazebo benchmark's
wall scenarios so the real drone flies Gate 2's comparison (tuned PX4 against Structure Hold)
in measured air, and it gives the first measured data for the near-wall wind term
(`docs/real-flight-gap.md`, row 2). Nothing is built or bought yet.

Parts, prices, the layout drawing, the safety checklist and what Zain decides are in the
"Fan rig build plan" doc (https://claude.ai/artifact/GuiVwxa7hrwdZBK8zKpay1). The drone-side
items (prop guards, tether, lidar, truth system) are in the "Test drone order" doc. This page
keeps the part that ties the rig to the code: the targets, the calibration and the procedure.

## Targets

The cases are `sim2real/gazebo/scenarios/wall1m_*.yaml`: hold 1 m from the wall, mean wind
into the wall, 60 s hold, two gusts at 15 s and 40 s into the hold. A scenario's
`mean_speed` is the free-stream wind. The disturbance model shelters a point `d` from the
wall to `1 - (1 - g0) exp(-d / L)` of it (`envs/disturbance/core.py`; nominal `L` 1.06 m,
`g0` 0), about 0.61 at the hold point. So the rig is set on free-stream speed with the wall
panel rolled away, and the speed with the wall back in is a measurement of `L` and `g0`.

| Case | Scenario | Free stream, wall away (m/s) | Model at 1 m, wall in (m/s) | Turbulence sigma (m/s) | Gust adds (m/s), 1 to 2 s |
| --- | --- | --- | --- | --- | --- |
| Calm | `wall1m_calm_0ms` | 0 | 0 | 0 | none |
| 4 m/s | `wall1m_wind_4ms` | 4.0 +- 0.3 | about 2.4 | 0.5, correlation about 1 s | 1 to 3 |
| 8 m/s | `wall1m_wind_8ms` | 8.0 +- 0.4 | about 4.9 | 1.0, correlation about 1 s | 2 to 4 |

Fans cannot make the sim's clean gust step, and a fan jet is not uniform. The rule: match the
mean and sigma to the table, log the real wind at 20 Hz, and replay the measured trace in the
mock and Gazebo, so each real case has a sim twin flown in the same air. The rig holds at
1.5 m height (the scenarios use 2.5 m), so 1.5 m copies of the three scenarios are flown in
Gazebo by both controllers before the first rig session.

## Wind sensors and logging

- S1 at the hold point for calibration, 0.8 m to the side along the wall during flights.
- S2 on the jet axis 0.4 m downwind of the fan screen (1.6 m from the hold point), every run.
  Calibration gives the S2 to hold point ratio; a run whose S2 mean over the hold is more
  than 5 % off its calibration value is re-flown, not scored.
- S3 0.3 m from the wall, for the shelter fit.
- One clock: the logger sends each sensor's speed and the gust switch to PX4 as MAVLink
  `NAMED_VALUE_FLOAT`, which PX4 logs as `debug_key_value`. That needs the debug bit in the
  logging profile (`SDLOG_PROFILE` 147 + 32 = 179). Checked in SIH on v1.17.0
  (`python -m sim2real.px4.rig_logging_check <PX4_DIR>`): `debug_key_value` has a queue of
  one, so keys sent back to back are lost (all but the last); send them one at a time,
  10 ms or more apart (100 % logged), or all values in one `DEBUG_FLOAT_ARRAY` per tick
  (`debug_array`). Keys keep 9 characters. Mocap is logged at most every 30 ms (~33 Hz).

## Calibration, every session

1. Hot-wire sensors against the reference anemometer at six speeds from 0 to 10 m/s.
2. Wall away: a 5 x 5 grid at 0.25 m over 1 m x 1 m at the hold point, 30 s per point. Set
   the bank to the table's free-stream speeds; every grid point within 15 % of the centre.
3. Turbulence: S1 at the centre, 120 s at 20 Hz per case. Sigma after removing the mean;
   correlation time where the autocorrelation falls to 1/e.
4. Gusts: 10 shots per case; rise time and peak added speed at S1, tuned into the table's
   range. The averaged shape is the gust of the sim twin.
5. Wall in: the centre point, and points 0.3, 0.5, 1, 1.5, 2 and 3 m from the wall. Fit `L`
   and `g0`; a change to the disturbance ranges is a Proposed row and its own PR.

## Procedure

On a partner's drone (software only since 2026-10-30) the build, checks, safety limits and
the session folder they send are in `docs/partner-pack.md`, and `eval/partner_session.py`
scores the session. Before the first session: a real tuned-PX4 baseline (autotune on the real airframe, then the
position-loop values of `eval/baselines/px4_tuned_v1.params`, saved as its own file), the
logging flight done and the estimate model refitted (`docs/logging-flight.md`). Ready in
code: `eval/ulog.py` scores against `vehicle_mocap_odometry` (`docs/metrics.md`), and the
1.5 m scenarios are `sim2real/gazebo/scenarios/rig_wall1m_*.yaml` (fly them with
`run_baseline --scenarios rig_wall1m_calm_0ms rig_wall1m_wind_4ms rig_wall1m_wind_8ms`).

Per session: calibrate, then calm, 4 m/s, 8 m/s. Within a case the two controllers alternate
run by run, 8 runs each, as Gate 2's 8 seeds. Run i of both uses the gust schedule drawn
from seed i. One run: fans on 30 s before take-off; take off in Position mode 2 m from the
wall; move to the hold point; engage (Structure Hold, or stay in Position mode for the
baseline); the script fires the gusts at +15 s and +40 s; hold 60 s untouched; disengage,
back off to 2 m, land. Tethered shakedown runs are never scored.

The wind fit: the calm, free-air (wall away, anemometer reading) and wall-in holds go
through `eval/fit_wind.py` (`docs/wind-fit.md`), which returns the drag, turbulence and the
shelter `L` and `g0` from the drone's own accelerometer, so the hot-wire grid of step 5 is a
cross-check rather than the only measurement.

Scoring: `eval/metrics.py` on the truth over the 60 s hold, gusts from the logged switch
times. Report each case as in `docs/baseline.md` (mean p95 against tuned PX4, worst toward
the wall, crashes, handbacks) with the sim twin beside it, and log every scored run in
`experiments/LOG.md`, aborts included.
