# RTK as the truth for outdoor flights

Every number we report is measured on an independent truth, never on the drone's own
estimate (`docs/metrics.md`). Indoors that truth is motion capture. At a real bridge or pier
there is no motion capture, and inspection buyers fly at real bridges (`docs/market-research.md`,
gap 7; `docs/partner-pack.md`, open item 4). An RTK GPS receiver with a fixed solution knows
where its antenna is to a centimetre or two, so since 2026-11-09 a partner's outdoor session
can be scored on it:

```bash
python -m eval.partner_session <folder>/session.yaml     # session.yaml says  truth: rtk
```

The code is `eval/rtk_truth.py`, wired into `eval/partner_session.py` and named on the page by
`eval/partner_report.py`. What the partner sets up is in `docs/partner-pack.md`, "Truth for
scoring outdoors (RTK)". This page says what the scorer does, what it refuses, and how it was
checked.

## What the scorer does

| Step | What happens | Where |
| --- | --- | --- |
| Find the receiver | Every `sensor_gps` instance in the log and `vehicle_gps_position` are looked at; the topic with the most RTK-fixed samples is read (a drone may carry two receivers) | `read_rtk` |
| Keep fixed solutions only | Only `fix_type` 6 (RTK fixed) is truth. Float (5), differential and plain 3D samples are off by decimetres to metres: they are dropped, never scored | `read_rtk` |
| Into metres | Latitude, longitude and altitude become ENU metres about an origin (the first surveyed wall point, or `rtk.origin`) on the WGS84 ellipsoid | `envs/frames.py` `geodetic_to_enu` |
| Receiver lag | The receiver stamps a sample when it arrives, a little after it was true. The lag (0 to 0.5 s) is measured as the shift that best lines the truth up with the drone's estimate over the take-off; `rtk.delay_s` in the sheet overrides it | `fit_delay` |
| Fit the estimate's frame | As the mocap path does: the EKF estimate is fitted to the truth by a yaw and an offset from arming to the mode switch, the hold point is carried through that fit, and the residual is reported with every run | `fit_frame` |
| The wall | From points surveyed on its face with the RTK rover (`wall.survey`), as the mocap wand gives it indoors. Two points, or more along one line, make a vertical wall; three or more spread over the face fit its plane (a leaning pier, a deck underside). The normal points from the drone to the wall | `wall_from_survey` |
| Score | `eval/metrics.py`, the same code as everywhere, with stretches that have no fixed sample left out rather than drawn through (`max_gap_s`; `scored_s` is the time scored) | `compute_metrics` |

Three things are different from mocap, and the scorer says each of them out loud.

**RTK comes and goes.** Under a deck the fix drops to float. Those samples are left out, and
so is the time they cover: a gap is any stretch where fixed samples are more than 0.5 s apart
(or 2.5 receiver periods, if that is longer). Every run reports `truth_coverage`, the share of
its hold that had a fixed solution, and `truth_coverage_flight`, the same from arming to
disarm. A hold under 95 % is listed in the report. A hold under 50 % is not scored at all: its
numbers would describe some other part of the hold. Nothing can be said about the time left
out, including whether the drone touched the wall in it.

**Truth must be logged, not fused.** If EKF2 fuses the receiver (the GPS-on envelope,
`docs/field-faults.md`), the estimate and the truth share it: a wrong solution is in both and
does not show, and the moments the receiver loses its fix, which are the estimate's worst,
are the moments left out of the score. The scorer reads what EKF2 really did from its aiding
flags (`estimator_status_flags`, or the same bits in `estimator_status`) between arming and
the end of the hold. A hold with GPS fused is still scored, and flagged: `truth_independent`
false on the row, a warning at the top of `report.md` and of the one-page report, and a line
per hold. When the log has neither topic the row says the check could not be made.

**A straight-up take-off is fitted by offset alone.** The yaw fit needs 0.5 m of sideways
motion. Mocap must refuse a take-off without it, because a mocap frame points anywhere. RTK's
frame is north-up, and so is the EKF's by its compass, so on RTK the yaw is then taken as zero
and only the offset is fitted (`align_yaw_fitted` false). What that can cost is the EKF's
heading error times the sideways distance from the take-off to the hold point, which is small
exactly when the take-off did not move sideways.

## What it refuses

| Message | Meaning |
| --- | --- |
| `log has no sensor_gps` | The receiver was not logged at all |
| `has no RTK-fixed samples (fix types logged: ...)` | The receiver never had a fixed solution: no corrections reached it |
| `is logged at 1.0 Hz; a truth needs 4 Hz or more` | PX4's logger thinned it (below). Load `sim2real/partner/rtk_truth.params` |
| `RTK fixed for N % of the hold, under the 50 % a score needs` | Too little of that hold had a fixed solution |
| `with truth: rtk the wall is wall.survey` | The sheet gave a mocap-style `point` and `normal` |

**Logging.** PX4 v1.17 logs `sensor_gps` once a second unless `SDLOG_PROFILE` has bit 11
(2048, high-rate sensors), and `vehicle_gps_position` (10 Hz) only exists with `SYS_HAS_GPS`
1, which a GPS-free set-up does not have. The first check flight below came back with one
truth sample a second for exactly that reason. `sim2real/partner/rtk_truth.params` sets
`SDLOG_PROFILE` 2195 (the flight set's 147 plus bit 11), `EKF2_GPS_CTRL` 0 and `SYS_HAS_GPS`
0, and the scorer refuses a receiver logged slower than 4 Hz.

## Checked in Gazebo, where the truth is known

Nobody can say how right an RTK score of a real flight was. In Gazebo somebody can. The
estimate bridge's scripted receiver (`sim2real/estimate_bridge/gps.py`: 10 Hz, RTK fixed,
1 cm noise per horizontal axis and 2 cm vertical, scripted drops to float) can now be logged
without being fused (`gps: {fused: false}`; the runners lay `gps_logged.params` over the PX4
set), so one log holds the receiver's samples and the simulator's truth. Four profiles fly
the benchmark hold that way: `rtk_truth_at_engage` and `rtk_truth_at_hover`, and
`rtk_truth_float_at_*`, where the fix drops to float for 15 s, 20 s into the hold, with the
solution 0.8 m off.

`eval/rtk_check.py` scores each such flight through `eval.partner_session`, the path a
partner's session takes, with the wall as two surveyed points:

```bash
python -m eval.baselines.run_baseline --runner hold --label rtk_v22 \
    --params eval/baselines/px4_tuned_v2.params --scenarios wall1m_wind_4ms --seeds 1 \
    -- --profile rtk_truth_at_engage --rangefinder
python -m eval.rtk_check runs/<flight folder> [...]
```

It prints the score on the receiver as logged, the same score with every receiver sample
moved onto Gazebo's truth (a perfect receiver with the same fix types), and that with a fixed
solution all flight. The first difference is what the receiver's own error did to the score;
the second is what the moments without a fix hid. The last column is the score with no frame
fit, as the Gazebo benchmark reads the hold point.

Six flights, 2026-11-09, cloud container, `wall1m_wind_4ms`, `px4_tuned_v2.params`, v22 with
the rangefinder and tuned PX4, one or two seeds each (`experiments/LOG.md`). The last two
flew the GPS-on profiles, where EKF2 fuses the receiver, to see the warning raised. A check
of the scorer, not a benchmark: the benchmark is `docs/baseline.md`.

| flight | profile | p95 m: RTK / Gazebo truth, same samples / whole hold | closest to wall m: RTK / Gazebo truth | RTK fixed | GPS fused | fit residual m | receiver off by m (horizontal rms) | p95 m with no frame fit |
|---|---|---|---|---|---|---|---|---|
| structure_hold seed 1 | rtk_truth_at_engage | 0.178 / 0.178 / 0.178 | 0.969 / 0.986 | 100 % | no | 0.029 | 0.015 | 0.202 |
| tuned_px4 seed 1 | rtk_truth_at_hover | 0.661 / 0.654 / 0.654 | 0.916 / 0.903 | 100 % | no | 0.026 | 0.015 | 0.659 |
| structure_hold seed 2 | rtk_truth_float_at_engage | 0.171 / 0.167 / 0.165 | 0.939 / 0.944 | 74 % | no | 0.032 | 0.015 | 0.185 |
| tuned_px4 seed 2 | rtk_truth_float_at_hover | 0.173 / 0.172 / 0.158 | 0.924 / 0.937 | 75 % | no | 0.034 | 0.015 | 0.189 |
| structure_hold seed 1 | gps_on_at_engage | 0.147 / 0.149 / 0.149 | 0.954 / 0.968 | 100 % | YES | 0.033 | 0.015 | 0.169 |
| structure_hold seed 1 | field_rtk_float_at_engage | 0.268 / 0.265 / 0.267 | 0.706 / 0.724 | 74 % | YES | 0.034 | 0.015 | 0.269 |

What the six flights say:

- **The receiver's own error barely moves the score.** The scripted receiver was 1.5 cm off
  horizontally and 2.0 cm vertically (rms; 7 cm at worst). The score on it differed from the
  score on a perfect receiver by at most 0.007 m on p95, 0.004 m on rms, 0.021 m on the
  largest error and 0.018 m on the closest distance to the wall. The last two are single
  worst samples, so one noisy sample moves them; quote the wall distance from RTK to the
  nearest 2 cm, not the nearest millimetre.
- **Losing the fix for 15 s of a 60 s hold moved p95 by up to 0.014 m.** On the two flights
  flown for it the hold was reported as 74 and 75 % covered, and the whole-hold p95 on
  Gazebo's truth was 0.002 and 0.014 m lower than the score on the covered part (the 15 s
  left out happened to be calmer). That is the size of what coverage hides on a quiet hold;
  had the drone struck the wall in those 15 s, RTK would not have seen it, which is why the
  report lists every hold under 95 %.
- **The frame fit works on a straight-up take-off.** Residual 0.026 to 0.034 m by offset
  alone, measured receiver lag 0.00 s (the scripted receiver has none).
- **Fused GPS is caught.** Both GPS-on flights came back `truth_independent` false, from
  EKF2's own aiding flags. Their scores still agreed with Gazebo's truth here, because the
  scripted receiver is right whenever it says fixed; on a real receiver nothing checks that,
  and the warning stands.
- **The logging trap is real.** The first flight of this check logged the receiver once a
  second (see "Logging" above); the scorer now refuses that and the parameter files prevent
  it.
- **A side finding about the benchmark's own numbers, not changed here.** The last column
  scores the same flights with no frame fit, as the Gazebo benchmark's runners read the hold
  point. EKF2's frame sat 0.04 to 0.27 m off the world in height on these flights (its
  barometer height at take-off; under 0.03 m horizontally), a constant the partner path fits
  out and the benchmark counts as hold error. It added 0.00 to 0.03 m to p95 and up to
  0.06 m to rms. Whether the benchmark should count it is a question for `docs/baseline.md`,
  raised with the owner of those numbers and left alone in this change.

## Limits

- **No real receiver has been through it.** The scripted receiver is a clean one: white
  noise, no multipath, no wrong fixes, no latency. A real log is the next check, and its
  receiver lag (`rtk_delay_s`) and fit residual are the first two numbers to look at.
- **A wrong fix is believed.** Under steel a receiver can report a fixed solution that is
  decimetres off. The scorer has no second truth to catch it; a jump in the truth with no
  matching move in the estimate or the rate commands is the sign, and the fit residual grows.
- **The antenna is taken as the drone.** The receiver reports its antenna, not the drone's
  centre. A fixed offset drops out of the hold error (the hold point is carried through the
  same fit) but moves the wall distance by up to the offset, and a tall mast adds its length
  times the tilt. Keep the antenna within about 10 cm of the centre or tell us the offset.
- **Altitude kinds must match.** `rtk.altitude` names which of the receiver's two altitudes
  the surveyed points are in (`msl` or `ellipsoid`); a vertical wall from two points does not
  use altitude at all.
- **The simulated message is coarser than a real one.** MAVLink `HIL_GPS` carries latitude
  and longitude in 1e-7 degrees (1.1 cm); PX4's `sensor_gps` from a real driver is finer.
- **Gust recovery times** need gust marks (`gusts:` in the sheet), which an outdoor session
  rarely has; without them the column is empty, as on mocap.
