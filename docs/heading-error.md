# Heading error near steel

Gap 10 of `docs/market-research.md`, item 6 of the GPU server plan. Heimdall named magnetic
interference on stage, and inspection work happens next to exactly what bends a compass: steel
decks, towers, girders, racks, rebar. This page says what a wrong heading does to a hold, how
the mock and Gazebo now put one in, what v22 does with it today, and what the server runs next.

## What a wrong heading does

On the benchmark set-up, and on most partner drones, EKF2's heading comes from the compass
and its position from the VIO, which carries no heading (`EKF2_EV_CTRL` 7, no vision yaw).
Next to steel the compass is off by up to tens of degrees. EKF2 turns its attitude with it,
while the vision positions stay in their own frame. Two things follow:

1. Every loop that turns a position error into a tilt does it in a frame turned by the error.
   A correction meant to go toward the hold point goes off to one side by the same angle.
   With no lag in the loop that would still converge up to 90 degrees, but the turn eats the
   loop's phase margin degree for degree, so a real loop starts to circle much sooner and the
   circle grows until something limits it (the "toilet bowl" PX4 users know from GPS flights
   with a bad compass). How far a loop goes depends on its margin: the mock's PD yardstick
   settles at 10 degrees, slowly circles out from about 20, holds a circle over a metre wide at
   30 and touches the wall at 45. Stock PX4's Position mode and Structure Hold both have this;
   nothing in either knows the heading is wrong. Where tuned PX4's margin runs out is what the
   Gazebo batches measure.
2. Structure Hold takes the heading at engage as the wall's normal (`docs/integration-guide.md`).
   A compass already off at engage turns the structure frame by that angle. A compass that
   drifts after engage makes the policy, which holds the latched heading, turn the vehicle by
   the drift, so the camera and the forward rangefinder no longer face the wall square.

Whatever the cause (compass, or a VIO's own yaw), what the controller gets is the same: a
heading off by some angle over positions in the true frame. That is what both simulators
model.

## In the mock (`envs/estimate/heading.py`)

`HeadingErrorConfig` (`heading_error:` in any Structure Hold env config, off by default) gives
each episode, with probability `prob`, a heading error built from:

| Part | Keys | Stands for |
| --- | --- | --- |
| Offset at engage | `offset_deg` (magnitude range, random sign) | The compass already wrong where the module engages |
| Drift | `drift_rate_deg_s`, `drift_start_s`, `drift_max_deg` | The vehicle edging toward steel, or EKF2 slowly giving in to a disturbed compass |
| Steps | `step_rate_hz`, `step_deg`, `step_tau_s` | The field changing (a girder passing under), which EKF2's compass fusion follows over a few seconds |

The sum is clipped at `max_deg`. The env (`envs/structure_hold/env.py`) latches the error at
every engage and re-engage, as the module latches EKF2's heading: positions and velocities are
read in the frame turned by the error at engage, the heading input by how far the error has
moved since. `env.believed_quat` is the attitude the module believes it has; the PD yardstick
steers its thrust with it, as PX4's own attitude loop would, so the PD meets the same turned
frame as the policy. Its own random stream: turning it on changes no other draw. The tests
(`envs/structure_hold/tests/test_heading_env.py`) pin that the module's body-frame error is the
true one turned by the current error, and that the PD still settles at 10 degrees and circles
out at 30.

## In Gazebo (`sim2real/estimate_bridge/vision_yaw.py`)

The compass path cannot carry a scripted error into PX4 v1.17's SITL. Compass calibration is
reloaded only while disarmed (`VehicleMagnetometer::ParametersUpdate`), and once EKF2 has
aligned its compass in flight its declination comes from its own earth-field states, which it
resets to the World Magnetic Model whenever it knows where it is, so `EKF2_MAG_DECL` moves
nothing in flight. So the error goes in through the vision stream: a profile with a
`vision_yaw` block makes the bridge send the vehicle's attitude in ODOMETRY (Gazebo's true
attitude turned about the vertical by the scripted error), and `vision_yaw.params`
(`EKF2_EV_CTRL` 15) makes EKF2 take its heading from it instead of the compass. Positions keep
coming in `MAV_FRAME_LOCAL_NED`, which EKF2 fuses as they are once its yaw is aligned: the same
turned heading over true-frame positions as the compass case.

EKF2 follows a slow change in the vision yaw. A step larger than its heading gate
(`EKF2_HDG_GATE`) is rejected for a second and then taken as a heading reset, which the module
follows by turning its frame (`docs/decisions.md`, heading resets), so the Gazebo profiles use an
offset from takeoff and slow drifts, not steps. Each flight's summary reports the heading error
EKF2 really had over the scored window (`estimate_error.heading_error`, EKF2's heading minus
Gazebo's, from the ulog), so the size the profile asked for is checked, not assumed.

| Profile (`_at_engage` hold runner, `_at_hover` stock runner) | Heading error | What it is for |
| --- | --- | --- |
| `field_heading_none` | 0, but EKF2 on the vision yaw | The control: what the heading source change alone does |
| `field_heading_offset` | 30 deg clockwise from takeoff on | A compass already wrong at takeoff and engage |
| `field_heading_drift` | 0 at engage, then 1 deg/s from 20 s to 30 deg | Drifting in under steel during the hold |

All three keep `vio_default`'s delay, noise and bias walk and turn its random jumps and
dropouts off, as the field-fault profiles do (`docs/field-faults.md`). The bridge sends the
true attitude at the moment it sends the sample, not delayed with the position; at a hover's
yaw rates the difference is far below the error.

## v22 and the PD in the mock (2026-11-09, cloud)

_Results go here once the check has run (`eval/configs/gazebo_check_heading.yaml`)._

## Server runs (`training/queues/heading_error.yaml`)

| Job | What | Hours | Go / no-go |
| --- | --- | --- | --- |
| `heading_s0`, `heading_s1` | `structure_hold_ppo_heading.yaml`: v22 continued 800 iterations (plus 200 of critic warm-up) with a wrong heading on 70 % of episodes (offset to 45 deg, drift to 30 deg, steps of 10 to 30 deg), seeds 0 and 1, one GPU each | 0.9 each | Trains and exports |
| `heading_check` | `gazebo_check_heading_candidates.yaml`: both candidates against v22 on the heading cells | 0.8 | A candidate beats v22 pooled over the heading cells and stays within slack on `calm_clean` and `head_none` (true heading) |
| `gz_heading_tuned` | Tuned PX4 on the three profiles: 30 deg at 4 and 8 m/s, the drift and the control at 4 m/s, 8 seeds each (32 flights) | 0.55 | None: the result is the comparison |
| `gz_heading_v22` | v22, the same 32 flights | 0.55 | v22 against its tuned twin per batch (`run_baseline --compare`) |
| `gz_heading_v22_range` | v22 with the forward rangefinder, 30 deg at 4 and 8 m/s (16 flights) | 0.3 | Whether the wall guard keeps a curving hold off the wall |
| `gz_heading_candidate` | Off until `heading_check` passes: the candidate's image and the same flights | 1.0 | Release only if Gazebo confirms (CLAUDE.md, Releasing a policy) |

At most two GPUs at once and every job under an hour, so nothing here needs a separate yes.

## What is not done

- No module change. A heading check in the module (EKF2's heading against the rangefinder's
  wall angle, or a yaw innovation trip) would be the classical answer; it waits until Gazebo
  shows how far v22 and tuned PX4 go wrong.
- The compass itself is not simulated in Gazebo (the reasons above). A partner's flight next to
  real steel is the check that the vision-yaw stand-in moved EKF2 the way a compass does.
- Steps of the compass are in the mock only.
