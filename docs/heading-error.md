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
   with a bad compass). How far a loop goes depends on its margin, so a tighter loop gives out
   sooner: on the benchmark x500 in the mock the PD yardstick still holds at 30 degrees and
   hits the wall at 45, while v22, which holds more than twice as tight, already circles into the
   wall at 30 (results below). Stock PX4's Position mode and Structure Hold both have this;
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
true one turned by the current error, and that the PD on the default airframe (slower than the
x500 preset) still settles at 10 degrees and circles out at 30.

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

`eval/configs/gazebo_check_heading.yaml` on CPU in 14 minutes (run
`20261109-161657_gazebo_check_heading_s7001_bba529b`, `experiments/LOG.md`): 2 seeds x 64
episodes per cell, 60 s holds 1 m from the wall on the x500 preset, 4 m/s wind and
`vio_default` unless the cell says otherwise. Yardstick: the cascaded PD in the same envs,
reading the same wrong heading.

| Cell | v22 crash | v22 p95 | PD crash | PD p95 |
| --- | --- | --- | --- | --- |
| `calm_clean` (no wind, clean, true heading) | 0 % | 0.055 m | 0 % | 0.046 m |
| `head_none` (true heading) | 0 % | 0.173 m | 12.5 % | 0.392 m |
| `head_off15` | 2.3 % | 0.186 m | 12.5 % | 0.363 m |
| `head_off30` | **100 %** | 1.12 m | 13.3 % | 0.364 m |
| `head_off45` | 100 % | 1.20 m | 100 % | 0.765 m |
| `head_drift` (1 deg/s from 10 s to 30 deg) | **76.6 %** | 0.581 m | 13.3 % | 0.352 m |
| `head_steps` (10 to 30 deg steps, about one every 20 s) | 50.8 % | 0.400 m | 32.8 % | 0.433 m |
| `head_off30_wind8` (30 deg, 8 m/s into the wall) | 93.0 % | 0.943 m | 14.8 % | 0.388 m |

Crash is any contact with the wall (0.40 m radius) or leaving the box in the hold; the PD's
12 to 15 % in every cell are the estimate jumps of `vio_default`, which it rides out worse than
v22 with or without a heading error.

What it says:

- **v22 is fragile to a wrong heading.** It is fine at 15 degrees, and at 30 degrees it circles
  into the wall in every episode, also in calm air with a clean estimate (checked by hand, 16
  episodes: off the hold by 1.6 m within 10 s). The PD, more than twice as loose, still holds
  at 30 degrees and goes at 45. That is the phase-margin picture above: the policy's tight,
  fast loop has the least to give.
- A slow drift is as bad once it passes about 20 degrees: three quarters of the drift episodes
  end on the wall, and the steps cell, where the error comes and goes, crashes half.
- So the server's first job is the right one: train v22 on a wrong heading
  (`heading_s0`, `heading_s1`) and see if a candidate keeps its tightness on the true heading
  while holding through 30 degrees. If none can, the classical answer, a heading check in the
  module, moves up (What is not done, below).
- The mock says nothing yet about tuned PX4, whose position loop is looser again; the Gazebo
  batches (`gz_heading_tuned`, `gz_heading_v22`) are the comparison that counts.

## v22 and tuned PX4 in Gazebo (2026-11-09, cloud)

The queue's three CPU-lane jobs (`gz_heading_tuned`, `gz_heading_v22`, `gz_heading_v22_range`),
flown in a cloud container one flight at a time (`--jobs 1`, 16-flight batches in seed halves,
pooled with `run_baseline --pool`), 80 flights. Gazebo's real-time factor, sampled every 15 s,
was 0.97 to 1.00 over 465 samples (mean 1.00; one sample at each of 12 flight starts, before
the simulation runs, left out). The stock runner's own finer sampling caught two tuned drift
flights dipping to 0.61 and 0.84; both were flown again at 1.00 and the re-flies count. Wind
into the wall, 1 m, 8 seeds per cell, `px4_tuned_v2.params` with `vision_yaw.params` over it.
Yardstick: tuned PX4 in Position mode on the same profile. "Held" is a full 60 s scored hold
with no wall contact. The heading error is what EKF2 really had (EKF2's heading minus Gazebo's,
from each ulog).

| Batch | Held | Not held | p95 of the holds (mean / worst) | Heading error in the holds |
| --- | --- | --- | --- | --- |
| tuned, drift to 30 deg, 4 m/s | 3 of 8 | 5 wall contact | 0.59 / 0.63 m | 0 up to 18 to 22 deg |
| **v22**, drift to 30 deg, 4 m/s | **8 of 8** | 0 | 0.23 / 0.32 m | 0 up to 29 to 30 deg |
| tuned, 30 deg from takeoff, 4 m/s | 3 of 8 | 5 wall contact | 0.61 / 0.68 m | 14 deg mean |
| tuned, 30 deg from takeoff, 8 m/s | 3 of 8 | 5 wall contact | 0.53 / 0.59 m | 14 deg mean |
| **v22**, 30 deg from takeoff, 4 m/s | 5 of 8 | 2 out of control before engage, 1 watchdog trip | 0.32 / 0.50 m | 22 deg mean |
| **v22**, 30 deg from takeoff, 8 m/s | 4 of 8 | 3 out of control before engage, 1 watchdog trip | 0.48 / 0.95 m | 22 deg mean |
| v22 + rangefinder, 30 deg, 4 m/s | 5 of 8 | 2 out of control before engage, 1 watchdog trip | 0.42 / 0.70 m | 22 deg mean |
| v22 + rangefinder, 30 deg, 8 m/s | 5 of 8 | 2 out of control before engage, 1 watchdog trip | 0.35 / 0.59 m | 22 deg mean |
| tuned, no error (control) | 8 of 8 | 0 | 0.24 / 0.36 m | 0 |
| **v22**, no error (control) | 8 of 8 | 0 | 0.23 / 0.34 m | 0 |

What it says:

- **The drift batch is the clean comparison, and v22 wins it.** Both controllers start on the
  true heading and the error grows during the hold. v22 held all 8 flights at 0.23 m p95, the
  same as its no-error control, while EKF2's heading went 29 to 30 deg wrong. Tuned PX4 hit the
  wall in 5 of 8, and its three holds were 0.59 m against 0.24 m in its control.
- **The 30 deg batches say less about v22 than their counts suggest**, because the hold runner
  takes off and hovers in Position mode on the same wrong heading before it engages, and stock
  PX4 is already circling by then. At engage the vehicle was moving at 0.6 to 6.0 m/s; in the
  control and drift batches it is 0.02 to 0.08 m/s. Over the 32 v22 flights (the rangefinder
  made no difference, so both batches together):
  - 19 held. In 17 the vehicle was at 0.6 to 1.7 m/s at engage and v22 caught it. In 2 stock
    PX4 had already carried it 2 to 4 m off and down to about 0.8 m height, and v22 held it
    there, which is a full hold by the scorer but not the hold that was asked for.
  - 9 were out of control before Structure Hold could do anything: 2 crashed before the
    engage, and 7 were moving at 3.4 to 6.0 m/s when it came and the module handed back
    within half a second.
  - 4 are v22's own: engaged at 1.4 to 1.9 m/s, v22 tilted 34 to 36 deg to stop the vehicle and
    the watchdog (`SH_WD_POS_ERR` or `SH_WD_TILT`) tripped 1.7 to 2.8 s later. Four other
    flights engaged in that speed range and held, so the speed alone does not decide it.
  - No flight touched the wall while Structure Hold was flying. After a handback the vehicle
    is back with stock PX4 on the same wrong heading; one of those went on past the wall and
    another came within 0.1 m of it.
  - By seed against tuned PX4, without the rangefinder: of the 10 flights tuned PX4 crashed,
    v22 held 4; of the 6 it held, v22 held 5 and lost one to a watchdog trip (8 m/s, seed 5).
    With the rangefinder: 4 of the 10, and all 6.
- **The rangefinder changes nothing here**: 10 of 16 held against 9 of 16, the same failure
  kinds, holds within the seed spread. No hold drifted toward the wall, so the wall guard had
  nothing to do.
- **Gazebo's EKF2 does not keep a steady 30 deg.** While the bridge kept sending a vision yaw
  30 deg off, EKF2's heading error over the holds averaged 22 deg in v22's flights and 14 deg
  in tuned PX4's surviving ones, moving between about 0 and 30. The vision positions and
  velocities arrive in the true frame and disagree with the turned heading whenever the
  vehicle accelerates, which is the likely reason; a vehicle that moves more would then be
  corrected more, as tuned PX4's was. This was looked at by hand in one flight only (EKF2's
  heading stayed put while the vehicle yawed 20 deg, and its corrected yaw rate was about
  2 deg/s off Gazebo's), so take the explanation as a guess. Either way these batches are
  milder than the mock's `head_off30` cell, where the error is fixed and v22 crashed every
  episode. Whether a compass-fused EKF2 next to real steel behaves like this is not shown.
- So the mock's verdict (v22 more fragile than a loose loop) did not show up in Gazebo at the
  error sizes EKF2 reached, and it is untested for a steady 30 deg and above, which Gazebo
  did not produce. The server's heading candidates (`heading_check`) are still worth their GPU
  hour, and a candidate flies these same batches (`gz_heading_candidate`) before any release.
- The two controls agree (0.23 and 0.24 m): taking the heading from the vision yaw changes
  neither controller by itself. They are not comparable to the `vio_at_*` benchmark, since the
  field profiles turn the VIO's jumps and dropouts off.

Two things these flights point at, neither done here: the 30 deg comparison would be fairer
with the offset switched on at engage instead of at takeoff (a profile change), and the module
has no check on how fast the vehicle is moving when it is asked to engage.

Runs: `experiments/LOG.md` (2026-11-09, heading batches). Run folders stay in the cloud
container (`runs/` is not committed).

## Candidates trained in the cloud (2026-11-09)

The GPU server was full, so `heading_s0` and `heading_s1` ran in the cloud container on 4 CPU
cores with `structure_hold_ppo_heading_cloud.yaml` (the same recipe on 1024 drones, about
25 min each), then `gazebo_check_heading_candidates.yaml` judged them against v22 (32 min).
A third try, `heading_s1b`, continued `heading_s1` 400 iterations with the wrong heading on
35 % of episodes instead of 70 %, to win back v22's tight hold on a true heading. Runs in
`experiments/LOG.md`; yardstick v22 (the PD's numbers are in the table above).

| Cell | v22 crash / p95 | `heading_s0` | `heading_s1` | `heading_s1b` |
| --- | --- | --- | --- | --- |
| `calm_clean` | 0 % / 0.055 m | 0 % / 0.067 m | 0 % / 0.081 m | 0 % / 0.068 m |
| `head_none` (true heading) | 0 % / 0.173 m | 0 % / 0.231 m | 0 % / 0.237 m | 0 % / 0.193 m |
| `head_off15` | 2.3 % / 0.186 m | 0 % / 0.236 m | 0 % / 0.242 m | 0 % / 0.195 m |
| `head_off30` | 100 % / 1.12 m | 1.6 % / 0.228 m | 0 % / 0.243 m | 0 % / 0.198 m |
| `head_off45` | 100 % / 1.20 m | 50.8 % / 0.482 m | 28.9 % / 0.314 m | 46.1 % / 0.434 m |
| `head_drift` | 76.6 % / 0.581 m | 0.8 % / 0.233 m | 0 % / 0.251 m | 0 % / 0.197 m |
| `head_steps` | 50.8 % / 0.400 m | 11.8 % / 0.252 m | 3.2 % / 0.273 m | 5.5 % / 0.206 m |
| `head_off30_wind8` | 93.0 % / 0.943 m | 1.6 % / 0.218 m | 0.8 % / 0.226 m | 0 % / 0.192 m |
| Verdict against v22 | reference | fails `head_none` slack | fails both true-heading cells | fails the fast ring |

**No candidate passes, so none goes to Gazebo yet.** Training on a wrong heading works in
the mock: every candidate holds through a fixed 30 degrees and the drift with almost no
crashes, where v22 hits the wall. The price is paid when the compass is right. `heading_s0`
and `heading_s1` hold about 35 % looser on a true heading (0.23 against 0.17 m), outside the
check's 2 cm slack. `heading_s1b` wins most of that back (0.193 m, inside the slack) but works
its rate commands harder: its 2.2 to 6 Hz roll and pitch rate content is 0.022 rad/s against
v22's 0.018 pooled over the heading cells (0.022 against 0.015 on `head_none`, 0.014 against
0.009 in calm air), just over the check's 25 % limit, and a fast ring is what made v18 ring in
Gazebo (`docs/gazebo_check.md`). It is the trade the phase margin predicts: a loop that
tolerates a turned frame gives up some tightness or some smoothness.

Read with the Gazebo flights above, the case for a heading-trained release is weaker than the
mock alone made it. In Gazebo v22 as released held all 8 drift flights at its no-error p95
while EKF2's heading went 29 to 30 degrees wrong; the mock's `head_drift` cell, the same
1 deg/s drift to 30 degrees, loses 77 % of v22's episodes. Why the mock is harsher is not
known. The Gazebo drift spends about 10 s at 30 degrees against the mock's 20 s, but about
three quarters of the mock's crashes come within those first 10 s, so the timing does not
explain it. Until that gap is understood the mock's heading cells overstate the danger to v22,
and the candidates buy robustness Gazebo has not shown v22 to lack, at a cost on the true
heading that is certain. Nothing is released or flown from this.

If a steady 30 degrees turns out to be a real case (the engage-time offset profile suggested
above, or a partner's flight near steel), `heading_s1b` is the starting point: continue it
with a heavier `action_band` weight to bring the fast ring back under the limit (one sub-hour
part), then fly it on the heading batches.

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
  wall angle, or a yaw innovation trip) would be the classical answer. In Gazebo (above) v22
  held through a drift to 30 deg where tuned PX4 hit the wall, so it has not moved up; a
  steady 30 deg and above is still open.
- The compass itself is not simulated in Gazebo (the reasons above). A partner's flight next to
  real steel is the check that the vision-yaw stand-in moved EKF2 the way a compass does.
- Steps of the compass are in the mock only.
