# Surface passes: the hold point moves 5 m along the wall

The product's Decided row (`docs/decisions.md`, 2026-10-21) is "hold **and 5 m passes** 1 to
3 m from a structure". Until November 2026 the env only held a fixed point. This page is the
pass: the hold point slides along the wall (a girder, a deck underside, a rack face) at the
same standoff, so the camera sweeps a strip of the surface while the drone keeps its distance.
`docs/market-research.md` row 12 is the buyer case.

Status (2026-11-09): in the mock, with the PD yardstick, a Gate 1 style scorer and a training
config queued for the GPU server; **in the PX4 module** (`SH_PASS_LEN`, off by default; "In
the module" below); and **in Gazebo against tuned PX4 flying the same pass** ("In Gazebo"
below). v22 as it is flies all of it. No policy trained on passes is released yet.

## What a pass is

Per episode, with probability `surface_pass.prob` (`envs/structure_hold/config.py`
`SurfacePass`, off by default), the hold point:

1. stays at the engage point for `start_s` (the settle after engage, 2 to 4 s by default);
2. moves `length_m` (5 m) along the wall, either way, or up it with `vertical_prob` (a pier
   or a column), on a trapezoidal speed profile: `accel_mps2` up to `speed_mps`, cruise,
   `accel_mps2` down to a stop;
3. for a raster (`legs` above 1), pauses `pause_s`, steps `row_step_m` across the pass (up
   for a horizontal pass, sideways for a vertical one), pauses, and flies the next leg back;
4. holds where it stopped for the rest of the episode.

Every segment lies in a plane parallel to the wall, so the standoff never changes. The path is
`envs/structure_hold/passes.py`; its own random generator means switching passes on changes
nothing else an episode draws (same engage point, wind, estimate, airframe).

A 5 m pass at 0.5 m/s takes 11 s after the settle, at 1 m/s 7 s
(`passes.pass_duration_s`). Inspection drones close to a structure fly slower than they could;
0.2 to 1 m/s spans what the training sees.

## What the policy reads, and why no new input

The position error input is to the moving hold point, and the velocity input is the
estimate's velocity **minus the hold point's** (`env.hold_vel`). On a plain hold that
difference is the velocity itself, so:

- every released policy reads exactly what it read before, and v22 flies passes as it is
  (scored below);
- a pass-trained policy starts from v22's weights (`init_actor_onnx`) with the same 21
  inputs, and its header drops into the module unchanged;
- the module needs a path generator and one subtraction, not a new input.

A separate "pass velocity" input was the alternative. It would tell the policy it is on a pass,
but it is a new input for every policy and for the module, and a relative velocity already
gives the controller the feedforward it needs (for the PD it is exactly velocity
feedforward, as PX4's position controller uses). If a pass-trained policy cannot keep up at the
start and stop of a pass, that is the evidence to add it.

Reward and termination follow the same point: the position terms are to the moving hold point,
and the speed penalty (and the toward-the-wall speed term) is on the speed relative to it, so
keeping up costs nothing. The critic's true velocity is relative too (`training/ppo_env.py`).
The heading term still faces the wall. Mid-episode re-engages (`reengage_rate_hz`) are for
plain holds only; a pass is flown from one engage.

## How a pass is scored

`eval/gate1.py` scores against the hold point at every step (it used to keep the engage point;
a pass makes the difference). Beside the 3-D p95 it reports:

- `standoff_p95_m`: the p95 of the error along the wall normal alone, the camera's distance
  to the surface, which is what an inspection cares about most;
- `pass_done_rate`: the share of pass episodes that lived past the end of their path.

`eval/configs/gate1_x500_ekf_pass.yaml` is the Gate 1 set for passes: the held-out seeds 9001
to 9003, the x500 through the EKF2-like fusion on barometer height, the standard estimate
corruption, both gusts inside the episode, the PD yardstick on the same episodes, Gate 1's
rules (headline cell 30 % under the PD, every cell significantly tighter, no significantly more
crashes, pooled crashes under 5 %). Seven cells, 256 episodes each per seed:

| Cell | Pass | Wind | Standoff |
| --- | --- | --- | --- |
| `p05_w0_d1` | 5 m along, 0.5 m/s | 0 m/s | 1 m |
| `p05_w4_d1` | 5 m along, 0.5 m/s | 4 m/s | 1 m |
| `p05_w8_d1` (headline) | 5 m along, 0.5 m/s | 8 m/s | 1 m |
| `p05_w8_d2` | 5 m along, 0.5 m/s | 8 m/s | 2 m |
| `p10_w4_d1` | 5 m along, 1 m/s | 4 m/s | 1 m |
| `up05_w4_d1` | 5 m up the wall, 0.5 m/s | 4 m/s | 1 m |
| `raster2_w4_d1` | 2 legs, 1 m row step, 0.5 m/s | 4 m/s | 1 m |

It judges v22 by default; with a candidate in `params.policies` (`{name: onnx}`), it judges
the candidate and reports v22 beside it.

## First result: v22 as it is, against the PD

Run `20261104-171727` (commit `720d0e9`, 21 min on the 4 cloud CPU cores, `experiments/LOG.md`):
v22, which never trained on a moving hold point, against the cascaded PD yardstick (mock, not
tuned PX4) on the same 768 held-out episodes per cell. **PASS, 7 of 7 cells.**

| Cell | v22 p95 (m) | PD p95 (m) | v22 vs PD | v22 standoff p95 (m) | PD standoff p95 (m) | Crash v22 / PD |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 m/s, calm, 1 m | 0.173 | 0.333 | -48 % | 0.114 | 0.184 | 0 / 4.7 % |
| 0.5 m/s, 4 m/s, 1 m | 0.174 | 0.348 | -50 % | 0.115 | 0.204 | 0 / 3.9 % |
| 0.5 m/s, 8 m/s, 1 m (headline) | 0.186 | 0.380 | -51 % | 0.122 | 0.241 | 0 / 3.5 % |
| 0.5 m/s, 8 m/s, 2 m | 0.207 | 0.458 | -55 % | 0.136 | 0.311 | 0 / 0 % |
| 1 m/s, 4 m/s, 1 m | 0.173 | 0.404 | -57 % | 0.115 | 0.204 | 0 / 3.9 % |
| 0.5 m/s up the wall, 4 m/s, 1 m | 0.164 | 0.311 | -47 % | 0.114 | 0.206 | 0 / 3.6 % |
| 2-leg raster, 4 m/s, 1 m | 0.174 | 0.393 | -56 % | 0.113 | 0.235 | 0 / 6.2 % |

Every interval is tight (the full table with 95 % intervals is the run's `report.md`).
Pooled: v22 0.18 m and no crash, the PD 0.38 m and 3.7 % crashes, every PD crash a wall
contact in a 1 m cell. v22 finished every pass, the PD 98 %.

What it says: the pass costs v22 almost nothing. On the plain-hold set with the same fusion
(`gate1_x500_ekf_baro_v22.yaml`, run `20261103-023211`) v22 held 0.19 m pooled against the PD's
0.34 m and 1.5 % crashes; on passes v22 is the same and the PD is looser and touches the wall
more than twice as often. The relative velocity input is enough for a hold policy to follow a
slow pass. These are different episode sets (25 s passes, 20 s holds), so read "the same", not
"better". The pass training on the server has to beat these v22 numbers, not just the PD.

## Training on the server

`training/configs/structure_hold_ppo_pass.yaml`: v22 continued (its ONNX, fresh critic warmed
up 200 iterations) on v22's recipe with 70 % of episodes passes (3 to 6 m, 0.2 to 1 m/s, a
quarter up the wall, up to three legs) and 30 % plain holds, so the hold is still trained on.
GPU-sized (4096 drones, 1200 iterations a part); its selection and end-of-run eval fly 0.5 m/s
passes against the PD. The smoke run passes on the cloud CPU (`--smoke`), and so does a resumed
part 2.

In `training/queues/surface_passes.yaml` (2 GPUs, every run under 1 h, nothing needs Zain's
yes):

| Job | What | Go/no-go |
| --- | --- | --- |
| `hold_pass`, `hold_pass_s1` | the config on seeds 0 and 1, one GPU each, chained twice (part 2 resumes part 1's `policy.pt`), about 45 min a part | the end-of-run eval beats the PD on passes |
| `hold_pass_gate` | `gate1_x500_ekf_pass.yaml` with both added as `pass_s0` (judged) and `pass_s1` | PASS against the PD, and a lower p95 than v22 in every cell (v22: 0.16 to 0.21 m) |

## In the module

`sim2real/px4/structure_hold/PassPath.hpp` is the path, plain C++ with no heap, stepped once
per control tick. Four parameters, clamped in `ParamLimits.hpp` and read at each engage:

| Parameter | Default | Range | What it does |
| --- | --- | --- | --- |
| `SH_PASS_LEN` | 0 m (no pass) | -10 to 10 m | How far the hold point moves. Positive to the right of the nose, negative to the left; up and down with `SH_PASS_DIR` 1 |
| `SH_PASS_DIR` | 0 | 0 or 1 | 0 along the wall, 1 up or down it |
| `SH_PASS_SPD` | 0.5 m/s | 0.1 to 1 m/s | Cruise speed; it speeds up and slows down at 0.5 m/s^2 |
| `SH_PASS_T0` | 3 s | 0 to 60 s | The still hold after the engage before the point moves |

With a length set, each engage flies one pass: still for `SH_PASS_T0`, the line at right
angles to the heading at engage (the engage heading is the wall normal, so that line is
parallel to the face), then a hold at the end. The policy reads its velocity relative to the
moving point (`PolicyController.hpp`, `hold_vel_ned`), the one subtraction the mock promised;
the watchdog's position error, the fallback rule and the status topic follow the moving
point too (`structure_hold_status.pass_state`, `pass_offset_ned`).

What the module does that the mock's path does not:

- **It waits for the estimate.** While the position fix is older than `SH_FB_T` the point
  brakes at 0.5 m/s^2 and waits, and goes on when the fix is fresh. The mock's path runs on a
  clock; a real drone should not chase a point that ran ahead while it could not see.
- **The guards act on the moving point.** The wall guard moves it back off a face that steps
  toward the drone and the pass goes on at the new standoff. The keep-out looks ahead along
  the line: an obstacle within the keep-out of where the point would come to rest (the
  braking distance plus 0.1 m) ends the pass.
- **A pass that ends early brakes to a stop** and stays there for the rest of the hold: on
  that obstacle, when the wall guard has used up `SH_SHIFT_MAX`, or when a pass down comes
  within 1 m of the ground. The first version stopped the point dead; in SIH the drone then
  overshot 0.3 m into a 1 m keep-out, which is why it brakes and looks ahead now.
- **One line per engage.** No raster, no route. The pilot (or a companion computer setting
  the parameter and the mode) starts the next one. Structure Land and Structure Hover do not
  fly passes; a handback during a pass starts from where the point was.

The pilot sees `Structure Hold: pass 5.0 m at 0.5 m/s`, then `pass done`, or `pass stopped
after 2.4 m: keep-out` (`docs/pilot-status.md`, safety manual section 4). The interface is
in `docs/integration-guide.md` ("Surface pass"), the hazards are F47 to F52 of
`docs/safety-manual.md` (version 1.10, not signed). The Auterion app (`docs/auterion-app.md`)
flies the same headers but does not start a pass yet.

Tests:

- `sim2real/px4/tests/test_pass_path.cpp` (CI): the profile against the closed form, 5 m in
  11 s at 0.5 m/s, both signs, up and down, the wait on a stale fix, the braking stop
  (before the start, on the move, while waiting, close to the end), the parameter limits.
  `test_keep_out.cpp` has the look-ahead.
- PX4 SIH, `sitl_test.sh` with `SITL_EXTRA_ARGS` (2026-11-09, `experiments/LOG.md`):

| Flight | Options | Result |
| --- | --- | --- |
| 5 m along | `--pass-len 5` | started 3.1 s after the engage, done 11.0 s later, ended 4.97 m along, never more than 0.18 m behind the point |
| 3 m up | `--pass-len 3 --pass-up` | done, same checks |
| A 0.5 m step in the face half way | `--wall --pass-len 5 --pass-wall-step-m 0.5` | the wall guard moved the point back; settled 0.79 m from the stepped face, closest 0.57 m, the pass finished |
| A pole on the line at 3.5 m | `--keep-out --pass-len 5 --pass-pole-m 3.5` | stopped after 2.4 m, closest 0.92 m to the pole with a 1 m keep-out |

The usual sequence (`--wall --keep-out`, 37 checks) still passes with the pass code in and
off. Flash: `px4_fmu-v6x_structurehold` is at 99.43 % (1,954,944 B, about 11 KB spare; it
was 99.27 %), so it fits, and the next feature of this size will not without making room.

## In Gazebo: against tuned PX4 flying the same pass

The yardstick is stock PX4 with the tuned set (`eval/baselines/px4_tuned_v2.params`) flying
the same moving point the way an integrator would without the module: **Offboard mode with
trajectory setpoints** (`sim2real/px4/pass_flight.py`: position, velocity and acceleration
of the same trapezoid at 50 Hz, the feedforward PX4's position controller is built to use).
That is a fair, strong baseline: it knows the path exactly and ahead of time.

Three scenarios (`sim2real/gazebo/scenarios/wall1m_pass_*.yaml`: calm, 4 m/s and 8 m/s with
two gusts, 1 m from the wall): 3 s still, the 5 m pass at 0.5 m/s (11 s), 16 s still at the
end, 30 s in all. To keep the two like for like:

- both controllers start their hold when PX4's clock reads 60 s (`pass.engage_at_s`), so
  both meet the same gusts at the same point of the pass (each seed puts its two gusts 6 to
  18 s into the hold: in the cruise or in the hold at the end);
- the same estimate profile (`vio_at_engage`: the scripted VIO jumps and dropouts start at
  the engage, which for stock PX4 is the switch to Offboard);
- both scored by `eval/metrics.py` on Gazebo's **true** position against the moving point
  (`eval/pass_score.py`), over the same 28 s (the 30 s less the first 2 s after the engage,
  as the hold benchmark trims); crash and wall contact judged on it;
- one flight at a time in a cloud container, real-time factor recorded per flight.

Beside the 3-D p95 the table has the error along the wall normal (the standoff, the camera's
distance to the surface) and along the pass (how far behind or ahead of the point).

RESULTS_PLACEHOLDER

## Not done yet

- **A policy trained on passes.** v22 never trained on a moving hold point. The server queue
  (`training/queues/surface_passes.yaml`) trains two; one that passes its mock gate and the
  release check gets these same Gazebo batches before it can replace v22 on passes.
- **Guards in Gazebo.** The Gazebo pass flights have no rangefinder and no lidar, so they
  score the pass, not the wall guard or the keep-out on it; those are SIH only.
- **Faster, longer, up the wall.** Gazebo flew 5 m along at 0.5 m/s. 1 m/s, a pass up a pier
  and the raster are mock only.
- **A raster or a route in the module.** One line per engage; chaining lines is the
  integrator's for now.
- **Curved and stepped surfaces.** The wall is a flat plane; the module's wall guard follows
  a step (SIH), but a pass along a curved pier face is not modelled anywhere.
