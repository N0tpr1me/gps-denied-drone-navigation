# Structure Hold PX4 module

One onboard PX4 flight mode, built into `px4_sitl` first and later into the target
airframe. Since 2026-10-24 it flies the released policy (v19, `training/policies/`, trained on the Gazebo x500; v10 from 2026-10-22 to 2026-10-24) with the
brake-and-hover fallback and a yaw guard available as options; the v0 placeholder cascade
stays selectable (`SH_CTRL 0`).

What it does today:

- **Registers "Structure Hold" as a named flight mode** with commander through PX4's
  external-component handshake (`register_ext_component_request` / `_reply`,
  `arming_check_request` / `_reply`, `config_control_setpoints`), driven from inside the
  firmware the way `mc_nn_control` does. On the pinned release (v1.17.0) the mode gets
  `nav_state` 23 (EXTERNAL1), shows up in QGC and on an RC mode switch, and PX4's
  failsafes and arming checks apply to it. No commander patch needed.
- **Holds the position and yaw it had when engaged** with the policy
  (`PolicyController.hpp`, `SH_CTRL 1`, default) or the placeholder P cascade
  (`HoldController.hpp`, `SH_CTRL 0`). Both output FRD body rates and a normalised
  collective thrust on `vehicle_rates_setpoint`. PX4's stock rate controller and control
  allocation stay in charge; the attitude and position controllers are idle while the mode
  is active.
- **Policy inference** is a compiled-in MLP (`MlpPolicy.hpp`, no heap, no PX4 headers, no
  third-party library; docs/decisions.md). `tools/export_policy_header.py` turns the
  released `training/policies/structure_hold_v19.onnx` + `.json` into
  `structure_hold/PolicyWeights.hpp`: weights, the history length (15, from
  `policy.json input.history`, never hardcoded), the observation layout, the action scaling,
  the mock airframe the policy was trained on, and the policy's own defaults for the two
  options below (`output.yaw`, `output.fallback`, `output.max_rate_rad_s`). The module builds the env's observation
  (structure frame = heading at engage, ENU-style, FLU body) from PX4's NED / FRD state
  (the age input is the env's: 0 while the fix is fresh, the time since the last fix while
  stale; the sample latency is reported in the status message only),
  keeps the rolling history, and maps the action back (FLU → FRD rates, thrust in newtons for
  the mock airframe scaled through the hover thrust). `tests/test_policy_header.py` fails
  when the header no longer matches the released policy.
- **Per-policy defaults from `policy.json`, overridable in the field.** `SH_YAW_GUARD`
  and `SH_FB_EN` default to -1 (the policy's own default), `SH_RATE_MAX` to 0 (the
  policy's action scale); 0 / 1 (and `SH_FB_EN 2`) override them without a rebuild. v3
  declares `output.yaw = "policy"` (it holds the heading itself; the guard drives it into
  the wall, 71 % crashes in the mock), `output.fallback = "off"` (it flies through dropouts
  itself; the rule costs it 0.06 m p95 and 1.1 points of crash rate) and 2 rad/s actions.
  For v2, whose `policy.json` has neither key, the guard is on, the fallback on, 6 rad/s.
- **Yaw guard** (`SH_YAW_GUARD`, `SH_YAW_P`): the module holds the wall-facing heading
  captured at engage with its own P loop and ignores the policy's yaw output. Needed by v2
  (no heading term), harmful to v3. The body-rate setpoints are clamped by `SH_RATE_MAX`
  (v2 flew SIH only at 1 to 2 rad/s, see below).
- **Brake-and-hover fallback** (docs/fallback.md) once the position fix has been stale for
  `SH_FB_T` (0.5 s): acceleration setpoint `-SH_FB_K × v_est` (1.5, retuned from 2.5 for the
  lagged rate loop) clamped at 6 m/s², no position term, P attitude loop (`SH_FB_TILT` 4),
  heading held on the structure; hands back on the first fresh fix. The policy keeps running
  meanwhile so its history stays current. A fix is a new `vehicle_visual_odometry` sample
  (VIO / the estimate bridge) or, with no odometry source, a local position update
  (`SH_FIX_SRC`). `SH_FB_EN 2` forces the fallback every tick for tests.
- **Watchdog** (`Watchdog.hpp`, plain C++, no PX4 headers, unit-tested): trips on a
  test fault, an invalid or stale estimate (no fix for `SH_WD_EST_TO`, 5 s so the fallback
  can fly through dropouts; EKF2-invalid flags trip at once), too much tilt or too much
  hold error, in that priority order. A trip latches until disarm. On a trip the module hands back:
  Position mode (Altitude mode if the horizontal estimate is gone), then Hold, then Land
  if the previous request was refused, holding level at hover thrust in between.
- **Engage hold-off after an EKF reset** (`SH_RST_HOLD`, seconds; default 0 = off): for
  that long after EKF2 reset its horizontal position (`vehicle_local_position.xy_reset_counter`
  changed) the mode's arming check reports it unavailable, so commander keeps the current
  mode, and if the switch slipped through anyway (commander polls every 300 ms) the module
  hands straight back with `trip_reason` `estimate_reset` (6), nothing latched. A hold point
  captured while the estimate settles after a reset is not where the vehicle is: v11
  re-engaged 0.1 to 0.4 s after a 1.1 m vision reset, captured a point on the wall face and
  struck the wall (`docs/baseline.md`, v11 section). It ships off because stock PX4
  captures the same settling estimate: in the mock replay of that sequence a 10 s hold-off
  left v10 at 92 % crashes against 91 % without it, most of them moved into the stock
  phase. It is an operator choice, not a fix; a fix needs a direct distance to the
  structure (`docs/real-flight-gap.md`). A reset during a hold never trips the watchdog;
  the policy is trained on them. `structure_hold_status.reset_age_s` says how long ago the
  last reset was (−1 if none). `sitl_test.py` retries its scripted re-engage once a second
  for up to 14 s, and records the refusals and the wait in `summary.json`.
- **Wall guard: a measured floor under the hold point** (`WallGuard.hpp`, plain C++,
  unit-tested; `SH_WALL_MIN`, default 0.7 m, 0 = off; `SH_WALL_REQ`). With a forward-facing
  rangefinder on `distance_sensor` (any instance, orientation forward; a downward one for
  EKF2's height is skipped), each tick turns the range into the hold point's distance to
  the wall along the engage heading, `range × cos(body x, normal) − (hold − pos)·normal`,
  and moves the hold point straight back when that is under `SH_WALL_MIN`. The estimate's
  own offset cancels in that sum, so it is the case the hold-off above cannot fix: a hold
  point captured on a jumped estimate that lies on the wall face (Gazebo seed 3,
  `docs/baseline.md` v11 section). The point only ever moves away from the wall during a
  hold. Samples count only when under 0.2 s old, inside the sensor's own range, not flagged
  bad, and with the body x axis within 35 deg of the wall normal; without one the guard
  does nothing, so the Gazebo x500 (no rangefinder) flies exactly as before. The policy sees
  the move as a step in its position error, like an estimate jump. `SH_WALL_REQ 1` makes
  the mode unavailable while no usable forward range comes in. Stock Position mode
  between two engages is not covered: that is PX4's Collision Prevention (`CP_DIST`) on
  the same rangefinder. `structure_hold_status` gains `wall_distance_m`,
  `hold_clearance_m` and `standoff_shift_m`.
- **Publishes `structure_hold_status`** every control tick while engaged (state, trip
  reason, setpoint, error, commanded rates and thrust, controller, fallback active, yaw
  guarded, stale time, action flown, loop time), logged by default. `structure_hold status`
  on the PX4 shell prints the policy's defaults and what is actually flying.
- **Parameters** under `SH_`: `SH_EN`, `SH_CTRL`, yaw guard `SH_YAW_GUARD` / `SH_YAW_P`,
  rate clamp `SH_RATE_MAX`, fallback `SH_FB_EN` / `SH_FB_T` / `SH_FB_K` / `SH_FB_TILT`, fix
  source `SH_FIX_SRC` / `SH_FIX_GAP`, watchdog thresholds `SH_WD_*`, the engage hold-off
  after an EKF reset `SH_RST_HOLD`, the wall guard `SH_WALL_MIN` / `SH_WALL_REQ`, the test hook `SH_TEST_FAULT`, placeholder gains
  `SH_P_*`, `SH_V_MAX`, `SH_VEL_SRC`.

Not here yet: choosing the standoff (the hold point is the engage point, kept outside
`SH_WALL_MIN` by the wall guard), passes along the surface, the fault injector that corrupts
the estimate the policy sees.

## Layout

```
sim2real/px4/
  structure_hold/        the module: CMakeLists.txt, Kconfig, StructureHold.{hpp,cpp},
                         Watchdog.hpp, WallGuard.hpp, PolicyController.hpp, MlpPolicy.hpp,
                         PolicyWeights.hpp (generated), HoldController.hpp,
                         structure_hold_params.c
  msg/StructureHoldStatus.msg
  tools/                 export_policy_header.py (onnx -> PolicyWeights.hpp),
                         make_test_vectors.py (Python env -> tests/PolicyTestVectors.hpp)
  install.sh             wires the module into a PX4-Autopilot tree (idempotent)
  sitl_test.sh / .py     acceptance flight in PX4 SITL (SIH quadrotor, no Gazebo)
  bench_test.py          props-off bench checks on a flight controller (docs/bench-test.md)
  run_gazebo.py          the same flight in Gazebo with wind and the corrupted estimate
                         (Docker image, sim2real/estimate_bridge), scored on the truth
  tests/                 test_watchdog.cpp, test_wall_guard.cpp, test_policy.cpp (+ PolicyTestVectors.hpp),
                         test_policy_header.py
```

After a policy release, regenerate both headers and commit them:

```bash
python -m sim2real.px4.tools.export_policy_header training/policies/structure_hold_v19.onnx
python -m sim2real.px4.tools.make_test_vectors training/policies/structure_hold_v19.onnx
```

`RATE_HZ` in the generated header is the tick the policy trained at (v9, v10 and v19: 83 Hz, the
12 ms tick the module's 10 ms `CONTROL_INTERVAL` gives on Gazebo's 4 ms gyro). It is
informational: the module keeps `CONTROL_INTERVAL` at 10 ms and never derives it from
`RATE_HZ`.

## Build

```bash
git clone --depth 1 --branch v1.17.0 --recurse-submodules https://github.com/PX4/PX4-Autopilot.git
sim2real/px4/install.sh /path/to/PX4-Autopilot          # --link to symlink while developing
cd /path/to/PX4-Autopilot && make px4_sitl_default
```

`install.sh` copies (or links) `structure_hold/` to `src/modules/structure_hold`, adds the
message to `msg/CMakeLists.txt`, enables `CONFIG_MODULES_STRUCTURE_HOLD` in the SITL board
config (`--board` for another), adds `structure_hold start` to `rc.mc_apps` behind
`SH_EN`, and makes the logger record `structure_hold_status`. Run it again after editing
the module when you copied rather than linked.

PX4's own build needs its Python packages (`Tools/setup/requirements.txt`; `empy<4`
does not build on Python 3.13, use 3.10 to 3.12).

## Fly it

```bash
pip install pymavlink pyulog
sim2real/px4/sitl_test.sh /path/to/PX4-Autopilot          # writes runs/<time>_sitl_structure_hold_<commit>/
```

The flight: take off to 2.5 m in stock PX4, engage Structure Hold over MAVLink (DO_SET_MODE,
auto / external 1), hold 20 s, disengage to Position mode, re-engage, trip the watchdog
with `SH_TEST_FAULT`, confirm the handback and that a further engage is refused, land.
`summary.json` has one boolean per check plus the numbers (drift while holding, handback
latency from the ulog, status samples logged). Exit 0 means every check passed.

`SITL_EXTRA_ARGS=--wall` adds the wall guard check: a forward rangefinder simulated over
MAVLink (`DISTANCE_SENSOR` at 20 Hz) against a virtual wall 1 m ahead of the engage point;
after the hold the wall jumps to 0.3 m ahead of the hold point, inside `SH_WALL_MIN` 0.7 m,
and the vehicle must back off to at least 0.55 m and log a nonzero `standoff_shift_m`.

From the PX4 shell: `commander mode ext1` engages, `commander mode posctl` disengages,
`structure_hold status` prints the module state, `structure_hold trip` trips the watchdog. `structure_hold bench [n]` times n policy
steps on synthetic samples (refused while engaged); `bench_test.py` runs it and the other
props-off checks on a flight controller over MAVLink (`docs/bench-test.md`).

### SIH and the velocity estimate

The cloud container has no Docker daemon and no Gazebo, so the test flies PX4's built-in
SIH quadrotor (`PX4_SIM_MODEL=sihsim_quadx`, lockstep, real time). In that setup on
v1.17.0 the EKF's horizontal velocity is unusable: correlation with ground truth about
0.05 (position is fine, 0.96). Stock Position mode wanders about 0.6 m there too. The test
therefore runs the placeholder controller with `SH_VEL_SRC=1` (velocity differentiated
from the estimated position inside the module, low-pass filtered), which holds within
0.3 m. The default stays 0 (estimator velocity) for Gazebo and real flights. The Gazebo
image in `sim2real/docker/` is the simulator the baseline and benchmark use.

Position mode needs a manual-control source, so the test streams a centred joystick
(`MANUAL_CONTROL` at 20 Hz). With `--no-joystick` the stock mode is Hold instead, which
exercises the handback escalation (Position refused → Hold).

## Unit tests

```bash
cd sim2real/px4/tests
g++ -std=c++17 -Wall -Wextra -Werror -I../structure_hold test_watchdog.cpp -o test_watchdog && ./test_watchdog
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_policy.cpp -o test_policy && ./test_policy
g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal \
    -I../structure_hold test_wall_guard.cpp -o test_wall_guard && ./test_wall_guard
python -m pytest sim2real/px4/tests
```

CI runs all four; they need no PX4 tree. `test_policy.cpp` checks the compiled-in MLP
against onnxruntime (within 1e-6), and the observation and fallback action against values
recorded from the Python env (within 3e-7), plus the hand-back, the yaw guard, the
action-to-setpoint mapping and the parameter resolution against the policy's defaults. The
c++14 flags mirror PX4's own build.

## SIH flights (2026-10-22, cloud)

`sitl_test.sh` with `SH_CTRL=0` (cascade) and with `SH_FB_EN=2` (fallback rule alone) pass
every check (experiments/LOG.md): the frame conversions and the attitude loop hold on PX4
dynamics. Policy v2 with the policy's own yaw (`SH_CTRL=1`, guard off, before the guard
existed) tripped the position-error watchdog within 3.4 s: a saturated yaw rate from the
first tick, which is what the same observation gives in onnxruntime and what v2 does in the
mock too (the reward has no heading term). With the yaw guard the heading holds but the
roll/pitch bang-bang at 6 rad/s tips the SIH quad over in 1.5 s: PX4's rate loop reaches the
setpoint 100 to 200 ms late, the mock's does not. With the guard and `SH_RATE_MAX` 1 to
2 rad/s, v2 flies the whole acceptance sequence with no trip (drift 0.58 m at 1 rad/s,
stock Position mode wanders 0.8 m in the same run); 3 rad/s trips. SIH is not a
hold-quality test (see below); the numbers say the integration is sound and that the
policy's rate commands are the sim-to-PX4 gap to close in training (docs/decisions.md).

**Policy v3 (same day, later):** trained with a heading term and a 100 to 200 ms rate-loop
lag in the mock, 2 rad/s actions. With its own defaults (guard off, fallback off, 2 rad/s)
it flies the whole sequence with no trip and no guard: heading within 8 deg of the wall
(as in the mock), altitude within 0.1 m (v2: 0.3 to 0.8 m), horizontal drift growing
slowly to 0.84 m over the 20 s hold (stock Position mode: 0.6 m in the same run), rate
commands still at the 2 rad/s limit much of the time. The forced fallback (`SH_FB_EN=2`)
with the retuned gains 1.5 / 4 is calm (rates under 0.3 rad/s, heading within 1 deg) but,
being a pure velocity brake with no position term, follows the bias of SIH's differentiated
velocity: 1 m after 10 s, position-error trip at 2 m after 19 s (with the old 2.5 / 8 gains
it reached 0.99 m in 20 s). That is a 20 s forced test of a rule meant for 0.5 to 3 s
dropouts, not a fault; the Gazebo flight with real dropouts is where the gains get judged.
`SH_YAW_GUARD=1` on v3 is the field override only (the mock says it hurts).

**Next step:** the Gazebo flight below on the Mac, then the benchmark against tuned PX4
(M1).

## Gazebo flight (needs Docker: the Mac)

`run_gazebo.py` flies the acceptance sequence in the wall-and-slab world with wind and the
VIO-like estimate (dropouts, jumps, bias) through EKF2, GPS off, and scores the TRUE
position over the hold against the hold point the module captured. It is the first flight
that judges the policy rather than the plumbing. The image must carry the current module:

```bash
sim2real/docker/run.sh build                                   # after any change under sim2real/px4 (~5 min)
python -m sim2real.px4.run_gazebo calm_0ms --profile clean     # 1. plumbing: no wind, exact estimate
python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default                # 2. v3 defaults
python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default --param SH_FB_EN=1   # 3. fallback on
python -m sim2real.px4.run_gazebo wind_8ms --profile vio_default --param SH_CTRL=0    # 4. cascade, regression
```

Each writes `runs/<time>_sitl_hold_<scenario>_<profile>_<commit>/` with `summary.json`
(the sequence's checks), `hold.json` (truth metrics, fallback / guard shares, stale time),
`ekf.json`, the verified wind and estimate traces, and `hold_summary.json` for the LOG
entry. About 3 minutes per run (60 s hold, two settles, the trip and the landing).

Expected results, in order; stop and look at the log the first time one is missed:

1. **calm, clean** passes every check (`"ok": true`): engage accepted, 60 s hold with
   `hold.json` p95 well under 0.5 m and `estimate_error` about zero, disengage, re-engage,
   handback under 1 s, latched refusal, landing; `fallback_share` 0, `yaw_guard_share` 0,
   `stale_max_s` about 0.02 (50 Hz odometry), `gps_free` and `vision_fused` true. If this
   one fails, the problem is plumbing (image not rebuilt, port 14540 busy, the bridge not
   started: see `bridge.log`), not the policy.
2. **wind 8 m/s, vio_default** is the policy's number: compare `hold.json metrics.p95_m`
   with the tuned-PX4 baseline (`docs/baseline.md`) and the mock (v3: 0.57 m). Expect
   `stale_max_s` up to 3 s (the profile's longest dropout), `fallback_share` 0 (off for v3),
   and `ekf.json aiding` showing dead-reckoning during dropouts. Known stock behaviour: a
   dropout longer than `COM_POS_FS_DELAY` (1 s) can make commander run its position-loss
   failsafe and leave the mode on its own; the summary then shows an unexpected mode before
   the trip step. That is PX4, not the module; note it and rerun with another `--seed`.
3. **fallback on** should show `fallback_episodes` > 0 and `fallback_share` roughly the
   share of the hold spent more than 0.5 s into a dropout, with the policy back in charge
   on each fresh fix; the hold number tells whether the rule helps or costs on PX4 as the
   mock says it costs v3.
4. **cascade** (`SH_CTRL=0`) must still pass as it did in PR #9's SIH runs; it is the
   regression check that the module's plumbing did not move while the policy went in.

Log every run in `experiments/LOG.md` against the tuned-PX4 baseline.

For the benchmark itself, fly the baseline's scenarios and seeds with the baseline's own
batch runner so the rows compare like for like (same wind and estimate draws per seed,
same PX4 parameter set underneath, same truth scoring over a 60 s window):

```bash
python -m eval.baselines.run_baseline --runner hold --label structure_hold_v19 \
    --params eval/baselines/px4_tuned_v1.params
python -m eval.baselines.run_baseline --compare runs/<tuned_v1>/batch.json runs/<structure_hold_v19>/batch.json
```

Result on 8 seeds (Gate 2): `docs/baseline.md`, "Gate 2 evidence: v19 vs tuned PX4 on 8 seeds, corrected estimate stream" (v19: p95 −47 to −48 % on the mean, −13 to −15 % on the median, lower on 6 of 8 seeds, 0 crashes; the earlier v19 and v10 tables are kept above it as the record of the protocol).

From PR #50 on, new comparisons should pass `-- --profile vio_at_engage` to
`run_baseline --runner hold` for every policy, and `-- --profile vio_at_hover` to the stock
runner for tuned PX4 (which never engages the mode, so the same schedule is keyed to
reaching the hover altitude instead): the estimate's jumps and dropouts are then keyed to
the moment the mode engages (fault clock 25 s at engage, where the seed 1 to 8 holds
opened), so every policy meets the same faults at the same time after engage and none
during takeoff (`sim2real/estimate_bridge/README.md`). The seed 1 to 8 numbers above were
flown on `vio_default` (clock from the bridge's first sample) and compare only with each other.
Known property: on seed 3 the big jump lands 67 s after the engage, 1.3 s after the 60 s
scripted hold, so stock PX4 meets it between the two engages (`docs/baseline.md`, engage
guard confirmation).

## Frames

PX4 boundary: NED world, FRD body, quaternion body → world (`docs/conventions.md`).
`structure_hold_status` positions are NED metres, rates rad/s, thrust normalised 0..1.
