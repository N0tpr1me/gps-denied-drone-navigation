# Structure Hold PX4 module

One onboard PX4 flight mode, built into `px4_sitl` first and later into the target
airframe. Since 2026-10-22 it flies policy v2 with the brake-and-hover fallback; the v0
placeholder cascade stays selectable (`SH_CTRL 0`).

What it does today:

- **Registers "Structure Hold" as a named flight mode** with commander through PX4's
  external-component handshake (`register_ext_component_request` / `_reply`,
  `arming_check_request` / `_reply`, `config_control_setpoints`), driven from inside the
  firmware the way `mc_nn_control` does. On the pinned release (v1.17.0) the mode gets
  `nav_state` 23 (EXTERNAL1), shows up in QGC and on an RC mode switch, and PX4's
  failsafes and arming checks apply to it. No commander patch needed.
- **Holds the position and yaw it had when engaged** with policy v2 and the fallback
  (`PolicyController.hpp`, `SH_CTRL 1`, default) or the placeholder P cascade
  (`HoldController.hpp`, `SH_CTRL 0`). Both output FRD body rates and a normalised
  collective thrust on `vehicle_rates_setpoint`. PX4's stock rate controller and control
  allocation stay in charge; the attitude and position controllers are idle while the mode
  is active.
- **Policy inference** is a compiled-in MLP (`MlpPolicy.hpp`, no heap, no PX4 headers, no
  third-party library; docs/decisions.md). `tools/export_policy_header.py` turns the
  released `training/policies/structure_hold_v2.onnx` + `.json` into
  `structure_hold/PolicyWeights.hpp`: weights, the history length (15, from
  `policy.json input.history`, never hardcoded), the observation layout, the action scaling
  and the mock airframe the policy was trained on. The module builds the env's observation
  (structure frame = heading at engage, ENU-style, FLU body) from PX4's NED / FRD state,
  keeps the rolling history, and maps the action back (FLU → FRD rates, thrust in newtons for
  the mock airframe scaled through the hover thrust). `tests/test_policy_header.py` fails
  when the header no longer matches the released policy.
- **Yaw guard** (`SH_YAW_GUARD`, `SH_YAW_P`): the module holds the wall-facing heading
  captured at engage with its own P loop and ignores the policy's yaw output; forced on for
  a policy whose `policy.json` lacks `output.yaw = "policy"` (v2 has no heading term). The
  body-rate setpoints are clamped by `SH_RATE_MAX` (6 rad/s default, the mock's limit; 1 to
  2 rad/s flew in SIH, see below).
- **Brake-and-hover fallback** (docs/fallback.md) once the position fix has been stale for
  `SH_FB_T` (0.5 s): acceleration setpoint `-2.5 × v_est` clamped at 6 m/s², no position
  term, P attitude loop, heading held on the structure; hands back on the first fresh fix.
  The policy keeps running meanwhile so its history stays current. A fix is a new
  `vehicle_visual_odometry` sample (VIO / the estimate bridge) or, with no odometry source,
  a local position update (`SH_FIX_SRC`). `SH_FB_EN 2` forces the fallback every tick for
  tests.
- **Watchdog** (`Watchdog.hpp`, plain C++, no PX4 headers, unit-tested): trips on a
  test fault, an invalid or stale estimate (no fix for `SH_WD_EST_TO`, 5 s so the fallback
  can fly through dropouts; EKF2-invalid flags trip at once), too much tilt or too much
  hold error, in that priority order. A trip latches until disarm. On a trip the module hands back:
  Position mode (Altitude mode if the horizontal estimate is gone), then Hold, then Land
  if the previous request was refused, holding level at hover thrust in between.
- **Publishes `structure_hold_status`** every control tick while engaged (state, trip
  reason, setpoint, error, commanded rates and thrust, controller, fallback active, stale
  time, action flown, loop time), logged by default.
- **Parameters** under `SH_`: `SH_EN`, `SH_CTRL`, yaw guard `SH_YAW_GUARD` / `SH_YAW_P`,
  rate clamp `SH_RATE_MAX`, fallback `SH_FB_EN` / `SH_FB_T` / `SH_FB_K` / `SH_FB_TILT`, fix
  source `SH_FIX_SRC` / `SH_FIX_GAP`, watchdog thresholds `SH_WD_*`, the test hook
  `SH_TEST_FAULT`, placeholder gains `SH_P_*`, `SH_V_MAX`, `SH_VEL_SRC`.

Not here yet: the standoff logic (the hold point is still the engage point), passes along
the surface, the fault injector that corrupts the estimate the policy sees.

## Layout

```
sim2real/px4/
  structure_hold/        the module: CMakeLists.txt, Kconfig, StructureHold.{hpp,cpp},
                         Watchdog.hpp, PolicyController.hpp, MlpPolicy.hpp,
                         PolicyWeights.hpp (generated), HoldController.hpp,
                         structure_hold_params.c
  msg/StructureHoldStatus.msg
  tools/                 export_policy_header.py (onnx -> PolicyWeights.hpp),
                         make_test_vectors.py (Python env -> tests/PolicyTestVectors.hpp)
  install.sh             wires the module into a PX4-Autopilot tree (idempotent)
  sitl_test.sh / .py     acceptance flight in PX4 SITL (SIH quadrotor, no Gazebo)
  tests/                 test_watchdog.cpp, test_policy.cpp (+ PolicyTestVectors.hpp),
                         test_policy_header.py
```

After a policy release, regenerate both headers and commit them:

```bash
python -m sim2real.px4.tools.export_policy_header training/policies/structure_hold_v2.onnx
python -m sim2real.px4.tools.make_test_vectors training/policies/structure_hold_v2.onnx
```

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

From the PX4 shell: `commander mode ext1` engages, `commander mode posctl` disengages,
`structure_hold status` prints the module state, `structure_hold trip` trips the watchdog.

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
python -m pytest sim2real/px4/tests
```

CI runs all three; they need no PX4 tree. `test_policy.cpp` checks the compiled-in MLP
against onnxruntime (within 1e-6), and the observation and fallback action against values
recorded from the Python env (within 3e-7), plus the hand-back and the action-to-setpoint
mapping. The c++14 flags mirror PX4's own build.

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

**Next step:** the Gazebo SITL flight on the Mac (`sim2real/docker`, `sim2real/gazebo`
wall_slab world, `sim2real/estimate_bridge` for the VIO-like estimate with dropouts):
engage with `SH_CTRL=1` (try `SH_RATE_MAX` 2 first), confirm the fallback takes over
0.5 s into a dropout and hands back, then the benchmark against tuned PX4 (M1). Policy v3
(heading term, scored with the same yaw guard) is in training; the rate-loop gap needs the
x500's response in the mock before any flight.

## Frames

PX4 boundary: NED world, FRD body, quaternion body → world (`docs/conventions.md`).
`structure_hold_status` positions are NED metres, rates rad/s, thrust normalised 0..1.
