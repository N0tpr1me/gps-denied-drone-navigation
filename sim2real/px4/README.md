# Structure Hold PX4 module (skeleton)

One onboard PX4 flight mode, built into `px4_sitl` first and later into the target
airframe. This is the v0 skeleton: it proves the mode plumbing before any learned policy
exists.

What it does today:

- **Registers "Structure Hold" as a named flight mode** with commander through PX4's
  external-component handshake (`register_ext_component_request` / `_reply`,
  `arming_check_request` / `_reply`, `config_control_setpoints`), driven from inside the
  firmware the way `mc_nn_control` does. On the pinned release (v1.17.0) the mode gets
  `nav_state` 23 (EXTERNAL1), shows up in QGC and on an RC mode switch, and PX4's
  failsafes and arming checks apply to it. No commander patch needed.
- **Holds the position and yaw it had when engaged.** A placeholder P cascade
  (`HoldController.hpp`: position → velocity → acceleration → attitude → body rates)
  stands in for the policy. It outputs what the policy will output: FRD body rates and a
  normalised collective thrust on `vehicle_rates_setpoint`. PX4's stock rate controller
  and control allocation stay in charge; the attitude and position controllers are idle
  while the mode is active.
- **Watchdog** (`Watchdog.hpp`, plain C++, no PX4 headers, unit-tested): trips on a
  test fault, an invalid or stale estimate, too much tilt or too much hold error, in
  that priority order. A trip latches until disarm. On a trip the module hands back:
  Position mode (Altitude mode if the horizontal estimate is gone), then Hold, then Land
  if the previous request was refused, holding level at hover thrust in between.
- **Publishes `structure_hold_status`** every control tick while engaged (state, trip
  reason, setpoint, error, commanded rates and thrust, loop time), logged by default.
- **Parameters** under `SH_`: `SH_EN`, watchdog thresholds `SH_WD_*`, the test hook
  `SH_TEST_FAULT`, placeholder gains `SH_P_*`, `SH_V_MAX`, `SH_VEL_SRC`.

Not here yet: policy inference, the structure frame and standoff logic, passes along the
surface, the fault injector that corrupts the estimate the policy sees.

## Layout

```
sim2real/px4/
  structure_hold/        the module: CMakeLists.txt, Kconfig, StructureHold.{hpp,cpp},
                         Watchdog.hpp, HoldController.hpp, structure_hold_params.c
  msg/StructureHoldStatus.msg
  install.sh             wires the module into a PX4-Autopilot tree (idempotent)
  sitl_test.sh / .py     acceptance flight in PX4 SITL (SIH quadrotor, no Gazebo)
  tests/test_watchdog.cpp
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
cd sim2real/px4/tests && g++ -std=c++17 -Wall -Wextra -Werror -I../structure_hold test_watchdog.cpp -o test_watchdog && ./test_watchdog
```

CI runs this; it needs no PX4 tree.

## Frames

PX4 boundary: NED world, FRD body, quaternion body → world (`docs/conventions.md`).
`structure_hold_status` positions are NED metres, rates rad/s, thrust normalised 0..1.
