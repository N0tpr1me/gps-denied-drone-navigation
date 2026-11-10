# PX4 versions the module builds and flies on

A drone maker runs the PX4 release their own fork is on, not ours. This page says which
releases the Structure Hold module has been built and flown on in simulation, what had to
change in the module for each, and how to check a new release in one command.

**PX4 v1.17.0 is still the only benchmark release** (`docs/decisions.md`). The tuned
baseline, every hold number in `docs/baseline.md`, the safety manual's envelope, the bench
test and the partner pack are all on v1.17.0 and nothing here changes that. On the other
releases the claim is narrower: the module compiles, registers its three modes, and passes
the same scripted acceptance flights in the simulators. No hold quality is claimed there,
because a fair number needs a baseline tuned on that release first.

## What was checked

Checked 2026-11-09 and 2026-11-10 in the cloud container (amd64, 4 cores), on the newest tag
of each release line. **PX4 v1.18.0 has no final release yet**: v1.18.0-rc1 is its newest
tag, so the v1.18 column is a release candidate and is worth one more run when the final
comes out ("Checking a release" below).

| Check | v1.16.2 | v1.17.0 (benchmark) | v1.18.0-rc1 |
| --- | --- | --- | --- |
| Module builds into `px4_sitl_default` with `install.sh` | yes | yes | yes |
| The 34 `SH_` parameters PX4 generates (names, defaults, limits, units, texts) match `structure_hold_params.c` | yes | yes | yes |
| SIH acceptance flight (`sitl_test.sh`: ground refusals, engage, 20 s hold, disengage, trip and handback, latched refusal, Structure Land, the pilot's status) | passed, 23 of 23 checks | passed, 23 of 23 | passed, 23 of 23 |
| SIH with a simulated VIO that keeps moving on the ground (`--prearm`: arming refused, allowed again once it is still or silent) | passed, 29 of 29 | passed, 29 of 29 | passed, 29 of 29 |
| SIH with a simulated lidar and a pole (`--keep-out`) | passed, 29 of 29 | passed, 29 of 29 | passed, 29 of 29 |
| SIH with a simulated rangefinder and a creeping wall (`--wall`) | passed, 31 of 31, on one flight of three (see below) | passed, 31 of 31 | passed, 31 of 31 |
| Gazebo image builds (`cloud_build.sh`) and flies the hover test | yes, passed | yes (every benchmark flight) | yes, passed |
| Gazebo Structure Hold flight on the estimate bridge, GPS off, forward rangefinder, 8 m/s wind, 1 m from the wall (`run_gazebo wall1m_wind_8ms --profile vio_at_engage --rangefinder`, seed 1) | passed every check twice, no wall contact | passed (the benchmark) | passed every check twice, no wall contact |

The first three rows are what `version_check.sh` does in one command.

The Gazebo flights were flown twice per release, before and after main gained the pre-arm
VIO check, on an image rebuilt each time; the SIH rows are from the later commit. Run
folders and commits are in `experiments/LOG.md` (2026-11-09 20:15 to 2026-11-10 03:36).

## What PX4 v1.18 changed, and how the module copes

Three things in v1.18 stopped the v1.17 module from compiling. The module now detects each
one at build time from the PX4 tree it is installed into, so one copy of the source builds
on all three releases and there is no version switch to set.

1. **How an external mode names the controllers it wants.** Up to v1.17 the mode publishes
   a `vehicle_control_mode` on `config_control_setpoints`. v1.18 removed that topic: the
   mode publishes a setpoint type on `setpoint_config` instead (rates for Structure Hold,
   trajectory for Structure Hover and Structure Land). `StructureHold.hpp` picks by whether
   the `setpoint_config` header exists; `sendControlModeConfig` has both forms.
2. **`ModuleBase`.** v1.18 made it a plain class with a per-module `Descriptor` instead of a
   template. The module's `CMakeLists.txt` looks for the descriptor in PX4's `module.h` and
   defines `SH_MODULE_DESCRIPTOR`; the four places that start, find and stop the module go
   through two small helpers (`setInstance`, `runningInstance`).
3. **Parameter definitions.** v1.18 no longer reads `*_params.c` files; it reads only YAML
   module configs. `structure_hold_params.c` stays the one source of truth (the module's
   limits, the integration guide and the safety manual are pinned to it), and
   `structure_hold_params.yaml` is generated from it by
   `python -m sim2real.px4.tools.params_yaml`. CMake hands the YAML to PX4 only on a tree
   that no longer reads the C file, so no release sees a parameter twice. A test fails when
   the YAML is stale, and `params_yaml --against <build>/parameters.json` compares what a
   PX4 build really generated with the C file (the second row of the table).

Nothing else differed: the registration and arming-check handshake, the topics the module
reads and the setpoints it publishes are the same on the three releases. `install.sh` is
unchanged; its anchors exist in all three trees.

## v1.16 notes

- The module needed no change for v1.16.2.
- A shallow clone of v1.16 does not build until NuttX's tags are fetched (its version
  header reads them): `git -C platforms/nuttx/NuttX/nuttx fetch --depth 1 --tags origin`.
  `version_check.sh` does it; the Docker image did not need it.
- v1.16's Gazebo start-up script uses only x, y, z of `PX4_GZ_MODEL_POSE` and ignores the
  yaw. Every Structure Hold scenario spawns at yaw 0, so they fly the same; the Route Follow
  worlds spawn along the route's first leg and would start facing the wrong way on v1.16.
  Route Follow was not flown on v1.16 or v1.18.

## Two SIH results that were about the test bench, not the module

**The guarded hover in SIH (`--wall`).** After a trip next to a measured wall the module
hands to Structure Hover, which PX4's own position controller flies on EKF2's velocity. In
SIH that velocity is poor on every release (`sim2real/px4/README.md`, "SIH and the velocity
estimate"): about 0.1 m/s off by the end of the flight on all three, sometimes toward the
wall and sometimes away. The test wants the hover to stay 0.55 m from the creeping wall, and
SIH lands close to that on every release: v1.17.0 ended at 0.59 and 0.65 m, v1.18.0-rc1 at
0.76 and 0.77 m, and v1.16.2 at 0.57 m on one flight and 0.44 and 0.46 m on two others,
which failed that one check. In Gazebo, where EKF2's velocity is good, the same hand-over
kept 0.86 m on v1.16.2. So a `--wall` miss on the hover check alone says more about SIH
than about the release; fly it again, and trust the Gazebo flight for the guarded hover.

**The keep-out flight's last check.** The first `--keep-out` flights passed on v1.17.0 and
failed `stock_mode_flying_after_disengage` on v1.16.2 and v1.18.0-rc1. The cause was in
the test, on every release: it switched its simulated lidar off after the pole stretch, and
PX4's own collision prevention commands Hold once its obstacle data is 5 s old, the same
5 s the check waits. v1.17.0 had been passing by a tenth of a second. The test now leaves
the lidar on with the pole out of range, as a real lidar would be, and has passed on all
three releases twice since.

## Not checked

- `run.sh build` from plain Ubuntu on v1.16 and v1.18: the cloud container cannot reach the
  Gazebo package host, so the images here were built with `cloud_build.sh` (PX4's own
  Gazebo base image). On the GPU server or the Mac it is `PX4_TAG=<tag> ./run.sh build`.
- Flight controller firmware (`px4_fmu-v6x_structurehold` and the flash budget) on v1.16
  and v1.18. The flash numbers in `sim2real/px4/README.md` are v1.17.0's.
- The props-off bench test, which checks for v1.17.0 firmware on purpose, the partner pack
  and the safety manual: all stay on v1.17.0 until a partner flies another release.
- Hold quality, tuned PX4 and the field faults on v1.16 or v1.18. One Gazebo flight per
  release says the modes work there, not how well they hold.
- The Auterion app (`sim2real/auterion/`), which follows AuterionOS's own PX4.
- Route Follow's Mission flights.

## Checking a release

One command clones the tag, installs and builds the module, compares the generated
parameters with the C file and flies the SIH acceptance flight:

```bash
sim2real/px4/version_check.sh v1.18.0              # tree in ~/px4-trees/v1.18.0
sim2real/px4/version_check.sh v1.16.2 /data/px4 -- --wall
```

It needs PX4's build tools and a Python 3.10 to 3.12 with PX4's `Tools/setup/requirements.txt`,
pymavlink and pyulog (PX4's `empy<4` does not build on Python 3.13). Exit 0 means all four
steps passed; add the result to the table above.

For Gazebo, build an image for the tag and hand it to any runner with `--image`. Images are
named after their PX4 tag, so they sit side by side and the default stays v1.17.0:

```bash
PX4_TAG=v1.18.0 sim2real/docker/run.sh build          # or cloud_build.sh in the cloud
PX4_TAG=v1.18.0 sim2real/docker/run.sh hover-test
python -m sim2real.px4.run_gazebo wall1m_wind_8ms --profile vio_at_engage --seed 1 \
    --rangefinder --params eval/baselines/px4_tuned_v2.params \
    --image structure-hold/px4-sitl:v1.18.0
```

When the module stops building on a new release, look first at the three places above:
PX4's external-mode messages, `ModuleBase`, and how parameters are declared.
