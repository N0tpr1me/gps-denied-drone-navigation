# Experiment log

One line per real run, appended by `training/launch.py` (smoke runs are not logged).
Hand-run experiments (for example SITL baseline batches) add a line in the same format.

Columns: date (UTC), run folder, git commit, config, device, status, result, vs baseline.
"vs baseline" compares against tuned PX4 (`eval/baselines/px4_tuned_v1.params`) or, before
that exists, says which classical baseline was used.

| Date | Run | Commit | Config | Device | Status | Result | vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-21 17:43 | runs/20261021-174342_sitl_hover_9d515d8 | 9d515d8 | sim2real/docker hover-test (gz_x500, 2.5 m, 30 s) | MacBook (Apple Silicon), Docker arm64, 10 CPU | ok | takeoff 9.7 s, hover alt 2.50 ± 0.03 m (max err 0.18 m), xy drift 0.04 m, land 6.2 s, RTF mean 0.98 min 0.85 | stock PX4 v1.17.0 defaults; acceptance run, no tuned baseline yet |
| 2026-10-21 | sitl_structure_hold (cloud, SIH quad, no Gazebo) | see PR | sim2real/px4/sitl_test.sh, SH_VEL_SRC=1, joystick / --no-joystick | CPU | passed (both) | Structure Hold engages (nav_state 23), holds 20 s with 0.25 m max drift, handback after trip in 16 ms (joystick, to Position) / 112 ms (no joystick, PX4 RC-loss failsafe), status topic logged | plumbing test, no baseline comparison; stock Position mode in the same SIH run wandered 0.66 m |
| 2026-10-21 20:41 | runs/20261021-204141_sitl_wind_calm_0ms_027ee90 | 027ee90 | sim2real/gazebo calm_0ms (wall_slab world, spawn 2 m from wall, adapter on, 2.5 m, 30 s) | MacBook (Apple Silicon), Docker arm64, 10 CPU | ok | hover alt 2.50 ± 0.04 m (max err 0.20 m), xy drift 0.07 m, RTF mean 0.92 min 0.70; trace 14109 steps at 4 ms, applied force 0 N throughout, max error vs Python model 1.7e-18 N, lag 0 steps | stock PX4 v1.17.0 defaults in the wall-and-slab world; still-air control for the wind runs |
| 2026-10-21 20:43 | runs/20261021-204344_sitl_wind_wind_4ms_4d1b4fb | 4d1b4fb | sim2real/gazebo wind_4ms (4 m/s into the wall, gusts at 15 s and 40 s, turbulence 0.5 m/s, seed 1) | MacBook (Apple Silicon), Docker arm64, 10 CPU | ok | hover alt 2.52 ± 0.04 m (max err 0.18 m), xy drift 0.08 m, RTF mean 1.17 min 0.78; trace 13844 steps, applied force max 0.98 N rms 0.40 N, shelter 0.93 to 0.95, max error vs Python model 7.5e-15 N, lag 0 steps | stock PX4 v1.17.0 defaults; same seed as the torch_quad env would draw |
| 2026-10-21 20:44 | runs/20261021-204448_sitl_wind_wind_8ms_4d1b4fb | 4d1b4fb | sim2real/gazebo wind_8ms (8 m/s into the wall, gusts at 15 s and 40 s, turbulence 1 m/s, seed 1) | MacBook (Apple Silicon), Docker arm64, 10 CPU | ok | hover alt 2.48 ± 0.04 m (max err 0.20 m), xy drift 0.30 m, RTF mean 0.92 min 0.75; trace 14075 steps, applied force max 3.35 N rms 1.56 N, shelter 0.91 to 0.95, max error vs Python model 3.0e-14 N, lag 0 steps | stock PX4 v1.17.0 defaults; drift 4x the calm run under 8 m/s, tuned baseline comes with T8 |
