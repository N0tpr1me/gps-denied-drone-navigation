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
