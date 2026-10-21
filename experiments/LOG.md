# Experiment log

One line per real run, appended by `training/launch.py` (smoke runs are not logged).
Hand-run experiments (for example SITL baseline batches) add a line in the same format.

Columns: date (UTC), run folder, git commit, config, device, status, result, vs baseline.
"vs baseline" compares against tuned PX4 (`eval/baselines/px4_tuned_v1.params`) or, before
that exists, says which classical baseline was used.

| Date | Run | Commit | Config | Device | Status | Result | vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
