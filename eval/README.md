# eval

Metrics, baselines and the PX4 SITL benchmark harness.

- `metrics.py`: the metrics every result is scored on (RMS, p95, worst excursion toward the
  surface, gust recovery to 10 cm, contacts, command roughness). Definitions in
  `docs/metrics.md`. `python -m eval.metrics flight.ulg ...` scores a PX4 log.
- `ulog.py`: reads a PX4 `.ulg` into an ENU `Trace` (needs `pip install -e .[ulog]`).

- `baselines/px4_tuned_v1.params`: the frozen tuned-PX4 baseline (docs/baseline.md). Every
  later result is compared against it.
- `baselines/run_baseline.py`: fly a parameter set over the `wall1m_*` scenarios and seeds in
  Gazebo SITL (GPS-free, corrupted estimate), tabulate p95, worst toward the wall, crash and
  failsafe, and compare two sets.
- `baselines/autotune.py` (+ `autotune_flight.py`, in the image): PX4's attitude autotune,
  GPS-free.

Planned: `report.py`.
