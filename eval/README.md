# eval

Metrics, baselines and the PX4 SITL benchmark harness.

- `metrics.py`: the metrics every result is scored on (RMS, p95, worst excursion toward the
  surface, gust recovery to 10 cm, contacts, command roughness). Definitions in
  `docs/metrics.md`. `python -m eval.metrics flight.ulg ...` scores a PX4 log.
- `ulog.py`: reads a PX4 `.ulg` into an ENU `Trace` (needs `pip install -e .[ulog]`).

Planned: `baselines/px4_tuned_v1.params` (the frozen tuned-PX4 baseline), the MAVSDK
scenario runner and `report.py`. Every later result is compared against the tuned-PX4
baseline.
