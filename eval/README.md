# eval

Metrics, baselines and the PX4 SITL benchmark harness.

Planned: `metrics.py` (one ulog in, RMS / p95 / worst excursion / recovery / roughness out),
`baselines/px4_tuned_v1.params` (the frozen tuned-PX4 baseline), the MAVSDK scenario runner
and `report.py`. Every later result is compared against the tuned-PX4 baseline.
