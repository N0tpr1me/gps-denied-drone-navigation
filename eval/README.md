# eval

Metrics, baselines and the PX4 SITL benchmark harness.

- `metrics.py`: the metrics every result is scored on (RMS, p95, worst excursion toward the
  surface, gust recovery to 10 cm, contacts, command roughness). Definitions in
  `docs/metrics.md`. `python -m eval.metrics flight.ulg ...` scores a PX4 log.
- `ulog.py`: reads a PX4 `.ulg` into an ENU `Trace` (needs `pip install -e .[ulog]`).
- `gate1.py` + `configs/gate1.yaml`, `configs/gate1_stress.yaml`: the Gate 1 protocol
  (`docs/gate1.md`). A held-out test set on the mock: seeds 9001 to 9003 (training refuses
  9000 to 9999), fixed cells of mean wind 0 / 4 / 8 m/s and hold point 1 m / 2 m from the
  wall, gusts inside every episode, standard estimate corruption. Each env flies exactly its
  first episode and the PD flies the same episodes, so the comparison is paired; crash rates
  get Wilson intervals, mean p95 and the policy-vs-PD differences get bootstrap intervals,
  and each cell gets the hold and safety verdicts. The stress config varies one thing that
  will differ on the real drone at a time (airframe, motor lag, estimate delay, long
  dropouts, near-wall yaw moment). Needs torch (`pip install -e .[torch]`).

```
python -m training.launch --config eval/configs/gate1.yaml --device cpu          # ~30 min
python -m training.launch --config eval/configs/gate1_stress.yaml --device cpu   # ~40 min
```

To judge another policy, add it under `params.controllers` and set `gate_controller`.

Planned: `baselines/px4_tuned_v1.params` (the frozen tuned-PX4 baseline), the MAVSDK
scenario runner and `report.py`. Every later result is compared against the tuned-PX4
baseline.
