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
  dropouts, near-wall yaw moment). `configs/gate1_x500.yaml` is the same test set on the
  PX4 Gazebo x500 (`airframe: gz_x500`); a policy trained on another airframe flies there
  through the module's thrust mapping for its own mass (`ModuleThrustScale`), as on PX4.
  Needs torch (`pip install -e .[torch]`).

```
python -m training.launch --config eval/configs/gate1.yaml --device cpu          # ~30 min
python -m training.launch --config eval/configs/gate1_stress.yaml --device cpu   # ~40 min
python -m training.launch --config eval/configs/gate1_x500.yaml --device cpu     # ~30 min
```

To judge another policy, add it under `params.controllers` and set `gate_controller`.

- `baselines/px4_tuned_v1.params`: the frozen tuned-PX4 baseline (docs/baseline.md). Every
  later result is compared against it.
- `baselines/run_baseline.py`: fly a parameter set over the `wall1m_*` scenarios and seeds in
  Gazebo SITL (GPS-free, corrupted estimate), tabulate p95, worst toward the wall, crash and
  failsafe, and compare two sets.
- `baselines/autotune.py` (+ `autotune_flight.py`, in the image): PX4's attitude autotune,
  GPS-free.

Planned: `report.py`.
