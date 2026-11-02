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

- `gazebo_check.py` + `configs/gazebo_check.yaml`: the Gazebo-like check (`docs/gazebo_check.md`),
  run before any policy goes to the Mac for a Gazebo re-fly. It flies the policies on the
  `gz_x500_sitl` airframe (PX4's yaw torque and gyro filters, the 83 Hz module tick, the
  bridge's 0.24 m lever arm, EKF2-like fusion with its resets) through the benchmark's
  conditions: 60 s, 1 m from the wall, hold point 2.4 m up, calm on a clean estimate and
  calm / 4 / 8 m/s on `vio_default`, plus 8 m/s blowing into the wall, where v9 drifts. It
  scores the lift above the hold point and the 1.6 Hz roll wobble that lifted v6 and v8 in
  Gazebo, as well as p95 and crashes, and a policy passes only if it beats the released
  policy (v10) on all of them. Gazebo's numbers for the policies flown there are printed
  beside the mock's.

```
python -m training.launch --config eval/configs/gazebo_check.yaml --device cpu   # ~25 min
```

- `baselines/px4_tuned_v2.params`: the tuned-PX4 baseline since 2026-11-01: v1 with
  barometer height (`EKF2_HGT_REF 0`; it read 1, GPS in PX4 v1.17, until 2026-11-02 and flew
  on the barometer by fallback; docs/decisions.md). Every result from then on is
  compared against it.
- `baselines/px4_tuned_v1.params`: the first tuned-PX4 baseline (docs/baseline.md), vision
  height; frozen, kept for the results flown on it up to 2026-11-01.
- `baselines/run_baseline.py`: fly a parameter set over the `wall1m_*` scenarios and seeds in
  Gazebo SITL (GPS-free, corrupted estimate), tabulate p95, worst toward the wall, crash and
  failsafe, and compare two sets.
- `baselines/autotune.py` (+ `autotune_flight.py`, in the image): PX4's attitude autotune,
  GPS-free.

- `field_faults.py`: scores Gazebo flights flown on the scripted field faults
  (`docs/field-faults.md`).
- `eval_kit.py` (run by `../evaluate.sh`): the evaluation kit, the five field faults flown
  with Structure Hold and with tuned PX4 in one command, one comparison table
  (`docs/eval-kit.md`).
- `partner_session.py` + `partner_report.py`: score a partner's real flight session and turn
  it into a one-page benchmark report (`docs/partner-pack.md`, `docs/eval-kit.md`).
