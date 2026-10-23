# sim2real/estimate_bridge

GPS-free flight in PX4 SITL on a VIO-like estimate: Gazebo's ground truth goes through
`envs/estimate` (delay, noise, bias walk, jumps, dropouts) and into PX4's EKF2 as external
vision odometry, with GPS fusion off. This is what makes the SITL benchmark GPS-free.

```
bridge.py          EstimateBridge: truth sample in, corrupted sample out; trace CSV
run_bridge.py      the in-container process: gz pose/info -> bridge -> MAVLink ODOMETRY
config.py          estimate profiles (profiles/clean.yaml, profiles/vio_default.yaml)
params.py          QGC .params reader -> PX4_PARAM_* environment for the container
ekf2_ev.params     EKF2 on vision, GPS off (loaded at PX4 start, before EKF2 runs)
heartbeat.py       PX4 HEARTBEAT custom_mode -> is Structure Hold flying (for the fault anchor)
verify_trace.py    replay the trace through envs.estimate: PX4 got exactly the model
ulog_checks.py     EKF2 aiding, estimate-vs-truth error, vision gaps and modes from the ulog
run_estimate.py    host runner: scenario + profile -> run folder, verify, score
```

## How the estimate reaches EKF2

1. `run_estimate.py` writes the scenario's wind (`sim2real/gazebo`) and the estimate profile
   to the run folder, mounts it at `/out` and the repo at `/repo`, and starts the image's
   `hover-test` with the `ekf2_ev.params` values as `PX4_PARAM_*` variables, which PX4's
   startup script applies before EKF2 starts.
2. The entrypoint starts `run_bridge.py` beside PX4. It subscribes to Gazebo's
   `/world/wall_slab/pose/info` (50 Hz, the same stream PX4 logs as
   `vehicle_local_position_groundtruth`), differentiates it for velocity, steps
   `EstimateCorruptor(cfg, num_envs=1, dt=0.02, seed)` and sends every non-stale sample as
   a MAVLink ODOMETRY message (local NED position and velocity, no attitude, no covariance,
   estimator type VIO) to PX4's second onboard link on port 14030. PX4 turns it into
   `vehicle_visual_odometry`; EKF2 fuses position, height and velocity from it
   (`EKF2_EV_CTRL 7`, `EKF2_HGT_REF 3`, `EKF2_GPS_CTRL 0`, `SYS_HAS_GPS 0`). Heading still
   comes from the magnetometer.
3. During a dropout nothing is sent: EKF2 sees a gap and dead-reckons on the IMU, as it
   would with a real camera that lost tracking. Jumps arrive as ordinary samples; the EKF's
   innovation gate decides what to do with them (see the LOG entries for what it did).
4. Samples are stamped on arrival: the bridge answers no TIMESYNC, so PX4's timesync
   offset stays zero and `sync_stamp` uses the receive time. The corruptor's delay is real
   latency to the EKF, compensated only by the constant `EKF2_EV_DELAY` (the mean of the
   profile's delay range), as on a VIO without clock sync.
5. After the flight, `verify_trace.py` replays the bridge's trace through `envs.estimate`
   with the run's profile and seed (must match to 1e-9 m, with nothing published while
   stale), `ulog_checks.py` reads what EKF2 did, and `eval.metrics` scores the TRUE position
   from the ulog over the hover window.

Seed semantics: the bridge builds the corruptor exactly as `envs/structure_hold` does
(config, one env, dt, seed) with the profile's seed, independent of the wind scenario's
seed, as in training where both generators are seeded separately. NumPy and PyTorch draw
different streams for the same seed (`envs/estimate/torch_adapter.py`), so "same seed"
means the same NumPy stream here, not bit-identical to a torch_quad episode.

## Use

```bash
sim2real/docker/run.sh build                                        # once after a Docker change
python -m sim2real.estimate_bridge.run_estimate calm_0ms --profile clean --hover-s 60
python -m sim2real.estimate_bridge.run_estimate calm_0ms --profile vio_default --hover-s 60
python -m sim2real.estimate_bridge.run_estimate wind_8ms --profile vio_default --seed 2
```

Exit code 0 means: hover passed (the hover test's own numbers come from PX4's estimate, so
they only show PX4 believed it held), estimate and wind traces verify, GPS was never used,
EKF2 fused vision for most of the flight. The truth-based numbers are in `metrics.json`
and `ekf.json`.

Profile keys: `name`, `seed`, `divisor` (corruptor every N-th 50 Hz sample), `estimate`
(any `EstimateCorruptionConfig` key, or `clean`), `fault_anchor` (`start`, the default,
`engage` or `altitude`), `fault_anchor_s` (25.0) and `fault_anchor_alt_m` (the `altitude`
anchor's height; `null` lets the runner fill it from the scenario).

Fault anchor. With `start` the fault clock runs from the bridge's first sample, so where a
jump or dropout lands relative to the hold depends on how long PX4 took to arm and take
off (0.8 s of spread over the seed 1 to 8 batches; seed 7's jumps landed during arming and
broke both controllers, and v11's seed 3 strike came from a jump 0.1 to 0.4 s before the
scripted disengage that v10 had met 1.3 s before it). With `engage` the bridge watches
PX4's heartbeat (asked for at 20 Hz on its link) and keys the schedule to the first sample
with Structure Hold flying: before it the same seed's draws run with every jump and
dropout suppressed; at it the corruptor is rebuilt from the seed with the fault clock at
`fault_anchor_s`, pre-rolled with faults suppressed, and the delay history, bias and
dropout state carried over so the stream is continuous (`bridge.py`, `EstimateBridge._anchor`).
From then on the draws are the ones `vio_default` makes at fault clock `fault_anchor_s` +
time since engage, so every policy and seed meets the same faults at the same time after
engage, and 25 s makes the in-hold faults the ones the seed 1 to 8 batches flew. The trace
gains an `engaged` column and `verify_trace` replays with it (`anchor_t`, `anchor_step` in
its report). Shipped as `vio_at_engage`; pass `-- --profile vio_at_engage` to
`eval.baselines.run_baseline --runner hold`.
Stock PX4 never engages the mode, so for the tuned baseline the same schedule is keyed to
height instead: `vio_at_hover` (`fault_anchor: altitude`) anchors on the first truth
sample `fault_anchor_alt_m` above the first one, and `run_estimate` fills that height in
as the scenario's hover altitude minus the 0.3 m margin of `ulog_checks.hover_window`,
where the scored hover begins. The two anchors differ by the hold runner's settle time
between reaching altitude and engaging (a few seconds); the bridge log says which fired
and `verify_trace` replays either from the trace. Pass `-- --profile vio_at_hover` to the
stock runner's `run_baseline` for the tuned-PX4 side of a matched comparison.

## Notes

- Why MAVLink ODOMETRY rather than uXRCE-DDS: see `docs/decisions.md`.
- `/model/x500_0/odometry_with_covariance`, which PX4's gz bridge would also turn into
  visual odometry, is not published by the stock x500 model, so the bridge is the only
  vision source. Check `ekf.json` → `vision.rate_hz` ≈ 50 if in doubt.
- The gz Python bindings (`gz.transport13`, `gz.msgs10`) and pymavlink are in the image
  already; the host needs only the repo's `[dev]` extras.

## What EKF2 did with the corruption (runs of 21 Oct 2026, `experiments/LOG.md`)

- **Dropouts** (0.2 to 3 s, nothing sent): EKF2 dead-reckons on the IMU (`cs_inertial_dead_reckoning` 15 to 17 % of the flight) and resumes vision fusion when samples return, resetting horizontal position to vision (3 xy resets per 60 s run). A 2.5 s dropout during the takeoff tripped PX4's position-loss failsafe (`COM_POS_FS_DELAY` 1 s): Land for about a second, then Takeoff resumed and the hover completed. That is the stock failover a real drone shows when its VIO drops out longer than the failsafe delay; it is logged, not hidden.
- **Jumps and bias**: with `EKF2_EVP_NOISE` 0.1 m the 5 sigma gate is 0.5 m, so the bias walk and small jumps pass as ordinary measurements (0 rejected in these runs; innovation max 0.08 m); the estimate follows the corrupted stream and sits about 0.07 m off truth. A jump above the gate would be rejected until it decays or the fusion times out and resets to vision.
- **Delay**: samples stamped on arrival plus `EKF2_EV_DELAY` 100 ms; EKF2 reports a 154 ms stamp-to-sample offset (delay plus its fusion horizon), no visible lag effect in a hover.
- **Cost of the corruption on stock PX4 at 8 m/s**: p95 error 0.145 m (clean) to 0.219 m (vio_default), worst excursion toward the wall 0.30 to 0.36 m.
