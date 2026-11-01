# Fitting the estimate model to real VIO error

Every robustness number so far is measured against `envs/estimate`, a VIO model whose
numbers came from papers, not from data (`docs/real-flight-gap.md`, row 1). The plan was one
logging flight with our own drone (`docs/logging-flight.md`). Since the project went
software only (`docs/decisions.md`, 2026-10-30), the real VIO error comes from public
real-drone datasets with ground truth instead. This page is how the fit works, the first
result, and what it does not cover yet.

## The method (`eval/vio_fit.py`)

Input: pairs of time-stamped trajectories, the VIO's output and the dataset's ground truth,
one pair per flight. Per flight the fit aligns the VIO onto truth (a yaw rotation and a
shift, the standard 4-DoF alignment), takes the error `e(t)` = VIO minus truth, and measures:

| Corruptor field | How it is measured |
| --- | --- |
| `pos_noise_std_m`, `bias_walk_std_m_per_sqrt_s` | The drift since engage: the rms of `e(t + tau) - e(t)` for `tau` from one sample to 20 s. The corruptor's error is white noise plus a random walk, whose drift is `2 noise^2 + walk^2 tau`; both are fitted by least squares on the relative misfit. Measured twice: over every window, and over near-hover windows only (truth moved at most 0.3 m/s on average), which is what a hold sees. |
| `bias_max_m` | The 95th percentile of the drift at 60 s (the Gazebo hold length). Near-hover windows when at least 3 flights have enough of them at 60 s, else every window. |
| `jump_*` | A single-step change of `e` above 0.1 m and 8 times the usual step-to-step spread. Rate, size range, and how much is left 5 s later (the decay). |
| `dropout_*` | A gap between VIO samples longer than 0.2 s and 3 sample periods: a real VIO sends nothing while it has lost tracking. |
| `vel_noise_std_mps` | VIO velocity minus truth velocity, when the file carries velocity. |
| `delay_*` | **Not measured.** Offline VIO output is stamped with the image time, so it has no latency in it. Latency needs the VIO's processing time on a live run. |

Fields with no events in the data keep their default rather than being switched off: no
tracking loss in a textured lab is not evidence of none in front of a bare wall.

The method is checked by a round trip (`eval/tests/test_vio_fit.py`): our own corruptor
corrupts a known trajectory at 20 Hz with known numbers, and the fit gets them back (noise
within 15 %, walk within 30 %, jump and dropout rates within 35 %, sizes and lengths inside
their ranges).

## First result: 12 EuRoC flights (2026-10-30)

The dataset hosts (ETH, UZH, TUM) are blocked by the cloud environment's network policy, so
the first fit uses the example results published with the RPG trajectory evaluation toolbox
(University of Zurich, MIT licence, pinned in `eval/configs/vio_fit_rpg_euroc.yaml`): one
monocular and one stereo VIO run on six EuRoC MAV flights (MH_01, MH_03, MH_05, V2_01,
V2_02, V2_03), real VIO output at 20 Hz against the dataset's laser-tracker and Vicon truth.
The first 10 s of each flight (VIO start-up and take-off) are left out. Run
`20261030-174839_vio_fit_rpg_euroc_s0_53457f3` (`experiments/LOG.md`), 1399 s of VIO:

| Drift since engage, rms per axis | 0.5 s | 1 s | 5 s | 20 s | 60 s |
| --- | --- | --- | --- | --- | --- |
| Real VIO, near-hover windows | 0.011 | 0.016 | 0.023 | 0.033 | (too few) |
| Real VIO, every window (flying 0.3 to 1 m/s) | 0.017 | 0.027 | 0.067 | 0.129 | 0.230 |
| Fitted model (near-hover) | 0.007 | 0.010 | 0.022 | 0.045 | 0.078 |
| Current default (0.02 m noise, 0.01 m/sqrt(s) walk) | 0.029 | 0.030 | 0.036 | 0.053 | 0.082 |

| Field | Current default | Fitted | Reading |
| --- | --- | --- | --- |
| Bias walk | 0.01 m/sqrt(s) | 0.010 (near hover); 0.021 over every window | The default matches a hovering VIO. Moving at 0.3 to 1 m/s drifts about twice as fast (scale and yaw error grow with distance flown). |
| Bias cap | 0.3 m | 0.41 m | Taken from every window, because near-hover 60 s windows exist in only one flight: an upper bound, but above the default. The worst flight (mono, V2_03, fast motion) drifts 0.84 m rms in 60 s. |
| Position noise | 0.02 m | 0.001 m | VIO output is smooth sample to sample. The default is 20 times rougher than this: pessimistic, so safe. |
| Jumps | 0.1 to 2 m at 0.02 Hz, decay 5 s | 0.1 to 0.2 m at 0.002 Hz, decay 9 s | 3 small jumps in 1399 s. Pure VIO has no loop closure, so nothing relocalises; the default's large jumps stand for resets the data does not show. |
| Dropouts | 0.2 to 3 s at 0.02 Hz | none seen | Well-textured lab, offline on a laptop: no frame drops above 0.1 s. Kept. |
| Delay, velocity noise | 50 to 150 ms; 0.05 m/s | not measurable | Kept. |

In short: the made-up model was right about drift when hovering, pessimistic about noise and
jumps, and somewhat optimistic about how far drift can build up over a minute.

## Second fit: Basalt on raw EuRoC (2026-11-01)

Our own VIO run on the raw dataset, on the Mac: Basalt (TUM, stereo-inertial, the EuRoC
calibration and config shipped with it) built in Docker for arm64 (`eval/vio_basalt/`,
pinned commit, patched to also write its velocity), run offline on the EuRoC ASL images and
IMU, scored against the dataset's own `state_groundtruth_estimate0` by
`eval/configs/vio_fit_basalt_euroc.yaml`. Five Vicon-room flights so far (V1_01, V1_02,
V1_03, V2_02, V2_03): the ETH Research Collection rate-limits downloads hard, and the
machine-hall flights (MH_01 to MH_05) and V2_01 had not come through when this was run.
Same fit settings as the first fit (first 10 s skipped). Run
`20261101-015732_vio_fit_basalt_euroc_s0_b49b1dc` (`experiments/LOG.md`), 512 s of VIO:

| Flight | ATE (m) | Mean speed (m/s) | Drift at 60 s, rms per axis (m) | Jumps | Gaps |
| --- | --- | --- | --- | --- | --- |
| V1_01_easy | 0.045 | 0.43 | 0.041 | 0 | 0 |
| V1_02_medium | 0.046 | 0.97 | 0.051 | 0 | 0 |
| V1_03_difficult | 0.055 | 0.80 | 0.054 | 0 | 0 |
| V2_02_medium | 0.045 | 0.73 | 0.045 | 0 | 0 |
| V2_03_difficult | 0.254 | 0.79 | 0.176 | 5 | 0 |

Tracking: no flight lost tracking or reset, and no flight has a gap in the VIO output.
V2_03 (fast, with motion blur) is the weak one: Basalt keeps going but its error is
five times the others', and all five jumps come from it.

| Field | Current default | First fit (RPG examples) | Second fit (Basalt) | Reading |
| --- | --- | --- | --- | --- |
| Bias walk | 0.01 m/sqrt(s) | 0.010 near hover; 0.021 every window | 0.009 every window; 0.006 near hover (short windows only) | Agrees with the default and the first fit: drift when hovering is about 0.01 m/sqrt(s). |
| Bias cap | 0.3 m | 0.41 m | 0.19 m | Every window, all five flights reach 60 s. Basalt drifts less than the RPG example VIOs; V2_03 alone makes most of it. |
| Position noise | 0.02 m | 0.001 m | 0.000 m | Smooth output again; the default stays pessimistic. |
| Velocity noise | 0.05 m/s | not in the data | 0.018 m/s | New: Basalt writes its velocity. The default is about 3 times rougher. |
| Jumps | 0.1 to 2 m at 0.02 Hz, decay 5 s | 0.1 to 0.2 m at 0.002 Hz, decay 9 s | 0.1 to 0.4 m at 0.010 Hz, no decay | All in V2_03. They do not come back: a jump here is a lasting shift, not a blip. |
| Dropouts | 0.2 to 3 s at 0.02 Hz | none seen | none seen | Offline, so no frame is ever dropped for time; kept. |
| Delay | 50 to 150 ms | not measurable | see processing time below | Still not measured as latency; the processing time bounds part of it. |

| Drift since engage, rms per axis (m) | 0.5 s | 1 s | 5 s | 20 s | 60 s |
| --- | --- | --- | --- | --- | --- |
| Basalt, every window | 0.009 | 0.014 | 0.035 | 0.058 | 0.081 |
| Basalt, near-hover windows | 0.005 | 0.007 | 0.023 | (too few) | (too few) |
| Fitted model (every window) | 0.006 | 0.009 | 0.020 | 0.040 | 0.069 |
| Current default | 0.029 | 0.030 | 0.036 | 0.053 | 0.082 |

**Processing time** (`stats_sums` from Basalt, 11,040 frames, one container on an Apple M4
with 8 CPUs given to Docker): the back end (the sliding-window optimiser, from new tracked
features to the state estimate) takes a median of 2.1 ms per frame, 95th percentile 4.4 ms,
99th 6.2 ms, worst 25.8 ms. Per flight the median ranges from 1.3 to 2.5 ms and the 95th
percentile from 3.6 to 4.8 ms. The whole pipeline, front end (optical flow) and image loading
included, ran at about 5 ms per frame on average (12 s for a 115 s flight). So on this
hardware the VIO itself adds only a few milliseconds; the 50 to 150 ms default delay would
have to come from the camera, the transport into PX4 and EKF2's own buffering, not from the
VIO. A small companion computer is several times slower than an M4, so this is a floor.

What the two fits agree on: drift near 0.01 m/sqrt(s), smooth output, and fewer, smaller
jumps than the default. Where they differ is the 60 s cap (0.19 m here, 0.41 m from the RPG
examples, which include a monocular VIO). The defaults are not changed by this page; any
change to `envs/estimate` defaults is its own decision.

## v19 under the fitted model

`eval/configs/gate1_x500_ekf_vio_fit_v19.yaml` flies the Gate 1 x500 test set through the
EKF2-like fusion (same seeds, cells and rules as `gate1_x500_ekf_v19.yaml`) with the fitted
noise, walk, cap and jumps, and the standard delay, velocity noise and dropouts. A re-check,
not a new gate. Run `20261030-174900_gate1_x500_ekf_vio_fit_v19_s9001_6cc20b8`, 3072 paired
episodes per cell, against the cascaded PD (the mock's yardstick, not tuned PX4):

| Cell (wind, hold distance) | v19 p95 | PD p95 | Change | Standard corruption: v19 / PD, change (run `20261024-041240`) |
| --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.206 | 0.169 | +22 % | 0.24 / 0.32, -26 % |
| 0 m/s, 2 m | 0.207 | 0.174 | +19 % | 0.24 / 0.34, -31 % |
| 4 m/s, 1 m | 0.207 | 0.180 | +15 % | 0.24 / 0.32, -27 % |
| 4 m/s, 2 m | 0.209 | 0.202 | +3 % | 0.24 / 0.36, -34 % |
| 8 m/s, 1 m | 0.213 | 0.235 | -9 % | 0.24 / 0.37, -34 % |
| 8 m/s, 2 m | 0.226 | 0.316 | -29 % | 0.26 / 0.44, -42 % |
| Pooled | 0.211 | 0.213 | -1 % | 0.24 / 0.36 |

Crashes: 5 of 18,432 episodes for v19, none for the PD. Verdict under the Gate 1 rules: 1 of 6
cells. The gate itself is unchanged (it is judged on the standard corruption, where v19
passed); this says what happens if the real estimate is as clean as these lab VIOs.

What it shows: on a near-real estimate the PD tightens from 0.36 to 0.21 m pooled, because
most of its error came from chasing the standard model's large jumps and 2 cm noise, which
the real data does not show. v19 barely moves (0.24 to 0.21 m): it sits on a floor of about
0.2 m in every cell, calm or not. So v19's edge over a simple controller is robustness to a
bad estimate and to strong wind; with a good estimate in calm or light wind the PD holds
tighter. Two follow-ups come out of this: find what sets v19's 0.2 m floor (done: it is the settle
after engage, not a standing offset, `docs/hold-offset.md`) (the next policy
should hold tighter when the estimate is good, not only survive when it is bad), and train
on a mix of the fitted and the standard corruption. Gate 2 (v19 against tuned PX4 in Gazebo,
`docs/baseline.md`) is not touched by this: it ran on the standard corruption, and tuned
PX4's own numbers there are far looser than this PD's.

## v22 and v19 under the Basalt fit

The same re-check with the second fit's numbers (noise 0, velocity noise 0.018 m/s, walk
0.009 m/sqrt(s), cap 0.19 m, jumps 0.1 to 0.4 m at 0.01 Hz that stay; delay and dropouts
kept), for v22 (released) with v19 alongside, against the cascaded PD on the same episodes.
The mock has no tuned PX4: that comparison exists only in Gazebo, on the standard corruption
(`docs/baseline.md`: v22 -54 to -55 % p95 against tuned PX4, v19 -47 to -48 %).

Gate 1 x500 test set through the fusion (`eval/configs/gate1_x500_ekf_basalt_fit_v22.yaml`,
run `20261101-020237`, 3072 paired episodes per cell):

| Cell (wind, hold distance) | v22 p95 | v19 p95 | PD p95 | v22 vs PD |
| --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.177 | 0.219 | 0.190 | -7 % |
| 0 m/s, 2 m | 0.178 | 0.220 | 0.195 | -9 % |
| 4 m/s, 1 m | 0.179 | 0.220 | 0.200 | -11 % |
| 4 m/s, 2 m | 0.183 | 0.222 | 0.222 | -18 % |
| 8 m/s, 1 m (headline) | 0.192 | 0.226 | 0.252 | -24 % |
| 8 m/s, 2 m | 0.214 | 0.239 | 0.328 | -35 % |
| Pooled | 0.187 | 0.224 | 0.231 | -19 % |

No crashes for v22 or the PD; v19 crashed in 5 of 18,432 episodes. v22 beats the PD in every cell, with
confidence intervals clear of zero; 5 of 6 cells pass the Gate 1 rules, the headline cell
misses the -30 % bar (-24 %). v19 loses to the PD in calm and light wind, as under the first
fit. So v22 fixed what the first re-check found: the time-since-engage input removed the slow
settle (`docs/hold-offset.md`), and v22 now holds tighter than the PD even when the estimate
is good, not only when it is bad.

Stress set (`eval/configs/gate1_x500_ekf_stress_basalt_fit_v22.yaml`, run `20261101-023637`,
1536 paired episodes per cell, each cell one change from the fitted base):

| Cell | v22 p95 | v19 p95 | PD p95 | v22 vs PD |
| --- | --- | --- | --- | --- |
| nominal | 0.184 | 0.224 | 0.218 | pass |
| mass x0.7 / x1.5 | 0.192 / 0.180 | 0.228 / 0.221 | 0.244 / 0.201 | pass |
| hover thrust error +/-15 % | 0.186 | 0.224 | 0.346 | pass |
| thrust/weight 1.5 | 0.184 | 0.225 | 0.218 | pass |
| inertia x0.5 / x2 | 0.184 / 0.187 | 0.226 / 0.228 | 0.219 / 0.217 | pass |
| motor lag 50 ms (80 ms outside) | 0.185 (0.188) | 0.218 (0.215) | 0.218 (0.217) | pass |
| delay 150-250 ms (250-350 outside) | 0.191 (0.208) | 0.236 (0.255) | 0.222 (0.238) | pass |
| **dropouts 2-3 s, IMU error 0.1** | **0.442** | 0.421 | **0.366** | **fail (+21 %)** |
| (dropouts 3-5 s, outside) | 0.813 | 0.797 | 0.700 | crash 11.5 / 10.7 / 8.4 % |
| wall yaw moment 25 / 50 % (100 % outside) | 0.184 / 0.184 (0.185) | 0.224 | 0.218 | pass |
| combined, moderate | 0.186 | 0.224 | 0.283 | pass |
| Pooled (13 envelope cells) | 0.205 | 0.240 | 0.245 | 12/13 |

Crashes pooled: v22 0.04 %, v19 0.08 %, PD 0.03 %. The one miss is the capped dropout cell.
Under the standard corruption v22 and v19 passed it (v19 0.44 vs PD 0.48 m, run
`20261024-141925`); here the PD tightens to 0.37 m while v22 stays at 0.44 m. The policy's
error while the fix is out is set by the dead-reckoning drift, which the fit does not change,
so it does not gain from the cleaner estimate around the gaps; the PD, whose standard-model
error was mostly chasing jumps and noise, does. Two things to keep in mind: no flight in
either fit lost tracking, so this cell's dropouts are the standard model's, not measured; and
the 3 to 5 s cell stays the known estimator limit where the 5 s watchdog takes over.

Neither run is a new gate: Gate 1 stays judged on the standard corruption, where v22 passed
(13/13 stress). What this adds is that v22's lead over the PD holds on a realistic estimate in
every nominal cell, and that dead-reckoning through a 2 to 3 s dropout is the place where a
simple controller does as well or better once the estimate is clean. That is the input for
the next policy's training mix (fitted plus standard corruption, `docs/decisions.md`).

The stress run took 73 min on the 4 cloud cores, over its 0.9 h estimate and over the 1 h
limit; its config now says 1.25 h, so a re-run needs a go-ahead or a split into two halves.

## What is still open

- **More raw datasets.** The second fit (Basalt on five EuRoC Vicon flights) added
  processing time and VIO velocity. Still missing: the EuRoC machine-hall flights, tracking
  loss on genuinely hard data (UZH-FPV's fast flight, TUM-VI's dark rooms and outdoor parts,
  `rpg.ifi.uzh.ch` and `cdn3.vision.in.tum.de`), and a third VIO (OpenVINS).
- **Hover is under-sampled.** Dataset flights move most of the time; only 10 to 45 % of each
  flight is near hover, and 60 s near-hover windows exist in one flight only.
- **A wall at 1 to 3 m.** No dataset flies a VIO this close to one surface. The first
  partner flight (`docs/decisions.md`, software only) is the check.
