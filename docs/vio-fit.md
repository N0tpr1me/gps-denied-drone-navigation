# Fitting the estimate model to real VIO error

Every robustness number so far is measured against `envs/estimate`, a VIO model whose
numbers came from papers, not from data (`docs/real-flight-gap.md`, row 1). The plan was one
logging flight with our own drone (`docs/logging-flight.md`). Since the project went
software only (`docs/decisions.md`, 2026-10-30), the real VIO error comes from public
real-drone datasets with ground truth instead. This page is how the fit works, the three
fits so far (EuRoC, then UZH-FPV and TUM-VI with a second VIO), and what it does not cover
yet. In one line: near hover the default drift is right or on the safe side, its noise and
jumps are pessimistic, and drift grows with distance flown (0.3 to 0.5 % at walking pace,
2 to 5 % racing).

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
| (drift per distance) | The 3-D change of `e` over each stretch of 1, 5, 10 and 20 m along the true path, as a % of the stretch. Reported, not a corruptor field: the corruptor's drift is in time. |
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
`eval/configs/vio_fit_basalt_euroc.yaml`. Ten of the eleven flights: all five machine-hall
flights (Leica laser-tracker truth) and five Vicon-room flights; V2_01 never came through the
ETH Research Collection's rate limit. Same fit settings as the first fit (first 10 s
skipped). Run `20261101-035741_vio_fit_basalt_euroc_s0_be13ede` (`experiments/LOG.md`),
1135 s of VIO. (A first pass on the five Vicon flights only, run
`20261101-015732_vio_fit_basalt_euroc_s0_b49b1dc`, gave walk 0.009, cap 0.19 m, velocity
noise 0.018 m/s and jumps at 0.010 Hz; the re-check in the next-but-one section uses those.)

| Flight | ATE (m) | Mean speed (m/s) | Drift at 60 s, rms per axis (m) | Jumps | Gaps |
| --- | --- | --- | --- | --- | --- |
| MH_01_easy | 0.066 | 0.45 | 0.063 | 0 | 0 |
| MH_02_easy | 0.069 | 0.50 | 0.064 | 0 | 0 |
| MH_03_medium | 0.062 | 1.05 | 0.061 | 0 | 0 |
| MH_04_difficult | 0.115 | 1.00 | 0.128 | 1 | 0 |
| MH_05_difficult | 0.143 | 0.94 | 0.155 | 0 | 0 |
| V1_01_easy | 0.045 | 0.43 | 0.041 | 0 | 0 |
| V1_02_medium | 0.046 | 0.97 | 0.051 | 0 | 0 |
| V1_03_difficult | 0.055 | 0.80 | 0.054 | 0 | 0 |
| V2_02_medium | 0.045 | 0.73 | 0.045 | 0 | 0 |
| V2_03_difficult | 0.254 | 0.79 | 0.176 | 5 | 0 |

Tracking: no flight lost tracking or reset, and no flight has a gap in the VIO output. The
hard flights cost accuracy, not tracking: MH_04 and MH_05 (fast, dark parts of the hall) are
2 to 3 times the easy flights' error, and V2_03 (fast, with motion blur) five times; five of
the six jumps come from V2_03.

| Field | Current default | First fit (RPG examples) | Second fit (Basalt) | Reading |
| --- | --- | --- | --- | --- |
| Bias walk | 0.01 m/sqrt(s) | 0.010 near hover; 0.021 every window | 0.009 every window; 0.004 near hover (2 flights reach 20 s) | Agrees with the default and the first fit; a hovering Basalt drifts less still. |
| Bias cap | 0.3 m | 0.41 m | 0.19 m | Every window, all ten flights reach 60 s. Basalt drifts less than the RPG example VIOs; the three hard flights make most of it. |
| Position noise | 0.02 m | 0.001 m | 0.000 m | Smooth output again; the default stays pessimistic. |
| Velocity noise | 0.05 m/s | not in the data | 0.017 m/s | New: Basalt writes its velocity. The default is about 3 times rougher. |
| Jumps | 0.1 to 2 m at 0.02 Hz, decay 5 s | 0.1 to 0.2 m at 0.002 Hz, decay 9 s | 0.1 to 0.4 m at 0.005 Hz, no decay | Six in 1135 s, five in V2_03. They do not come back: a jump here is a lasting shift, not a blip. |
| Dropouts | 0.2 to 3 s at 0.02 Hz | none seen | none seen | Offline, so no frame is ever dropped for time; kept. |
| Delay | 50 to 150 ms | not measurable | see processing time below | Still not measured as latency; the processing time bounds part of it. |

| Drift since engage, rms per axis (m) | 0.5 s | 1 s | 5 s | 20 s | 60 s |
| --- | --- | --- | --- | --- | --- |
| Basalt, every window | 0.008 | 0.013 | 0.034 | 0.064 | 0.085 |
| Basalt, near-hover windows | 0.004 | 0.005 | 0.010 | 0.015 | (one flight) |
| Fitted model (every window) | 0.006 | 0.009 | 0.020 | 0.039 | 0.068 |
| Current default | 0.029 | 0.030 | 0.036 | 0.053 | 0.082 |

**Processing time** (`stats_sums` from Basalt, 24,767 frames, one container on an Apple M4
with 8 CPUs given to Docker): the back end (the sliding-window optimiser, from new tracked
features to the state estimate) takes a median of 2.6 ms per frame, 95th percentile 6.1 ms,
99th 8.6 ms, worst 25.8 ms. Per flight the median ranges from 1.3 to 3.5 ms and the 95th
percentile from 3.6 to 7.8 ms (the machine hall, with more features, is the slower end). The
whole pipeline, front end (optical flow) and image loading included, ran at 4.4 to 6.4 ms per
frame on average (10 to 23 s per flight). So on this hardware the VIO itself adds only a few
milliseconds; the 50 to 150 ms default delay would have to come from the camera, the
transport into PX4 and EKF2's own buffering, not from the VIO. A small companion computer is
several times slower than an M4, so this is a floor.

What the two fits agree on: drift near 0.01 m/sqrt(s), smooth output, and fewer, smaller
jumps than the default. Where they differ is the 60 s cap (0.19 m here, 0.41 m from the RPG
examples, which include a monocular VIO). Adding the machine hall moved nothing but the jump
rate (0.010 to 0.005 Hz) and the velocity noise (0.018 to 0.017 m/s). The defaults are not
changed by this page; any change to `envs/estimate` defaults is its own decision.

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

## Third fit: Basalt and OpenVINS on UZH-FPV and TUM-VI (2026-11-26)

Two more datasets and a second VIO, all run in the cloud container (4 cores, no GPU) once
the dataset hosts were allowed in its network settings:

- **UZH-FPV** (University of Zurich): a racing quadrotor flown by a pilot through a hangar
  and a field, 16 flights with Leica total-station truth (the IMU's pose), Snapdragon Flight
  stereo fisheye 640x480 at 30 Hz, IMU 500 Hz. Truth covers 18 to 86 s of each flight, 618 s
  in all, at 2.8 to 8 m/s (median 5.1 m/s). Laid out for the VIOs by
  `eval/vio_basalt/prepare_uzhfpv.py` (Kalibr calibration to Basalt's, image stamps moved
  onto the IMU clock by Kalibr's time shift); `eval/vio_basalt/run_uzhfpv.sh`.
- **TUM-VI** (TU Munich): a handheld stereo-inertial rig carried through a motion-capture
  room, the six room sequences (the only ones with truth over the whole run), stereo
  512x512 at 20 Hz, IMU 200 Hz, 723 to 750 s in all at 0.5 to 1.1 m/s;
  `eval/vio_basalt/run_tumvi.sh`, Basalt's own TUM-VI calibration and config.
- **OpenVINS** (University of Delaware, an MSCKF filter, v2.7) beside Basalt (an optimiser):
  built without ROS in Docker (`eval/vio_openvins/`), fed by our own runner
  (`run_euroc.cpp`) the way its ROS serial runner reads a bag, with OpenVINS's own config
  for each dataset. GPL-3.0, evaluation only: nothing from it ships.

Both VIOs ran with the same IMU noise on every dataset: Basalt's (the numbers it ships for
EuRoC and TUM-VI; OpenVINS's UZH-FPV config uses the same EuRoC values). UZH-FPV's own Kalibr
file says 0.1 and 0.05, far above a MEMS datasheet; with those Basalt reached 0.69 m ATE on
indoor_forward_3 against 0.20 m. OpenVINS needed one setting its TUM-VI config itself notes:
room6 does not start with its room1-5 initialisation threshold (0.45, no pose in 2636
frames), and does with 0.25.

EuRoC was not run again: its host now answers every request with "too many requests"
(`www.research-collection.ethz.ch`, 429) and the old `robotics.ethz.ch` mirror is gone, so
the second fit's EuRoC numbers stand and OpenVINS has no EuRoC run.

Runs `20261126-204558_vio_fit_basalt_uzhfpv`, `-204601_vio_fit_openvins_uzhfpv`,
`-204601_vio_fit_basalt_tumvi`, `-204602_vio_fit_openvins_tumvi` (`experiments/LOG.md`;
configs `eval/configs/vio_fit_{basalt,openvins}_{uzhfpv,tumvi}.yaml`):

| | EuRoC, Basalt (second fit) | TUM-VI, Basalt | TUM-VI, OpenVINS | UZH-FPV, Basalt | UZH-FPV, OpenVINS |
| --- | --- | --- | --- | --- | --- |
| Flights, VIO seconds | 10, 1135 | 6, 750 | 6, 723 | 16, 618 | 16, 618 |
| Speed (median of flights) | 0.4 to 1.1 m/s | 1.0 m/s | 1.0 m/s | 5.1 m/s | 5.1 m/s |
| ATE, median (range) | 0.06 (0.05 to 0.25) m | 0.12 (0.05 to 0.19) m | 0.09 (0.02 to 0.15) m | 0.66 (0.20 to 1.78) m | 0.30 (0.13 to 1.04) m |
| Lost tracking, gaps, resets | none | none | none (after the room6 setting) | none | none |
| Drift rms per axis at 1 / 5 / 20 / 60 s, every window | 0.013 / 0.034 / 0.064 / 0.085 m | 0.047 / 0.079 / 0.087 / 0.115 m | 0.042 / 0.061 / 0.066 / 0.089 m | 0.21 / 0.61 / 0.55 / 1.00 m | 0.14 / 0.35 / 0.44 / 1.09 m |
| Drift at 1 / 5 s, near-hover windows | 0.005 / 0.010 m | 0.008 / 0.017 m | 0.006 / 0.018 m | 0.014 / too few | 0.015 / too few |
| Drift per distance flown, median (p95), 10 m | not measured | 0.49 (2.1) % | 0.35 (1.6) % | 4.8 (12.7) % | 2.0 (6.7) % |
| Walk fitted (every window) | 0.009 | 0.021 | 0.018 | 0.057 | 0.029 m/sqrt(s) |
| Cap (95th percentile of 60 s drift) | 0.19 m | 0.27 m | 0.16 m | 1.63 m | 2.08 m |
| Position noise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 m |
| Velocity noise | 0.017 | 0.059 | 0.054 | 0.24 | 0.15 m/s |
| Jumps | 6, 0.1 to 0.4 m, stay | none | none | 8, 0.10 to 0.19 m | 5, 0.10 to 0.21 m |

The current defaults for comparison: noise 0.02 m, velocity noise 0.05 m/s, walk 0.01
m/sqrt(s) (drift 0.030 / 0.036 / 0.053 / 0.082 m at 1 / 5 / 20 / 60 s), cap 0.3 m, jumps 0.1
to 2 m at 0.02 Hz decaying in 5 s, dropouts 0.2 to 3 s at 0.02 Hz, delay 50 to 150 ms.

"Drift per distance flown" is new in `eval/vio_fit.py`: the 3-D change of the error over each
stretch of 1, 5, 10 and 20 m along the true path, as a share of the stretch (the relative
error VIO papers quote, here in the fixed frame the hold sees, so heading error counts).

**What the three datasets say together.**

1. **Drift goes with distance flown, not with time.** At walking pace (TUM-VI) both VIOs drift
   0.3 to 0.5 % of the distance (median, 10 m stretches); racing at 5 m/s with motion blur
   (UZH-FPV) 2 to 5 %. A VIO that barely moves drifts least: in near-hover windows TUM-VI's
   drift at 5 s is 0.017 to 0.018 m, half the default's 0.036 m, and EuRoC's 0.010 m. So for
   a hold the default walk (0.01 m/sqrt(s)) stays right or on the safe side on all three
   datasets; the every-window walks of 0.02 (walking) to 0.03 to 0.06 (racing) apply to a
   VIO that is carried somewhere, which a hold is not.
2. **The 0.3 m cap holds for a slow VIO, not a fast one.** Over 60 s the 95th percentile is
   0.16 to 0.27 m at walking pace (0.19 m on EuRoC), 1.6 to 2.1 m racing. A hold never flies
   60 s at 5 m/s, so the default stands for Structure Hold.
3. **Noise is pessimistic, velocity noise about right.** No VIO in any dataset has white
   position noise; the default's 2 cm is a safety margin. Velocity noise is 0.05 to 0.06 m/s
   at walking pace (the default's 0.05) and 0.02 m/s on EuRoC's slower flights.
4. **Jumps are small and rare; tracking was never lost.** 0 to 0.013 per second, every one
   0.1 to 0.4 m, against the default's up to 2 m at 0.02 Hz. On UZH-FPV the "jumps" are
   single-frame steps of 0.1 to 0.2 m at 5 to 8 m/s, inside error that moves more than that
   within the next seconds, not resets. None of the 44 runs (22 flights, two VIOs each)
   lost tracking or left a gap, even racing outdoors; the default's dropouts and large jumps stand for the field faults
   (`docs/field-faults.md`) and dust (`docs/vio-render.md`) that no public dataset shows.
5. **The two VIOs agree at walking pace, not racing.** On TUM-VI the walks are 0.021 and
   0.018 and the drift within about 30 % at every horizon (the 60 s caps differ more, 0.27
   and 0.16 m); on UZH-FPV OpenVINS (with the configs its authors ship for these flights)
   drifts half as much as Basalt (with its EuRoC config). The racing numbers are a VIO
   tuning question as much as a dataset one.
6. **Processing time** (OpenVINS on the cloud's 4 cores, sharing them with downloads and
   other runs): a median 10 ms per stereo frame on UZH-FPV and 13 ms on TUM-VI, 95th
   percentile 21 ms, worst 181 ms. With Basalt's 2.6 ms median on an M4 (second fit), the VIO
   is still a small part of the default's 50 to 150 ms delay.

**For Route Follow** (`docs/route-follow.md`): its estimate's drift walk was raised to
0.02 m/sqrt(s) with a 0.5 m cap "until a tunnel VIO fit". TUM-VI is the closest thing yet: a
VIO carried at 0.5 to 1.1 m/s, Route Follow's cruise of 0.5 to 1.5 m/s. Both VIOs fit 0.018
to 0.021 over every window, and a 60 s cap of 0.16 to 0.27 m, so the guess matches the walk
and is on the safe side on the cap. A tunnel (dark, repetitive) is still unmeasured.

**v22 under the TUM-VI fit.** The Gate 1 x500 test set through the fusion, as in the two
re-checks above, with the estimate fitted to TUM-VI over every window (walk 0.02, cap 0.27 m,
noise 0, velocity noise 0.057 m/s; no jumps in TUM-VI, so the EuRoC fit's 0.1 to 0.4 m at
0.01 Hz; delay and dropouts kept): twice the walk of the EuRoC fit, a VIO that is being
carried about rather than hovering. `eval/configs/gate1_x500_ekf_tumvi_fit_v22.yaml`, run
`20261126-204602`, 3072 paired episodes per cell, against the cascaded PD (the mock's
yardstick, not tuned PX4):

| Cell (wind, hold distance) | v22 p95 | PD p95 | v22 vs PD | Under the Basalt EuRoC fit (v22 / PD) |
| --- | --- | --- | --- | --- |
| 0 m/s, 1 m | 0.207 | 0.222 | -7 % | 0.177 / 0.190 |
| 0 m/s, 2 m | 0.208 | 0.228 | -9 % | 0.178 / 0.195 |
| 4 m/s, 1 m | 0.209 | 0.232 | -10 % | 0.179 / 0.200 |
| 4 m/s, 2 m | 0.212 | 0.251 | -15 % | 0.183 / 0.222 |
| 8 m/s, 1 m (headline) | 0.219 | 0.273 | -20 % | 0.192 / 0.252 |
| 8 m/s, 2 m | 0.237 | 0.340 | -30 % | 0.214 / 0.328 |
| Pooled | 0.22 | 0.26 | | 0.187 / 0.231 |

No crashes for either. v22 holds tighter than the PD in every cell with confidence intervals
clear of zero; 5 of 6 cells pass the Gate 1 rules and the headline misses the -30 % bar at
-20 % (-24 % under the EuRoC fit). Both controllers loosen by about 0.03 m against the
EuRoC fit, which is the doubled walk, and the order between them does not change. Not a
gate: Gate 1 stays judged on the standard corruption.

The defaults in `envs/estimate` are not changed by this fit; a change is its own decision.

## What is still open

- **More raw datasets.** The third fit added UZH-FPV, TUM-VI's rooms and OpenVINS. Still
  missing: EuRoC V2_01 and OpenVINS on EuRoC (the ETH host refuses with 429), TUM-VI's
  corridor, hall, outdoor and slide sequences (truth only at start and end: they would give
  end-to-end drift over hundreds of metres, not windows), and any tracking loss: no run on
  the three datasets lost it.
- **Hover is under-sampled.** Dataset flights move most of the time; only 10 to 45 % of each
  flight is near hover, and 60 s near-hover windows exist in one flight only.
- **Dust, fog and low light.** No dataset has them; `docs/vio-render.md` renders hold flights
  in each and runs Basalt on them.
- **A wall at 1 to 3 m.** No dataset flies a VIO this close to one surface. The first
  partner flight (`docs/decisions.md`, software only) is the check.
