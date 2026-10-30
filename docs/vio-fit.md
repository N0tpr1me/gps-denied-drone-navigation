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

## What is still open

- **An own VIO run on the raw datasets.** With the hosts allowed (`research-collection.ethz.ch`
  for EuRoC, `rpg.ifi.uzh.ch` for UZH-FPV, `cdn3.vision.in.tum.de` for TUM-VI), run an
  open-source VIO (OpenVINS or Basalt) in the cloud. That adds what this fit lacks: latency
  (processing time per frame), VIO velocity, tracking loss on the hard sequences (UZH-FPV's
  fast flight, TUM-VI's dark rooms and outdoor parts), and a second and third VIO.
- **Hover is under-sampled.** Dataset flights move most of the time; only 10 to 45 % of each
  flight is near hover, and 60 s near-hover windows exist in one flight only.
- **A wall at 1 to 3 m.** No dataset flies a VIO this close to one surface. The first
  partner flight (`docs/decisions.md`, software only) is the check.
