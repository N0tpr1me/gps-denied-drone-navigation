# Airframe sizes: bring your own drone

Item 4 of the GPU server plan for client needs: does one policy hold on drones of very
different sizes, after an airframe fit (`docs/airframe-fit.md`) and the drone's own PX4
autotune, without a retrain? In the mock, the released policy v22 already does on a 300 g
quad and on the x500 with up to a 35 % payload: 41 to 53 % tighter than the cascaded PD with
no crashes (cloud CPU, 2026-11-09). It breaks at a 40 % payload, where the x500's hover sits
at the module's 0.9 thrust clamp. The GPU server trains v27 on a wide airframe spread and
flies the payloads in Gazebo against tuned PX4 (`training/queues/airframe_sizes.yaml`).

## The airframes

All of them fly through the same PX4 pieces as the Gazebo benchmark (rate loop and its
filters, the module's 83 Hz tick, the EKF2-like fusion on the barometer), so only the vehicle
differs.

| Airframe | Mass | Thrust / weight | Hover command | Rate loop | Rotor lag | Wind push per kg | Where the numbers come from |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gz_x500_sitl` | 2.06 kg | 1.69 | 0.73 | tuned x500 | 19 ms | 1.0 | x500 SDF, PX4 4001_gz_x500, `px4_tuned_v1.params` (`envs/structure_hold/airframes.py`) |
| x500 + 25 % payload | 2.58 kg | 1.35 | 0.84 | same gains | 19 ms | 0.80 | `airframe_rand` mass x1.25, max thrust unchanged, inertia +7 % |
| x500 + 35 % payload | 2.79 kg | 1.25 | 0.88 | same gains | 19 ms | 0.74 | mass x1.35, inertia +10 % |
| x500 + 50 % payload | 3.10 kg | 1.13 | 0.93 | same gains | 19 ms | 0.67 | mass x1.5, inertia +15 % |
| `small_300g` | 0.285 kg | 2.60 | 0.55 | the x500's loop gain | 12 ms | 1.58 | ModalAI Starling 2 datasheet and forum (below); the rest scaled |

**The 300 g class.** ModalAI's Starling 2 (the VOXL 2 Mini size, a common inspection
development drone) is published at 285 g take-off weight, 230 mm motor to motor and 120 mm
props ([datasheet](https://docs.modalai.com/starling-2-datasheet/)), with about 185 g of
thrust per motor, a thrust/weight of about 2.6, and 2.0 the recommended minimum
([ModalAI forum, topic 5202](https://forum.modalai.com/topic/5202/starling-2-payload)). The
rest is an assumption, stated in `airframes.py`: inertia scaled from the x500 by mass and arm
length squared, rotor lag 8 ms up and 16 ms down, the tracking camera 0.02 m from the IMU,
and drag scaled by frontal area (motor-to-motor diagonal squared), which leaves the wind
pushing 1.6 times as hard per kilogram as on the x500.

**Why the rate loop keeps the x500's gain.** The module sends body rates to the drone's own
PX4 rate controller, and a partner tunes that with PX4's autotune, which aims at the same
closed-loop rise time (`MC_AT_RISE_TIME`) on any airframe. So the preset scales each axis's
rate gains to keep the x500's loop gain (rate P times torque per unit over inertia); a roll
step reaches 63 % in 60 ms on both (`test_small_300g_rate_loop_answers_like_the_x500s`).

**Why thrust needs nothing.** The module maps the policy's thrust through the drone's own
hover thrust (`thrust_n / (MASS_KG g) x hover_thrust`), so the policy asks for thrust as a
share of weight. A policy trained on the x500 flies the others through that mapping
(`eval.gate1.ModuleThrustScale`), capped as the module caps it: at most the x500's 1.69
times weight, and never above a normalised 0.9.

## v22 in the mock (cloud CPU, 2026-11-09)

`eval/configs/airframe_sizes.yaml`: v22 against the cascaded PD on the same episodes, 3 seeds
x 512 episodes per cell, wind 0 to 8 m/s drawn per episode (Gate 1's gust schedule and
estimate), hold 1 to 2 m. Runs `20261109-160318_airframe_sizes_s7101_e3e006e` and
`20261109-161617_airframe_sizes_s7101_b8120f0` (the 35 and 40 % cells; `experiments/LOG.md`).
p95 is the 95th-percentile hold error on the true position.

| Cell | v22 p95 | PD p95 | v22 vs PD | v22 crash | PD crash |
| --- | --- | --- | --- | --- | --- |
| x500 | 0.180 m | 0.320 m | -44 % | 0.0 % | 1.1 % |
| x500 + 25 % payload | 0.178 m | 0.313 m | -43 % | 0.0 % | 1.0 % |
| x500 + 35 % payload | 0.181 m | 0.307 m | -41 % | 0.0 % | 1.2 % |
| x500 + 40 % payload | 0.432 m | 0.320 m | +35 % | 2.7 % | 1.2 % |
| x500 + 50 % payload | 1.56 m | 1.54 m | (both sink) | 100 % | 100 % |
| 300 g class | 0.218 m | 0.379 m | -42 % | 0.0 % | 0.7 % |
| 300 g class, 8 m/s | 0.259 m | 0.481 m | -46 % | 0.0 % | 0.8 % |
| 300 g class, hover thrust error +/-15 % | 0.223 m | 0.470 m | -53 % | 0.0 % | 0.7 % |

What it says:

- **Size is not the problem.** v22 never trained on anything like a 300 g quad, and holds it
  4 cm looser than the x500 (0.218 against 0.180 m), mostly from the harder wind per
  kilogram (0.259 m at 8 m/s); the PD loses 6 cm there.
- **Payload is limited by thrust headroom, not by the policy.** Up to 35 % the hold does not
  move. At 40 % the x500's hover command (0.89) sits just under the module's 0.9 clamp, so
  there is no thrust left to climb out of a gust and v22 sags (the PD, which commands thrust
  differently, holds a little longer). At 50 % the hover needs 0.93: neither can hover and
  both sink. The Gazebo x500 has a low thrust/weight (1.69); a real X500 V2 or a partner's
  drone has more. The rule for the datasheet: **hover thrust at or below 0.88 normalised
  (thrust/weight 1.25 or more)**. That is a property of the 0.9 clamp, not of the policy;
  lifting the clamp is a module change with its own safety review, not part of this work.

## v27: the wide-spread candidate

`training/configs/structure_hold_ppo_v27_airframes.yaml` continues v22 (same 21 inputs, same
reward and estimate, so the module header could take it unchanged) on a much wider spread:
thrust/weight x0.75 to x1.6 (1.27 to 2.7), mass x0.7 to x1.6, inertia x0.6 to x1.6, rotor lag
8 to 40 ms, drag x0.8 to x2.0 (wind push per kilogram 0.5 to 2.9 times the x500's), hover
thrust error up to 20 %. It goes on only if it beats v22 on the 300 g and payload cells
without losing on the x500 (`eval/configs/gazebo_check_v27.yaml`, the usual release check);
since v22 already holds them, the bar is a clear gain, not a pass.

## Payloads in Gazebo

A scenario's `payload` block (`mass_kg`, `offset_down_m`) fixes a 0.10 x 0.10 x 0.05 m box
under the x500's `base_link` on a fixed joint; the SITL image adds it to the x500 model before
PX4 spawns it (`sim2real/docker/add_payload.py`), so Gazebo carries its mass and inertia and
the model keeps its name. PX4's hover thrust estimator learns the new hover in flight, as on
a real drone. `wall1m_calm_0ms_payload{25,35,50}` and `wall1m_wind_8ms_payload{25,35,50}` are
the benchmark scenarios with 0.52, 0.72 and 1.03 kg.

## The server queue

`training/queues/airframe_sizes.yaml`, after `server_week1`'s setup, two GPUs at most, every
job under 1 h, nothing that needs Zain's yes:

| Job | Lane | Hours | What |
| --- | --- | --- | --- |
| `v27_s0`, `v27_s1` | GPU 0, 1 | 0.9 each | train v27, two seeds |
| `sizes_score` | GPU | 0.6 | `airframe_sizes.yaml` with v22, both v27 seeds and the PD |
| `sizes_check_s0`, `_s1` | GPU 0, 1 | 0.75 each | the x500 release check, each v27 against v22 |
| `gz_payload_image` | CPU | 0.2 | rebuild the SITL image with `add_payload.py` |
| `gz_payload_tuned`, `gz_payload_v22` | CPU | 0.5 each | 48 flights each: tuned PX4 and v22 with the three payloads, calm and 8 m/s, 8 seeds |
| `gz_payload_v27` | CPU | 0.6 | off: only if a v27 passes both checks |

About 2.2 h of wall clock and 3.9 GPU-hours. The Gazebo rows compare against the same
scenarios without a payload (`server_week1`'s `gz_tuned_baseline` and `gz_v22_baseline`).

## Limits

- The 300 g class is a model built from published figures and scaling, not a fit; a
  partner's drone replaces it with `eval/fit_airframe.py` on its own autotune log.
- The mock carries a payload as mass, max thrust and a little inertia; it ignores the shift
  of the centre of mass and the extra drag of a hanging load. Gazebo carries the first.
- Nothing here has flown a real drone, and no Gazebo model of the 300 g class exists.
