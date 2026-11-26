# Airframe sizes: bring your own drone

Item 4 of the GPU server plan for client needs: does one policy hold on drones of very
different sizes, after an airframe fit (`docs/airframe-fit.md`) and the drone's own PX4
autotune, without a retrain? In the mock (cloud CPU, 2026-11-10, against the cascaded PD) the
released v22 and v24 hold a 300 g quad, the x500 with up to a 35 % payload, a 3 kg quad up to
its 6.3 kg take-off limit and a 16 kg heavy lifter up to 35 kg, with next to no crashes.
Weight is not what limits it. Two things are:

- **Thrust headroom.** The module clamps thrust at 0.9 normalised, so the drone must hover at
  or below 0.88 (thrust/weight 1.25 or more). The x500 breaks at a 40 % payload.
- **How fast the drone answers.** v22 holds while the rotors answer within about 40 ms (a
  roll-rate step reaching 63 % in 0.12 s), v24 to about 60 ms (0.17 s). At 120 ms (0.31 s),
  which is what a 16 kg drone on 33 inch props is assumed to have here, nothing holds, the
  PD included. That drone needs a policy trained on slow rotors: v27 heavy, below.

A 25 % payload hover flown in Gazebo in the cloud confirmed the payload model (thrust 0.835
against 0.836 predicted). The GPU server trains v27 on a wide airframe spread and v27 heavy on
slow rotors, and the payloads fly in Gazebo against tuned PX4
(`training/queues/airframe_sizes.yaml`).

## The airframes

All of them fly through the same PX4 pieces as the Gazebo benchmark (rate loop and its
filters, the module's 83 Hz tick, the EKF2-like fusion on the barometer), so only the vehicle
differs.

| Airframe | Mass | Thrust / weight | Hover command | Rate loop | Rotor lag | Wind push per kg | Where the numbers come from |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gz_x500_sitl` | 2.06 kg | 1.69 | 0.73 | tuned x500 | 19 ms | 1.0 | x500 SDF, PX4 4001_gz_x500, `px4_tuned_v1.params` (`envs/structure_hold/airframes.py`) |
| x500 + 25 % payload | 2.58 kg | 1.35 | 0.84 | same gains | 19 ms | 0.80 | `airframe_rand` mass x1.25, max thrust unchanged, inertia +7 %, hover estimate -4 % |
| x500 + 35 % payload | 2.79 kg | 1.25 | 0.88 | same gains | 19 ms | 0.74 | mass x1.35, inertia +10 %, hover estimate -9 % |
| x500 + 50 % payload | 3.10 kg | 1.13 | 0.93 | same gains | 19 ms | 0.67 | mass x1.5, inertia +15 %, hover estimate -14 % |
| `small_300g` | 0.285 kg | 2.60 | 0.55 | the x500's loop gain | 12 ms | 1.58 | ModalAI Starling 2 datasheet and forum (below); the rest scaled |
| `mid_3kg` | 3.0 kg | 3.00 | 0.50 | x500's loop gain x 0.5 | 38 ms | 1.20 | Holybro X650 kit page (below); the rest scaled |
| `mid_3kg` at 6.3 kg | 6.3 kg | 1.43 | 0.81 | same gains | 38 ms | 0.57 | mass x2.1, max thrust unchanged, inertia +30 %, hover estimate -1 % |
| `heavy_16kg` | 16.0 kg | 4.29 | 0.39 | x500's loop gain x 0.16 | 120 ms (assumed) | 1.07 | Freefly Alta X specs and performance pages (below); the rest scaled |
| Alta X at 35.5 kg | 35.5 kg | 1.93 | 0.67 | same gains | as scored | 0.48 | mass x2.22, max thrust unchanged, inertia +30 % |

**The 300 g class.** ModalAI's Starling 2 (the VOXL 2 Mini size, a common inspection
development drone) is published at 285 g take-off weight, 230 mm motor to motor and 120 mm
props ([datasheet](https://docs.modalai.com/starling-2-datasheet/)), with about 185 g of
thrust per motor, a thrust/weight of about 2.6, and 2.0 the recommended minimum
([ModalAI forum, topic 5202](https://forum.modalai.com/topic/5202/starling-2-payload)). The
rest is an assumption, stated in `airframes.py`: inertia scaled from the x500 by mass and arm
length squared, rotor lag 8 ms up and 16 ms down, the tracking camera 0.02 m from the IMU,
and drag scaled by frontal area (motor-to-motor diagonal squared), which leaves the wind
pushing 1.6 times as hard per kilogram as on the x500.

**The 3 kg class.** Holybro's X650 development kit (a common PX4 development frame) is
published at 650 mm wheelbase, 15 inch props on MN4014 KV330 motors, 2.0 kg without a
battery, a 6S battery of 0.75 to 1.2 kg, and a take-off limit of 6.3 kg "at 70 % throttle"
([X650 overview](https://docs.holybro.com/drone-development-kit/x650-development-kit/overview)).
The preset takes a 1.0 kg battery (3.0 kg) and reads the limit as 70 % of full thrust: 9.0 kg
of thrust, thrust/weight 3.0. Assumed: rotor lag 25 ms up and 50 ms down, the camera 0.08 m
below the IMU, inertia and drag scaled as for the 300 g class.

**The 16 kg class.** No 16 kg drone with a full public spec sheet was found, so the preset is
the Freefly Alta X's airframe at 16 kg: 1415 mm unfolded diameter without props, 840 mm (33
inch) props
([technical specs](https://docs.freeflysystems.com/products/products/alta-x/untitled-3/technical-specs)),
thrust/weight 3.5 at its 19.63 kg no-payload weight
([performance specs](https://docs.freeflysystems.com/products/products/alta-x/untitled-3/performance-specs)),
which makes 68.7 kg of thrust and thrust/weight 4.3 at 16 kg. The Alta X itself weighs
10.86 kg empty, 19.63 kg with its standard battery pair and up to 35.5 kg loaded
(thrust/weight 1.9); the 35.5 kg cell scores that. Freefly publishes no rotor lag. The preset
assumes 80 ms up and 160 ms down for the 33 inch props, and since the hold turns on that
number the 16 kg drone is scored at 40, 60 and 120 ms.

**Slow rotors slow the rate loop.** A rate loop cannot outrun its motors, and PX4's autotune
fits the lag it identifies. So where the rotor lag is above the x500's 19 ms, each axis's loop
gain is the x500's times 19 ms over the lag. A roll-rate step reaches 63 % in 0.06 s on the
x500 and the 300 g class, 0.11 s on the 3 kg class, and 0.12, 0.17 and 0.31 s on the 16 kg
drone at 40, 60 and 120 ms (`test_slow_rotors_slow_the_rate_loop_in_proportion`).
`airframe_rand.rate_loop_follows_lag` applies the same rule to a rotor lag drawn per episode.

**Why the rate loop keeps the x500's gain.** The module sends body rates to the drone's own
PX4 rate controller, and a partner tunes that with PX4's autotune, which aims at the same
closed-loop rise time (`MC_AT_RISE_TIME`) on any airframe. So the preset scales each axis's
rate gains to keep the x500's loop gain (rate P times torque per unit over inertia); a roll
step reaches 63 % in 60 ms on both (`test_small_300g_rate_loop_answers_like_the_x500s`).

**PX4's hover estimate stops at 0.8.** The 25 % payload hover in Gazebo (run
`20261109-164535_sitl_wind_wall1m_calm_0ms_payload25_cd2d28b`, stock PX4, calm) commanded a
median thrust of 0.835, against 0.836 predicted and about 0.73 without a payload, but PX4
v1.17's hover thrust estimate stayed at 0.80, the top of `MPC_THR_HOVER`'s range. The module
maps the policy's thrust through that estimate, so on a heavy x500 its hover reference reads
low: by 4, 9 and 14 % at 25, 35 and 50 %. The payload cells carry that as a hover-thrust error.

**Why thrust needs nothing.** The module maps the policy's thrust through the drone's own
hover thrust (`thrust_n / (MASS_KG g) x hover_thrust`), so the policy asks for thrust as a
share of weight. A policy trained on the x500 flies the others through that mapping
(`eval.gate1.ModuleThrustScale`), capped as the module caps it: at most the x500's 1.69
times weight, and never above a normalised 0.9.

## v22 and v24 in the mock (cloud CPU)

Both configs score v22 and v24 against the cascaded PD on the same episodes, 3 seeds x 512
episodes per cell, wind 0 to 8 m/s drawn per episode (Gate 1's gust schedule and estimate),
the forward rangefinder on (v24 reads it), hold 1 to 2 m (2 to 3 m on the 16 kg drone). p95 is
the 95th-percentile hold error on the true position; the verdict is judged on v22, the
released policy when the configs were written.

**The x500, payloads and the 300 g class** (`eval/configs/airframe_sizes.yaml`, run
`20261126-200150_airframe_sizes_s7101_552295e`; v22's numbers match the 2026-11-09 runs
`20261109-160318` and `20261109-164704` to the millimetre):

| Cell | v22 p95 | v24 p95 | PD p95 | v22 vs PD | v24 vs PD | Crash v22 / v24 / PD |
| --- | --- | --- | --- | --- | --- | --- |
| x500 | 0.180 m | 0.155 m | 0.320 m | -44 % | -52 % | 0.0 / 0.0 / 1.1 % |
| x500 + 25 % payload | 0.177 m | 0.154 m | 0.352 m | -50 % | -56 % | 0.0 / 0.0 / 1.1 % |
| x500 + 35 % payload | 0.187 m | 0.160 m | 0.461 m | -60 % | -65 % | 0.0 / 0.0 / 1.1 % |
| x500 + 40 % payload | 0.662 m | 0.543 m | 0.854 m | -23 % | -36 % | 5.8 / 4.5 / 2.8 % |
| x500 + 50 % payload | 1.58 m | 1.56 m | 1.57 m | (all sink) | (all sink) | 100 / 100 / 100 % |
| 300 g class | 0.218 m | 0.188 m | 0.379 m | -42 % | -50 % | 0.0 / 0.0 / 0.7 % |
| 300 g class, 8 m/s | 0.259 m | 0.233 m | 0.481 m | -46 % | -52 % | 0.0 / 0.0 / 0.8 % |
| 300 g class, hover thrust error +/-15 % | 0.223 m | 0.189 m | 0.470 m | -53 % | -60 % | 0.0 / 0.0 / 0.7 % |

**The 3 kg and 16 kg classes** (`eval/configs/airframe_sizes_big.yaml`, run
`20261110-123321_airframe_sizes_big_s7101_cd43264`):

| Cell | v22 p95 | v24 p95 | PD p95 | v22 vs PD | v24 vs PD | Crash v22 / v24 / PD |
| --- | --- | --- | --- | --- | --- | --- |
| 3 kg class | 0.216 m | 0.192 m | 0.331 m | -35 % | -42 % | 0.0 / 0.0 / 2.0 % |
| 3 kg class at its 6.3 kg limit | 0.220 m | 0.184 m | 0.306 m | -28 % | -40 % | 0.1 / 0.0 / 2.2 % |
| 16 kg, rotor lag 40 ms | 0.264 m | 0.244 m | 0.363 m | -27 % | -33 % | 0.1 / 0.0 / 0.6 % |
| Alta X at 35.5 kg, rotor lag 40 ms | 0.253 m | 0.227 m | 0.323 m | -22 % | -30 % | 0.1 / 0.3 / 0.6 % |
| 16 kg, rotor lag 60 ms | 0.329 m | 0.263 m | 0.355 m | -7 % | -26 % | 3.0 / 0.3 / 0.6 % |
| 16 kg, rotor lag 120 ms (the preset) | 1.69 m | 1.86 m | 1.52 m | (all crash) | (all crash) | 100 / 100 / 100 % |

What it says:

- **Size is not the problem.** Neither policy trained on anything but the x500, and both hold
  the 300 g quad, the 3 kg quad and the 16 kg drone with next to no crashes, 22 to 53 %
  (v22) and 30 to 60 % (v24) tighter than the PD. Each is a few centimetres looser than on
  the x500, mostly from the wind: harder per kilogram on the 300 g quad (0.259 m for v22 at
  8 m/s), and on the bigger drones from the slower rate loop their slower rotors allow.
- **Payload is limited by thrust headroom, not by the policy.** Up to 35 % the hold does not
  move (v22 0.18 to 0.19 m, v24 0.15 to 0.16 m), while the low hover estimate costs the PD 3
  to 14 cm. At 40 % the x500's hover command (0.89) sits just under the module's 0.9 clamp,
  so there is no thrust left to climb out of a gust: all three sag and crash more. At 50 %
  the hover needs 0.93 and nothing can hover. The 3 kg class at 6.3 kg hovers at 0.81 and
  holds. The rule for the datasheet: **hover thrust at or below 0.88 normalised
  (thrust/weight 1.25 or more)**. That is a property of the 0.9 clamp, not of the policy;
  lifting the clamp is a module change with its own safety review, not part of this work.
- **Slow rotors are the real limit.** The 16 kg drone holds as well as the 3 kg one when its
  rotors answer within 40 ms, even loaded to the Alta X's 35.5 kg. At 60 ms v22 starts to
  crash (3.0 %, more than the PD) while v24 still holds (0.3 %). At 120 ms nothing holds
  with the rate loop slowed to match, the PD included; a softer PD would, so the drone itself
  can be flown, but neither released policy can fly it. Whether a real 16 kg drone answers in
  40 or in 120 ms is the open question: Freefly publishes no rotor lag, and one autotune log
  (`docs/airframe-fit.md`) would measure it. v27 heavy is the training answer.

## v27 and v27 heavy: the training candidates

`training/configs/structure_hold_ppo_v27_airframes.yaml` continues v22 (same 21 inputs, same
reward and estimate, so the module header could take it unchanged) on a much wider spread:
thrust/weight x0.75 to x1.8 (1.27 to 3.0), mass x0.7 to x1.6, inertia x0.6 to x1.6, rotor lag
8 to 40 ms, drag x0.8 to x2.0 (wind push per kilogram 0.5 to 2.9 times the x500's), hover
thrust error up to 20 %. That covers the 300 g and 3 kg classes and the payloads. Since v22
already holds them, its bar is a clear gain there without losing on the x500
(`eval/configs/gazebo_check_v27.yaml`, the usual release check), not a pass.

`training/configs/structure_hold_ppo_v27_heavy.yaml` is the same recipe with slow rotors: lag
8 to 150 ms, each episode's rate loop as slow as a tune on those rotors would be
(`rate_loop_follows_lag`), thrust/weight up to x2.6 (4.4). It is the candidate for the 16 kg
drone at a 60 to 120 ms lag, where v22 and v24 lose it. Its bar: hold those two cells with
under 1 % crashes. Its release check tells what the slow rotors cost the x500; if they cost
too much, a heavy lifter gets its own policy rather than the shared one.

## Payloads in Gazebo

A scenario's `payload` block (`mass_kg`, `offset_down_m`) fixes a 0.10 x 0.10 x 0.05 m box
under the x500's `base_link` on a fixed joint; the SITL image adds it to the x500 model before
PX4 spawns it (`sim2real/docker/add_payload.py`), so Gazebo carries its mass and inertia and
the model keeps its name. PX4's hover thrust estimator learns the new hover in flight, as on
a real drone. `wall1m_calm_0ms_payload{25,35,50}` and `wall1m_wind_8ms_payload{25,35,50}` are
the benchmark scenarios with 0.52, 0.72 and 1.03 kg.

The 300 g, 3 kg and 16 kg drones have no Gazebo model: the SITL image flies PX4's x500
(airframe 4001) only. Each needs its own model, PX4 airframe file and tuned-PX4 baseline
(an autotune in simulation) before a benchmark flight means anything; that is a job of its
own, not started.

## The server queue

`training/queues/airframe_sizes.yaml`, after `server_week1`'s setup, two GPUs at most, every
job under 1 h, nothing that needs Zain's yes:

| Job | Lane | Hours | What |
| --- | --- | --- | --- |
| `v27_s0`, `v27_s1` | GPU 0, 1 | 0.9 each | train v27, two seeds |
| `v27h_s0`, `v27h_s1` | GPU 0, 1 | 0.9 each | train v27 heavy, two seeds |
| `sizes_score` | GPU | 0.9 | `airframe_sizes.yaml`: v22, v24, the four candidates and the PD on the x500, its payloads and the 300 g class |
| `sizes_score_big` | GPU | 0.75 | `airframe_sizes_big.yaml`: the same on the 3 kg and 16 kg drones |
| `sizes_check_s0`, `_s1`, `_h0`, `_h1` | GPU 0, 1 | 0.75 each | the x500 release check, each candidate against v22 |
| `gz_payload_image` | CPU | 0.2 | rebuild the SITL image with `add_payload.py` |
| `gz_payload_tuned`, `gz_payload_v22` | CPU | 0.5 each | 48 flights each: tuned PX4 and v22 with the three payloads, calm and 8 m/s, 8 seeds |
| `gz_payload_v27` | CPU | 0.6 | off: only if a v27 passes both checks |

About 4.2 h of wall clock on two GPUs and 8.2 GPU-hours (about 8 h on one). The Gazebo rows
compare against the same scenarios without a payload (`server_week1`'s `gz_tuned_baseline`
and `gz_v22_baseline`); Gazebo batches fly in the cloud, not on the server (`docs/gpu-server.md`).

## Limits

- The 300 g, 3 kg and 16 kg classes are models built from published figures and scaling, not
  fits; a partner's drone replaces them with `eval/fit_airframe.py` on its own autotune log.
- The rotor lag of the 3 kg and 16 kg drones is assumed, and the rule that slows the rate
  loop with it is a rule of thumb, not an autotune. The 16 kg result hangs on both: a real
  heavy lifter's own autotune log settles which of the three scored cells it is.
- The cascaded PD keeps its x500 hand gains on every airframe, so on slow rotors it is a
  loose yardstick. The 120 ms drone can be held: the same PD with gains about half as stiff
  (position 1.0, velocity 1.6, tilt 1.5) held it at 0.49 m p95 with no crash in a quick
  64-episode look (not a logged run). The released policies are simply too quick for it.
- The mock carries a payload as mass, max thrust and a little inertia; it ignores the shift
  of the centre of mass and the extra drag of a hanging load. Gazebo carries the first.
- The 16 kg drone is 2.3 m across its props and holds at 2 to 3 m; the others at 1 to 2 m.
- Nothing here has flown a real drone, and only the x500 has a Gazebo model.
