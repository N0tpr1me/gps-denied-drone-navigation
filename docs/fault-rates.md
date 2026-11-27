# Failure rates per fault, with confidence bounds

The failure table in `docs/safety-manual.md` (section 5) says what happens on each estimate
fault, from Gazebo flights on 1 to 3 seeds (`docs/field-faults.md`). That says what, not how
often. A buyer's safety reviewer asks for the sample size behind a row, and TC 922 and ASTM
F3201 accept simulation as evidence. This page is item 2 of the week-2 plan Zain approved on
2026-11-09 (`docs/gpu-server-plan.md`, "Week 2"): every scripted fault flown tens of thousands
of times in the mock, with a 95 % bound on the crash rate, and the Gazebo field faults taken
from 3 seeds to 8.

## What is measured

`eval/fault_rates.py` with `eval/configs/fault_rates_v22.yaml`: 4096 drones x 5 seeds =
20,480 episodes per fault, 40 s each on the `gz_x500_sitl` airframe at 4 m/s into the wall
(the evaluation kit's scenario), the fault landing at 10 s in every episode. The released v22
and the cascaded PD yardstick fly the same episodes. A crash is a contact with the wall or the
ground, a tilt past the limit or leaving the box (`envs/structure_hold/env.py`); the bound is
the Wilson 95 % upper bound (`eval.gate1.wilson`), written as the rarest rate the sample can
rule out: with no crash in 20,480 episodes, "fewer than 1 in 5,000".

| Fault | FMEA row | Scripted as |
| --- | --- | --- |
| nominal | F8 | vio_default as flown, random jumps and dropouts on; the reference row |
| reset | F1 | the reported position snaps to the origin at 10 s and stays |
| restart | F2 | a 2 s gap, then the reset |
| dropout3 | F3 | a 3 s dropout, dead-reckoned with a 0.3 m/s^2 bias toward the wall |
| silence5 | F4 | a 5 s dropout: the seconds before the module's `SH_WD_EST_TO` trip |
| freeze1 | F6 | the fix frozen for 1 s: the second before the module's `SH_WD_FRZ_T` trip |
| drift | F7 | the reported position walks 3 cm/s away from the wall for 20 s (0.6 m) |
| degrade | F36 | `field_degrade`'s levels for the whole hold: 5 cm noise, a 0.1 m/sqrt(s) walk, 0.5 s losses every 4 s |
| jumps5x | F8 | random jumps and dropouts at five times the vio_default rates |

## What it does not measure

The policy flies alone. No watchdog, wall guard, keep-out or handback is in the mock, so the
`silence5` and `freeze1` rows are what the drone does in the seconds before the module's own
trip would fire, not what the module does after it, and the `drift` row has no wall guard to
stop it at `SH_WALL_MIN`. The Gazebo field faults (`docs/field-faults.md`) remain the evidence
for the handback and the guards; these rates go beside them in the FMEA, in a column of
their own, not in their place. The mock's faults are also idealised: the reset snaps the whole
position at once, where a real VIO reset may come with a few bad frames first.

## Running it

```
python -m training.launch --config eval/configs/fault_rates_v22.yaml --smoke --device cpu
python -m training.launch --config eval/configs/fault_rates_v22.yaml --device cuda --set "params.only=[reset]"
python -m eval.fault_rates collect runs/<reset> runs/<restart> ... --out runs/fault_rates/fault_rates.md
```

All nine faults in one run is about 2 h on a 24 GB card, so the queue
(`training/queues/server_week2.yaml`) runs one fault per GPU job with `params.only` and
collects the reports into one table. The Gazebo half is the same queue's `gz_faults_v22` and
`gz_faults_tuned` jobs: the nine field-fault profiles (the six of the kit plus the GPS jump,
RTK float and GPS loss) on seeds 1 to 8, 144 flights in all.

## The Gazebo half: nine field faults on 8 seeds (2026-11-10, cloud)

Flown in a Claude cloud container (Linux, 4 CPUs, Docker, one flight at a time) on the SITL
image built from `e1fc624` (PX4 v1.17.0, v22 in the module with the VIO runaway trip and the
pre-arm check of 2026-11-09/10), `eval/baselines/px4_tuned_v2.params` (barometer height),
`wall1m_wind_4ms` (1 m from the wall, 4 m/s into it), seeds 1 to 8: Structure Hold on
`field_<fault>_at_engage --rangefinder`, tuned PX4 on `field_<fault>_at_hover`, 144 flights.
Every flight ran at a mean real-time factor of 0.97 to 1.04. Scored by `eval.field_faults` on
the TRUE position in the 25 s after the fault (contact is closer than 0.40 m, as everywhere).
Run folders and the `experiments/LOG.md` lines are dated 2026-11-10.

| Fault | FMEA | Controller | Touched the wall | Closest to the wall, worst (median) | Largest move, worst (median) | Still flying at the end | Trip or failsafe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reset | F1 | Structure Hold v22 | 0 of 8 | 0.91 m (1.00) | 0.12 m (0.09) | 8 of 8 | no trip in 8 |
| reset | F1 | tuned PX4 | 0 of 8 | 0.91 m (0.96) | 0.11 m (0.09) | 8 of 8 | failsafe in 0 |
| restart | F2 | Structure Hold v22 | 0 of 8 | 0.93 m (1.00) | 0.21 m (0.15) | 8 of 8 | no trip in 8 |
| restart | F2 | tuned PX4 | 0 of 8 | 0.79 m (0.95) | 0.29 m (0.14) | 8 of 8 | failsafe in 0 |
| silence | F4 | Structure Hold v22 | 0 of 8 | 0.92 m (1.00) | 0.93 m (0.40) | 8 of 8 | `estimate_invalid` trip in 8 |
| silence | F4 | tuned PX4 | 2 of 8 (seeds 1, 3) | 0.24 m (0.58) | 2.11 m (0.47) | 0 of 8 | failsafe in 8 |
| freeze | F6 | Structure Hold v22 | 0 of 8 | 0.93 m (0.96) | 1.24 m (0.27) | 8 of 8 | `estimate_frozen` trip in 8 |
| freeze | F6 | tuned PX4 | 2 of 8 (seeds 4, 7) | 0.14 m (0.88) | 12.91 m (1.31) | 6 of 8 | failsafe in 0 |
| drift | F7 | Structure Hold v22 | 0 of 8 | 0.67 m (0.69) | 0.34 m (0.33) | 8 of 8 | no trip in 8 |
| drift | F7 | tuned PX4 | 4 of 8 (seeds 2, 4, 5, 7) | 0.31 m (0.39) | 0.71 m (0.62) | 8 of 8 | failsafe in 0 |
| degrade | F36 | Structure Hold v22 | 1 of 8 (seed 8) | 0.36 m (0.86) | 0.74 m (0.53) | 8 of 8 | no trip in 8 |
| degrade | F36 | tuned PX4 | 2 of 8 (seeds 1, 8) | 0.37 m (0.53) | 0.80 m (0.73) | 8 of 8 | failsafe in 0 |
| degrade | F36 | Structure Hold v24 (2026-11-27) | 0 of 8 | 0.77 m (0.85) | 1.09 m (0.72) | 8 of 8 | no trip in 8 |
| gps jump | F27 | Structure Hold v22 | 0 of 8 | 0.68 m (0.70) | 0.35 m (0.31) | 8 of 8 | no trip in 8 |
| gps jump | F27 | tuned PX4 | 8 of 8 (seeds 1, 2, 3, 4, 5, 6, 7, 8) | 0.15 m (0.25) | 16.55 m (5.53) | 4 of 8 | failsafe in 0 |
| rtk float | F38 | Structure Hold v22 | 0 of 8 | 0.70 m (0.72) | 0.31 m (0.28) | 8 of 8 | no trip in 8 |
| rtk float | F38 | tuned PX4 | 0 of 8 | 0.65 m (0.68) | 0.34 m (0.32) | 8 of 8 | failsafe in 0 |
| gps loss | F39 | Structure Hold v22 | 0 of 8 | 0.73 m (0.75) | 0.31 m (0.29) | 8 of 8 | no trip in 8 |
| gps loss | F39 | tuned PX4 | 0 of 8 | 0.64 m (0.69) | 0.35 m (0.33) | 8 of 8 | failsafe in 0 |

What the 8 seeds say, in plain words:

- **Reset and restart (F1, F2)** no longer disturb either controller. On barometer height the
  height does not snap with the VIO, EKF2 absorbs the snap, and both moved 0.3 m at most; the
  module did not trip. The FMEA rows still describe the vision-height flights (a 2.2 m climb
  and a trip): their effect column is out of date, not their rating.
- **Silence (F4)**: the module tripped on all 8 (`estimate_invalid`), Structure Hover held the
  wall distance on the rangefinder, all 8 still flying, closest 0.92 m. Tuned PX4's failsafe
  landed all 8; 2 touched the wall on the way.
- **Freeze (F6)**: the frozen-fix trip fired on all 8, closest 0.93 m. One seed (7) slid
  1.24 m along the wall in Structure Hover after the trip (the others 0.18 to 0.40 m). Tuned
  PX4 has no trip: 2 of 8 flew off (7 and 13 m), touched the wall and came down; the row
  says it wandered 0.46 m on one seed.
- **Drift (F7)**: the wall guard stopped v22 at 0.67 to 0.70 m on all 8. Tuned PX4 followed
  the walk into the wall on 4 of 8.
- **Degrade, dust and fog (F36)**: **v22 touched the wall on 1 of 8 (seed 8, closest
  0.36 m)**; tuned PX4 on 2 of 8. This is inside the envelope (4 m/s, a VIO that still runs)
  and the FMEA row says 0 of 6 touched, residual 2 with the rangefinder. What happened on
  seed 8: the VIO walked about 0.45 m and carried the drone toward the wall; the guard saw
  the wall close and moved the hold point 0.55 m back, as it should, but v22 stayed 0.6 to
  0.7 m short of the new point with almost no rate command for the last 20 to 25 s of the
  hold and never closed the gap. Seed 6 left 0.35 m the same way; seeds 1, 3, 4, 5 and 7
  closed theirs in 10 to 15 s. Nothing tripped (the hold error stayed under `SH_WD_POS_ERR`
  2 m and the VIO-IMU velocity gap under 0.8 m/s). Our reading, not proven: v22 treats a
  sudden step of the hold point like an estimate jump and does not chase it; slow moves,
  as in the drift fault, it always follows. Same-seed flights are not bit-repeatable (v22's
  reset seed 1 flew 0.34, 0.17 and 0.20 m p95 on three flights), so 1 of 8 is a rate, not a
  seed that always fails.
- **GPS jump (F27)**: v22 held at the guard (closest 0.68 m) on all 8. Tuned PX4 touched the
  wall on 8 of 8 and 4 of them ended on the ground.
- **RTK float and GPS loss (F38, F39)**: neither touched; v22 held at the guard (0.70 m and
  0.73 m closest), tuned PX4 came to 0.64 to 0.65 m.

How much 8 flights can say: no contact in 8 rules out only rates above about 1 in 3 (Wilson
95 % upper bound 0.32); 1 in 8 is consistent with anything from 2 % to 47 %. These flights
show what happens and roughly how often; the mock runs above (on the server queue, not
logged yet) are what will give tight bounds.

One tuned PX4 GPS-loss flight (seed 8) never got its fault: the stock runner fires the fault
when the true height reaches the hover height less 0.3 m, and on that flight EKF2's
barometer height sat about 0.43 m high, so the drone hovered at 2.07 m true and the anchor
(2.2 m) never came. It is void and was flown again on 2026-11-26 (same image; the fault fired, 0.69 m closest, 0.28 m moved); the table counts that flight. The anchor would be safer keyed
to EKF2's own height or set lower; that is in "Next" below rather than changed here.

### What this changes in the failure table (safety manual 1.14, for Zain to re-sign)

These went into the safety manual as version 1.14 (2026-11-27), not yet signed:

| Row | Manual now | 8 seeds in Gazebo |
| --- | --- | --- |
| F1, F2 | Stock PX4 climbed 2.2 m; the module trips | On barometer height: no climb, no trip, both moved 0.3 m at most (vision-height text out of date) |
| F4 | Tuned PX4 touched on 6 of 12 (three winds) | 2 of 8 at 4 m/s; v22 0 of 8, closest 0.92 m (agrees) |
| F6 | Tuned PX4 wandered 0.46 m | Tuned PX4 flew off and came down on 2 of 8; v22 moved up to 1.24 m along the wall after the trip (row says 0.21 m) |
| F7 | Tuned PX4 followed it to 0.42 m | Tuned PX4 touched on 4 of 8; v22 0.67 m closest (agrees) |
| **F36** | **v22 0 of 6 touched, residual 2 with the rangefinder** | **v22 touched on 1 of 8 (0.36 m)**; tuned PX4 2 of 8; v24 (released 2026-11-26) 0 of 8, closest 0.77 m. 1.14: residual 2 with v24, 3 for a v22 build |
| F27 | Tuned PX4 touched on 3 of 3 | 8 of 8; v22 0 of 8 (agrees) |

### Next

- The degrade contact: flown with v24 on the same 8 seeds (2026-11-27, image from `0fb68ad`,
  run `20261127-010621`): 0 of 8 touched, closest 0.77 m (median 0.85), largest move 1.09 m
  (median 0.72), no trip, every flight at a real-time factor of 0.92 or more. Its range
  input keeps it off the wall where v22 stayed short of the guard's moved point; it follows
  the walk along the wall a little further than v22 (one flight moved 1.09 m, past the
  runner's 1 m position check). Left: whether v22 really ignores a hold-point step (a mock
  test that steps the hold point by 0.5 m), which matters only for a v22 build.
- Key the stock runner's fault anchor to EKF2's height, or lower it, so no flight goes
  without its fault.

## Where the numbers go

Each run writes its `experiments/LOG.md` line. The collected table is quoted in the safety
manual's failure table as a new column ("rate in the mock, 95 % bound, 20,480 episodes"),
which is a change to section 5 and so a new manual version for Zain to re-sign. A fault whose
bound is worse than the row's current rating is a finding for Zain, not a quiet edit.
