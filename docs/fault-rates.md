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

## Where the numbers go

Each run writes its `experiments/LOG.md` line. The collected table is quoted in the safety
manual's failure table as a new column ("rate in the mock, 95 % bound, 20,480 episodes"),
which is a change to section 5 and so a new manual version for Zain to re-sign. A fault whose
bound is worse than the row's current rating is a finding for Zain, not a quiet edit.
