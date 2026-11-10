# The hold datasheet: where Structure Hold breaks

Buyers read drone datasheets that quote a wind rating (Elios 3: 5 m/s, 7 in Sport; Skydio
X10: 12 m/s) and nobody publishes hold accuracy against wind. The datasheet is our version
of that number, with the honest extra dimensions: how far from the wall, which way the wind
blows, and how good the position estimate is. It also puts numbers on the safety manual's
operating envelope (`docs/safety-manual.md` section 2), which today says "up to 8 m/s, 1 to
3 m" from the Gazebo benchmark alone. It is item 1 of the week-2 plan Zain approved on
2026-11-09 (`docs/gpu-server-plan.md`, "Week 2").

## The sweep

Three `eval.gate1` runs, one per wind direction, so each stays under the hour rule on one GPU:

| Config | Wind | Cells |
| --- | --- | --- |
| `eval/configs/hold_datasheet_into.yaml` | into the wall (`mean_heading_deg` 0), 0, 4, 8, 10, 12, 14 m/s | 48 |
| `eval/configs/hold_datasheet_along.yaml` | along the wall (90), 4 to 14 m/s | 40 |
| `eval/configs/hold_datasheet_lee.yaml` | away from the wall, the drone in its lee (180), 4 to 14 m/s | 40 |

Every cell is one wind speed, one distance (1 or 3 m) and one estimate quality, 512 episodes
of 60 s on the `gz_x500_sitl` airframe (EKF2-like fusion on the barometer, as the benchmark
flies), the released v22 and the cascaded PD yardstick on the same episodes, with 95 %
intervals from `eval.gate1`. Turbulence scales with the wind as the Gazebo scenarios do
(`turb_sigma` = wind / 8). The four estimates:

| Estimate | What it is |
| --- | --- |
| clean | no delay, noise, bias, jumps or dropouts: the controller's own limit |
| vio_default | the benchmark's estimate (`sim2real/estimate_bridge/profiles/vio_default.yaml`) |
| degraded | the `field_degrade` levels for the whole hold: 5 cm noise, 0.2 m/s velocity noise, a 0.1 m/sqrt(s) walk, 0.5 s losses every 4 s (dust, fog, low light) |
| dropout3 | vio_default with 2 to 3 s dropouts about every 20 s (the top of the Gate 1 dropout cell) |

Cells inside the released envelope (8 m/s or less, clean or vio_default) are `envelope`
cells, so the run's verdict still says whether v22 beats the PD there; every other cell is
measured, not judged. Cell names carry the coordinates (`into_w10_d1_vio`), which is what
`eval/hold_datasheet.py` reads.

## The table

```
python -m eval.hold_datasheet runs/<into> runs/<along> runs/<lee> --out runs/hold_datasheet/datasheet.md
```

writes two tables and a JSON file. The envelope table gives, per direction, distance and
estimate, the highest wind at which each controller still holds (p95 at most `--p95-limit`,
0.5 m, with the crash rate's 95 % upper bound under `--crash-limit`, 1 %) and the first wind at
which it does not. The cell table gives every p95 and crash rate. The limits are arguments,
not decisions: a buyer with a tighter requirement reruns the script.

## Gazebo at the edges

The mock has not predicted Gazebo on its own (`CLAUDE.md`, "Releasing a policy"), so the
queue flies the sweep's edge cells in Gazebo against tuned PX4, 8 seeds each, as new
scenarios in `sim2real/gazebo/scenarios/`: `wall1m_wind_14ms` (above the training range),
`wall1m_along_10ms` and `wall1m_along_12ms` (cross wind), `wall1m_lee_12ms` (wind from behind
the wall, without the near-wall flow terms). Crashes are allowed and are the result. The
Gazebo rows go beside the mock's in the datasheet when it is published, with the same rule
as `docs/baseline.md`: where the two disagree, Gazebo wins and the mock's cell is marked.

## Where the numbers go

The sweep's `experiments/LOG.md` lines come from `launch.py`; the Gazebo batches write
`log_lines.md` to paste. The datasheet itself is a result page, not a decision: it is quoted
in `docs/safety-manual.md` section 2 (a new manual version, re-signed) and in the outreach
pack. A cell where v22 crashes inside the current envelope is a finding for Zain and a
Proposed row, not a quiet edit.
