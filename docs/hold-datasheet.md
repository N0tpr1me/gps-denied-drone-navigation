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

### Flown (2026-11-09/10, cloud)

64 flights in a Claude cloud container (Linux, 4 CPUs, Docker, one flight at a time) on the
SITL image built from `9893e4b` (PX4 v1.17.0, v22 in the module),
`eval/baselines/px4_tuned_v2.params` (barometer height), seeds 1 to 8 in two halves under the
hour, pooled: Structure Hold on `vio_at_engage --rangefinder`, tuned PX4 on `vio_at_hover`.
Mean real-time factor 0.97 to 1.01 on every flight. Scored like for like (the decision of
2026-11-10: the height EKF2 read on the ground taken out, a fixed hold point for both;
`run_baseline --rescore`), mean p95 over the 8 seeds with the worst in brackets:

| Scenario | Tuned PX4 | v22 | Change | Worst toward the wall, mean: tuned PX4 → v22 | Closest to the wall: tuned PX4 / v22 | Crashes, failsafes |
| --- | --- | --- | --- | --- | --- | --- |
| `wall1m_wind_14ms` (into the wall) | 0.344 (0.741) m | 0.233 (0.319) m | -32 % | 0.183 → 0.153 m | 0.64 / 0.70 m | 0 of 8 each |
| `wall1m_along_10ms` | 0.346 (0.748) m | 0.251 (0.487) m | -27 % | 0.160 → 0.070 m | 0.53 / 0.86 m | 0 of 8 each |
| `wall1m_along_12ms` | 0.345 (0.742) m | 0.256 (0.401) m | -26 % | 0.164 → 0.075 m | 0.54 / 0.87 m | 0 of 8 each |
| `wall1m_lee_12ms` (from behind the wall) | 0.343 (0.753) m | 0.465 (0.740) m | **+36 %** | 0.198 → 0.009 m | 0.49 / 0.94 m | 0 of 8 each |

As scored before the decision (against the setpoint as logged) the same flights read tuned
PX4 0.50 to 0.52 m and v22 0.33 to 0.36 m at the first three edges (-31 to -33 %) and
0.52 against 0.53 m in the lee (+2 %); pooled files `runs/week2_pooled/*_edges_seeds1-8.json`.

What it says:

- **Up to 14 m/s into the wall and 12 m/s along it, nothing broke.** Neither controller
  crashed or hit a failsafe, v22 never came closer than 0.70 m, and it held 26 to 32 %
  tighter than tuned PX4: a little less than its 39 to 41 % inside the envelope
  (`docs/baseline.md`), so the margin narrows with the wind but does not close.
- **In a 12 m/s lee v22 is looser than tuned PX4**, by 36 %. It is loose in the safe
  direction: on 5 of 8 seeds it settled 0.2 to 0.6 m further from the wall than its hold
  point and stayed there (a steady offset, not a wobble), worst toward the wall 0.06 m. This
  wind is outside the 8 m/s envelope, so no envelope claim changes, but a buyer reading a
  hold number behind a wall would see it. It is a candidate for training (the lee case, with
  the near-wall flow terms off, as flown here).
- Tuned PX4's two loose seeds (1 and 6, 0.67 to 0.75 m) are loose in every wind, so they
  come from the estimate those seeds draw, not from the wind.

The mock sweep's matching cells are not run yet (the server's week-2 queue), so there is no
mock row to set beside these; when there is, Gazebo wins where the two disagree.

**The same edges on the module with the newer trips.** The image above predates the VIO
runaway trip (`SH_WD_VIO_DV`) and the pre-arm check (`SH_WD_GND_V`), so v22 flew the edges
again on the `e1fc624` image (2026-11-26, same container): seeds 1 to 4 in full (16 flights)
and the first 5 flights of seeds 5 to 8 before the container stopped. In all 21 the only
watchdog trip was the test sequence's own scripted one after the hold: neither new trip fired
at 10 to 14 m/s. Like for like, seeds 1 to 4 came out at 0.241 / 0.280 / 0.260 / 0.440 m
mean p95 (into 14, along 10, along 12, lee 12) against 0.245 / 0.238 / 0.267 / 0.452 m on
the first image, no crash, closest to the wall 0.74 m: the same hold, with one seed (2, along
10 m/s) 0.16 m looser, inside flight-to-flight spread.

## Where the numbers go

The sweep's `experiments/LOG.md` lines come from `launch.py`; the Gazebo batches write
`log_lines.md` to paste. The datasheet itself is a result page, not a decision: it is quoted
in `docs/safety-manual.md` section 2 (a new manual version, re-signed) and in the outreach
pack. A cell where v22 crashes inside the current envelope is a finding for Zain and a
Proposed row, not a quiet edit.
