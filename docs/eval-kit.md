# Evaluation kit and partner benchmark report

Confidential. Two tools that turn "robust to a bad position estimate" into something a
drone maker can run, read and pass to their safety reviewer. Both come from the "Features
that sell" ranking (items 1 and 2, 2026-11-02): an engineering lead wants to see the module
handle the failures they know before they take a call, and after a partner flight they want
one page they can show their manager.

## 1. The evaluation kit: one command, five failures, one table

```bash
./evaluate.sh                          # checks Docker, builds the image if missing, flies
./evaluate.sh --load kit-image.tar     # use an image we ship instead of building it
./evaluate.sh --faults reset drift     # a subset
./evaluate.sh --dry-run                # the plan and how long it should take
python -m eval.eval_kit --rescore runs/<kit folder>    # print an earlier table again
```

`evaluate.sh` does four things and says which step it is on: checks that Docker runs;
loads or builds the SITL image (PX4 v1.17.0, Gazebo Harmonic, the Structure Hold module and
the `wall_slab` world, `sim2real/docker/`); makes a small Python environment
(`.venv-kit/`, `pip install -e .[sitl]`) if the machine's Python lacks the packages; then
runs `eval/eval_kit.py`.

The kit flies the five scripted field faults of `docs/field-faults.md` twice each:

| Fault | Structure Hold flight | Tuned PX4 flight |
| --- | --- | --- |
| VIO resets to its origin in the air | `field_reset_at_engage`, rangefinder on | `field_reset_at_hover` |
| VIO restarts (3 s gap, then reset) | `field_restart_at_engage` | `field_restart_at_hover` |
| Position link lost for 8 s | `field_silence_at_engage` | `field_silence_at_hover` |
| Estimate freezes for 5 s | `field_freeze_at_engage` | `field_freeze_at_hover` |
| Estimate drifts 0.6 m away from the wall | `field_drift_at_engage` | `field_drift_at_hover` |

Every flight: scenario `wall1m_wind_4ms` (1 m from the wall, 4 m/s mean wind into it,
gusts), seed 1, tuned parameters `eval/baselines/px4_tuned_v2.params` (barometer height),
the fault 20 s into the hold. The module flies the released policy with its defaults
(including `SH_HB_RANGE` and `SH_WD_FRZ_T`); tuned PX4 holds in Hold mode. Each pair is
scored by `eval/field_faults.py` on the TRUE position over the 25 s after the fault, and the
kit prints one table: per fault, each controller's closest approach to the wall, largest
move, lowest height and what it did (trip and handback, climb, drop, still flying or on the
ground), plus a verdict.

The verdict is a fixed rule so nobody has to trust a judgement call: fewer wall contacts
wins, then more wall clearance by more than 5 cm, then a smaller largest move by more than
5 cm, otherwise "about the same". Landing is not judged: stock PX4's failsafe landing on a
lost estimate is its designed answer and counts against it only if it touches the wall on
the way down, which the first rule catches.

Output: `runs/<time>_eval_kit_<commit>/` with `config.yaml`, `flights.json` (one run folder
per flight, each with its PX4 log, bridge trace and scores), `comparison.json` and
`comparison.md`.

### How long it takes

Gazebo runs headless on CPU cores, no GPU. `--jobs auto` (the default) flies as many
flights at once as the machine has free blocks of 4 cores, each pinned to its own:

| Machine | Flights at once | Time for all 10 |
| --- | --- | --- |
| 40 or more free cores (the GPU server's Threadripper, a build box) | 10 | about 10 minutes |
| 16 cores | 4 | about 15 minutes |
| 4 cores (laptop, the cloud container) | 1 | see the first run below |

The first build of the image takes 20 to 40 minutes; ship a saved image
(`docker save structure-hold/px4-sitl:v1.17.0 -o kit-image.tar`, about 3 GB) to skip it.

### First run (2026-11-02, cloud container)

FIRST_RUN_PLACEHOLDER

### Before it goes to a buyer

- **What they receive.** The kit as written needs this repository (the Python side
  launches the containers and scores them) and the image, which carries the module's
  source. Whether a buyer gets source under the evaluation agreement or a binary-only image
  is the open "source or binary" item of `docs/partner-pack.md`, for Zain.
- **One seed, one scenario.** The table is a first look, as the field-faults page says;
  more seeds come from `eval.baselines.run_baseline` on the same profiles.
- **The cloud image** (`sim2real/docker/cloud_build.sh`) has a different Gazebo base from
  the real image (`run.sh build`); quote numbers from the real image.

## 2. The one-page benchmark report from a partner session

```bash
python -m eval.partner_session <folder>/session.yaml      # scores; writes scored/ including
                                                          #   benchmark_report.html
python -m eval.partner_report <folder>/scored/session_summary.json --pdf   # re-render, + PDF
```

`eval/partner_session.py` scores a partner's session folder (`docs/partner-pack.md`, step 6)
and now also writes `scored/benchmark_report.html` through `eval/partner_report.py`. The
page reads every number from `session_summary.json`, so it can never disagree with the
engineer's tables in `report.md`. It holds:

- what was flown (drone, site, date, firmware, policy, position source, truth source);
- the headline: Structure Hold's mean p95 distance from the hold point against tuned PX4's,
  as a percentage, on the same drone and day;
- four tiles: that change, holds with a wall contact, holds that ended early, and the
  closest either came to the wall;
- per case: holds, p95 mean (worst), worst move toward the wall, closest approach,
  contacts and early ends, and the change;
- every hold's p95 as a dot (Structure Hold circles, tuned PX4 squares), so a single bad
  run cannot hide in a mean;
- every handback in plain words ("hold ended after 18 s, position estimate jumped; handed to
  Structure Hover"), every aborted or unscored run, and any run whose EKF-to-mocap fit is off
  by more than 5 cm;
- the set-up and how it was scored.

It is one self-contained HTML file (no scripts or web fonts) that prints on one A4 page;
`--pdf` prints it with Playwright where that is installed, otherwise print it from a browser.
No real session exists yet; the tests (`eval/tests/test_partner_report.py`) build one from
synthetic logs, and the sample below came from Gazebo logs scored through the same path.

SAMPLE_PLACEHOLDER
