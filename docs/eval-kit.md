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
| 40 or more free cores (the GPU server's Threadripper, a build box) | 10 | about 3 minutes |
| 16 cores | 4 | about 7 minutes |
| 4 cores (laptop, the cloud container) | 1 | about 20 minutes (measured, below) |

A flight takes about 2 minutes of wall clock (Structure Hold 2.2, tuned PX4 1.8, measured on
the cloud container); `--dry-run` prints the estimate for the machine it runs on.

The first build of the image takes 20 to 40 minutes; ship a saved image
(`docker save structure-hold/px4-sitl:v1.17.0 -o kit-image.tar`, about 3 GB) to skip it.

### First run (2026-11-02, cloud container)

`./evaluate.sh` from a clean checkout with the image already built (`cloud_build.sh`): all
four steps ran, 10 flights one at a time took 20 minutes (16:34 to 16:54 UTC), exit 0. Kit
folder `runs/20261102-163357_eval_kit_e185a80` (flight folders after the first three carry
the next local commit in their names; same image). The printed table:

| Fault | Controller | Closest to wall, m | Largest move, m | Lowest height, m | What happened | Better |
|---|---|---|---|---|---|---|
| VIO resets to its origin in the air | Structure Hold | 1.02 | 0.07 | 2.43 | no trip, still flying at the end | **Structure Hold**: stayed 0.05 m further from the wall |
|  | tuned PX4 | 0.96 | 0.09 | 2.47 | still flying at the end |  |
| VIO process restarts (3 s gap, then reset) | Structure Hold | 1.00 | 0.08 | 2.40 | no trip, still flying at the end | **Structure Hold**: stayed 0.10 m further from the wall |
|  | tuned PX4 | 0.90 | 0.11 | 2.44 | still flying at the end |  |
| position link lost for 8 s | Structure Hold | 0.95 | 0.43 | 2.41 | tripped (estimate lost) and handed back, still flying at the end | **Structure Hold**: tuned PX4 touched the wall |
|  | tuned PX4 | 0.23 (contact) | 0.81 | -0.00 | touched the wall, dropped 2.5 m, on the ground at the end |  |
| estimate freezes for 5 s | Structure Hold | 0.97 | 0.19 | 2.39 | tripped (estimate frozen) and handed back, still flying at the end | **Structure Hold**: stayed 0.05 m further from the wall |
|  | tuned PX4 | 0.92 | 0.74 | 2.29 | still flying at the end |  |
| estimate drifts 0.6 m away from the wall | Structure Hold | 0.70 | 0.33 | 2.43 | no trip, still flying at the end | **Structure Hold**: stayed 0.28 m further from the wall |
|  | tuned PX4 | 0.42 | 0.59 | 2.47 | still flying at the end |  |


Structure Hold better on 5 of 5 faults, tuned PX4 on 0, about the same on 0. Wall contacts: Structure Hold 0, tuned PX4 1.

What it says, in plain words:

- **Link loss is the clear case.** Tuned PX4 held for about 4.5 s, then its failsafe
  descended and it touched the wall on the way down. Structure Hold tripped, handed back to
  its range hover and stayed 0.95 m off the wall, still flying.
- **Drift:** the wall guard stopped Structure Hold at 0.70 m; tuned PX4 followed the drift to
  0.42 m, 2 cm from prop contact. The same result as the first flights in
  `docs/field-faults.md`.
- **Freeze:** the frozen-fix trip fired and the guarded hover moved 0.19 m; tuned PX4
  wandered 0.74 m. The verdict names wall clearance first, which differed by only 5 cm.
- **Reset and restart are mild on barometer height.** The VIO snaps back to the takeoff
  point, which here is straight below the hold point, so only its height jumps, and with
  the barometer as EKF2's height reference (`px4_tuned_v2.params`) neither controller
  moved more than 11 cm. The 2.2 m climb in `docs/field-faults.md` was on vision height
  (`px4_tuned_v1`). A takeoff away from the hold point would make the reset a horizontal
  jump too; that is a different scenario, not flown here.

### Before it goes to a buyer

- **What they receive: binary only** (Decided 2026-11-02, `docs/decisions.md`).
  `sim2real/docker/build_kit.sh` writes `dist/kit-<commit>/`: `kit-image.tar` (about 3 GB,
  the SITL image with the module's source and build objects deleted and the layers
  flattened, so no lower layer still holds them; the compiled module is in PX4's binary)
  and `kit-src.tar.gz` (about 130 KB: `evaluate.sh`, the runners, the estimate bridge, the
  Gazebo scenarios and the scorers; no module source, training code, policies or docs). The
  buyer unpacks the second and runs `./evaluate.sh --load kit-image.tar`. Checked
  2026-11-02: unpacked in an empty folder, the local image deleted, `./evaluate.sh --load
  kit-image.tar --faults silence drift` loaded the stripped image and flew all 4 flights
  (exit 0). Link loss: Structure Hold closest 0.91 m and still flying, tuned PX4 touched the
  wall and landed. Drift: 0.71 against 0.42 m. Module source in the image: none (checked);
  the module's strings are in PX4's binary.
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

### Sample (2026-11-02, Gazebo, not a partner flight)

To check the whole path on real PX4 logs, four clean 60 s holds were flown in the cloud on
`wall1m_wind_4ms`, seeds 1 and 2 (`run_baseline`, v22 with the rangefinder on
`vio_at_engage`, tuned PX4 on `vio_at_hover`, `px4_tuned_v2.params`), written up as a
`session.yaml` with `truth: Gazebo` (the logs' ground-truth topic stands in for mocap) and
scored by `eval.partner_session`. The page said Structure Hold held 41 % tighter (p95 0.238
against 0.407 m, worst run 0.323 against 0.649 m), 0 contacts and 0 early ends on either
side. Two holds per side on one wind: a check that the report works, not a result; the
benchmark is `docs/baseline.md`.

Two things it found, both fixed in `eval/partner_session.py`:

- The module refuses to engage on the ground and hands straight back, which leaves a blip
  of Structure Hold in `vehicle_status`; the scorer took that blip as the hold ("stretch is
  0.0 s"). It now takes the first stretch that lasts longer than the settle time plus 1 s.
- PX4's SITL sits in Hold mode on the ground before arming, which the scorer took as tuned
  PX4's hold. It now looks only at stretches that start after arming.

The Gazebo stock runner switches to Hold straight after its climb, so its first seconds in
Hold are still the climb settling; the sample gives those runs the runner's own hover
window (`window:` in the sheet). A partner pilot reaches the hold point before switching,
as the pack says, so real sessions need no override. The scorer also holds against the
setpoint at the start of the hold, not PX4's moving setpoint as `run_estimate` does, so
the sample's tuned-PX4 p95 on seed 1 (0.649 m) is lower than that runner's own (0.895 m),
where PX4 moved its setpoint after the estimate jump.
