# The GPU server: the first week's work, every card busy from day 1

`docs/gpu-server.md` says how to set the server up and give Claude access. This page says what
runs on it once it is up, in what order, on which GPU, for how long, and what each result
decides. The whole queue is one file, `training/queues/server_week1.yaml`, run by
`training/queue.py`; everything in it was smoke-tested on the cloud CPUs before the server
existed, so day 1 is running, not writing.

Expected machine (Zain, 2026-11-02): several RTX 4500 Ada (24 GB each), about 512 GB RAM, an
AMD Threadripper Pro. Nothing below depends on the exact count: the queue fills whatever
`nvidia-smi` shows free and leaves the SITL flights on the CPU cores.

## What the server does that the cloud and the Mac could not

| Work | Why it waited for the server |
| --- | --- |
| Isaac Lab (M3, `docs/isaac-lab.md`) | Isaac Sim needs an RTX GPU with RT cores; the cloud has no GPU and the Mac is not Linux |
| The v24 Gazebo re-fly (`docs/wall-range.md`) | 24 flights with a rangefinder and a rebuilt SITL image, a Mac afternoon; simulations do not go on the Mac any more (Zain, 2026-10-31) |
| Route Follow for narrow pipes (`docs/route-follow.md`) | The harder mix needs thousands of iterations on thousands of drones; the cloud's 4 cores gave 1000 iterations an hour on 1024 |
| 10 to 12 m/s wind (`docs/market-research.md` row 15) | Gazebo cells at datasheet winds, 16 flights per controller; the cloud flies one or two at a time |
| Bigger Gazebo batches, Isaac-trained policies, seed sweeps | Each needs a few GPU-hours or tens of CPU-cores for an hour; the Mac and the cloud each managed one such job a day |

The 5 m surface passes (`docs/surface-passes.md`) have a queue of their own,
`training/queues/surface_passes.yaml` (2026-11-09): two seeds of the pass training on 2 GPUs,
chains of sub-1 h parts, then `hold_pass_gate`. It runs on its own or beside this one.

## How the queue works

```
python -m training.queue training/queues/server_week1.yaml --dry-run --gpus 4   # the plan
python -m training.queue training/queues/server_week1.yaml --smoke              # every smoke, ~5 min
python -m training.queue training/queues/server_week1.yaml                     # jobs under 1 h
python -m training.queue training/queues/server_week1.yaml --approved          # with Zain's yes
```

- Every free GPU is a lane; the queue starts the first runnable GPU job on each and starts
  the next one the moment a lane frees. `launch.py --gpu N` pins each run to its card (new,
  with the queue), so two runs started together never land on the same GPU, and each GPU job
  gets four host cores of its own.
- SITL flights are the CPU lane: one `run_baseline --jobs auto` batch at a time, on all the
  cores the GPU jobs do not hold (`reserve_cpus` 4 stay free for the shell). On a 64-core
  Threadripper with 4 GPUs that is 44 cores, 11 flights at once, so a 24-flight batch takes
  about 6 minutes instead of the Mac's 50.
- A job starts when the jobs in its `after` list finished ok. A failed job blocks only the
  jobs that depend on it; the rest go on. Jobs over 1 h, or marked `needs_approval`, are
  skipped unless the queue runs with `--approved`.
- Long training is a `chain`: the same config run N times, each part resuming the last
  part's `policy.pt` (`launch.py --set params.resume=...`, also new), so every part is under
  the 1 h rule and a stopped queue resumes at the last good part.
- `{run:ID}` in a job names another job's run folder, so a scoring job points at the policy
  a training job just exported.
- State is `runs/queue_server_week1/state.json`; `summary.md` there is the table to paste in
  the project. Re-running the same command skips what finished ok.
- Real runs go through `training/launch.py` as always: the 1 h and 1 GPU rules, the free-GPU
  check, one folder per run and one `experiments/LOG.md` line each. launch.py no longer
  counts appended LOG.md lines as a dirty tree, so a chain's second part is not refused by
  the first part's log line.

## The queue, in order

Times are estimates from the cloud CPU runs (`experiments/LOG.md`) scaled by what a 24 GB
GPU usually gives the mock: about 10x the env steps per second of 4 CPU cores at 4096
drones. The first smoke runs on the server (`--smoke`, the `env_steps_per_s` line in each
`stdout.log`) replace these; if a part would pass 1 h, cut `iterations` in the config.

### Day 1 (needs Zain for about ten minutes, then nothing until the evening)

| # | Job | Lane | Time | What it decides |
| --- | --- | --- | --- | --- |
| 1 | `report`: `server_report.sh` + `nvidia-smi` | CPU | 3 min | Driver 580.65.06+, every GPU listed and free, Docker without sudo, 80 GB disk, nvcr.io reachable. Anything missing is the admin's step in `docs/gpu-server.md` section 1 |
| 2 | `setup`: `server_setup.sh` | CPU | ~1 h | Python env, lint, tests, a smoke run on CPU and on one GPU, the SITL image (PX4 v1.17.0, v22 compiled in), a hover test, several hovers at once at real time, the Isaac Lab image. Ends with `report.md`. Over 1 h, so it is in the batch approval below |
| 3 | `isaac_parity`: `training/isaac_parity.sh` | GPU 0 | 45 min | `docs/isaac-lab.md` steps 3 to 9 in one container: repo tests in Isaac's Python with no Isaac skips, `parity: PASS`, v22 recipe smoke on both backends with steps/s, v22 scored on Isaac within the mock's 0.20 to 0.21 m p95 and under 1 % crashes. **Go/no-go for M3.** A GPU without RT cores or a parity FAIL stops the Isaac lane; the decided fallback is the mock on the GPU, which the other lanes already use |
| 4 | `route_pipes`, `route_pipes_s4`, `route_pipes_s5` | GPU 1, 2, 3 | 3 x 45 min each | Route Follow from scratch on the mix of worlds weighted to pipes (70 % tunnels, radius 0.8 to 2.0 m) with the estimate's jumps and dropouts five times the tunnel defaults, 4096 drones, three seeds because PPO moves more between seeds than between recipes. Each part's `summary.json` compares the policy with both PD yardsticks on the pipes (`pipe_nominal`) |
| 5 | `hold_wind_check`: `eval/configs/gazebo_check_wind12.yaml` | first free GPU | 30 min | v22 and v24 in the mock at 8, 10 and 12 m/s into the wall, with and without the near-wall flow terms, v22 the reference. A table for v22 and a pass/fail for v24 at high wind. Decides whether `hold_wind_finetune` (below, off) is worth switching on before the Gazebo cells even fly |
| 6 | `v24_check_range`: `eval/configs/gazebo_check_v24_range.yaml` | first free GPU | 45 min | The pre-re-fly gate for v24 on the server's own numbers; must pass against v22 as it did on the cloud (p95 ratio 0.77, 1.5 cm lift, `docs/wall-range.md`); those numbers were on vision height and the check fuses on the barometer since 2026-11-03, so expect close, not identical, numbers, and the lift rule is soft (under 5 cm passes) since the same day. A different answer means the GPU build differs from the CPU one, which `v22_eval_mock_gpu` then pins down |
| 7 | `v22_eval_mock_gpu`: `structure_hold_eval_v22.yaml` | first free GPU | 10 min | v22 scored on the mock on the GPU: p95 0.211 / 0.206 / 0.202 m, crash 0 / 0.4 / 0 %, the same numbers as the cloud CPU to a few mm, or the torch build is suspect. Also the reference Isaac's step 9 is held to |

Steps 3 to 7 start the moment `setup` passes and run side by side: four GPUs busy by hour 2
of day 1 and until about hour 4.5, then the three pipe chains finish at about hour 2.4 and
the scoring jobs take the lanes.

### Day 1 evening to day 2: the CPU lane (Gazebo + PX4 SITL)

All of these are `run_baseline --jobs auto`, scored on true position by `eval/metrics.py`,
8 seeds, the `px4_tuned_v2.params` set (barometer height, Decided 2026-11-01). None uses a GPU.

| # | Job | Flights | Time | What it decides |
| --- | --- | --- | --- | --- |
| 8 | `gz_tuned_baseline`: tuned PX4, `vio_at_hover` | 24 | ~25 min | The server's tuned-PX4 table against the Mac's (`docs/baseline.md`: mean p95 0.427 / 0.424 / 0.428 m). The machine changed, so a few % either way is expected; more means Gazebo's real-time factor is off (`rtf_min` in `batch.json`; fewer jobs) |
| 9 | `gz_v22_baseline`: v22, `vio_at_engage` | 24 | ~25 min | v22 on the server: mean p95 40 % or more below tuned, 0 crashes in the 60 s holds, as on the Mac (-43 to -45 %). This pair becomes the server's benchmark row in `docs/baseline.md`; the Mac's stays the record |
| 10 | `gz_v24_image`: export v24's header, build `structure-hold/px4-sitl:v24`, restore the committed header | - | ~30 min | A SITL image with v24 compiled in. The committed `PolicyWeights.hpp` stays v22's until a release |
| 11 | `gz_v24_refly`: v24, `--rangefinder`, the v24 image | 24 | ~25 min | **The v24 release decision.** Beats `gz_v22_baseline` on mean p95 and worst toward the wall with 0 crashes, and the mock predicted it (p95 ratio about 0.77): v24 goes to Zain as a release candidate, with the lift rule miss (+1.5 cm in the mock) stated. Loses or crashes: v24 is parked and the mock check's lift rule was right |
| 12 | `gz_v22_range`: v22, `--rangefinder` | 24 | ~25 min | v22 with the same rangefinder, so the wall guard is on for both and v24's gain is the policy's, not the guard's |
| 13 | `gz_wind12_tuned`: tuned PX4 at 10 and 12 m/s into the wall | 16 | ~17 min | New scenarios `wall1m_wind_10ms` and `wall1m_wind_12ms` (the 8 m/s scenario with more wind and turbulence). Crashes are allowed and are the result: the stock controller's limit at datasheet winds |
| 14 | `gz_wind12_v22`: v22 at 10 and 12 m/s, `--rangefinder` | 16 | ~17 min | **The datasheet-wind evidence** for `docs/market-research.md` row 15: hold p95 and closest approach at 10 and 12 m/s against tuned PX4. v22 trained on 0 to 12 m/s, so it should hold; if it loses or crashes, switch `hold_wind_finetune` on |

About 3 hours of CPU-lane time, so by the end of day 2 the benchmark, the v24 decision and
the wind evidence are all in `experiments/LOG.md`.

### Day 2 to 3: what the results switch on

| Job | Lane | Time | Runs when |
| --- | --- | --- | --- |
| `isaac_train_v22`: v22's recipe on Isaac Lab (`--set params.sim=isaac`) | GPU 0 | 45 min | After `isaac_parity` passes, automatically. The M3 training; the policy then takes `gazebo_check_v24_range.yaml`'s place in a copy of that config and a Gazebo re-fly like any candidate (`gz_isaac_policy`, off until then) |
| `route_pipes_stress`: the 13-cell stress table with the three pipe policies | first free GPU | 25 min | After the three chains finish, automatically. **Go/no-go for the pipes work:** the best seed has fewer narrow-pipe crashes than v0's 11 % and still beats both PD yardsticks on every cell; then it is the Route Follow candidate for the Gazebo route worlds (`sim2real/route_follow/`) |
| `hold_wind_finetune`: `structure_hold_ppo_v26_wind.yaml` | first free GPU | 50 min | By hand, `enabled: true`, only if v22 loses at 10 or 12 m/s in Gazebo. v22 continued on 6 to 12 m/s winds blowing within 60 degrees of the wall, same 21 inputs so the module header takes it unchanged; then the usual check and re-fly |
| `gz_isaac_policy` | CPU | ~1 h | By hand, after `isaac_train_v22` passes its mock check: `gz_v24_image` and `gz_v24_refly` with its run folder |

### Day 3 to 5: the week's second queue (not written yet; written from the results)

- If v24 is released: regenerate both module headers and the parts list, `training/policies/
  README.md`, the decision row (`CLAUDE.md` "Releasing a policy"), rebuild the kit image, and
  re-fly the evaluation kit's six field faults with v24 (`./evaluate.sh`, 12 flights, 3 min
  on the server).
- If Isaac passes M3: Isaac Lab's speed decides whether later recipes use more drones or
  longer episodes inside the hour rule (`docs/isaac-lab.md` "M3"). A seed sweep of v22's recipe
  on Isaac and on the mock (3 seeds each, 6 GPU runs of 45 min) says how much of a candidate's
  mock score is seed luck; today nobody knows.
- Gate 1 stress set and the Basalt-fitted estimate configs (`eval/configs/gate1_x500_ekf_*`)
  re-run on the GPU for v24 and the Isaac policy, 10 min each.
- If the pass policy passes `hold_pass_gate`: Gazebo pass scenarios with stock PX4 flying the
  same moving setpoint, the pass's real yardstick (`docs/surface-passes.md`).
- Heading error near steel (`training/queues/heading_error.yaml`, `docs/heading-error.md`): two
  heading-trained candidates (0.9 GPU-h each, side by side), their check against v22 (0.8 h),
  and 80 Gazebo flights of v22 and tuned PX4 with EKF2's heading 30 deg off or drifting
  (about 1.4 CPU-h). Every job under an hour, at most two GPUs: no separate yes needed.

### Route Follow toward a real test (its own queue, any time)

`training/queues/route_gazebo.yaml` (GPU plan item 7, `docs/route-follow.md` "Degraded
sensors" and "Gazebo") runs beside week 1 and needs none of its results: two seeds of the
pipes recipe with dust-spoiled rangefinders and a degraded VIO (`route_degraded`,
`route_degraded_s7`, 3 x 45 min chains on 2 GPUs), their 17-cell stress table, and on the CPU
lane stock PX4 Mission mode through the four Gazebo route worlds (`gz_route_clean` first as
the plumbing go/no-go, then 8 seeds each on `route_vio`, `route_drift`, `route_degrade`).
Run it with `--max-gpus 2`; `--dry-run --gpus 2` shows the plan.

## Week 2: the evidence buyers ask for (approved 2026-11-09)

Zain approved the eight-item plan for the server's second week on 2026-11-09 (the research
behind it: what inspection drone makers ask for, what the market research found missing, and
what one card can prove in under an hour). Nothing in it waits on the first week's results.
The first three items are one queue, `training/queues/server_week2.yaml`, every job under 1 h
on 1 GPU and the whole queue capped at 2 GPUs at once (Zain's limit on the shared machine);
it was smoke-tested on the cloud CPUs:

```
python -m training.queue training/queues/server_week2.yaml --dry-run --gpus 2
python -m training.queue training/queues/server_week2.yaml --smoke
python -m training.queue training/queues/server_week2.yaml --max-gpus 2
```

| Item | GPU hours | What it proves to a buyer | Where |
| --- | --- | --- | --- |
| 1. the hold datasheet | 3 x 0.9 | the wind at which Structure Hold and the PD break, per direction, distance and estimate quality: the number every datasheet quotes, which nobody publishes for hold accuracy; then the edge cells in Gazebo | `docs/hold-datasheet.md` |
| 2. failure rates per fault | 9 x 0.3 | a 95 % bound on the crash rate behind every FMEA row (20,480 episodes each), and the Gazebo field faults from 3 seeds to 8 | `docs/fault-rates.md` |
| 3. seed repeatability | 3 x 0.8 + 0.75 | how much of a check score is seed luck: v22's recipe on three more seeds through the Gazebo-like check | `docs/gazebo_check.md` |
| 4. other drone sizes | about 5 | the policy on a 250 g and a 2 kg airframe fitted by `eval/fit_airframe.py`: whether a buyer's drone needs retraining | own thread |
| 5. deck and pier undersides | about 4 | holds under a surface, the pier scenario the kit does not fly yet | own thread |
| 6. heading error | about 3 | holds with the estimate's heading off by 5 to 15 degrees (the `log_health.py` finding, flown) | `training/queues/heading_error.yaml`, `docs/heading-error.md` |
| 7. Route Follow on the GPU | about 5 | the pipes recipe with degraded sensors and stock PX4 Mission mode through the Gazebo route worlds, so the second capability has a server number | `training/queues/route_gazebo.yaml` (above) |
| 8. a real VIO in the loop | about 10, over 1 h | Basalt on EuRoC frames feeding the estimate instead of the model; needs Zain's yes per run and Isaac parity first | own thread |

Items 4, 5 and 8 need an environment or a world change before they can be queued; each is
its own thread and lands in a `server_week3.yaml` when ready. Where a week-2 result changes the
safety manual (the envelope in section 2, the failure table in section 5) it is a new manual
version for Zain to re-sign, not an edit in place.

## What needs Zain's yes, as one batch

`launch.py` and the queue refuse these without `--approved`. Everything else in the queue is
under 1 h and 1 GPU per run and runs without asking (the project rule; chains of sub-1 h
parts are fine).

| Job | Why over the limit | Ask |
| --- | --- | --- |
| `setup` (`server_setup.sh`) | About 1 h the first time, mostly the SITL image build and the 25 GB Isaac pull | Run it on arrival |
| Running the queue on all GPUs at once | Each job is 1 GPU, but the queue keeps every card busy, so the machine runs 4 GPU jobs side by side | Fill every free GPU (the point of the server), or cap it with `--max-gpus` |
| The three `route_pipes` chains | Each part is 45 min; a chain is three parts, about 2.4 h per seed, three seeds | Run them (the rule allows checkpointed chains; listed here because it is the biggest block of GPU time) |

Nothing in the queue spends money, installs system software, or changes a Decided row. The
v24 release, if its re-fly wins, is a separate decision card.

## Go/no-go checks, in one place

| Check | Pass | Fail means |
| --- | --- | --- |
| `report` | Driver 580+, all GPUs free, Docker without sudo, disk 80 GB+ | Admin steps first (`docs/gpu-server.md` section 1); nothing else starts |
| `setup` | `report.md` all PASS, hover test rtf about 1.0, parallel hovers each at real time | The failing step's log; the SITL lane waits, the GPU lanes can start if the GPU smoke passed |
| `isaac_parity` | `parity: PASS`; no Isaac SKIPs; v22 on Isaac p95 0.20 to 0.21 m, crash under 1 % | M3 stops; mock on the GPU is the decided fallback; tell Zain with the parity table |
| `v22_eval_mock_gpu` | p95 0.211 / 0.206 / 0.202 m to a few mm | The GPU torch build differs from the CPU one; stop the GPU scoring jobs until explained |
| `gz_tuned_baseline` | Mean p95 within about 5 % of the Mac's 0.427 / 0.424 / 0.428 m, `rtf_min` near the Mac's 0.85 | Fewer parallel jobs (`--jobs 8`), re-fly; the server's numbers do not replace the Mac's until this passes |
| `gz_v22_baseline` | 40 % or more below tuned, 0 crashes | A regression on main since the Mac batches; bisect before any new policy flies |
| `gz_v24_refly` vs `gz_v22_range` | v24 tighter on mean p95 and worst toward the wall, 0 crashes | v24 parked; the lift rule stands as the gate |
| `route_pipes_stress` | Best seed: narrow-pipe crashes under 11 %, beats both PDs on all 13 cells | More jumps in training did not help; next try is a rangefinder-weighted reward, not more iterations |
| `hold_pass_gate` | PASS against the PD on the pass cells, lower p95 than v22 in every cell | v22 as it is stays the pass controller to beat; look at the start and stop of each pass (a pass-velocity input is the next idea) |
| `gz_wind12_v22` | Holds at 10 and 12 m/s with 0 crashes, tighter than tuned | Switch on `hold_wind_finetune` |

## Where the numbers go

Every real run writes its own `experiments/LOG.md` line (launch.py) or `log_lines.md` to
paste (run_baseline); the server is named in the line (`SITL_HOST_LABEL` overrides it).
Results that change a page go to that page: `docs/baseline.md` (the server's benchmark row),
`docs/wall-range.md` (the v24 re-fly), `docs/route-follow.md` (the pipes table),
`docs/market-research.md` row 15 and `docs/field-faults.md` (the wind cells),
`docs/isaac-lab.md` (parity and M3). Policies that pass go to `training/policies/` with a
README row; everything else stays in `runs/`.
