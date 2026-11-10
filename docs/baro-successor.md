# A v22 successor on barometer height

v22, the released policy, trained and was gated with the estimate's height coming from the
vision fix. The benchmark and the safety manual fly on barometer height (`EKF2_HGT_REF` 0,
Decided 2026-11-01). The cloud's 8-seed re-fly on that set-up scored v22 at 0.29 to 0.33 m
p95 (against the setpoint as logged, the scoring until 2026-11-10) against 0.19 to 0.20 m
on vision height, and the client-needs review asked for a
candidate trained on the barometer to win that back. This page is that candidate: whether it
is still needed, what already covers it, the recipe, the check and the server queue.

Status (2026-11-09): configs and queue written and smoke-tested on CPU. **Parked: not sent
to the server**, for the two reasons in the next section. Nothing is trained, and nothing here is
released on a mock score (CLAUDE.md "Releasing a policy").

## Is it still needed?

Probably not: the loss it was asked to win back is not there, and the retraining it
would do has in effect been done.

While this was being written the cloud benchmark thread split its 48 barometer flights into
horizontal and vertical error and compared EKF2's height with Gazebo's true height (its
tables and the tool, `eval/height_split.py`, go into `docs/baseline.md` with that thread's
PR; until it is merged this is its report, not a logged result here):

- On barometer height EKF2's height sits a constant 0.16 to 0.23 m off the true height, and
  nearly all of that offset is already there on the ground before take-off. The scorer
  measures the true position against the commanded point, so it counts that offset as hold
  error, for tuned PX4 and for v22 alike.
- Inside a hold the barometer height is no worse than the vision height: it wanders less,
  both controllers show the same, and v22 does not react badly to it.
- With the offset taken out, v22 holds on the barometer what it holds on vision height, to
  within a centimetre.

So the looser number was the scoring frame, not the policy, and there is no loss for a
retrained policy to win back. A policy holds the estimate; an offset between the estimate
and the truth is the same for any policy. The fix belongs in the scorer (or the estimate's
frame), which is the cloud thread's work, not in training.

The mock had said the same all along, which is why it "did not reproduce" the loosening: in
the mock v22 holds a little tighter on the barometer than on vision height (rough cells
0.176 m against 0.196 m, `docs/baseline.md` "The mock fusion on barometer height"), and the
cloud flights found the fitted barometer walk (0.005 m/sqrt(s)) matches what EKF2 does.

**It has also been run once already, in effect.** The seed-repeatability job of week 2
(`docs/gazebo_check.md` "Seed repeatability", the same day) retrained v22's own recipe on
seeds 1 to 3, and since `gz_x500_sitl` fuses on the barometer those three trained on
barometer height. In the check they hold 6 to 10 % tighter than v22 with no crashes, the
seeds differ from each other by about 5 %, and one of the three passes every rule. That is
the answer this queue would give: v22 never trained on the barometer's behaviour (no height
resets, no frozen height), and a policy that did holds a centimetre or two tighter in the
mock. This queue continues v22 instead of retraining it from v19 and uses four times the
drones, so its numbers would differ in detail, not in kind.

What a 6 to 10 % gain in the mock is worth:

- It is at the edge of one seed's luck, which is why the recipe now needs three seeds
  (CLAUDE.md "Releasing a policy") and why this queue has three.
- Earlier candidates kept from none to half of their check gain in Gazebo
  (`docs/gazebo_check.md`), so expect a centimetre or less on a 0.20 m hold there. An 8-seed
  re-fly cannot tell that from Gazebo's own scatter between two flights of one seed.
- v24, the rangefinder candidate, is outside the luck band against v22 (ratio 0.77 on the
  server) and is already in line for the cloud re-fly. It is the successor worth a re-fly.

So the queue stays ready and is run only if Zain asks for it, or as the control when an
airframe, bridge or heading candidate beats v22 and the question "was it the barometer?"
comes up. v22's own row on this check is in `experiments/LOG.md` (run
`20261109-200353_gazebo_check_v28`: pooled rough p95 0.176 m, the same as in the
seed-repeatability check).

## Does it need its own candidate line?

No. Barometer height is the base every line trains on now, not a third line:

- The mock's `gz_x500_sitl` airframe has fused on the barometer since 2026-11-03
  (`fusion.height_source: baro`, `envs/estimate/ekf.py`). Every v22 continuation written
  since trains on it without saying so: wind (`structure_hold_ppo_v26_wind.yaml`), bridge
  decks (`..._v27_bridge.yaml`), heading error (`..._heading.yaml`), surface passes
  (`..._pass.yaml`) and the airframe sizes (`..._v27_airframes.yaml`, PR #149). Each of them
  changes one more thing (the wind, a deck, a wrong heading, a moving hold point, a wide
  airframe spread), so none of them says what the barometer alone gives.
- **v28** here is that missing control: v22's recipe with only the height source moved.
  (v27 already names two candidates, so this one is v28.)
- **v24**, the rangefinder candidate, trained on vision height. The server scored it on the
  barometer as it is and it is waiting for its cloud Gazebo re-fly. A barometer step should
  start from whichever policy is the release: if v24 is released, it continues v24, not v22.
  That config and its jobs are in the queue, switched off.

## The recipe

`training/configs/structure_hold_ppo_v28_baro.yaml`: v22's recipe
(`structure_hold_ppo_v22.yaml`) from v22's ONNX (`params.init_actor_onnx`, fresh critic
warmed up 200 iterations), the same 21 inputs, reward, estimate fault rates and airframe
randomisation, so the released module header could take the result unchanged. The height
source is stated in the config (`env.fusion: {height_source: baro}`) rather than left to the
airframe preset. GPU-sized as the other continuations: 4096 drones, 800 iterations, one part
under 1 h. Three seeds: a margin inside about 10 % is within one seed's luck.

`training/configs/structure_hold_ppo_v28_baro_range.yaml` is the same step for v24 (its
recipe, its 23 inputs, from its ONNX).

## The check

`eval/configs/gazebo_check_v28.yaml` is the Gazebo-like check (`docs/gazebo_check.md`) with
v22 the reference and the candidates added by name by the queue (`params.candidates`). Its
cells and rules are `gazebo_check_v24.yaml`'s; the barometer height is stated in `base_env`,
so the preset cannot move the check. The barometer's walk and noise stay at their fit.

## The server queue

`training/queues/baro_successor.yaml` (GPU only, no SITL image, no Gazebo):

```bash
python -m training.queue training/queues/baro_successor.yaml --dry-run --gpus 2
python -m training.queue training/queues/baro_successor.yaml --smoke
python -m training.queue training/queues/baro_successor.yaml --max-gpus 2
```

| Job | What | Hours |
| --- | --- | --- |
| `baro_s0`, `baro_s1`, `baro_s2` | the recipe, seeds 0 to 2, one GPU each | 0.9 each |
| `baro_check_s0` to `_s2` | each candidate against v22 in the check, one GPU each | 0.8 each |
| `gz_baro_candidate` (off) | a note: the re-fly is the cloud's | – |
| `baro_range_*` (off) | the same for v24: three trainings, each judged against v24 with and without a rangefinder | 0.9 and 0.8 each |

Every job is under 1 h on one GPU and at most two GPUs are busy at once, so nothing needs a
separate yes. About 5.1 GPU-hours, done about 2.6 h after the start on two free GPUs.

**Go/no-go:** a seed passes if its check says it beats v22 with every rule passed: pooled
p95 lower, lift, ring and fast ring not higher, no more crashes, the three seed replays no
worse, the re-engage clean, `calm_clean` within 0.02 m. The recipe goes on only if at least
two of the three seeds pass; a rule one seed fails alone is a watch item, not a verdict
(`docs/gazebo_check.md` "Seed repeatability"). The seed with the lowest pooled p95 among
those that pass is the one to fly.

## After the check, if it is run

- **The recipe passes:** the chosen seed flies the benchmark in the cloud (Gazebo flies there,
  `docs/gpu-server.md`): export its header and build an image as `server_week1`'s
  `gz_v24_image`, then three scenarios on seeds 1 to 8 with the hold runner on
  `px4_tuned_v2.params`, one flight at a time, against the cloud's tuned PX4 and v22 batches
  on the same image, scored with the frame offset taken out. It is released only if Gazebo
  confirms it, and that is Zain's decision; then the steps in CLAUDE.md "Releasing a policy".
- **It does not:** v22 stays and the result goes in `docs/gazebo_check.md` "Policies judged".
- **v24 is released first:** switch the `baro_range_*` jobs on (`enabled: true`) and run the
  queue again; it skips what already finished.

Runs are logged in `experiments/LOG.md` by `training/launch.py`, the check against v22 and
the training's own score against the cascaded PD in the mock.
