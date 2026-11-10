# Route Follow in Gazebo

Route Follow's SITL side (`docs/route-follow.md`, "Gazebo"): fixed routes through tunnels
and corridors as Gazebo worlds, and stock PX4 Mission mode flying them GPS-free on the
corrupted estimate, scored on the true position against the same walls the mock uses. This
is the classical baseline a Route Follow policy has to beat before it becomes a PX4 mode
(Decided 2026-11-01).

| File | What it is |
| --- | --- |
| `routes/*.yaml` | The benchmark routes: `tunnel_bend` (3 m tube, S-bend), `corridor_ramp` (3 x 2.6 m corridor, two 90 degree turns, a 5 degree ramp), `pipe_narrow` (1.8 m pipe, 0.5 m/s), `drift_arched` (arched mine drift with a stretch narrowed to 70 %) |
| `world.py` | Loads a route file into the mock's own `Routes` (same speed plan) and writes its Gazebo world from `sim2real/gazebo/worlds/wall_slab.sdf` (same physics, systems, wind plugin) |
| `walls.py` | The mock's route geometry (`envs/route_follow/geometry.py`) for one route in NumPy: clearance, beams, cross-track, the moving target; pinned to the torch version by `tests/test_walls.py` |
| `mission_flight.py` | Runs in the SITL image: uploads the mission (MAVSDK), takes off from the tunnel floor, flies it, holds, lands |
| `run_mission.py` | Host runner: builds each flight's world, mission, wind and estimate files, flies it in Docker (`--jobs` for several at once), scores the ulog, writes the batch table and LOG.md lines |

The estimate profiles are `sim2real/estimate_bridge/profiles/route_*.yaml`: `route_clean`
(the plumbing check), `route_vio` (the mock's tunnel estimate, the baseline), `route_drift`
(drift doubled) and `route_degrade` (20 s of dust or fog).

```
# plan only (no Docker): every route's world and mission
python -m sim2real.route_follow.run_mission --label check --dry-run \
    --params eval/baselines/px4_tuned_v2.params
# one flight
python -m sim2real.route_follow.run_mission --label tuned_mission \
    --params eval/baselines/px4_tuned_v2.params --routes tunnel_bend --seeds 1
# the baseline batch (GPU server CPU lane; training/queues/route_gazebo.yaml)
python -m sim2real.route_follow.run_mission --label tuned_mission --jobs auto \
    --params eval/baselines/px4_tuned_v2.params --profile route_vio --seeds 1 2 3 4 5 6 7 8
```

Design points, each explained in its module docstring:

- **The walls are drawn, not solid.** Contact is judged on the true position against the
  route's exact walls (0.40 m, the x500, as the mock's configs), so the Gazebo score and the
  mock score are one measurement.
- **The world is generated per flight** and mounted into the image, so the existing SITL
  image flies every route; no rebuild.
- **The floor is the ground:** the drone takes off from the floor of the first segment, and
  the scored window starts when PX4 enters Mission mode.
- **The draft blows along the first segment** (Gazebo's wind plugin has one direction; the
  mock turns it along the route).
- **Each waypoint's acceptance radius is 0.5 m** (a tunnel plan; PX4's 2 m default would cut
  every corner into the wall) and every corner is fly-through, as QGroundControl plans it.
- **Crash is a wall contact or a mission PX4 never finished.** Ending more than 0.5 m from
  the last waypoint (the estimate drifted) is "not completed", not a crash, as in the mock.
  `--rescore BATCH_DIR` re-scores a batch's ulogs after a scoring change.
- **A benchmark batch flies one flight at a time, in seed halves.** Each half stays under
  an hour; `--pool BATCH_DIR ... --out DIR` merges them into one batch, and `--replace` lets
  a re-flight (a later batch) take the place of a flight of the same route and seed.
- **The real-time factor is part of the result.** Start `sim2real/docker/rtf_sampler.sh`
  beside a batch: it writes `rtf_samples.txt` into each flight's folder, and the tables show
  each flight's mean and worst sample and flag any under 0.9 (Gazebo's start-up samples are
  skipped). Three flagged flights re-flown at real time gave the same numbers, so the flag
  is a prompt to look, not proof the flight is wrong.
- **Arming is tried again for 30 s** when PX4 denies it (the heading estimate can settle a
  moment after the health stream says armable). A flight that never arms is "not flown",
  not a crash.
- The host side needs the `torch` extra, as `envs/route_follow` imports it.
