# Near-wall flow

The disturbance model (`envs/disturbance`) used to know one thing about walls: the wind
fades near them (the shelter factor `g(d)`, fitted to the one published data point). Real air
next to a structure does more than fade. This page says what was added, where the numbers
come from, and what it does to v19 and the PD yardstick in the mock.

All four terms are **off by default**. They draw their random numbers after everything
else, so a config that does not turn them on flies bit-identical air to before (tested in
`envs/disturbance/tests/test_near_wall.py`), and every earlier result still reproduces.

## What is modelled

The drone sits at distance `d` from one flat surface whose normal `n` points into the free
air. `w` is the free-stream wind (mean + gusts + turbulence) and `w_n = w . n` its part along
the normal: negative blows into the surface (the drone is on the windward side), positive
blows away from it (the drone is in the structure's lee). The air the drone feels is

    air = g(d) w                                         shelter, as before
        + deflect * (1 - g(d)) * max(-w_n, 0) * t        windward deflection
        - recirc * (d / Lr) exp(1 - d / Lr) * max(w_n, 0) * n   leeward recirculation
        + exp(-d / Lw) * (wake turbulence + shedding * s)        wake

and the force is drag on `air - velocity`, as before.

| Term | What it is | Config keys | Eval range | Why that range |
| --- | --- | --- | --- | --- |
| Windward deflection | Wind blowing into a wall stops at it and turns along it: the stagnation flow that becomes upwash above the stagnation point of a facade (about two-thirds up a building face) and downwash below it. The blocked part of the wind, `(1 - g) * |w_n|`, partly reappears along the wall in direction `t`. | `wall_deflect`, `wall_deflect_angle_deg` (0 along the wall horizontally, 90 up; default any direction, because where on the face the drone sits decides up or down) | 0.3 to 0.8 | Potential flow turns all of it; viscous losses and flow escaping round the edges take some. |
| Leeward recirculation | Wind blowing away from the structure leaves a separation bubble behind it, with reverse flow back toward the wall. Zero at the surface, strongest at `Lr`. | `wall_recirc`, `wall_recirc_length` | 0.2 to 0.5 of the normal wind, `Lr` 0.5 to 3 m | Reverse flow of a few tenths of the free stream is typical behind surface-mounted bluff bodies (for example the cube measurements of Martinuzzi and Tropea, 1993); the bubble scales with the structure, so `Lr` spans the 1 to 3 m hold distances. |
| Wake turbulence | Turbulence is stronger near structures (shear layers, separated flow) and scales with the wind speed, unlike the free-stream turbulence that the shelter fades out. Standard deviation `wake_ti * |mean wind|` per axis at the wall, same spectrum as the turbulence, its own phases. | `wake_ti`, `wake_length` | 0.1 to 0.3, `Lw` 1 to 4 m | Turbulence intensities of 20 to 40 % near buildings and piers against 10 to 15 % in open terrain. |
| Vortex shedding | A finite structure (a pier, a column, a deck edge) sheds vortices alternately from its two sides, so the drone in its lee sees a sideways wind that swings back and forth at `f = St * U / D`. Amplitude `shed_amp` times the mean wind blowing away from the wall. | `shed_amp`, `shed_width` (D), `shed_strouhal` (St), `shed_angle_deg` (default 0: sideways, as behind a vertical pier) | 0.1 to 0.4, D 0.5 to 5 m, St 0.1 to 0.21 | St is about 0.2 for a circular cylinder over a wide speed range, 0.1 to 0.15 for box-like bridge decks. At 8 m/s past a 1 to 2 m pier this is 0.8 to 1.7 Hz, inside the band the hold loop works in, which is why it matters. |

Not modelled: variation along the surface (every term depends on distance only, so the
whole wall behaves like the stagnation region or the bubble), more than one surface at once
(a wall and a deck), and the near-wall yaw torque. The ranges are literature-shaped, not
fitted; the Phase 3 fan rig fits them, as it does the shelter.

### Why not CFD now

Precomputed CFD wind fields around a wall, a pier and a deck would pin these profiles down
for specific geometries. They need a CFD set-up (OpenFOAM or similar) and a field lookup in
both the torch model and the Gazebo plugin, and the result would still be a guess until the
fan rig measures the real drone. The closed-form terms give the controllers the same kinds
of air (deflected, reversed, gustier, periodic) now, at no cost per step, and a fitted
lookup can replace any one profile later behind the same `air_velocity` function. Parked in
`docs/decisions.md`.

## Where it runs

- **Mock and training**: set the keys under `disturbance:` in any env config. With the flow
  off the extra cost per step is a few elementwise operations (no wake sinusoids are kept).
- **Isaac Lab**: the same `torch_adapter.disturbance_force`, so nothing extra to port.
- **Gazebo**: the plugin's C++ copy (`sim2real/gazebo/plugin/Disturbance.hh`) has the same
  terms; the parity test checks it against NumPy with the flow on. Two scenarios turn it on:
  `wall1m_wind_8ms_flow` (into the wall) and `wall1m_lee_8ms_flow` (the drone in the lee).
  Params files written before the flow read as the flow off. A Gazebo re-fly needs the Mac
  (and an image rebuild); it has not been flown yet.

## Effect on v19 and the PD yardstick (mock)

`eval/configs/near_wall_flow_v19.yaml`: the Gate 1 x500 fused set, 3 seeds (9101 to 9103)
x 512 episodes per cell, with the mean wind's heading fixed into the wall (0) or away from
it (180) so the windward and leeward terms each get cells, and the shelter-only cells beside
them. The yardstick is the cascaded PD (`envs/structure_hold/baseline.py`); tuned PX4 exists
only in Gazebo.

Run `20261028-010943_near_wall_flow_v19_s9101_11228b3` (commit 11228b3, 26 min on 4 CPU
cores). Mean p95 error over 1536 episodes per cell (lower is better), crash rate after it:

| Wind, distance | Shelter only: v19 | Shelter only: PD | Near-wall flow: v19 | Near-wall flow: PD | v19 vs PD with the flow |
| --- | --- | --- | --- | --- | --- |
| 8 m/s into the wall, 1 m | 0.225 m, 0.1 % | 0.350 m, 2.5 % | 0.229 m, 0.1 % | 0.382 m, 3.3 % | -40 % [-42, -37] |
| 8 m/s away from the wall, 1 m | 0.250 m, 0.1 % | 0.395 m, 4.3 % | 0.238 m, 0.1 % | 0.349 m, 3.2 % | -32 % [-35, -28] |
| 8 m/s any heading, 1 m | 0.240 m, 0.1 % | 0.379 m, 3.6 % | 0.241 m, 0.1 % | 0.382 m, 2.9 % | -37 % [-39, -34] |
| 8 m/s into the wall, 2 m | | | 0.234 m, 0.1 % | 0.441 m, 0.0 % | -47 % [-49, -45] |
| 8 m/s away from the wall, 2 m | | | 0.245 m, 0.1 % | 0.383 m, 0.0 % | -36 % [-39, -33] |
| 4 m/s into the wall, 1 m | | | 0.227 m, 0.1 % | 0.328 m, 3.5 % | -31 % [-34, -27] |
| 4 m/s away from the wall, 1 m | | | 0.231 m, 0.1 % | 0.324 m, 4.4 % | -29 % [-32, -25] |

What it says:

- v19 holds tighter than the PD in every cell, with the flow and without, and crashes less
  (0.1 % against 2.8 % pooled). The flow costs v19 at most 2 % of p95 (windward, 1 m).
- Windward, the deflected wind and wake turbulence hurt the PD more (+9 %) than v19 (+2 %).
- In the lee the flow makes both a little easier, not harder: the reverse flow in the
  bubble cancels part of the push away from the wall that the shelter-only model gives.
  Shedding at 0.3 to 3 Hz did not bring out a ring in v19 (rate roughness unchanged).
- With these ranges the extra force is small next to the drone's weight: drag on a 1 to
  3 m/s crossflow on top of an 8 m/s wind is a few tenths of a newton for the x500. If the fan
  rig measures stronger near-wall forces than the fitted drag gives, these terms are where to
  turn them up (and the first thing to re-run is this config).

This is the mock and the PD yardstick. The tuned-PX4 comparison is in Gazebo, below.

## Gazebo: v19 against tuned PX4 under the flow

Flown on the Mac on 2026-10-28 at `6eb3c3d` (image built the same day from sources identical
to main, so it carries the flow in the wind plugin; hover test passed). Scenarios
`wall1m_wind_8ms_flow` (8 m/s into the wall) and `wall1m_lee_8ms_flow` (8 m/s away from it,
the drone in the lee), seeds 1 to 3, 60 s hold 1 m from the wall. v19: hold runner,
`vio_at_engage`, batch `runs/20261028-134426_px4_baseline_v19_flow_6eb3c3d`. Tuned PX4:
`eval/baselines/px4_tuned_v1.params`, `vio_at_hover`, batch
`runs/20261028-135808_px4_baseline_tuned_v1_flow_6eb3c3d`. The wind trace check passed in all
12 flights: Gazebo applied the same force as the Python model with the flow on.

Mean over seeds 1 to 3 (worst in brackets), tuned PX4 → v19:

| Scenario | p95 (m) | Worst toward wall (m) | rms (m) | Crashes |
| --- | --- | --- | --- | --- |
| 8 m/s into the wall, flow | 0.533 (0.916) → 0.263 (0.286), -51 % | 0.255 (0.432) → 0.134 (0.291), -47 % | 0.216 → 0.127, -41 % | 0/3 → 0/3 |
| 8 m/s in the lee, flow | 0.535 (0.918) → 0.258 (0.292), -52 % | 0.264 (0.484) → 0.107 (0.270), -60 % | 0.217 → 0.130, -40 % | 0/3 → 0/3 |

Per seed (p95, m): into the wall v19 0.230 / 0.286 / 0.273, tuned 0.916 / 0.288 / 0.395; in
the lee v19 0.218 / 0.292 / 0.263, tuned 0.918 / 0.282 / 0.405. Seed 2 is a tie, seeds 1 and 3
are v19's. Seed 3 is still the closest approach to the wall for both (min clearance v19
0.71 / 0.73 m, tuned 0.57 / 0.52 m); no contacts.

The shelter-only benchmark (`docs/baseline.md`, 8 seeds, PR #67) had v19 at -47 % on
`wall1m_wind_8ms`; with the flow it is -51 to -52 % on seeds 1 to 3. The seed sets differ,
so this says the margin held, not that it grew.

Not about the hold, noted for the module threads: in v19's into-the-wall seed 3 flight, the
landing after the hold timed out (90 s waiting for touchdown). And in both seed 1 flights
the flight script's own `position_held` check failed (about 1.5 m drift reported against a
true p95 of 0.22 m); it probably reads the jumpy `vio_at_engage` estimate, not truth.
Neither was looked into here.

Followed up (2026-10-28): the seed 3 timeout is not Structure Land's. After the scripted
watchdog trip, stock Position mode followed the recovering estimate jump into the wall
(true gap 0.91 m at 93 s down to 0.28 m at 101 s) and struck it about 1 s before the land
command; Structure Land engaged on a tumbling vehicle, tripped on tilt at once and handed
back to Position mode, which hovered 0.12 m up and later flipped, so touchdown never
registered. This batch flew without `--rangefinder`, so neither the wall guard nor
Structure Hover could see the wall; the lee flight on the same seed walked in more slowly
and landed 0.37 m off it. The seed 1 `position_held` failures were the injected mid-hold
jump (1.5 m in the estimate, truth within 0.26 m); `run_gazebo.py` now judges the hold
checks on truth and keeps the estimate's verdict under `estimate_checks`.
