# Holds under a bridge deck and beside a pier

The pitch says "under bridges"; until November 2026 the evidence was one flat wall. This page
covers item 5 of the GPU server plan for client needs: the air under a deck and in a pier's
wake in the wind model, two Gazebo worlds to fly it in, v22 against the PD yardstick in the
mock, and the server queue that flies v22 against tuned PX4 there
(`training/queues/bridge_holds.yaml`).

## What was added to the wind model

`envs/disturbance` knew one surface. A hold under a bridge has two: the abutment or pier the
drone faces, and the deck over it. The model now takes an optional **second surface**
(`sample_params(..., surface2_point, surface2_normal, surface2_across)`), placed the same way
as the wall, with two effects:

- **Shelter.** The deck slows the wind near it as the wall does, `g2(d2) = 1 - (1 - g0) *
  exp(-d2 / L)`, drawn from its own ranges (`surface2_decay`, `surface2_floor`, by default
  the wall's). The two factors multiply: `air = g_wall * g_deck * wind + near-wall flow`.
- **Deck-edge vortex shedding.** A bluff deck in a crosswind sheds vortices from its edges
  at `f = St * U / D` (St 0.1 to 0.15 for box decks, D the deck's depth). Under the deck
  the drone feels it mostly as an up-and-down wind: amplitude `surface2_shed_amp` times the
  mean wind across the deck, along the deck's normal, decaying as
  `exp(-d2 / surface2_wake_length)`. "Across" is the horizontal direction perpendicular to
  the bridge's axis (`surface2_across`); wind along the bridge sheds nothing. Off by default
  (`surface2_shed_amp` 0).

A pier needs nothing new: it is the first surface with the existing vortex shedding on, its
width set to the pier's (`shed_width` 1.5 m, `shed_strouhal` 0.13 to 0.2, square to rounded
sections), the drone in its wake. At 8 m/s that is 0.7 to 1.1 Hz, inside the band the hold
loop works in.

Without a second surface nothing changes: no extra random draws, a shelter factor of
exactly 1 and a shedding term of exactly 0, so every earlier config and scenario flies
bit-identical air (`envs/disturbance/tests/test_surface2.py`). The Gazebo plugin's C++ copy
has the same terms and the parity test checks it against NumPy to 1e-9 N with a deck in
(`sim2real/gazebo/tests/test_parity.py`). Params files written before it read as no second
surface.

Not modelled: the rotor **ceiling effect** (thrust rises within a few rotor radii of a
surface overhead). At the 1 to 3 m holds the envelope covers, the x500's rotors are six or
more radii from the deck, where published gains are a few per cent at most; parked in
`docs/decisions.md` for holds under 1 m. Nor the deck's own deflection, recirculation and
wake turbulence (the first surface's near-wall flow is unchanged and still the abutment's).
The ranges are literature-shaped, not fitted, like the rest of `docs/near-wall-flow.md`.

## The mock task

`StructureHoldConfig.deck_clearance` puts a deck that many metres over each episode's hold
point (drawn per episode; `None`, the default, is open sky). It enters the wind as the
second surface, its across direction world y (along the wall, as in `bridge_deck.sdf`), and
coming within `contact_radius_m` of it ends the episode as a contact, as the wall does. No
new policy input: v22 flies it as it is.

## The Gazebo worlds and scenarios

| World | Structure | Scenarios |
| --- | --- | --- |
| `bridge_deck.sdf` | Abutment face at x = 0, the deck it carries with its underside at z = 4 m, 1 m deep, 14 m out over the free air and 20 m wide. The bridge's axis is x, so wind along y blows across the deck. | `deck_corner_8ms_flow`: 1 m from the abutment, 1.5 m under the deck, 8 m/s at 45 degrees (into the abutment and across the deck), all flow terms. `deck_open_8ms_flow`: 6 m out, 1.5 m under the deck, 8 m/s across the deck. |
| `bridge_pier.sdf` | A 1.5 m square pier, face at x = 0, 6 m tall, under a deck at z = 6 to 7 m along y. | `pier_lee_8ms_flow`: 1 m from the face, 8 m/s blowing away from it, the drone in the pier's wake with shedding at 0.7 to 1.1 Hz. |

A scenario now names its `world` (default `wall_slab`, so every older scenario flies where it
did) and an optional `surface2` block (`point`, `normal_toward`, `across`). The runners
(`run_scenario`, `run_estimate`, `run_gazebo`, `run_baseline`, `autotune`) start PX4 in the
scenario's world, and `eval.metrics` scores the second surface too:
`min_clearance_surface2_m`, `worst_toward_surface2_m`, and a contact with either surface
counts as a crash. The pier scenarios leave the deck 3.5 m overhead out of the wind model;
the forward rangefinder of the estimate bridge sees the pier face as a plane, which it is
while the hold stays within 0.75 m sideways.

## v22 against the PD in the mock

`eval/configs/bridge_holds_v22.yaml`: the Gate 1 x500 fused set on barometer height with
seven bridge cells, 3 seeds (9111 to 9113) x 512 episodes per cell. The yardstick is the
cascaded PD (`envs/structure_hold/baseline.py`); tuned PX4 exists only in Gazebo (below).
Run `20261109-173246_bridge_holds_v22_s9111_8026ac2` (commit 8026ac2, 19 min on 4 CPU cores).
Mean p95 error over 1536 episodes per cell (lower is better), crash rate after it:

| Cell | v22 | PD | v22 vs PD |
| --- | --- | --- | --- |
| Deck 1.5 m over, 1 m from the abutment, 8 m/s at 45 degrees | 0.180 m, 0.0 % | 0.334 m, 2.7 % | -46 % [-48, -44] |
| Deck 1.0 m over, 1 m from the abutment, 8 m/s at 45 degrees | 0.178 m, 0.0 % | 0.316 m, 2.5 % | -44 % [-46, -41] |
| Deck 1.5 m over, 6 m out, 8 m/s across the deck | 0.202 m, 0.0 % | 0.398 m, 0.0 % | -49 % [-51, -47] |
| Deck 1 to 3 m over, 1 to 3 m out, 8 m/s any heading | 0.192 m, 0.0 % | 0.363 m, 0.3 % | -47 % [-49, -45] |
| Pier wake, 1 m from the face, 8 m/s | 0.190 m, 0.0 % | 0.334 m, 2.7 % | -43 % [-46, -40] |
| Pier wake, 2 m from the face, 8 m/s | 0.202 m, 0.0 % | 0.370 m, 0.0 % | -45 % [-48, -43] |
| Pier wake, 1 m from the face, 10 m/s | 0.201 m, 0.0 % | 0.355 m, 3.1 % | -43 % [-46, -41] |

v22 holds tighter than the PD in all seven cells (pooled 0.19 against 0.35 m) and never
crashes (the PD: 1.6 % pooled, its contacts at 1 m from the abutment or pier face). v22's
numbers sit where they do at the flat wall with the near-wall flow (0.18 to 0.20 m on the
barometer Gate 1 set), so the deck and the pier's wake do not cost it a measurable amount in
the mock, and the fine-tune (`structure_hold_ppo_v27_bridge.yaml`) is not needed on this
evidence. The Gazebo flights against tuned PX4 decide it.

### A first look in Gazebo (cloud image, one seed)

Flown in the cloud container on 2026-11-09 on an image built there from this branch
(`cloud_build.sh`, PX4's published Gazebo base, so a preview, not the benchmark), seed 1:
v22 (hold runner, `vio_at_engage`, `--rangefinder`) and tuned PX4 (`vio_at_hover`,
`px4_tuned_v2.params`). The wind trace of a `deck_open_8ms_flow` hover matched the Python
model to 1e-15 N.

| Scenario, seed 1 | v22 p95 | Tuned PX4 p95 | Crashes |
| --- | --- | --- | --- |
| `deck_corner_8ms_flow` | 0.265 m (1.52 m under the deck at the closest) | 0.906 m | 0 and 0 |
| `pier_lee_8ms_flow` | 0.221 m | 0.881 m | 0 and 0 |

Seed 1 is one of tuned PX4's bad seeds at the flat wall too (0.92 m in
`docs/near-wall-flow.md`), so this says the worlds and runners work, not by how much v22
wins; the 8-seed batches on the server do that.

### Eight seeds in Gazebo (2026-11-09, cloud)

`gz_bridge_tuned` and `gz_bridge_v22` of the queue, flown in a cloud container on the
`cloud_build.sh` image rebuilt from main with the two worlds (Gazebo benchmarks fly in the
cloud since 2026-11-09, `docs/gpu-server.md`): one flight at a time (`--jobs 1`), each batch in
two seed halves pooled with `run_baseline --pool`, 24 flights each. Tuned PX4 is
`px4_tuned_v2.params` in Position mode on `vio_at_hover`; v22 is the hold runner on
`vio_at_engage` with `--rangefinder`. Mean p95 error over 8 seeds (worst seed in brackets):

| Scenario | Tuned PX4 | v22 | Change | Furthest toward the face, worst seed (tuned, v22) | Crashes |
| --- | --- | --- | --- | --- | --- |
| `deck_corner_8ms_flow` | 0.493 m (1.131) | 0.351 m (0.523) | -29 % | 0.48 m and 0.18 m | 0 and 0 |
| `deck_open_8ms_flow` | 0.492 m (1.110) | 0.318 m (0.526) | -35 % | 0.48 m and 0.22 m | 0 and 0 |
| `pier_lee_8ms_flow` | 0.520 m (1.140) | 0.384 m (0.492) | -26 % | 0.49 m and 0.16 m | 0 and 0 |

| p95 per seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deck corner, tuned | 0.905 | 0.179 | 0.472 | 0.281 | 0.316 | 1.131 | 0.306 | 0.358 | 0.337 |
| Deck corner, v22 | 0.384 | 0.205 | 0.523 | 0.483 | 0.176 | 0.332 | 0.397 | 0.307 | 0.358 |
| Deck open, tuned | 0.925 | 0.173 | 0.476 | 0.378 | 0.288 | 1.110 | 0.328 | 0.261 | 0.353 |
| Deck open, v22 | 0.267 | 0.321 | 0.389 | 0.265 | 0.146 | 0.526 | 0.333 | 0.300 | 0.311 |
| Pier wake, tuned | 0.898 | 0.370 | 0.525 | 0.311 | 0.224 | 1.140 | 0.324 | 0.369 | 0.369 |
| Pier wake, v22 | 0.407 | 0.306 | 0.482 | 0.492 | 0.241 | 0.339 | 0.454 | 0.352 | 0.380 |

What it says:

- **The go check passes**: v22's mean p95 is below tuned PX4's in all three scenarios and
  neither crashed in 48 flights, so the fine-tune (`bridge_ft_*`) stays off.
- **The margin is 26 to 35 % and it is not even across seeds.** The last flat-wall benchmark
  had v22 43 to 45 % below tuned PX4 (`docs/baseline.md`), but it was flown before
  `EKF2_HGT_REF` moved to 0 and is being flown again, so the two are not yet comparable.
  Tuned PX4 itself does here what it does at the flat wall on current main (0.49 to 0.53 m,
  `experiments/LOG.md` 2026-11-09 17:35): the deck and the wake cost it nothing measurable.
  The gain comes from tuned PX4's two bad seeds, 1 and 6 (0.9 to 1.1 m in every scenario), where
  v22 holds 0.27 to 0.53 m. Over the other six seeds the two are level (deck corner 0.32
  against 0.35 m for v22, deck open 0.32 against 0.29 m, pier wake 0.35 against 0.39 m), the
  medians are level, and v22 is the tighter of the two on 4, 5 and 5 of the 8 seeds.
- **What v22 does buy here is the bad case and the distance to the structure.** Its worst
  seed is half of tuned PX4's (0.49 to 0.53 m against 1.1 m), and it comes 0.06 to 0.08 m
  toward the abutment or pier face on average (worst 0.22 m) where tuned PX4 comes 0.20 m
  (worst 0.49 m, half of the 1 m standoff). Under the deck both keep their height margin:
  the closest either came to the underside was 1.19 m (tuned PX4) and 1.33 m (v22) of the
  1.5 m asked for.
- v22's error here is as much height as position: its largest height error in a hold was
  0.26 to 0.35 m on average (0.5 m at worst) against 0.22 to 0.23 m sideways, and larger
  than the sideways one in 15 of the 24 flights. v22 was trained and gated on vision height
  and flies here on the barometer (`docs/baseline.md`), which probably matters more to these
  numbers than the deck's or the pier's air does; these flights do not separate the two.
- The mock had v22 at 0.18 to 0.20 m in these cells (about 45 % under the PD). Gazebo's
  0.32 to 0.38 m is 1.6 to 2 times that, so the mock's verdict (holds under a deck and in a
  wake, no crash) carries and its sizes do not.

Run quality. The real-time factor was sampled every 15 s inside the container (295 samples,
mean 0.99) and by the stock runner over each hover. Six flights showed a momentary dip (four
tuned PX4 flights with a runner minimum of 0.57 to 0.79 or a sample of 0.33, two v22 flights
with a sample of 0.38 and 0.81) and were flown again. Then the bridge logs, which print the
simulation time every 10 s of wall time, showed that no flight of the day ran slow for any
length of time: every 10 s stretch of all 136 flights (these and the heading batches) ran at
0.98 or above. The dips are stalls of well under a second, which PX4 and Gazebo, stepping
together, do not feel. So the first flights are the ones scored above, and the six second
flights are repeats of the same seed:

| Repeat of | First flight | Second flight |
| --- | --- | --- |
| Tuned, pier wake, seed 2 | 0.370 m | 0.367 m |
| Tuned, pier wake, seed 3 | 0.525 m | 0.476 m |
| Tuned, deck open, seed 7 | 0.328 m | 0.308 m |
| Tuned, deck open, seed 8 | 0.261 m | 0.312 m |
| v22, deck corner, seed 1 | 0.384 m | 0.170 m |
| v22, deck corner, seed 4 | 0.483 m | 0.348 m |

The same seed does not give the same flight. v22's two repeats moved by 0.14 and 0.21 m (the
one-seed preview above had seed 1 at 0.265 m, a third value); tuned PX4's four by 0.05 m at
most. The wind and the estimate noise repeat with the seed, so the likely cause is the moment
of engage against them, which does not; that was not checked. With the second flights in
place of the first, v22's deck corner mean would be 0.307 m (-38 %) and the other cells move
by 0.01 m. A difference of a few centimetres between two 8-seed means is inside this noise.

Runs: `experiments/LOG.md` (2026-11-09, bridge batches); the folders stay in the cloud
container.

## On the server

`training/queues/bridge_holds.yaml` (no job needs Zain's yes: all under 1 h, one GPU each,
at most two at once):

1. `bridge_check_v22`: the mock check above on the server's GPU; must match the cloud run.
2. `gz_bridge_image`: rebuild the SITL image so it has the two worlds and the plugin's
   second surface.
3. `gz_bridge_tuned`, `gz_bridge_v22` (flown in the cloud instead, results above): tuned PX4 (`vio_at_hover`) and v22 (hold runner,
   `vio_at_engage`, `--rangefinder`) in the three scenarios, seeds 1 to 8, 24 flights each.
   Go: v22's mean p95 below tuned PX4's in every scenario with 0 crashes, as at the flat
   wall (`docs/baseline.md`).
4. Only if v22 loses a scenario (or a mock cell): `bridge_ft_s0` and `bridge_ft_s1`
   (`training/configs/structure_hold_ppo_v27_bridge.yaml`, v22 continued under a deck 1 to
   6 m up and in pier wakes, two seeds) and `bridge_ft_check`. A candidate then goes
   through `gazebo_check` and a Gazebo re-fly like any other (CLAUDE.md, releasing a
   policy).
