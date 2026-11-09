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

## On the server

`training/queues/bridge_holds.yaml` (no job needs Zain's yes: all under 1 h, one GPU each,
at most two at once):

1. `bridge_check_v22`: the mock check above on the server's GPU; must match the cloud run.
2. `gz_bridge_image`: rebuild the SITL image so it has the two worlds and the plugin's
   second surface.
3. `gz_bridge_tuned`, `gz_bridge_v22`: tuned PX4 (`vio_at_hover`) and v22 (hold runner,
   `vio_at_engage`, `--rangefinder`) in the three scenarios, seeds 1 to 8, 24 flights each.
   Go: v22's mean p95 below tuned PX4's in every scenario with 0 crashes, as at the flat
   wall (`docs/baseline.md`).
4. Only if v22 loses a scenario (or a mock cell): `bridge_ft_s0` and `bridge_ft_s1`
   (`training/configs/structure_hold_ppo_v27_bridge.yaml`, v22 continued under a deck 1 to
   6 m up and in pier wakes, two seeds) and `bridge_ft_check`. A candidate then goes
   through `gazebo_check` and a Gazebo re-fly like any other (CLAUDE.md, releasing a
   policy).
