# sim2real/gazebo

The test worlds for PX4 SITL (a flat wall, a bridge deck, a bridge pier) and the adapter that
applies the shared wind and near-wall force (`envs/disturbance`) to the simulated vehicle.

```
worlds/wall_slab.sdf     wall at x = 0 (structure on +x), slab underside at z = 5 m
worlds/bridge_deck.sdf   abutment at x = 0 and the deck it carries, underside at z = 4 m
worlds/bridge_pier.sdf   a 1.5 m square pier, face at x = 0, under a deck at z = 6 m
plugin/                  StructureHoldWind: gz-sim 8 world system (C++), built into the image
scenarios/*.yaml         calm_0ms, wind_4ms, wind_8ms: spawn, wall, disturbance config, seed
params_io.py             DisturbanceParams <-> the text file the plugin reads
scenario.py              scenario YAML -> params, as envs/structure_hold would draw them
run_scenario.py          fly a scenario in the SITL image, verify the trace, write a run folder
verify_trace.py          recompute the plugin's trace with the Python model
parity.py                C++ model == NumPy model on random inputs (pytest, needs a compiler)
```

## How the wind reaches Gazebo

1. `scenario.py` samples the episode's disturbance with `envs.disturbance.sample_params`
   (one env, the scenario's seed and wall), exactly as `envs/structure_hold` does, so a
   Gazebo run and a `torch_quad` run with the same scenario see the same wind.
2. `params_io.write_params` writes it to `wind_params.txt` in the run folder, which is
   mounted at `/out` in the container.
3. The world file loads `StructureHoldWind`. Each physics step (250 Hz) the plugin reads
   the vehicle's world pose and velocity, evaluates `plugin/Disturbance.hh` (a line-for-line
   C++ copy of `core.py`) at the current sim time, subtracts the model's still-air drag
   (Gazebo's rotor model already has body drag; see the envs/disturbance README) and adds
   the result with `Link::AddWorldForce`. Same step, so zero lag between the commanded
   and the applied force. Every step it logs `t, pos, vel, wind, shelter, force` to
   `wind_trace.csv`.
4. `verify_trace.py` recomputes the trace with the NumPy model and fails above 1e-6 N.

The plugin reads `STRUCTURE_HOLD_WIND_PARAMS` / `STRUCTURE_HOLD_WIND_TRACE` (set by
`run_scenario.py`) or the `<params>` / `<trace>` elements in the world file. Without a
params file it is silent and applies nothing, so the world also works for plain flights.

Decision (why a C++ plugin rather than a Python bridge over gz-transport): see
`docs/decisions.md`. The parity test keeps the copy honest: `pytest sim2real` compiles
`parity_main.cc` with the system compiler and checks 1e-9 N agreement; it is skipped where
no compiler exists and runs in CI and in the image
(`docker run --rm -v "$PWD:/repo:ro" -w /repo structure-hold/px4-sitl:v1.17.0 python3 -m sim2real.gazebo.parity`).

## Use

```bash
sim2real/docker/run.sh build                               # once; adds the plugin layer
python -m sim2real.gazebo.run_scenario wind_4ms            # or calm_0ms, wind_8ms, a path
python -m sim2real.gazebo.run_scenario wind_8ms --hover-s 60
```

`run_scenario.py` needs the repo's Python (`pip install -e .[dev]`) on the host and writes
`runs/<time>_sitl_wind_<scenario>_<commit>/` with `config.yaml`, `wind_params.txt`,
`wind_trace.csv`, `wind_verify.json`, the hover test's `summary.json`, `flight.ulg` and
`px4.log`, and `wind_summary.json` (overall pass/fail). Exit code 0 means the hover passed
and the trace matched the model.

Scenario keys: `name`, `seed`, `world` (a file in `worlds/`, default `wall_slab`), `spawn`
(`x y z yaw_deg`, ENU, 1 to 3 m from the wall), `wall` (`point`, `normal_toward`, from the
drone into the structure), `surface2` (optional: `point`, `normal_toward` and `across`, the
second surface of the wind model, the deck in `bridge_deck`; docs/bridge-holds.md),
`disturbance` (any `DisturbanceConfig` key), `wind_heading_deg` (direction the mean wind
blows toward, 0 = +x, optional), `hover` (`altitude`, `hover_s`). The runners fly each
scenario in its own world and score contacts with both surfaces.

## Notes

- Episode time is Gazebo sim time (`<start_time>` 0), so gusts scheduled at 15 and 40 s in
  the shipped scenarios land inside the 30 s hover that starts after a ~10 s takeoff.
- The world copies PX4's `default.sdf` physics, scene and spherical coordinates. gz-sim
  loads PX4's `server.config` systems only for worlds with no `<plugin>` element, so the
  world declares the same stock systems itself, then the wind plugin. If PX4's
  server.config changes at a pin move, mirror it here. The slab is a static box, so the x500's sensors are not affected (no occlusion
  modelled); wall_slab's scenarios leave the slab out of the wind model (it is 2.5 m over
  the hold). `bridge_deck`'s scenarios put its deck in as the second surface.
- The plugin is a world system: it finds `x500_0/base_link` after PX4 spawns it. Other
  models need `<model_name>` / `<link_name>` in the world file.
- Force is applied at the link origin (no near-wall torque in the model yet).
