# Environmental factors: what the simulators model, and what they do not

Zain, 2026-11-10: the simulation has to take into account every environmental factor that
might affect the drone's flight. This page is the list. Every factor we could name for a
quadrotor flying on a camera-based position estimate next to a structure, indoors, in a
tunnel or under a bridge, with one row each: what it does to the flight, whether the mock
(the batched PyTorch simulator all training runs on) and Gazebo (PX4 SITL, the benchmark)
model it, and where the code is. The second half is what was added on 2026-11-10 to close
the gaps in the mock, and how v22, v24 and the PD yardstick fly with each new factor on.

Read the status words literally:

- **yes**: modelled, switched on from a config, with a test.
- **partly**: its main effect is modelled, something named in the row is not.
- **no**: not modelled. The row says why, or what stands in for it.
- **new**: added by this work (mock only), off by default.

Nothing here was fitted to a real flight. Where a range comes from physics (air density) it is
as good as the physics; where it comes from the literature (rotors near a wall) it has the
right shape and a wide range; where it is a guess (how much worse a VIO gets on a wet wall)
the row says so. A partner's flight logs are what turn the guesses into numbers
(`docs/wind-fit.md`, `docs/vio-fit.md`, `docs/airframe-fit.md` are the tools for that).

## The list

### Moving air

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Steady wind, any direction | Pushes the drone off the hold point | yes | yes | `envs/disturbance` (`mean_speed`, `mean_heading_deg`); C++ wind plugin, parity-tested |
| Gusts | A step of wind for 0.5 to 3 s | yes | yes | `num_gusts`, `gust_*` |
| Turbulence | Random wind with a correlation time | yes | yes | `turb_sigma`, `turb_tau` |
| Shelter close to the wall | Less wind near the surface | yes | yes | `wall_floor`, `wall_decay` |
| Flow turned by the wall, the bubble in its lee, wake turbulence, vortex shedding | The air next to a facade is not the free wind | yes | yes | Off by default; `docs/near-wall-flow.md`, `*_flow` scenarios |
| A deck overhead, a pier's wake | A second surface's shelter and the vortices off its edges | yes | yes | `docs/bridge-holds.md`, `bridge_deck` and `bridge_pier` worlds |
| Up and down draughts (a shaft, a stairwell, ventilation, the updraught on a sunlit or windward face) | A steady vertical wind: the drone has to climb or sink through it, and it eats thrust headroom | **new** | partly | `disturbance.mean_vertical`. Gazebo's plugin takes a vertical mean wind already; no scenario flies one yet |
| Draught along a tunnel | Wind that follows the route | yes | yes | Route Follow only: `envs/route_follow` (`_draft_force`), the four Gazebo routes |
| A passing vehicle or train | A gust and a pressure pulse | partly | partly | The gust is a gust; the pressure pulse is the barometer row below (mock only) |
| Wind that changes along the surface (speed-up round a corner, a jet through a gap, wind rising with height) | The wind at one end of a pass is not the wind at the other | no | no | Every wind term depends on time and distance from the surface only. Matters for the 5 m passes more than for a hold. Needs a flow field, not a formula: parked |

### The air itself, and the weather

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Air density: temperature, altitude, humidity | Thinner air gives less thrust and less control torque for the same motor command, and less drag. PX4's hover-thrust estimate hides the thrust offset, so it shows as less headroom | **new** | no | `envs/ambient` (`temperature_c`, `altitude_m`, `humidity`). Gazebo's motor model has one fixed air |
| Rain | Water on the airframe (weight), wet props (thrust), drops pushing down (small), and above all drops in front of the camera and the rangefinder | **new** | partly | `ambient.rain_mm_h` for the airframe; the camera and rangefinder side through the rows below. Gazebo: the `field_degrade` profiles give the sensor side, the airframe side is not there. Rain is outside the operating envelope (`docs/safety-manual.md` section 2) |
| Icing | Ice on the props: thrust falls and keeps falling | no | no | Outside the envelope, and there is no model we would trust. The thin-air cells show what running out of thrust looks like |
| Dust, fog, steam, smoke in the air | Nothing to the airframe; everything to the camera | n/a | n/a | See the camera rows |
| Heat and cold on the electronics | Sensor biases move as the board warms up | partly | partly | The gyro bias and barometer walk rows below stand in for it; there is no temperature model of a sensor |

### Rotors close to a surface

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Ground effect | More lift for the same command close to the ground | **new** | no | `envs/ambient` (`rotor_radius_m`, `ground_coef`), Cheeseman and Bennett. Small at the 2 to 3.5 m the hold is flown at; it matters under 1 m |
| Ceiling effect | A deck close overhead pulls the drone up into it | **new** | no | `ceiling_coef`, acts when the env has a deck (`deck_clearance`) |
| Wall effect | The rotors nearest the wall lose inflow: a pull toward the wall | **new** | no | `wall_pull`, grows as (rotor radius / distance) squared |
| The drone's own downwash coming back in a confined space | Unsteady pushes that scale with thrust and closeness | **new** | no | `unsteady`, `unsteady_tau_s` |
| Turning moment near a wall | The published case shows a yaw moment that changes sign | partly | no | A placeholder (`wall_yaw_torque`, training has it on since v12): one data point is not enough to fit it. The fan rig on a partner's drone would give it |
| Downwash throwing up dust or spray | Blinds the camera | yes | yes | A camera effect: the dust rows below |

### Battery and motors

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Battery level | Thrust and torque per command fall with voltage squared | **new** | no | `ambient.battery_voltage`, per episode, relative to the nominal 3.7 V a cell |
| Sag under load | The pack's voltage dips when the motors pull hard, so the top of the thrust curve bends over | **new** | no | `ambient.battery_sag` |
| A cold pack | Lower voltage and more sag, in dense air | **new** | no | The `cold_pack` cell combines them |
| Voltage falling during one flight | Slow loss of headroom | partly | no | Drawn per episode, not ramped inside one: over a 60 s hold the change is under 2 % |
| Motor lag, weak or strong motors, hover thrust estimated wrong | The airframe is not the nominal one | yes | partly | `airframe_rand` (`motor_tau_s`, `thrust_to_weight_scale`, `hover_thrust_error`); Gazebo flies the one x500 |
| Payload, other drone sizes | Mass and inertia | yes | partly | `mass_scale`, `inertia_scale`, `docs/airframe-sizes.md`, `docs/airframe-fit.md` |
| One damaged prop or motor, weight off-centre | A constant torque the rate loop has to hold | no | no | The gyro bias row is the nearest thing (a constant rate error). A motor failure is PX4's failure detector's job, not the hold's |

### The camera position estimate (VIO)

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Noise, slow drift, jumps, dropouts, delay | The estimate the hold flies on is wrong in each of these ways | yes | yes | `envs/estimate`; `sim2real/estimate_bridge` profiles; fitted to EuRoC (`docs/vio-fit.md`) |
| Reset, restart, silence, freeze, runaway | The field faults | yes | yes | `eval/fault_rates.py`, `eval/vio_runaway.py`; `docs/field-faults.md` |
| Dust, fog, low light | Noisier, walking, losing track | yes | yes | Measured with a real VIO (Basalt) on rendered flights, `docs/vio-render.md`; `field_degrade` |
| Flicker, glare and low sun, blank walls, wet or mirror-like or glass surfaces, smoke, drops on the lens, people or vehicles moving through the picture, blur from vibration | Each makes the VIO worse in the same four ways: more noise, faster drift, more jumps, more and longer losses | **new** | partly | `estimate.degrade_*`: each episode draws how bad the seeing is. **The severities are assumed, not measured**: only fog, dust and low light went through a real VIO. Gazebo: `field_degrade` has the same shapes, scripted |
| Camera mounted away from the centre | The camera swings when the drone tilts | yes | yes | `vision_offset_down_m` |

### Heading

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Compass bent by steel, rebar, motor current | Corrections go off to one side; the hold circles | yes | yes | `envs/estimate/heading.py`, `field_heading_*`; `docs/heading-error.md` |

### Height (barometer)

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Barometer noise and slow walk | The height reference wanders | yes | yes | `fusion.height_source: baro`, numbers from EKF2 on Gazebo hovers |
| Pressure steps: a door or shutter opening, ventilation starting, a train entering the tunnel, a gust front | 12 Pa reads as 1 m. The height estimate steps and the drone follows it | **new** | no | `fusion.baro_step_*` |
| Wind on the barometer's port | A pressure error that grows with airspeed squared, so height moves with every gust | **new** | no | `ambient.baro_wind_m_per_mps2` |
| The rotors' own pressure near the ground | A height error on take-off and landing | no | no | Constant at a fixed height, so the hold does not see it. It would matter for Structure Land, which PX4's own controller flies |

### IMU

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Gyro and accelerometer noise | Noisy rate and attitude inputs | yes | yes | `sensor_noise`, `fusion.imu_noise_mps2` |
| Vibration (a worn prop, a loose arm, resonance) | The same, several times larger | yes | partly | Same knobs, larger: the `vibration` cell. Gazebo flies its own fixed IMU noise |
| Gyro bias, attitude tipped by an accelerometer bias | The rate loop holds a small wrong rate; "level" is a degree or two off, so the drone slides | **new** | partly | `sensor_noise.gyro_bias_rad_s`, `tilt_bias_deg`. In Gazebo EKF2 estimates its own biases against Gazebo's IMU |

### Forward rangefinder

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Noise, scale error, stray returns, dropouts, the beam meeting the wall at an angle | The wall distance the guard and v24 read is off | yes | yes | `cfg.range`; `--rangefinder`, `field_degrade` |
| Glass, water, a mirror-like face | The beam goes through or bounces away: it reads what is behind, 0.5 to 3 m too far, some of the time | **new** | no | `range.glass_*`. Outside the envelope ("do not use next to glass") |
| A black, wet or sunlit face | The sensor loses the wall for seconds | yes | partly | Existing dropout and outlier knobs, harder: the `range_lost` cell |

### GPS, when it is fused

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Multipath jumps, loss, RTK dropping to float under a structure | EKF2's position moves with the GPS | no | yes | `sim2real/estimate_bridge/gps.py`, `field_gps_*`. The mock has no GPS: the product is GPS-off, and GPS-on is a Gazebo-only check |

### The place

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| The wall, the ground, a deck overhead | Contact ends the episode | yes | yes | `envs/structure_hold`, the three worlds |
| Tunnel and corridor walls, narrowings, branches | Route Follow's world | yes | yes | `envs/route_follow`, `sim2real/route_follow` |
| Other obstacles (a pipe, a cable, scaffolding) | The module's keep-out holds the hold point off them | no | partly | `KeepOut.hpp`, flown in SIH with one pole. Avoiding obstacles is out of scope (`docs/decisions.md`) |
| A structure that moves (a swaying cable, a moored vessel, a crane) | The hold point moves under the drone | no | no | The surface pass moves the hold point on purpose, which is the nearest thing. Parked |
| People and animals | Stay 5 m away | n/a | n/a | Safety manual |

### Links and timing

| Factor | What it does | Mock | Gazebo | Where |
| --- | --- | --- | --- | --- |
| Estimate arriving late or at an uneven rate | Old information | yes | yes | `estimate.delay_*`, the bridge's rate |
| Radio or telemetry lost (steel, tunnel) | Nothing to the hold, which runs on board; PX4's own failsafes act | n/a | n/a | Safety manual section 3 |
| The flight controller running late, lag between the module and the rate loop | The command acts later than it was computed | partly | yes | `rate_lag_s`, drawn per episode; Gazebo runs the real module in PX4. A single late step (jitter) is not modelled; the policy step itself is tens of microseconds (`docs/bench-test.md`) |

## What was added on 2026-11-10

All of it is in the mock, all of it is **off by default**, and each part draws from a random
generator of its own, so every existing config flies exactly the episodes it flew before
(checked by hashing rollouts against `main`). Switching a factor on changes that factor and
nothing else.

| Where | What | Keys |
| --- | --- | --- |
| `envs/ambient/` (new, NumPy only) | Air density from temperature, altitude and humidity; battery level and sag; rain on the airframe; ground, ceiling and wall effect on the rotors; unsteady recirculation; wind on the barometer's port | `ambient:` in a Structure Hold env config |
| `envs/disturbance/` | A steady vertical wind | `disturbance.mean_vertical` |
| `envs/estimate/ekf.py` | Pressure steps on the barometer; an outside pressure offset | `fusion.baro_step_rate_hz`, `baro_step_m`, `baro_step_tau_s`, `baro_step_decay_s` |
| `envs/estimate/core.py` | Poor seeing: a per-episode factor on the VIO's noise, drift, jump rate and loss rate | `estimate.degrade_prob`, `degrade_noise_scale`, `degrade_walk_scale`, `degrade_jump_rate_scale`, `degrade_dropout_rate_scale` |
| `envs/structure_hold/` | Gyro bias and a tipped attitude; a rangefinder that reads through glass | `sensor_noise.gyro_bias_rad_s`, `tilt_bias_deg`; `range.glass_prob`, `glass_extra_m`, `glass_frac`, `glass_dwell_s` |
| `eval/environment_factors.py` | The scorer for the table below | `eval/configs/environment_factors.yaml` |

How the airframe side works, in one paragraph. PX4 estimates the hover thrust in flight, so
a drone in thin air, on a low battery or wet with rain still hovers at the right height: the
module scales the policy's thrust by that estimate. What changes is what is left above hover.
The env therefore keeps hover where it was and scales the maximum thrust and the control
torque by `power = density x voltage squared x (1 - rain loss)`, and drag by the density. Sag
bends the thrust curve above hover. Rotor effects are forces along the thrust axis or toward
the wall that scale with the thrust being made at that moment.

One thing this turned up about the benchmark drone. The Gazebo x500 has thrust for 1.69
times its weight, and 1.41 times through the module's 0.9 thrust limit. It can no longer
hover below 0.71 of standard-day power: standard air at about 3500 m, or 1500 m at 35 degC
with the pack down to 3.5 V a cell. That is the airframe, not the policy, and
`docs/airframe-sizes.md` has the matching rule for payload (hover at or under 88 % throttle).

## How the controllers fly with each factor on

Mock, the benchmark x500 through the EKF2-like fusion on barometer height
(`gz_x500_sitl`), holds 1 to 2 m from the wall in 0 to 8 m/s wind with two gusts, 20 s
episodes, 768 episodes a row (3 seeds x 256), the same episodes for every controller. v22 is
the released policy, v24 the wall-range policy being released, PD the cascaded PD yardstick.
Each number is the mean per-episode p95 error on true position, with the change from that
controller's own nominal row. **These are mock numbers against the PD, not Gazebo against
tuned PX4**, and neither policy was trained on any of these conditions. Runs
`20261110-122610`, `-122630`, `-122650` and `-125017` in `experiments/LOG.md`.

| Factor | v22 p95 (m) | v22 crash | v24 p95 (m) | v24 crash | PD p95 (m) | PD crash |
| --- | --- | --- | --- | --- | --- | --- |
| nominal (standard day, nominal battery, dry) | 0.175 | 0.0% | 0.153 | 0.1% | 0.338 | 0.9% |
| hot and humid (30-40 degC, sea level) | 0.174 (-0%) | 0.0% | 0.152 (-1%) | 0.1% | 0.334 (-1%) | 0.9% |
| hot and high (25-35 degC, 1000-2000 m) | 0.175 (-0%) | 0.0% | 0.153 (-0%) | 0.1% | 0.329 (-3%) | 0.9% |
| cold (-15-0 degC, cold battery, sag 10-20 %) | 0.177 (+1%) | 0.0% | 0.155 (+1%) | 0.1% | 0.342 (+1%) | 0.9% |
| low battery (3.4-3.6 V a cell, sag 5-15 %) | 0.176 (+0%) | 0.0% | 0.154 (+0%) | 0.1% | 0.338 (+0%) | 0.9% |
| rain 10-50 mm/h (airframe, camera and rangefinder) | 0.212 (+21%) | 0.0% | 0.182 (+19%) | 0.1% | 0.447 (+32%) | 2.7% |
| up and down draughts to 3 m/s | 0.176 (+0%) | 0.0% | 0.154 (+1%) | 0.1% | 0.339 (+0%) | 0.9% |
| a close hold (0.8-1.2 m), no rotor effects | 0.170 (-3%) | 0.0% | 0.148 (-3%) | 0.1% | 0.288 (-15%) | 6.2% |
| rotors near the wall and the ground (hold at 0.8-1.2 m) | 0.171 (-2%) | 0.0% | 0.148 (-3%) | 0.1% | 0.302 (-10%) | 6.4% |
| under a deck 0.8-1.5 m overhead (no rotor effects) | 0.168 (-4%) | 0.0% | 0.145 (-5%) | 0.1% | 0.320 (-5%) | 0.9% |
| under that deck, with the rotors feeling it | 0.169 (-3%) | 0.0% | 0.144 (-6%) | 0.0% | 0.321 (-5%) | 0.9% |
| pressure steps 0.2-1 m (doors, ventilation), one per 20 s | 0.511 (+192%) | 0.9% | 0.500 (+227%) | 0.7% | 0.673 (+99%) | 2.5% |
| wind on the barometer's port (4-12 % of dynamic pressure) | 0.193 (+10%) | 0.0% | 0.169 (+10%) | 0.1% | 0.351 (+4%) | 0.9% |
| IMU drift (gyro bias to 0.03 rad/s, attitude tipped to 2 deg) | 0.231 (+32%) | 0.0% | 0.185 (+21%) | 0.1% | 0.345 (+2%) | 0.9% |
| vibration (gyro 0.1 rad/s, accelerometer 0.5 m/s2) | 0.246 (+40%) | 0.0% | 0.209 (+37%) | 0.1% | 0.372 (+10%) | 0.9% |
| poor seeing (low light, flicker, glare): noise x1-3, drift x1-3, losses x1-5 | 0.215 (+23%) | 0.0% | 0.180 (+18%) | 0.1% | 0.453 (+34%) | 2.7% |
| bad seeing (smoke, blank or wet walls, people): noise x2-5, drift x2-6, losses x5-20 | 0.364 (+108%) | 0.0% | 0.301 (+96%) | 0.0% | 0.783 (+132%) | 9.0% |
| rangefinder on glass or water (reads 0.5-3 m too far, 20-80 % of the time) | 0.175 (+0%) | 0.0% | 0.816 (+433%) | 58.7% | 0.338 (+0%) | 0.9% |
| rangefinder losing the wall (black, wet or sunlit: dropouts 0.5 Hz, 5 % stray returns) | 0.175 (+0%) | 0.0% | 0.155 (+1%) | 0.1% | 0.338 (+0%) | 0.9% |
| a field day: everything above at once, moderate | 0.270 (+54%) | 0.0% | 0.354 (+132%) | 11.2% | 0.447 (+32%) | 2.3% |

What the table says, row group by row group:

- **Air, battery, draughts: no effect** (within 1 %) as long as the drone can still hover.
  PX4's hover-thrust estimate does the work; the hold needs little thrust above hover even in
  8 m/s of wind. The limit is the airframe's, described above, not the controller's.
- **Rotors near the wall, the ground or a deck: no effect on the policies.** The close-hold
  row is the same hold without the rotor effects: the PD's extra crashes (6 %) come from
  holding closer, the rotor effects add 5 % to its error and nothing to v22 or v24. A wall
  pull of up to 5 % of thrust at 1 m is a steady push like wind, which the policies already
  reject. The ranges are literature-shaped, so this is "not sensitive", not "measured".
- **Rain: about 20 % looser**, all of it from what the camera and the rangefinder make of
  the drops. The water on the airframe does nothing measurable.
- **Pressure steps on the barometer are the largest effect on everyone**: one 0.2 to 1 m step
  per 20 s takes v22 from 0.18 to 0.51 m and the PD from 0.34 to 0.67 m. It is not the
  controller. EKF2's height follows the barometer, the drone holds the height EKF2 reports,
  and the scorer measures the true height. Wind on the barometer's port is the same thing in
  small (+10 %).
- **A drifting or vibrating IMU: 20 to 40 % looser for the policies**, 2 to 10 % for the PD.
  The policies read the gyro directly and were trained on little gyro noise and no bias;
  they still hold a third tighter than the PD in absolute terms.
- **Poor seeing: graceful.** At the "poor" level v22 and v24 lose about 20 %, at the "bad"
  level (outside the envelope) they double to 0.30 to 0.36 m with no crash, where the PD
  goes to 0.78 m and 9 % crashes. This is the condition the policies were built for.
- **A rangefinder that reads through glass or water crashes v24: 59 % of episodes.** v22 and
  the PD do not read the rangefinder and are untouched. v24 uses the range to correct the
  position estimate, so when the beam reads long, v24 closes the gap: it moves toward the
  face by as much as the reading is long, and hits it when that is more than the standoff
  minus the prop reach (10 % crashes for readings 0.5 to 1 m long, 82 % for 2 to 3 m). The
  spike gate does not stop it, because the long readings agree with each other. A
  rangefinder that just loses the wall (black, wet, sunlit) is harmless: v24 falls back to
  flying like v22. Two cautions on this row: the glass numbers are assumed, and the mock
  takes the range reference as the true standoff, where the module latches it from the
  reading at engage (a beam already through the glass at engage would push the drone away
  from the face instead, the safe way).
- **The field day** (everything at once, moderate): v22 0.27 m with no crash. v24's 11 %
  crashes are the one episode in five with a rangefinder on glass.

## What follows from it

1. **v24 and glass.** The safety manual already rules out glass, mesh and water spray for the
   rangefinder, but with v22 the price was a lost guard; with v24 it is a crash, and the wall
   guard reads the same sensor so it does not catch it. Before v24 flies near any face the
   rangefinder may see through or bounce off, either the module refuses a range that steps
   by more than the gate while the position estimate did not move (v24 then flies like v22),
   or a policy is trained with glass in the mix
   (`training/configs/structure_hold_ppo_v24_field.yaml`, below). Until one of them is
   shown to work, v24's release notes and the safety manual need the stronger wording.
2. **Height where the pressure moves.** In a tunnel with trains, a plant room, a building
   with doors and ventilation, the barometer is the weak sensor. The envelope already allows
   a downward rangefinder as the height reference (`EKF2_HGT_REF` 2); this result is the
   reason to prefer it there. The mock does not model that height source yet, so the gain is
   not measured.
3. **Training.** Gyro bias, vibration, poor seeing and glass are things training can teach.
   Air, battery, draughts and rotor effects are not worth training time on this airframe.

## The field-conditions candidate

`training/configs/structure_hold_ppo_v24_field.yaml` continues v24 with every new factor on
at moderate strength (glass on one rangefinder in five, poor seeing in half the episodes, a
drifting IMU, pressure steps, draughts, air and battery inside what the x500 can hover).
Its first seed is training in the cloud as this page is written; the result goes here.

## Not done, and why

- **Gazebo has none of the new factors.** The benchmark against tuned PX4 is flown in
  Gazebo, so none of the numbers above is a benchmark number. The cheap ones to carry over
  are the vertical draught (the wind plugin already takes it: a scenario file) and the
  pressure steps and glass (the estimate bridge scripts sensors already). Air density,
  battery and rotor effects need the motor model or the wind plugin extended. Worth doing
  for whichever factors the table above shows the hold is sensitive to.
- **Nothing was trained on them.** v22 and v24 never saw these conditions; the table is how
  they cope untrained. A training config with the randomisation on is the next step for the
  factors that hurt, and it is a server job.
- **Route Follow** has the draught along the tunnel and the estimate faults, but not the
  `ambient` block or the barometer steps. Tunnels are where pressure steps and recirculating
  downwash are worst, so it should get them when Route Follow next moves.
- **Severities are assumed** for the poor-seeing rows, the rotor effects near a wall and
  rain. The tools to measure them exist: rendered flights through Basalt
  (`docs/vio-render.md`) for textures, glare and smoke; a partner's fan rig and flight logs
  for the wall effect.
- **Still missing outright:** wind that varies along the surface, a moving structure,
  icing, one bad motor. Each is a row above with the reason.

## Running it

```
# smoke (under a minute)
python -m training.launch --config eval/configs/environment_factors.yaml --smoke --device cpu
# the real thing, in three parts under an hour each (they can run side by side)
python -m training.launch --config eval/configs/environment_factors.yaml --device cpu \
    --set "params.only=[hot,hot_high,cold_pack,low_battery,rain,draughts]"
python -m training.launch --config eval/configs/environment_factors.yaml --device cpu \
    --set "params.only=[close_hold,rotors_wall,deck,rotors_deck,baro_doors,baro_wind,imu_drift]"
python -m training.launch --config eval/configs/environment_factors.yaml --device cpu \
    --set "params.only=[vibration,seeing_poor,seeing_bad,range_glass,range_lost,field_day]"
# one table from the three run folders
python -m eval.environment_factors collect runs/<a> runs/<b> runs/<c> --out runs/factors.md
```

To fly any config with a factor on, add its block to the env config, for example
`ambient: {temperature_c: [30, 40], battery_voltage: [0.92, 1.0]}` or
`fusion: {baro_step_rate_hz: 0.05, baro_step_m: [0.2, 1.0]}`.
