# Route Follow: flying a given route through tunnels and corridors

Structure Hold keeps a drone still next to a structure. Route Follow is the second
capability: fly a route the operator gives (a list of waypoints) through a tunnel, culvert,
mine drift or building corridor, without GPS, on the drone's existing position estimate,
using simple range sensors to stay off the walls. Structure Hold stays the product and is
not changed by any of this (`docs/decisions.md`, 2026-11-01 row).

## What it is and is not

- **Is:** follow a fixed route at a set speed, keep a safe distance from the walls, stop at
  the last waypoint, and hand back to PX4 (or to Structure Hold) cleanly.
- **Is not:** route planning, mapping, SLAM/VIO, obstacle avoidance or vision in the policy.
  Those stay out of scope as decided on 2026-10-21. The route is fixed before flight; the
  drone never decides where to go. Keeping its distance from the corridor walls it was sent
  down is part of following the route, not obstacle avoidance.

Who it is for: the same small and mid PX4 drone makers. Inspection of tunnels, culverts,
sewers, mine drifts and long indoor corridors, and indoor logistics runs along a fixed path.

## How it flies

The interface to PX4 is the one Structure Hold already uses: the learned outer loop sends
body rates and collective thrust to PX4's stock rate controller, a small MLP on a short
observation history, exported as ONNX and compiled into the module as a header.

**The target.** A point moves along the route's centreline (straight segments between the
waypoints) at the route speed: it starts from rest, slows for each corner (a 90 degree
corner at a fifth of the cruise speed, a straight joint at full speed, as PX4 Mission mode
slows for a corner), and stops on the last waypoint. The policy is asked to stay on that moving point. A moving target, rather
than "fly to the next waypoint", gives the policy one well-defined error to close at all
times and is the same shape as PX4's own position controller tracking a trajectory setpoint,
so the classical baseline is a fair copy of what PX4 does.

**What the policy reads** (all in the route frame at the target: x along the route, y to
the left, z up, so the same policy works on any heading):

| Input | Why |
| --- | --- |
| estimated position minus the target | the error to close |
| estimated velocity | damping |
| the route speed | how fast the target is moving |
| the target 1 m and 2.5 m further along the route, relative to the target | sees a turn or a climb coming, so it can start turning early |
| attitude (gravity in the body frame), heading against the route, body rates | as Structure Hold |
| previous action, estimate stale flag and age | as Structure Hold |
| distance left to the last waypoint, capped | knows when to stop |
| up to five 1D rangefinders (forward, left, right, up, down), each with a usable flag | measures the walls directly, independent of the estimate |

The rangefinders are the point. In a tunnel the position estimate drifts (dark, repetitive,
featureless walls are where visual odometry is worst), and a controller that only follows
the estimate flies into the wall once the drift is larger than the clearance. The side and
up/down ranges measure where the walls really are. Each rangefinder is optional (drawn per
episode in training), so a drone with none still flies on the estimate alone, and a drone
with only left and right still uses them.

**Reward** (on the true state, as Structure Hold): stay on the target (bounded), match its
velocity, face along the route, keep at least a margin from every wall (charged only inside
the margin), smooth commands, and a one-off crash cost on wall contact, tilt-over or losing
the target by more than 3 m.

## PX4 hand-off

- **Route in:** the route comes from PX4's mission (its waypoint items), so the operator
  plans it in the ground station they already use. The module reads the items; it does not
  change them.
- **Start:** the target starts at the vehicle's position and ramps up to speed toward the
  first waypoint.
- **End:** at the last waypoint the target stops and the policy holds there; then the module
  hands to Structure Hold if a wall is measured within range, otherwise to PX4 Position mode.
- **Watchdog:** the same classical, non-learned watchdog as Structure Hold, with two extra
  trips: a measured wall closer than a minimum clearance, and the estimate more than a set
  distance off the target. A trip hands back to PX4 Position mode (Structure Hover when a
  wall is measured), then Hold, then Land, as for Structure Hold.

The module side is not built in this phase; it comes only if the policy beats tuned PX4 in
Gazebo (below).

## The simulation (`envs/route_follow/`, mock, CPU)

- **Routes:** a few straight segments joined at random turns (up to 90 degrees left or
  right) with small climbs, a new route every episode.
- **Corridors:** a rectangular cross-section (width and height drawn per episode), built as
  one box per segment, each box extended at its ends so the corners are filled.
- **Tunnels:** a round cross-section (radius drawn per episode), one capsule per segment.
- **Walls:** contact with any wall, floor or ceiling is a crash (the drone's props radius).
  The rangefinders are cast against the same geometry.
- **Air:** the same wind model as Structure Hold, turned to blow along the tunnel (a draft)
  with gusts and turbulence, at lower speeds than outdoors.
- **Estimate:** the same corruption and EKF2-like fusion as Structure Hold, with a tunnel
  profile that lets the drift grow larger.
- **Airframe:** the Gazebo x500 presets (`gz_x500`, `gz_x500_sitl`), so the numbers carry
  over to SITL as they did for Structure Hold.

### Worlds

Beyond plain corridors and round tunnels, the route generator (`RouteGen`) builds the places
customers fly, each off by default so the first results' sets stay exactly as they were:

| World | Setting | What it is |
| --- | --- | --- |
| Arched drift | `arch_prob` (of the corridors) | vertical walls under a half-circle roof as wide as the passage: a mine drift, an arched culvert or rail tunnel |
| Narrow pipe | `tunnel_radius_m` down to 0.8 m | a sewer or culvert where the props are 0.4 m from the wall |
| Narrowed stretch | `pinch_prob`, `pinch_frac` | a segment whose cross-section shrinks (a doorway, a collapsed stretch, a smaller pipe section) |
| Side branches | `branches`, `branch_prob`, `branch_len_m` | dead-end passages leaving the route at right angles (crosscuts, side corridors, junctions); the side rangefinders see down them, so a controller that centres on the walls is pulled sideways there |
| Ramps | `climb_deg` up to 25 degrees | steep slopes between levels (ramps in mines, car parks) |

Each is one more piece in the same union-of-pieces geometry (`geometry.py`), so contact and
the rangefinder beams work the same way in all of them. `eval/configs/route_follow_worlds.yaml`
scores every world on its own and a mix of all of them, and
`training/configs/route_follow_ppo_worlds.yaml` trains on the mix.

## The code

| Piece | Where |
| --- | --- |
| Config (route, rangefinders, reward; airframes as Structure Hold) | `envs/route_follow/config.py` |
| Random routes and the target's speed plan (NumPy) | `envs/route_follow/routes.py` |
| Walls, target, route frame, rangefinder beams (sphere-traced) on tensors | `envs/route_follow/geometry.py` |
| The env (38 inputs per step, table in its docstring) | `envs/route_follow/env.py` |
| PD yardsticks (estimate only; with range centring) | `envs/route_follow/baseline.py` |
| Scoring and the yardstick run | `eval/route_follow.py`, `eval/configs/route_follow_baseline.yaml` |
| PPO training (smoke mode, exports `policy.onnx`) | `training/train_route_ppo.py`, `training/configs/route_follow_ppo.yaml` |

Defaults worth knowing: 4 segments with turns up to 90 degrees and slopes up to 10 degrees
(a short route turns less, one piece per 3 m at most); corridors 2 to 4 m wide and 2.4 to 4 m
tall, tunnels 1.1 to 2 m in radius; cruise 0.5 to 1.5 m/s at 0.5 m/s^2; 30 s episodes with
the last 5 s a hold on the last waypoint; draft up to 3 m/s along the tunnel; the estimate's
drift walk and cap raised to 0.02 m/sqrt(s) and 0.5 m (a guess until a tunnel VIO fit).

```
python -m training.launch --config training/configs/route_follow_ppo.yaml --smoke --device cpu
python -m training.launch --config eval/configs/route_follow_baseline.yaml --device cpu
```

## How it is scored

Always on the true position, never the estimate:

- cross-track error: distance from the route centreline (p95 and worst per flight);
- tracking error to the moving target (p95);
- closest approach to any wall, and wall contacts;
- completion: within 0.5 m of the last waypoint when the flight ends;
- crash rate.

**Baselines.** In the mock: a cascaded PD tracking the same moving target with the target's
velocity fed forward, on the estimate only (the shape of PX4's position controller), and the
same PD with a classical wall-centring term from the left/right/up/down ranges, the best a
hand-written controller does with the same sensors. Every mock result is reported against
both. In SITL: stock PX4 Mission mode with the tuned parameters
(`eval/baselines/px4_tuned_v2.params`) flying the same waypoints through the same estimate.

## First results (mock, 2026-11-01)

v0, the first policy: 30 minutes of PPO on the cloud container's 4 CPU cores (1024 drones,
1000 iterations, run `20261101-174152`), rangefinders present on 70 % of the beams in
training. Scored on 256 new routes per set (run `20261101-181304`; the yardsticks alone in
run `20261101-173858`), x500 as SITL flies it, true position:

| Set | v0 cross-track p95 | PD | PD + centring | v0 / PD / PD + centring done | crashes |
| --- | --- | --- | --- | --- | --- |
| corridor, no rangefinders | 0.28 m | 0.36 m | 0.36 m | 93 / 88 / 88 % | 1 / 4 / 4 % |
| corridor, rangefinders | 0.24 m | 0.39 m | 0.34 m | 94 / 89 / 92 % | 3 / 4 / 3 % |
| tunnel, no rangefinders | 0.26 m | 0.35 m | 0.35 m | 93 / 87 / 87 % | 3 / 7 / 7 % |
| tunnel, rangefinders | 0.24 m | 0.36 m | 0.33 m | 97 / 86 / 92 % | 1 / 7 / 5 % |

So v0 follows the route 22 to 38 % tighter than the PD on the estimate alone and 22 to 29 %
tighter than the PD that centres on the walls, and finishes more routes. Every crash, for
all three, is a wall contact; at 256 routes a set, 1 to 7 % is 2 to 19 flights, so the crash
numbers are rough. This is the mock only: the lesson from Structure
Hold is that the mock alone does not predict Gazebo, so nothing is claimed until a Gazebo
tunnel flight against tuned PX4 Mission mode.

## Worlds results (mock, 2026-11-02)

Every controller on 256 new routes per world (run `20261102-010928`,
`eval/configs/route_follow_worlds.yaml`), x500 as SITL flies it, true position. v0 is the
first policy (corridors and round tunnels only); v1 is v0 trained on for 1000 more
iterations (35 min) on the mix of worlds (run `20261102-003213`). Each cell: cross-track p95,
routes finished, crashes.

| World | v0 | v1 | PD | PD + centring |
| --- | --- | --- | --- | --- |
| arched drift | 0.27 m, 92 %, 4 % | 0.27 m, 92 %, 2 % | 0.36 m, 88 %, 6 % | 0.36 m, 88 %, 6 % |
| arched drift, rangefinders | 0.23 m, 95 %, 2 % | 0.23 m, 92 %, 4 % | 0.35 m, 90 %, 4 % | 0.33 m, 92 %, 4 % |
| narrow pipe | 0.22 m, 83 %, 16 % | 0.23 m, 82 %, 16 % | 0.26 m, 70 %, 29 % | 0.26 m, 70 %, 29 % |
| narrow pipe, rangefinders | 0.21 m, 89 %, 9 % | 0.20 m, 85 %, 14 % | 0.27 m, 69 %, 30 % | 0.25 m, 46 %, 53 % |
| narrowed stretches | 0.26 m, 91 %, 6 % | 0.27 m, 91 %, 6 % | 0.31 m, 86 %, 11 % | 0.31 m, 86 %, 11 % |
| narrowed, rangefinders | 0.22 m, 92 %, 7 % | 0.23 m, 90 %, 7 % | 0.32 m, 82 %, 15 % | 0.31 m, 76 %, 22 % |
| side branches | 0.29 m, 93 %, 3 % | 0.28 m, 93 %, 2 % | 0.37 m, 89 %, 5 % | 0.37 m, 89 %, 5 % |
| side branches, rangefinders | 0.31 m, 97 %, 0 % | 0.26 m, 94 %, 2 % | 0.38 m, 88 %, 4 % | 0.50 m, 59 %, 35 % |
| ramps to 25 degrees | 0.30 m, 95 %, 2 % | 0.28 m, 95 %, 1 % | 0.37 m, 95 %, 3 % | 0.37 m, 95 %, 3 % |
| ramps, rangefinders | 0.27 m, 96 %, 2 % | 0.26 m, 95 %, 1 % | 0.36 m, 90 %, 4 % | 0.37 m, 86 %, 12 % |
| all mixed, rangefinders on 70 % | 0.28 m, 91 %, 6 % | 0.25 m, 93 %, 5 % | 0.33 m, 87 %, 10 % | 0.37 m, 76 %, 19 % |
| plain corridor, rangefinders | 0.24 m, 96 %, 1 % | 0.28 m, 91 %, 2 % | 0.38 m, 89 %, 4 % | 0.33 m, 91 %, 7 % |
| plain tunnel, rangefinders | 0.24 m, 95 %, 2 % | 0.24 m, 95 %, 2 % | 0.39 m, 88 %, 6 % | 0.34 m, 93 %, 5 % |

What this says:

- **v0 already copes with worlds it never saw.** It follows the route tighter than both PD
  yardsticks in every world and crashes less, though nothing in its training had an arch, a
  side branch, a pipe under 1.1 m or a slope over 10 degrees.
- **Wall centring, the classical way to use rangefinders, breaks in real places.** A side
  branch makes one side beam read far, and the centring PD steers into the opening: 35 %
  crashes against 4 % for the plain PD. It also does worse in narrow pipes (53 %) and
  narrowed stretches. The policies use the same beams without that failure.
- **Narrow pipes are the hard world.** With the props 0.4 to 0.7 m from the wall and the
  estimate allowed to drift 0.5 m, every controller crashes there most (v0 9 to 16 %, PD
  about 30 %). All crashes are wall contacts.
- **Short CPU training does not move it much.** v1 (v0 plus 35 min on the mix) is level
  with v0: a little tighter on the mix, side branches and ramps, a little worse in the plain
  corridor and pipes; its action noise had shrunk to almost nothing, so it was barely
  exploring. A fresh policy on the mix (with pipes down to 0.8 m) was still behind the PD
  after 1000 iterations (runs `20261102-015210`, cross-track p95 0.41 m against 0.37 m on the
  mix) and only level with it after 2000 (`20261102-023213`, 0.34 m against 0.37 m, 12 %
  against 11 % crashes). The harder mix needs far longer training than the cloud CPUs give
  in an hour: that is GPU server work.

As before, this is the mock only; Gazebo against tuned PX4 Mission mode decides.

## Stress results (mock, 2026-11-02)

One thing made worse at a time on the corridor and tunnel mix, rangefinders on 70 % of the
beams, 256 routes per test (run `20261102-134815`, `eval/configs/route_follow_stress.yaml`).
Each cell: cross-track p95, routes finished, crashes.

| Test | v0 | v1 | PD | PD + centring |
| --- | --- | --- | --- | --- |
| nominal (drift walk 0.02, cap 0.5 m) | 0.25 m, 94 %, 2 % | 0.25 m, 94 %, 2 % | 0.37 m, 93 %, 4 % | 0.35 m, 91 %, 5 % |
| perfect estimate | 0.11 m, 100 %, 0 % | 0.11 m, 100 %, 0 % | 0.17 m, 100 %, 0 % | 0.21 m, 100 %, 0 % |
| low drift (walk 0.01, cap 0.3 m) | 0.20 m, 95 %, 1 % | 0.21 m, 95 %, 1 % | 0.36 m, 89 %, 5 % | 0.36 m, 89 %, 6 % |
| high drift (walk 0.04, cap 1 m) | 0.34 m, 89 %, 3 % | 0.35 m, 86 %, 3 % | 0.46 m, 74 %, 6 % | 0.39 m, 86 %, 4 % |
| estimate lost 1 to 3 s, about every 10 s | 0.26 m, 94 %, 2 % | 0.26 m, 93 %, 4 % | 0.35 m, 90 %, 6 % | 0.34 m, 89 %, 7 % |
| 0.2 to 1 m jumps, about every 10 s | 0.36 m, 91 %, 2 % | 0.37 m, 91 %, 2 % | 0.49 m, 83 %, 4 % | 0.43 m, 86 %, 6 % |
| estimate 0.2 to 0.4 s late | 0.29 m, 91 %, 4 % | 0.29 m, 93 %, 3 % | 0.48 m, 82 %, 14 % | 0.44 m, 82 %, 13 % |
| draft 3 to 6 m/s, gusts to 4 m/s | 0.27 m, 96 %, 2 % | 0.27 m, 93 %, 3 % | 0.39 m, 87 %, 7 % | 0.37 m, 91 %, 5 % |
| fast, 1.5 to 2.5 m/s (trained to 1.5) | 0.28 m, 96 %, 2 % | 0.30 m, 91 %, 4 % | 0.43 m, 86 %, 9 % | 0.38 m, 91 %, 5 % |
| 10 % rangefinder outliers | 0.26 m, 93 %, 2 % | 0.26 m, 94 %, 2 % | 0.37 m, 87 %, 6 % | 0.33 m, 95 %, 2 % |
| narrow pipe, perfect estimate | 0.14 m, 100 %, 0 % | 0.13 m, 100 %, 0 % | 0.17 m, 100 %, 0 % | 0.18 m, 79 %, 21 % |
| narrow pipe, low drift | 0.18 m, 90 %, 9 % | 0.19 m, 84 %, 13 % | 0.25 m, 68 %, 31 % | 0.26 m, 59 %, 41 % |
| narrow pipe, nominal | 0.22 m, 87 %, 11 % | 0.22 m, 89 %, 11 % | 0.28 m, 68 %, 32 % | 0.27 m, 58 %, 41 % |

What this says:

- **The policy holds up under every stress.** It beats both PD yardsticks on every test.
  Its worst cases are estimate jumps and high drift (0.34 to 0.37 m), and even there it
  crashes on 2 to 3 % of routes.
- **The estimate sets the limit, not the flying.** With a perfect estimate every controller
  finishes every route, and the policy's cross-track halves (0.11 m against 0.25 m).
- **Narrow-pipe crashes come from the estimate.** With a perfect estimate nobody crashes in
  the pipes except the centring PD. Lower drift alone barely helps (11 % to 9 %), so the
  remaining crashes come from the estimate's jumps and dropouts. In a 0.8 m pipe, one jump
  bigger than the 0.4 m gap is a crash. A pipe needs a better estimate, or a policy that
  trusts the rangefinders more when the estimate jumps. The next training target is
  estimate jumps in narrow pipes.
- **A late estimate hurts the classical controllers most** (13 to 14 % crashes against 3 to
  4 % for the policy), as it did for Structure Hold.
- **Wall centring is unsafe in round pipes** even with a perfect estimate (21 % crashes): a
  tilted drone's side beams hit the curved wall unevenly, so the centring term pushes it
  the wrong way.
- v1 is again level with v0 throughout.

## Plan, and what needs the GPU server

1. **Now, no GPU:** this design, the tunnel and corridor envs, the classical baselines and a
   training script with a smoke mode, run in the cloud container's CPUs.
2. **GPU server:** full training runs (thousands of iterations on thousands of parallel
   drones; on the 4-core cloud container one policy would take many hours), then Isaac Lab.
3. **Gazebo:** tunnel and corridor worlds, flown in SITL against tuned PX4 Mission mode on
   the same waypoints. The policy is released only if Gazebo confirms the mock.
4. **Module:** Route Follow as a second external mode with its watchdog, then a partner's
   drone.

Simulations run on the GPU server, not the Mac. Until the server is up, only short CPU runs
happen in the cloud container.
