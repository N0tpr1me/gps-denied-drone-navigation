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
waypoints) at the route speed, with a speed ramp at the start and the end, and stops on the
last waypoint. The policy is asked to stay on that moving point. A moving target, rather
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
(`eval/baselines/px4_tuned_v1.params`) flying the same waypoints through the same estimate.

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
