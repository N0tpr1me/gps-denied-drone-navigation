# Structure Hold as an Auterion app

Started 2026-11-02. Zain asked to do everything in the "Features that sell" research
(buyer role-play doc, Nov 2026), whose item 5 is an Auterion version of Structure Hold.
`docs/market-research.md` (gap 18) has the earlier, weaker case for it.

## Why

Some of the best-fit buyers do not run plain PX4. Freefly (top of the buyer scorecard) ships
Auterion's version of PX4 on the Astro, and Auterion sells AuterionOS on its Skynode
computers to other drone makers too. On those vehicles a third party cannot compile a
module into the autopilot the way `sim2real/px4/structure_hold` does: third-party code runs
as an **app** on the Linux side of Skynode and talks to the autopilot over ROS 2. Without an
app version, "wrong flight stack" was Freefly's second deal-breaker after "no real flight".

The app is a second way to ship the same product, not a new product. The onboard module
stays the main path and keeps PX4 pinned at v1.17.0.

## What Auterion offers (checked 2026-11-02)

All from docs.auterion.com unless noted.

| Piece | What it is | Open to us without Auterion? |
| --- | --- | --- |
| AuterionOS app | A Docker container (or Compose set) on Skynode, described by `auterion-app.yml` with an `auterion-api-version` (v3 or later for the SDK; v3 needs AuterionOS 3.2.9) | The format is documented |
| `auterion-cli` | Builds an app into an `.auterionos` file (`app build`, `--simulation` for Virtual Skynode) and installs it on a device | Yes, public on PyPI (1.21.1) |
| Auterion SDK | ROS 2 Humble C++ library (`ros-humble-auterion-sdk`) with a Flight Mode API: `auterion::Mode` with go-to, local and body-frame dynamics, fixed-wing and VTOL setpoints; rate and attitude setpoints exist but are "not recommended in general"; a thrust setpoint is not documented | Installs from Auterion's public package feed; source not public |
| PX4 ROS 2 Interface Library (`px4_ros2`, `ros-humble-px4-ros2-cpp`) | The library external modes register through, and which the SDK's package set installs alongside it; has a rates + thrust setpoint type | Yes: open source (BSD-3, github.com/Auterion/px4-ros2-interface-lib), works with stock PX4 |
| Compatibility | Package versions must match the vehicle's AuterionOS / APX4 or the mode does not register (table below) | Documented |
| Virtual Skynode | AuterionOS in a VM (Ubuntu 22.04 with KVM, or Mac), running APX4 in SIH by default or with Gazebo; apps install as on a real Skynode | No: the image comes from Auterion Suite's developer store and needs the Skynode Developer program |
| Distribution | Auterion's own apps deploy through the Suite Store (one vehicle) and Fleet Deploy (a fleet) | How a third party lists an app is not public: needs Auterion |

Compatibility rows we use (Auterion SDK installation page):

| AuterionOS | APX4 | auterion-sdk | px4-ros2-cpp | px4-msgs |
| --- | --- | --- | --- | --- |
| 4.6.2 | 1.17.7-3.5.x | 1.27.0 | 2.2.1 | 3.5.1 |
| 4.3.0 | 1.16.0-3.3.17 | 1.15.0 | 1.6.1 | 3.3.1 |

APX4 1.17 is Auterion's build of PX4 1.17, the same release line as our pin, so the 4.6.2
row is the target.

## Design

```
 Skynode (Linux, AuterionOS)                         Autopilot (APX4)
 +--------------------------------------+            +------------------------------+
 | structure-hold app (Docker)          |  uXRCE-DDS | commander: "Structure Hold"  |
 |  ROS 2 node, px4_ros2 external mode  | <--------> |  failsafes, arming checks,   |
 |  HoldCore.hpp                        |  ROS 2     |  external-mode timeout       |
 |   = the module's PolicyController,   |            | rate controller + allocation |
 |     Watchdog, Handback, ParamLimits  | rates +    |                              |
 |  100 Hz                              | thrust --> |                              |
 +--------------------------------------+            +------------------------------+
```

- **Same flight code.** `sim2real/auterion/structure_hold_app/include/structure_hold_app/HoldCore.hpp`
  is a ROS-free wrapper around the module's own PX4-free headers: `PolicyController.hpp`
  (observation, the released policy v22, fallback, yaw guard, setpoint mapping),
  `Watchdog.hpp`, `Handback.hpp` and `ParamLimits.hpp`. A policy release regenerates
  `PolicyWeights.hpp` once and both variants fly it. `test_hold_core.cpp` checks the app's
  setpoints against a bare `PolicyController` value for value.
- **Same interface to PX4.** Body rates and collective thrust to PX4's stock rate
  controller at 100 Hz, through `px4_ros2::RatesSetpointType`, exactly what the module
  publishes on `vehicle_rates_setpoint`. The mode registers with commander like the module
  does (it is the same external-mode handshake, from outside the autopilot), so it appears
  in Auterion Mission Control or QGroundControl and PX4's failsafes apply.
- **`px4_ros2` directly, not `auterion::Mode`.** The SDK's Flight Mode API wraps the same
  registration, documents rate setpoints as "not recommended" and does not document
  thrust; `px4_ros2` is installed in the same app image (the SDK's package set includes
  it), is open source, and runs against stock PX4 today, so we can build and test the app
  without an Auterion account. If Auterion wants apps on its own API, the wrapper is
  thin: `HoldCore` stays, only the ~300-line node changes.
- **Settings** are ROS 2 parameters named after the module's `SH_` parameters
  (`wd_est_to` for `SH_WD_EST_TO` and so on), clamped by the same `ParamLimits.hpp`.

### Where the app differs from the module

| Topic | Module (in PX4) | App (on the companion) |
| --- | --- | --- |
| Control path | In-process uORB | One DDS hop each way; a few ms more latency, not yet measured |
| Process dies | Cannot, short of PX4 itself | Commander leaves the mode to its failsafe on its own (2.2 s in SIH, below) |
| Engage on the ground or while latched | Starts, then hands back at idle thrust | Refused up front by the mode's arming check, with the reason in the app log |
| Trip handback | Structure Hover next to a measured wall, else Position / Altitude, Hold, Land | Position / Altitude, Hold, Land (no Structure Hover yet) |
| Wall guard, wall-range input, Structure Hover and Land | Yes | Not yet: PX4 v1.17's DDS bridge does not export `distance_sensor`, so the app needs the range from Auterion's sensor API or a custom topic list |
| Settings | PX4 parameters, editable from the ground station | ROS 2 parameters; not visible to the ground station yet |
| Disarm, NaN guards, parameter clamping, frozen-fix trip, stale trip, fallback | Yes | Yes, same code |

v22, the released policy, needs nothing the app lacks. The wall guard is the first thing to
add (next steps).

## What runs today, without Auterion hardware or an account

| What | Command | Result |
| --- | --- | --- |
| App core unit tests (CI) | `g++ ... test_hold_core.cpp` (`sim2real/auterion/README.md`) | Pass: same setpoints as the module's controller, refusals, stale trip and handback order, latch until disarm, frozen fix, tilt, test fault, parameter clamping |
| Package checks (CI) | `pytest sim2real/auterion/tests` | Header lists cover every include; manifest and pins match this doc; `stage_app.sh` writes a complete app folder |
| SIH flight on stock PX4 v1.17.0 | `sim2real/auterion/sitl_app_test.sh <PX4_DIR>` | 15/15 checks passed: see "SIH flight" below |

The app image for SITL (`docker/Dockerfile.sitl`) builds `px4_msgs` and `px4_ros2` from
their PX4 v1.17 release branches on stock ROS 2 Humble, plus the Micro XRCE-DDS Agent.
The Auterion image (`package/Dockerfile`) installs the same libraries from Auterion's feed
at the 4.6.2 row; it cannot be built in the cloud container (that feed's host is blocked
there) and has not been built yet.

### SIH flight (2026-11-02)

`sitl_app_test.sh` on stock PX4 v1.17.0 (SIH quad, no module compiled in), the app in its
SITL image with `vel_src` 1: **15 of 15 checks passed**.

| Check | Result |
| --- | --- |
| Mode registers with commander | 8 s after the app starts |
| Engage on the ground (armed) | Refused by the app's arming check ("landed"); the mode is never entered |
| Hold 20 s after a stock take-off | 0.29 m max horizontal drift, 0.11 m max height error (the module's SIH test allows 0.5 m; SIH does not judge hold quality) |
| Disengage to Position, re-engage | Accepted both ways |
| Watchdog trip (`test_fault`) | Handed back to Position mode 20 ms after the trip, still airborne; a further engage refused until disarm; latch cleared on disarm |
| App killed while engaged | Commander left the mode 2.2 s later by itself (its "no response" check) and stayed airborne |

Two things the flight taught, both fixed or recorded:

- PX4 v1.17.0's default DDS topic list leaves out `vehicle_angular_velocity`, so an app
  that waits for it never gets gyro data. The app reads the body rates from
  `vehicle_odometry` instead. An earlier run that missed this engaged without flying and
  drifted 15 m before the test ended it (logged as a failure in `experiments/LOG.md`).
- After the app was killed, commander's failsafe for a lost external mode chose **Return**
  in SIH, which has GPS. On a GPS-denied vehicle Return is not available and PX4 falls
  further down its chain (Land or Descend). A partner integrating the app should check
  which failsafe their parameters give for a lost external mode before flying it, and the
  partner pack should gain that step when the app is offered.

## What needs Auterion (Zain's call: sign-ups and money)

1. **Skynode Developer program** for the Virtual Skynode image: the only way to run the app
   on AuterionOS itself before touching a vehicle. Cost and terms are not public.
2. **An Auterion Suite account** to install apps and use Fleet Deploy.
3. **How a third-party app is listed, signed and sold** in the Suite Store: not
   documented; needs a partner conversation with Auterion.
4. **A Skynode or a partner's Auterion drone** (Freefly Astro) for the real-flight half,
   which the partner pack (`docs/partner-pack.md`) would cover the same way as for PX4.

None of these were signed up for or paid for.

## Next steps (no Auterion needed)

1. Measure the DDS round trip in SITL (rates setpoint sent to rates setpoint applied) and
   fly the app through the Gazebo benchmark next to the module once the GPU server is up:
   the app must match the module's numbers within noise before it is offered.
2. Wall guard and wall-range input in the app: the forward range from a custom
   `dds_topics.yaml` (stock PX4) or Auterion's sensor API (AuterionOS).
3. Structure Hover and Structure Land as two more external modes, and the guarded handback.
4. Mirror the settings as PX4 parameters (or Auterion app settings, API v6+) so a pilot can
   change them from the ground station.

## Results

| Date | Commit | Run | Result |
| --- | --- | --- | --- |
| 2026-11-02 | first commit of this doc | `sitl_app_test.sh`, SIH, `vel_src` 1 | 15/15 checks passed (table above) |
