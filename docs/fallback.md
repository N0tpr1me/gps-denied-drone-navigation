# Brake-and-hover fallback: the rule for the PX4 module

Measured in the mock sim on 2026-10-22 (runs `20261022-0518..0531_structure_hold_eval_fallback`,
`envs/structure_hold/fallback.py`). This page is the spec the C++ side of
`sim2real/px4/structure_hold` implements; the Python wrapper is the reference behaviour and
its tests say what "correct" means.

## Why

Policy v2 crashes more than the PD yardstick, and most of its crashes happen while the
estimate is stale: it keeps flying on a dead-reckoned estimate and drifts out of the box at
speed. A tiny classical rule that brakes once the estimate has been stale for a while removes
those crashes without touching the policy.

## The rule

Every control tick (100 Hz in the mock), with `stale_s` = time since the last fresh position
fix (0 while fixes arrive):

1. **Policy flies** while `stale_s < T_FB`. The policy keeps receiving its observation
   (including the stale flag and the capped age) so its history stays current.
2. **Fallback flies** while `stale_s >= T_FB`:
   - acceleration setpoint, world frame: `a_sp = clamp(-K_BRAKE * v_est, ±A_MAX)` where
     `v_est` is the estimator's dead-reckoned velocity. No position term: the dead-reckoned
     position is the least trustworthy part of the estimate, and holding velocity at zero
     is a level hover.
   - thrust vector `f = m * (a_sp + g_up)`; collective thrust = component of `f` along the
     current body z (clamped at 0); desired body z = `f / |f|`.
   - body rates: `rates_xy = K_TILT * (z_body × z_desired)` (small-angle rotation vector in
     the body frame), `rate_z = -K_YAW * yaw_error` with the heading held on the structure
     (+x of the structure frame). This is the same attitude loop the PD yardstick and the
     module's placeholder cascade already use.
3. **Hand back to the policy** on the first tick with a fresh fix (`stale_s` drops to 0).
   No hysteresis was needed in the mock; add a short minimum fallback time only if the
   estimator source flickers.
4. A new flight / engage starts with `stale_s = 0`.

| Constant | Value | Note |
| --- | --- | --- |
| `T_FB` | 0.5 s | sweep below; 0.25 s is as good, 1 s halves the gain, 2 s does nothing |
| `K_BRAKE` | 1.5 1/s | 2.5 (the PD's velocity gain) oscillated once the mock's rate loop lagged 100 to 200 ms like PX4's; 1.5 brakes 1.5 m/s to a hover in about 1 s |
| `A_MAX` | 6 m/s² | horizontal and vertical command limit |
| `K_TILT` | 4 rad/s per rad | attitude P gain, as the yardstick (8 before the rate-loop lag) |
| `K_YAW` | 3 rad/s per rad | heading P gain, as the yardstick |

The thresholds and gains are the mock airframe's (0.8 kg, 16 N, 2 rad/s rate limit since v3, 6 rad/s when measured here); scale
`K_TILT` / `K_YAW` with the real rate loop and keep `A_MAX` below what the airframe can tilt
to in the wind it is rated for.

## What it buys and costs

Measured with the mock before the rate-loop lag and the 2 rad/s action limit (v3 work, 2026-10-22) and with `K_BRAKE` 2.5, `K_TILT` 8; the v3 rows in `training/policies/README.md` carry the re-measurement under the lagged plant.

3 seeds x 2048 identical 20 s episodes, standard wind and corruption, dead-reckoning
estimate (dropouts 0.2 to 3 s at 0.02 Hz, jumps 0.02 Hz, delay 50 to 150 ms). Crash rate
s.e. about 0.3 points; p95 moves by about 0.04 m between otherwise identical runs.

| Controller | p95 hold error | crash rate | steps in fallback |
| --- | --- | --- | --- |
| PD yardstick | 0.83 m | 4.6 % | - |
| v2 bare | 0.51 m | 6.7 % | - |
| v2 + fallback, `T_FB` 2 s | 0.58 m | 6.6 % | 0.25 % |
| v2 + fallback, `T_FB` 1 s | 0.50 m | 4.2 % | 1.3 % |
| **v2 + fallback, `T_FB` 0.5 s** | **0.55 m** | **2.7 %** | 2.0 % |
| v2 + fallback, `T_FB` 0.25 s | 0.49 m | 3.2 % | 2.5 % |

With `T_FB` 0.5 s the pair beats the PD on both numbers: 34 % lower p95 and 40 % fewer
crashes (60 % fewer than bare v2). The hold-error cost of the fallback is within run noise
(+0.04 m p95 at 0.5 s, none at 0.25 s): the fallback flies 2 % of the steps and most of
them are steps the bare policy would have spent drifting anyway.

## Re-measured with policy v3 on the lagged plant (2026-10-22)

With the rate-loop lag, the 2 rad/s action limit and the retuned gains (`K_BRAKE` 1.5,
`K_TILT` 4), 3 seeds x 2048 episodes:

| Controller | p95 hold error | crash rate | heading error |
| --- | --- | --- | --- |
| PD yardstick (retuned) | 0.95 m | 7.9 % | 2 deg |
| v3 bare | 0.57 m | 2.9 % | 8 deg |
| v3 + fallback, `T_FB` 0.5 s | 0.63 m | 4.0 % | 8 deg |

v3 handles dropouts on its own and the fallback now costs it on both numbers, so v3 ships
with the fallback off (`SH_FB_EN` 0). The rule stays in the module as an option for a policy
that needs it (around v2 it halved the crash rate) and for flight-test safety if the real
estimator's dropouts look different from the mock's.

## For the module

- **Watchdog order.** `SH_WD_EST_TO` (stale-estimate trip, default 0.5 s) must be raised
  above the dropouts the fallback is meant to fly through (3 s in the mock), otherwise the
  watchdog hands back to stock PX4 before the fallback does anything. Keep an immediate trip
  on an estimate EKF2 marks invalid; the fallback is for a *valid but unaided* (IMU-only)
  estimate, which is what the mock's dead-reckoning models. Who decides: the PX4 module
  thread with Zain, since it changes a watchdog default.
- **Stale means "no fresh position fix"**, counted from estimate timestamps or EKF2's
  aiding status, not from the age value in the policy observation (capped at 1 s).
- **Log it.** Publish fallback active / `stale_s` in `structure_hold_status` so the SITL
  benchmark can report the share of steps in fallback, as the mock does.
- **Not measured here:** wind gusts during the brake beyond the standard 0 to 12 m/s, the
  real EKF2 dead-reckoning drift (the mock uses a 0.3 m/s² per-axis bias), and the estimate
  jump that often ends a real dropout. The SITL benchmark against tuned PX4 (M1) is where the
  rule is confirmed before any flight.

## Status in the module (2026-10-22)

Implemented unchanged in `sim2real/px4/structure_hold/PolicyController.hpp` (`fallbackAction`,
checked against `acceleration_to_action` to 3e-7 in `tests/test_policy.cpp`). Parameters:
`SH_FB_EN` (1 on, 0 off, 2 forced for tests), `SH_FB_T` (0.5 s), `SH_FB_K` (K_BRAKE),
`SH_FB_TILT` (K_TILT), `SH_YAW_P` (K_YAW, shared with the yaw guard), fix source `SH_FIX_SRC`,
stale flag gap `SH_FIX_GAP`. `SH_WD_EST_TO` is now 5 s (docs/decisions.md). The stale time is
`now - last fix`, so at engage it is the true fix age rather than zero; a hold engaged inside
a dropout therefore goes to the fallback immediately instead of 0.5 s later.
`structure_hold_status` logs `stale_s`, `fallback_active` and the action flown.
