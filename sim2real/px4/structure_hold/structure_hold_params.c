/**
 * Structure Hold parameters. All under SH_ (build plan, PX4 integration section).
 */

/**
 * Start the Structure Hold module at boot
 *
 * The mode registers with commander at start and is idle until engaged.
 *
 * @boolean
 * @reboot_required true
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_EN, 1);

/**
 * Watchdog: position fix timeout
 *
 * No fresh position fix (see SH_FIX_SRC) or no vehicle_local_position update for longer
 * than this trips the watchdog. Sits above the dropouts the brake-and-hover fallback is
 * meant to fly through (up to 3 s in the mock) and at EKF2's own no-aiding limit
 * (EKF2_NOAID_TOUT, 5 s): an estimate EKF2 marks invalid trips the watchdog at once
 * regardless of this value (docs/fallback.md, docs/decisions.md).
 *
 * @unit s
 * @min 0.05
 * @max 10.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_WD_EST_TO, 5.0f);

/**
 * Watchdog: maximum hold error
 *
 * Distance between the held setpoint and the estimate beyond which the watchdog trips.
 *
 * @unit m
 * @min 0.2
 * @max 10.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_WD_POS_ERR, 2.0f);

/**
 * Watchdog: maximum tilt
 *
 * @unit deg
 * @min 10
 * @max 80
 * @decimal 0
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_WD_TILT, 45.0f);

/**
 * Test hook: force a watchdog trip
 *
 * Set to 1 to trip the watchdog on the next tick (hands back to stock PX4). Clears
 * itself. For SITL and flight tests of the handback path only.
 *
 * @boolean
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_TEST_FAULT, 0);

/**
 * Placeholder hold controller: position gain
 *
 * @unit 1/s
 * @min 0.1
 * @max 3.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_P_POS, 0.95f);

/**
 * Placeholder hold controller: velocity gain
 *
 * @unit 1/s
 * @min 0.1
 * @max 6.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_P_VEL, 1.8f);

/**
 * Placeholder hold controller: attitude gain
 *
 * @unit 1/s
 * @min 1.0
 * @max 12.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_P_ATT, 6.5f);

/**
 * Placeholder hold controller: maximum velocity setpoint
 *
 * @unit m/s
 * @min 0.2
 * @max 5.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_V_MAX, 1.0f);

/**
 * Velocity source for the controllers
 *
 * 0: velocity from the local position estimate (normal). 1: velocity differentiated from
 * the estimated position inside the module and low-pass filtered, for simulators whose
 * velocity estimate is unusable (PX4 v1.17.0 SIH, see sim2real/px4/README.md).
 *
 * @value 0 Estimator velocity
 * @value 1 Differentiated estimator position
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_VEL_SRC, 0);

/**
 * Controller flying the hold
 *
 * 1: policy v2 (PolicyWeights.hpp, generated from training/policies) with the classical
 * brake-and-hover fallback. 0: the skeleton's placeholder P cascade (SH_P_*).
 *
 * @value 0 Placeholder cascade
 * @value 1 Policy with fallback
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_CTRL, 1);

/**
 * Brake-and-hover fallback enable
 *
 * While the estimate has had no fresh position fix for SH_FB_T or longer, the fallback
 * brakes on the dead-reckoned velocity and hovers level until the next fix
 * (docs/fallback.md). -1: the compiled-in policy's own default (policy.json
 * output.fallback: on for v2, off for v3, which handles dropouts itself). 0: the policy
 * flies through every dropout. 2: test hook, the fallback flies every tick (checks the
 * rule and the frame conversions on their own).
 *
 * @value -1 Policy default
 * @value 0 Off
 * @value 1 On
 * @value 2 Forced (test)
 * @min -1
 * @max 2
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_FB_EN, -1);

/**
 * Brake-and-hover fallback: take-over time
 *
 * Time without a fresh position fix after which the fallback takes over from the policy.
 * 0.5 s in the mock sweep (docs/fallback.md); keep below SH_WD_EST_TO.
 *
 * @unit s
 * @min 0.05
 * @max 5.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_FB_T, 0.5f);

/**
 * What counts as a fresh position fix
 *
 * 0: vehicle_visual_odometry samples (VIO / the estimate bridge) once any has arrived since
 * boot, otherwise vehicle_local_position updates. 1: visual odometry only. 2: local
 * position updates only (every EKF2 output counts, so the estimate is never stale; for
 * GPS or SIH flights).
 *
 * @value 0 Auto
 * @value 1 Visual odometry
 * @value 2 Local position
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_FIX_SRC, 0);

/**
 * Fix interval beyond which the estimate is flagged stale to the policy
 *
 * The policy's stale observation is 1 once the last fix is older than this; the fallback
 * uses the stale time itself (SH_FB_T). Set to about twice the fix source's period.
 *
 * @unit s
 * @min 0.01
 * @max 1.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_FIX_GAP, 0.1f);

/**
 * Yaw guard: hold the heading with the module's own loop
 *
 * 1: the module holds the wall-facing heading captured at engage with a P loop (SH_YAW_P)
 * and ignores the policy's yaw output. 0: the policy's yaw output flies. -1: the
 * compiled-in policy's own default (policy.json output.yaw): on for v2, which has no
 * heading term and yaws freely; off for v3, which holds heading itself and hits the wall
 * when the guard replaces its yaw.
 *
 * @value -1 Policy default
 * @value 0 Off
 * @value 1 On
 * @min -1
 * @max 1
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_YAW_GUARD, -1);

/**
 * Yaw guard gain
 *
 * Yaw rate per radian of heading error, as the mock's PD yardstick and the fallback.
 *
 * @unit 1/s
 * @min 0.5
 * @max 10.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_YAW_P, 3.0f);

/**
 * Body rate setpoint limit for the policy and the fallback
 *
 * Applies to the commanded roll, pitch and yaw rates after the action scaling. 0: the
 * compiled-in policy's own action scale (policy.json output.max_rate_rad_s: 6 rad/s for
 * v2, 2 rad/s for v3). A real airframe's rate loop may need a lower clamp (v2 flew SIH
 * only at 1 to 2 rad/s).
 *
 * @unit rad/s
 * @min 0.0
 * @max 6.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_RATE_MAX, 0.0f);

/**
 * Brake-and-hover fallback: braking gain
 *
 * Acceleration setpoint per m/s of dead-reckoned velocity (K_BRAKE in docs/fallback.md,
 * retuned 2.5 to 1.5 for the lagged rate loop with policy v3).
 *
 * @unit 1/s
 * @min 0.2
 * @max 5.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_FB_K, 1.5f);

/**
 * Brake-and-hover fallback: attitude gain
 *
 * Body rate per radian of tilt error of the fallback's attitude loop (K_TILT in
 * docs/fallback.md, retuned 8 to 4 for the lagged rate loop with policy v3); scale with
 * the airframe's rate loop.
 *
 * @unit 1/s
 * @min 1.0
 * @max 12.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_FB_TILT, 4.0f);
