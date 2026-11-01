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
 * Watchdog: frozen position fix
 *
 * Trips the watchdog when the position fix (the VIO odometry, SH_FIX_SRC) keeps arriving on
 * time but has not moved by more than 0.1 mm for this long while flying, and refuses an
 * engage while it is so. A stuck camera pipeline or VIO process can republish its last pose
 * forever: the stale trip never sees it and EKF2 keeps it valid. A real VIO on a flying
 * vehicle is never that still for long. After the trip the decided handback order applies;
 * with SH_HB_RANGE 1 Structure Hover treats a frozen fix as no estimate and holds the wall
 * distance on the rangefinder. Needs the odometry fix (no effect when the fix is EKF2's
 * local position). 0 = off (default; Proposed, docs/decisions.md 2026-11-01).
 *
 * @unit s
 * @min 0.0
 * @max 10.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_WD_FRZ_T, 0.0f);

/**
 * Watchdog: entry hold-off after an estimator reset
 *
 * Structure Hold refuses to engage (commander keeps the current mode, or the module hands
 * straight back if it was already switched in) for this long after EKF2 reset its
 * horizontal position (vehicle_local_position.xy_reset_counter), because a hold point
 * captured while the estimate settles after a reset is not where the vehicle is. Not
 * latched: the mode becomes available again when the time has passed. A reset during a
 * hold never trips the watchdog; the policy is trained on them. Off by default (0):
 * stock PX4 captures the same settling estimate, so in the mock replay of the v11 seed 3
 * sequence a 10 s hold-off moved the crashes into the stock phase without preventing
 * them (docs/baseline.md, v11 section). An operator choice, not a fix.
 *
 * @unit s
 * @min 0.0
 * @max 60.0
 * @decimal 1
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_RST_HOLD, 0.0f);

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
 * Bench test only: allow engaging on the ground
 *
 * Structure Hold refuses to engage while the land detector says the vehicle is on the
 * ground (a hold point on the ground is no hold, and the policy never learned a take-off),
 * and hands back at idle thrust. The props-off bench test (docs/bench-test.md) needs the
 * mode engaged on the ground, so this lifts that refusal. Props off only; never fly with
 * it set. The mode still cannot be armed into.
 *
 * @boolean
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_BENCH, 0);

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

/**
 * Wall guard: minimum standoff from the measured wall
 *
 * With a forward-facing rangefinder (distance_sensor, orientation forward), the hold point
 * is kept at least this far from the wall it measures, along the heading at engage: when
 * the range says the hold point is closer, the point moves straight back by the shortfall
 * and never forward again during that hold (WallGuard.hpp). This is the floor for the
 * Gazebo seed 3 strike, where a hold point captured on a jumped estimate lay on the wall
 * face (docs/real-flight-gap.md). Without a usable range sample (stale over 0.2 s, out of
 * the sensor's range, bad signal, or the vehicle yawed more than 35 deg off the wall
 * normal) the guard does nothing. Set it above the prop reach plus margin (x500: 0.36 m
 * reach). 0 turns the guard off; distances are still reported in structure_hold_status.
 *
 * @unit m
 * @min 0.0
 * @max 5.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_WALL_MIN, 0.7f);

/**
 * Wall guard: require a forward range to engage
 *
 * When set, the mode reports itself unavailable to commander while no usable forward
 * range sample is coming in, so it cannot be engaged without the floor of SH_WALL_MIN.
 * A range lost during a hold does not trip the watchdog: the hold point keeps the shift
 * it already has. Off by default so vehicles without a forward rangefinder (the Gazebo
 * benchmark x500) fly as before.
 *
 * @boolean
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_WALL_REQ, 0);

/**
 * Structure Land: register the guarded landing mode
 *
 * Registers a second external mode, "Structure Land" (after Structure Hold, so ext2), that
 * lands next to the structure: PX4's own position controller descends at MPC_LAND_SPEED
 * with its land detector and touchdown, and the module owns only the xy setpoint, which the
 * wall guard keeps at least SH_WALL_MIN from the measured wall the whole way down. Stock
 * Land holds xy on the estimate alone, so after a handback on a jumped estimate it walks
 * the vehicle into the wall as the estimate recovers (docs/baseline.md, wall guard on
 * Gazebo seed 3). Available in flight only, and also after a latched watchdog trip. Its
 * own watchdog trips (stale or invalid estimate, tilt, xy error) hand back to stock Land
 * first, since the landing was already asked for, then to Position mode, Hold and Land
 * as Structure Hold's do. Without a usable forward range it is a plain position-holding landing.
 *
 * @boolean
 * @reboot_required true
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_LAND_EN, 1);

/**
 * Handback: guarded hover first when a wall is measured
 *
 * Registers a third external mode, "Structure Hover" (ext3): the Structure Land setpoint
 * without the descent, so PX4's own position controller hovers on a point the wall guard
 * keeps at least SH_WALL_MIN from the measured wall. A watchdog trip in Structure Hold
 * hands back into it first when a usable forward range sees the wall within 5 m, then
 * falls through to Position mode as before if commander refuses it. Stock Position mode
 * holds the estimate alone, and after a trip on a jumped estimate it creeps toward the
 * wall as the estimate recovers (Gazebo seed 3: 0.59 to 0.31 m clearance before the
 * landing). The pilot leaves it for Position mode as usual. Available in flight only, also
 * after a latched trip. 0: hand back to Position mode directly.
 *
 * @boolean
 * @reboot_required true
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_HB_GUARD, 1);

/**
 * Structure Hover: hold the wall distance on the rangefinder when the position is lost
 *
 * Proposed, off by default (docs/decisions.md 2026-11-01, docs/field-faults.md). With the
 * horizontal position estimate gone, the decided handback goes to Altitude mode, which holds
 * the height and keeps the vehicle level: a wind blowing into the wall then pushes it onto
 * the face (the 8 s link-loss field fault in Gazebo). With 1, a trip in Structure Hold next
 * to a measured wall (as SH_HB_GUARD) goes to Structure Hover even without a horizontal
 * estimate, and Structure Hover, whenever the estimate is invalid or stale, holds the
 * measured wall distance on the forward rangefinder alone (never under SH_WALL_MIN), leans
 * into the wind it was holding against before the estimate went, and keeps the height on
 * PX4's own altitude control. Once the estimate is back and fresh for 1 s it holds the
 * point on the estimate again. A lost range hands back as before (Altitude mode, Hold,
 * Land). Structure Hover then no longer needs a local position to be selected. Needs
 * SH_HB_GUARD 1 and SH_WALL_MIN > 0. 0: Altitude mode as decided.
 *
 * @boolean
 * @group Structure Hold
 */
PARAM_DEFINE_INT32(SH_HB_RANGE, 0);
