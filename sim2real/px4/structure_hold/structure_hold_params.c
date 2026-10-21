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
 * Watchdog: local position estimate timeout
 *
 * No vehicle_local_position update for longer than this trips the watchdog.
 *
 * @unit s
 * @min 0.05
 * @max 5.0
 * @decimal 2
 * @group Structure Hold
 */
PARAM_DEFINE_FLOAT(SH_WD_EST_TO, 0.5f);

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
 * Placeholder hold controller: velocity source
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
