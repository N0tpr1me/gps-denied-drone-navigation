/**
 * Structure Hold: one onboard PX4 flight mode.
 *
 * Skeleton (v0): registers a named mode with commander through PX4's external-component
 * path (the same uORB handshake the ROS 2 interface library uses, driven from inside the
 * firmware as mc_nn_control does), holds the position and yaw it had when engaged, and
 * hands control back to stock PX4 (Position mode, or Altitude mode if the horizontal
 * estimate is gone) the moment the watchdog trips. PX4's stock rate controller stays in
 * charge of the low-level loop: this module publishes vehicle_rates_setpoint only.
 *
 * Controller (SH_CTRL): 1 = policy v2 with the brake-and-hover fallback
 * (PolicyController.hpp, docs/fallback.md), 0 = the placeholder P cascade of the skeleton
 * (HoldController.hpp). Both output FRD body rates and a normalised collective thrust.
 *
 * Structure Land (SH_LAND_EN), a second registered mode: a landing next to the structure
 * that keeps the wall guard. PX4's own position controller flies it (xy held, descent at
 * MPC_LAND_SPEED, its land detector and touchdown); the module only owns the xy setpoint,
 * which the forward range keeps at least SH_WALL_MIN from the wall. Stock Land holds xy on
 * the estimate alone, and after a handback on a jumped estimate that walks the vehicle into
 * the wall (docs/baseline.md, wall guard on seed 3).
 *
 * Structure Hover (SH_HB_GUARD), a third mode, is the same guarded setpoint with no
 * descent. A watchdog trip hands back into it first when a forward range sees the wall,
 * so stock Position mode's creep after a handback on a jumped estimate cannot walk the
 * vehicle into the wall; the pilot switches out of it to Position mode as usual. With
 * SH_HB_RANGE 1 (the default) it also flies with the horizontal estimate gone,
 * holding the measured wall distance on the rangefinder alone (RangeHover.hpp).
 *
 * Keep-out (SH_KO_DIST, following PX4's CP_DIST by default): the data PX4's collision
 * prevention reads (obstacle_distance and horizontal distance_sensors) keeps the hold point
 * of all three modes that far from any obstacle (KeepOut.hpp). Not avoidance: the point
 * only moves away from what is measured too close.
 */

#pragma once

#include <drivers/drv_hrt.h>
#include <lib/perf/perf_counter.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/posix.h>
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>
#include <systemlib/mavlink_log.h>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/SubscriptionMultiArray.hpp>
#include <uORB/topics/arming_check_reply.h>
#include <uORB/topics/arming_check_request.h>
#include <uORB/topics/debug_array.h>
#include <uORB/topics/distance_sensor.h>
#include <uORB/topics/hover_thrust_estimate.h>
#include <uORB/topics/obstacle_distance.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/register_ext_component_reply.h>
#include <uORB/topics/register_ext_component_request.h>
#include <uORB/topics/structure_hold_status.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/unregister_ext_component.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_odometry.h>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_status.h>

#include "Handback.hpp"
#include "HoldController.hpp"
#include "KeepOut.hpp"
#include "ParamLimits.hpp"
#include "PilotStatus.hpp"
#include "PolicyController.hpp"
#include "RangeHover.hpp"
#include "WallGuard.hpp"
#include "Watchdog.hpp"

using namespace time_literals;

class StructureHold : public ModuleBase<StructureHold>, public ModuleParams, public px4::WorkItem
{
public:
	StructureHold();
	~StructureHold() override;

	/** @see ModuleBase */
	static int task_spawn(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);
	int print_status() override;

	bool init();

	static constexpr const char *MODE_NAME = "Structure Hold";
	static constexpr const char *LAND_MODE_NAME = "Structure Land";
	static constexpr const char *HOVER_MODE_NAME = "Structure Hover";
	static constexpr float HOVER_WALL_RANGE_M = 5.f;  // a trip hands back to Structure Hover inside this
	static constexpr float GUARD_SPEED_TAU_S = 0.3f;  // low-pass on the guard's push rate (feed-forward)
	static constexpr float LAND_SLOW_MPS = 0.3f;      // Structure Land's descent while it lags the guard
	static constexpr hrt_abstime CONTROL_INTERVAL = 10_ms;  // 100 Hz outer loop
	static constexpr hrt_abstime STANDBY_STATUS_INTERVAL = 200_ms;
	static constexpr hrt_abstime LIVE_INTERVAL = 200_ms;    // the live array; MAVLink streams it at its own rate
	static constexpr uint16_t LIVE_ARRAY_ID = 0x5348;       // "SH": tells our array from another module's
	static constexpr hrt_abstime HANDBACK_RETRY = 300_ms;  // then Position, Hold, Land
	// Structure Hover on the range alone (SH_HB_RANGE): no usable reading for this long
	// trips it, and the estimate must be back and fresh for this long to leave it.
	static constexpr hrt_abstime RANGE_HOVER_LOST = 500_ms;
	static constexpr hrt_abstime RANGE_HOVER_XY_BACK = 1_s;
	static constexpr float RANGE_HOVER_STALE_S = 1.f;  // a fix older than this is not good
	// Run() is woken by vehicle_angular_velocity, which a flight controller can publish at
	// 1 to 8 kHz; the control tick is 10 ms, so wake-ups closer than this are skipped.
	static constexpr hrt_abstime WAKE_INTERVAL = 2_ms;

private:
	void Run() override;

	// Mode registration with commander
	void sendRegistration(const char *name, uint64_t &request_id);
	void sendUnregistration(const char *name, int8_t mode_id, int8_t arming_check_id);
	void checkRegistrationReplies();
	void sendControlModeConfig(int8_t mode_id, bool position_control);
	void replyToArmingCheck(uint8_t request_id);

	// Flight
	void engage(const vehicle_local_position_s &lpos);
	structure_hold::Trip entryRefusal(const hrt_abstime now) const;
	void disengage(const char *why);
	void handBack(structure_hold::Trip trip, bool xy_valid, bool latch = true);
	void endHandback(const char *why);
	void requestHandbackMode();
	void handbackStep(const hrt_abstime now);
	structure_hold::Trip checkWatchdog(const hrt_abstime now, const matrix::Dcmf &r, float position_error_m,
					   bool range_only = false);
	void controlStep(const hrt_abstime now);
	void engageGuarded(const vehicle_local_position_s &lpos, bool descend);
	void guardedStep(const hrt_abstime now);
	bool rangeHoverWanted(const hrt_abstime now) const;
	void rangeHoverStep(const hrt_abstime now, const matrix::Dcmf &r);
	float measuredWall(const hrt_abstime now, const matrix::Dcmf &r) const;
	bool estimateGood() const;
	bool fixFrozen() const;  // SH_WD_FRZ_T on and the fix has not moved for that long
	void publishAccelerationSetpoint(float ax_ned, float ay_ned);
	void updateWindFeedForward(const hrt_abstime now);
	void publishTrajectorySetpoint(float z_ned, const matrix::Vector3f &velocity_ned);
	bool guardedHandbackAvailable(const hrt_abstime now) const;
	void updateFix(const hrt_abstime now);
	void updateRange();
	bool rangeFresh(const hrt_abstime now) const;
	void applyWallGuard(const hrt_abstime now, const matrix::Vector3f &pos, const matrix::Dcmf &r);
	void applyKeepOut(const hrt_abstime now, const matrix::Vector3f &pos, const matrix::Dcmf &r);
	bool keepOutDataFresh(const hrt_abstime now) const;
	bool estimateFinite() const;
	void publishRates(const matrix::Vector3f &rates, float thrust);
	bool useOdometryFix() const;
	void publishStatus(const hrt_abstime now);
	// The pilot's live view (PilotStatus.hpp, docs/pilot-status.md)
	float watchdogStaleS(const hrt_abstime now, bool range_only) const;
	void wallText(char *buf, size_t len, float wall_m) const;
	void sayEngaged(const hrt_abstime now, const char *mode);
	void pilotStep(const hrt_abstime now, const char *mode, const matrix::Dcmf &r, float position_error_m,
		       bool range_only);
	void publishLive(const hrt_abstime now);
	void updateParams() override;

	// Subscriptions
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};
	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _register_reply_sub{ORB_ID(register_ext_component_reply)};
	uORB::Subscription _arming_check_request_sub{ORB_ID(arming_check_request)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _local_position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _hover_thrust_sub{ORB_ID(hover_thrust_estimate)};
	uORB::Subscription _visual_odometry_sub{ORB_ID(vehicle_visual_odometry)};
	uORB::Subscription _land_detected_sub{ORB_ID(vehicle_land_detected)};
	uORB::SubscriptionMultiArray<distance_sensor_s> _distance_sensor_subs{ORB_ID::distance_sensor};
	uORB::Subscription _obstacle_distance_sub{ORB_ID(obstacle_distance)};

	// Publications
	uORB::Publication<register_ext_component_request_s> _register_request_pub{ORB_ID(register_ext_component_request)};
	uORB::Publication<unregister_ext_component_s> _unregister_pub{ORB_ID(unregister_ext_component)};
	uORB::Publication<vehicle_control_mode_s> _config_control_setpoints_pub{ORB_ID(config_control_setpoints)};
	uORB::Publication<arming_check_reply_s> _arming_check_reply_pub{ORB_ID(arming_check_reply)};
	uORB::Publication<vehicle_rates_setpoint_s> _rates_setpoint_pub{ORB_ID(vehicle_rates_setpoint)};
	uORB::Publication<trajectory_setpoint_s> _trajectory_setpoint_pub{ORB_ID(trajectory_setpoint)};
	uORB::Publication<vehicle_command_s> _vehicle_command_pub{ORB_ID(vehicle_command)};
	uORB::Publication<structure_hold_status_s> _status_pub{ORB_ID(structure_hold_status)};
	// The live numbers for the ground station (MAVLink DEBUG_FLOAT_ARRAY "SH_STATUS").
	uORB::Publication<debug_array_s> _live_pub{ORB_ID(debug_array)};
	// Ground-station text (MAVLink STATUSTEXT): the pilot is told when a safety function
	// acts and why (docs/market-research.md, gap 2). Each call also prints to the console.
	orb_advert_t _mavlink_log_pub{nullptr};

	// Registration state
	bool _registration_sent{false};
	uint64_t _request_id{0};
	int8_t _mode_id{-1};
	int8_t _arming_check_id{-1};
	bool _land_registration_sent{false};
	uint64_t _land_request_id{0};
	int8_t _land_mode_id{-1};      // Structure Land; stays -1 with SH_LAND_EN 0
	int8_t _land_arming_check_id{-1};
	bool _hover_registration_sent{false};
	uint64_t _hover_request_id{0};
	int8_t _hover_mode_id{-1};     // Structure Hover; stays -1 with SH_HB_GUARD 0
	int8_t _hover_arming_check_id{-1};

	// Vehicle state cache
	vehicle_status_s _vehicle_status{};
	vehicle_local_position_s _local_position{};
	vehicle_attitude_s _attitude{};
	vehicle_angular_velocity_s _angular_velocity{};
	float _hover_thrust{0.5f};
	bool _hover_thrust_from_estimator{false};
	bool _armed{false};
	bool _landed{true};  // vehicle_land_detected; on the ground until it says otherwise

	// Mode state
	uint8_t _state{structure_hold_status_s::STATE_UNREGISTERED};
	int8_t _active_mode_id{-1};    // the mode that is flying (or handing back): hold or land
	uint8_t _land_xy_reset_counter{0};  // Structure Land and Structure Hover: estimator resets
	uint8_t _land_z_reset_counter{0};   // the setpoint follows, as in PX4's own modes
	uint8_t _land_heading_reset_counter{0};
	uint8_t _hold_heading_reset_counter{0};  // Structure Hold: the structure yaw follows a heading reset
	structure_hold::HandbackPlan _handback_plan{};  // targets of the running handback (Handback.hpp)
	float _guard_speed{0.f};            // [m/s] the guard's recent push rate, away from the wall
	uint8_t _config_resend{0};          // which mode's control config the next poll re-sends
	structure_hold::Watchdog _watchdog{};
	structure_hold::HoldGains _gains{};
	structure_hold::HoldSetpoint _setpoint{};
	structure_hold::HoldOutput _last_output{};
	structure_hold::PolicyController _policy{};
	float _last_action[4] {};
	bool _fallback_active{false};
	float _stale_s{0.f};
	structure_hold::FrozenFix _frozen_fix{};  // how long the odometry fix has not moved
	float _fix_frozen_s{0.f};
	float _last_estimate_age_s{0.f};

	// Forward rangefinder and the floor it puts under the hold point (WallGuard.hpp):
	// the newest forward-facing distance_sensor sample from any instance.
	structure_hold::RangeSample _range{};
	structure_hold::WallGuard _wall_guard{};
	structure_hold::WallGuardOutput _wall_out{};
	structure_hold::RangeReference _range_ref{};  // the wall-range input's reference

	// Keep-out from PX4's collision-prevention data (KeepOut.hpp): the newest
	// obstacle_distance scan, and per distance_sensor instance the newest reading of a
	// sensor facing a horizontal direction other than forward, with its bearing off the nose.
	static constexpr int SIDE_RANGES = ORB_MULTI_MAX_INSTANCES;  // distance_sensor instances
	structure_hold::ObstacleScan _scan{};
	structure_hold::RangeSample _side_range[SIDE_RANGES] {};
	float _side_yaw[SIDE_RANGES] {};
	structure_hold::KeepOut _keep_out{};
	structure_hold::KeepOutOutput _keep_out_out{};
	param_t _param_cp_dist{PARAM_INVALID};  // PX4's CP_DIST, if this build has it
	bool _keep_out_warned{false};          // "keep-out moved the hold point" sent this engage
	bool _no_room_warned{false};

	// Structure Hover on the range alone (SH_HB_RANGE, RangeHover.hpp): the controller, the
	// wind the vehicle leaned into while the estimate was good, when a usable range was
	// last seen, and since when the estimate has been back.
	structure_hold::RangeHover _range_hover{};
	structure_hold::WindFeedForward _wind_ff{};
	hrt_abstime _wind_ff_us{0};
	hrt_abstime _range_hover_seen_us{0};
	hrt_abstime _range_hover_xy_back_us{0};
	bool _range_hover_handback{false};  // the handback running started from a range hover

	// Position-fix tracking for the stale time (docs/fallback.md): a fix is a new
	// vehicle_visual_odometry sample with a finite position (the VIO / estimate-bridge path),
	// or, without any odometry source, a vehicle_local_position update.
	hrt_abstime _last_fix_us{0};
	hrt_abstime _last_fix_sample_us{0};
	bool _odometry_seen{false};
	uint32_t _last_loop_time_us{0};
	uint8_t _trip_count{0};
	structure_hold::Trip _last_trip{structure_hold::Trip::None};  // latched or not; cleared on disarm
	uint8_t _handback_step{0};
	hrt_abstime _handback_request_us{0};
	bool _handback_logged_last{false};  // "requesting Land" printed once, not at every repeat
	matrix::Vector3f _prev_pos{};
	matrix::Vector3f _vel_fd{};
	hrt_abstime _prev_pos_us{0};
	bool _vel_fd_valid{false};
	hrt_abstime _last_control_us{0};
	hrt_abstime _engaged_us{0};  // the policy's engage-time input counts from here
	hrt_abstime _last_status_us{0};

	// The pilot's live view: warnings near a watchdog limit, the status line, the live array
	structure_hold::PilotStatus _pilot{};
	structure_hold::Margin _margin{};    // closest watchdog limit at the last flying tick
	float _hold_error_m{-1.f};           // the watchdog's hold error then; -1 on the range alone
	float _wd_stale_s{0.f};              // the watchdog's no-fix time then
	hrt_abstime _last_live_us{0};

	perf_counter_t _loop_perf{perf_alloc(PC_ELAPSED, MODULE_NAME": control step")};
	perf_counter_t _tick_perf{perf_alloc(PC_INTERVAL, MODULE_NAME": control tick")};
	uint32_t _max_loop_time_us{0};  // worst control step since boot (print_status, bench test)
	bool _setpoint_replaced{false};  // the last rate setpoint needed sanitizeSetpoint

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::SH_WD_EST_TO>) _param_wd_est_to,
		(ParamFloat<px4::params::SH_WD_POS_ERR>) _param_wd_pos_err,
		(ParamFloat<px4::params::SH_WD_TILT>) _param_wd_tilt,
		(ParamFloat<px4::params::SH_WD_FRZ_T>) _param_wd_frz_t,
		(ParamFloat<px4::params::SH_RST_HOLD>) _param_rst_hold,
		(ParamInt<px4::params::SH_TEST_FAULT>) _param_test_fault,
		(ParamInt<px4::params::SH_BENCH>) _param_bench,
		(ParamFloat<px4::params::SH_P_POS>) _param_p_pos,
		(ParamFloat<px4::params::SH_P_VEL>) _param_p_vel,
		(ParamFloat<px4::params::SH_P_ATT>) _param_p_att,
		(ParamFloat<px4::params::SH_V_MAX>) _param_v_max,
		(ParamInt<px4::params::SH_VEL_SRC>) _param_vel_src,
		(ParamInt<px4::params::SH_CTRL>) _param_ctrl,
		(ParamInt<px4::params::SH_FB_EN>) _param_fb_en,
		(ParamFloat<px4::params::SH_FB_T>) _param_fb_t,
		(ParamFloat<px4::params::SH_FB_K>) _param_fb_k,
		(ParamFloat<px4::params::SH_FB_TILT>) _param_fb_tilt,
		(ParamInt<px4::params::SH_FIX_SRC>) _param_fix_src,
		(ParamFloat<px4::params::SH_FIX_GAP>) _param_fix_gap,
		(ParamInt<px4::params::SH_YAW_GUARD>) _param_yaw_guard,
		(ParamFloat<px4::params::SH_YAW_P>) _param_yaw_p,
		(ParamFloat<px4::params::SH_RATE_MAX>) _param_rate_max,
		(ParamFloat<px4::params::SH_WALL_MIN>) _param_wall_min,
		(ParamInt<px4::params::SH_WALL_REQ>) _param_wall_req,
		(ParamInt<px4::params::SH_LAND_EN>) _param_land_en,
		(ParamInt<px4::params::SH_HB_GUARD>) _param_hb_guard,
		(ParamInt<px4::params::SH_HB_RANGE>) _param_hb_range,
		(ParamFloat<px4::params::SH_KO_DIST>) _param_ko_dist,
		(ParamFloat<px4::params::SH_GCS_PER>) _param_gcs_per,
		(ParamFloat<px4::params::MPC_THR_HOVER>) _param_mpc_thr_hover,
		(ParamFloat<px4::params::MPC_LAND_SPEED>) _param_mpc_land_speed
	)
};
