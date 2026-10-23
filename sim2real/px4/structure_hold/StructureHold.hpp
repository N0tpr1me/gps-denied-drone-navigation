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
 */

#pragma once

#include <drivers/drv_hrt.h>
#include <lib/perf/perf_counter.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/posix.h>
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/topics/arming_check_reply.h>
#include <uORB/topics/arming_check_request.h>
#include <uORB/topics/hover_thrust_estimate.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/register_ext_component_reply.h>
#include <uORB/topics/register_ext_component_request.h>
#include <uORB/topics/structure_hold_status.h>
#include <uORB/topics/unregister_ext_component.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_odometry.h>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_status.h>

#include "HoldController.hpp"
#include "PolicyController.hpp"
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
	static constexpr hrt_abstime CONTROL_INTERVAL = 10_ms;  // 100 Hz outer loop
	static constexpr hrt_abstime STANDBY_STATUS_INTERVAL = 200_ms;
	static constexpr hrt_abstime HANDBACK_RETRY = 300_ms;  // then Hold, then Land

private:
	void Run() override;

	// Mode registration with commander
	void sendRegistration();
	void sendUnregistration();
	bool checkRegistrationReply();
	void sendControlModeConfig();
	void replyToArmingCheck(uint8_t request_id);

	// Flight
	void engage(const vehicle_local_position_s &lpos);
	void disengage(const char *why);
	void handBack(structure_hold::Trip trip, bool xy_valid, bool latch = true);
	void requestHandbackMode();
	void handbackStep(const hrt_abstime now);
	void controlStep(const hrt_abstime now);
	void updateFix(const hrt_abstime now);
	bool useOdometryFix() const;
	void publishStatus(const hrt_abstime now);
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

	// Publications
	uORB::Publication<register_ext_component_request_s> _register_request_pub{ORB_ID(register_ext_component_request)};
	uORB::Publication<unregister_ext_component_s> _unregister_pub{ORB_ID(unregister_ext_component)};
	uORB::Publication<vehicle_control_mode_s> _config_control_setpoints_pub{ORB_ID(config_control_setpoints)};
	uORB::Publication<arming_check_reply_s> _arming_check_reply_pub{ORB_ID(arming_check_reply)};
	uORB::Publication<vehicle_rates_setpoint_s> _rates_setpoint_pub{ORB_ID(vehicle_rates_setpoint)};
	uORB::Publication<vehicle_command_s> _vehicle_command_pub{ORB_ID(vehicle_command)};
	uORB::Publication<structure_hold_status_s> _status_pub{ORB_ID(structure_hold_status)};

	// Registration state
	bool _registration_sent{false};
	uint64_t _request_id{0};
	int8_t _mode_id{-1};
	int8_t _arming_check_id{-1};

	// Vehicle state cache
	vehicle_status_s _vehicle_status{};
	vehicle_local_position_s _local_position{};
	vehicle_attitude_s _attitude{};
	vehicle_angular_velocity_s _angular_velocity{};
	float _hover_thrust{0.5f};
	bool _hover_thrust_from_estimator{false};
	bool _armed{false};

	// Mode state
	uint8_t _state{structure_hold_status_s::STATE_UNREGISTERED};
	structure_hold::Watchdog _watchdog{};
	structure_hold::HoldGains _gains{};
	structure_hold::HoldSetpoint _setpoint{};
	structure_hold::HoldOutput _last_output{};
	structure_hold::PolicyController _policy{};
	float _last_action[4] {};
	bool _fallback_active{false};
	float _stale_s{0.f};
	float _last_estimate_age_s{0.f};

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
	bool _handback_xy_valid{true};
	hrt_abstime _handback_request_us{0};
	matrix::Vector3f _prev_pos{};
	matrix::Vector3f _vel_fd{};
	hrt_abstime _prev_pos_us{0};
	bool _vel_fd_valid{false};
	hrt_abstime _last_control_us{0};
	hrt_abstime _last_status_us{0};

	perf_counter_t _loop_perf{perf_alloc(PC_ELAPSED, MODULE_NAME": control step")};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::SH_WD_EST_TO>) _param_wd_est_to,
		(ParamFloat<px4::params::SH_WD_POS_ERR>) _param_wd_pos_err,
		(ParamFloat<px4::params::SH_WD_TILT>) _param_wd_tilt,
		(ParamFloat<px4::params::SH_RST_HOLD>) _param_rst_hold,
		(ParamInt<px4::params::SH_TEST_FAULT>) _param_test_fault,
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
		(ParamFloat<px4::params::MPC_THR_HOVER>) _param_mpc_thr_hover
	)
};
