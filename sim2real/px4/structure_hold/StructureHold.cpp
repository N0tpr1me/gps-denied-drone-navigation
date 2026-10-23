#include "StructureHold.hpp"

#include <commander/px4_custom_mode.h>
#include <px4_platform_common/log.h>

#include <cstring>

using namespace structure_hold;
using matrix::Dcmf;
using matrix::Quatf;
using matrix::Vector3f;

StructureHold::StructureHold() :
	ModuleParams(nullptr),
	WorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers)
{
	updateParams();
}

StructureHold::~StructureHold()
{
	perf_free(_loop_perf);
}

bool StructureHold::init()
{
	// Run() on every angular-velocity sample; the control step itself is decimated to 100 Hz.
	if (!_angular_velocity_sub.registerCallback()) {
		PX4_ERR("callback registration failed");
		return false;
	}

	return true;
}

void StructureHold::updateParams()
{
	ModuleParams::updateParams();

	WatchdogLimits limits{};
	limits.estimate_timeout_us = static_cast<uint64_t>(_param_wd_est_to.get() * 1e6f);
	limits.max_position_error_m = _param_wd_pos_err.get();
	limits.max_tilt_rad = math::radians(_param_wd_tilt.get());
	limits.reset_holdoff_us = static_cast<uint64_t>(math::max(_param_rst_hold.get(), 0.f) * 1e6f);
	_watchdog.setLimits(limits);

	_gains.p_pos = _param_p_pos.get();
	_gains.p_vel = _param_p_vel.get();
	_gains.p_att = _param_p_att.get();
	_gains.v_max = _param_v_max.get();

	// Per-policy defaults (policy.json: output.yaw, output.fallback, output.max_rate_rad_s)
	// unless the parameter overrides them (PolicyController.hpp, resolve*).
	FallbackRule rule{};
	rule.enabled = resolveFallback(_param_fb_en.get());
	rule.forced = _param_fb_en.get() == 2;
	rule.threshold_s = _param_fb_t.get();
	rule.brake_vel = _param_fb_k.get();
	rule.tilt = _param_fb_tilt.get();
	rule.yaw = _param_yaw_p.get();  // one heading gain for the guard and the fallback
	_policy.setRule(rule);

	YawGuard guard{};
	guard.enabled = resolveYawGuard(_param_yaw_guard.get());
	guard.gain = _param_yaw_p.get();
	_policy.setYawGuard(guard);

	SetpointLimits limits_sp{};
	limits_sp.thrust_min = _gains.thrust_min;
	limits_sp.thrust_max = _gains.thrust_max;
	limits_sp.rate_max = resolveRateMax(_param_rate_max.get());
	_policy.setLimits(limits_sp);
}

// ---------------------------------------------------------------------------------------
// Mode registration (PX4 external-component handshake, driven from inside the firmware)

void StructureHold::sendRegistration()
{
	register_ext_component_request_s request{};
	request.timestamp = hrt_absolute_time();
	_request_id = request.timestamp ^ 0x5348u;  // "SH": any value unique among registrations
	request.request_id = _request_id;
	strncpy(request.name, MODE_NAME, sizeof(request.name) - 1);
	request.px4_ros2_api_version = register_ext_component_request_s::LATEST_PX4_ROS2_API_VERSION;
	request.register_arming_check = true;  // required for a mode
	request.register_mode = true;
	request.register_mode_executor = false;
	_register_request_pub.publish(request);
	_registration_sent = true;
}

void StructureHold::sendUnregistration()
{
	unregister_ext_component_s request{};
	request.timestamp = hrt_absolute_time();
	strncpy(request.name, MODE_NAME, sizeof(request.name) - 1);
	request.arming_check_id = _arming_check_id;
	request.mode_id = _mode_id;
	request.mode_executor_id = -1;
	_unregister_pub.publish(request);
}

bool StructureHold::checkRegistrationReply()
{
	register_ext_component_reply_s reply;
	int tries = register_ext_component_reply_s::ORB_QUEUE_LENGTH;

	while (_register_reply_sub.update(&reply) && --tries >= 0) {
		if (reply.request_id != _request_id) {
			continue;
		}

		if (!reply.success) {
			PX4_ERR("mode registration rejected by commander");
			return false;
		}

		_mode_id = reply.mode_id;
		_arming_check_id = reply.arming_check_id;
		_state = structure_hold_status_s::STATE_STANDBY;
		PX4_INFO("mode registered: mode_id %d, arming_check_id %d", _mode_id, _arming_check_id);
		sendControlModeConfig();
		return true;
	}

	return false;
}

void StructureHold::sendControlModeConfig()
{
	// What commander enables while our nav_state is active: only the rate loop and
	// control allocation. Attitude and position controllers stay idle so our
	// vehicle_rates_setpoint is the one mc_rate_control follows.
	vehicle_control_mode_s config{};
	config.timestamp = hrt_absolute_time();
	config.source_id = _mode_id;
	config.flag_control_rates_enabled = true;
	config.flag_control_allocation_enabled = true;
	config.flag_control_attitude_enabled = false;
	config.flag_control_position_enabled = false;
	config.flag_control_velocity_enabled = false;
	config.flag_control_altitude_enabled = false;
	config.flag_control_climb_rate_enabled = false;
	config.flag_control_acceleration_enabled = false;
	config.flag_control_manual_enabled = false;
	config.flag_control_auto_enabled = false;
	config.flag_control_offboard_enabled = false;
	config.flag_control_termination_enabled = false;
	_config_control_setpoints_pub.publish(config);
}

void StructureHold::replyToArmingCheck(uint8_t request_id)
{
	arming_check_reply_s reply{};
	reply.timestamp = hrt_absolute_time();
	reply.request_id = request_id;
	reply.registration_id = _arming_check_id;
	reply.health_component_index = arming_check_reply_s::HEALTH_COMPONENT_INDEX_NONE;
	reply.num_events = 0;
	// Commander refuses to switch into the mode while a trip is latched or for SH_RST_HOLD
	// after an EKF position reset (Watchdog.hpp).
	reply.can_arm_and_run = _watchdog.latched() == Trip::None && !_watchdog.inResetHoldoff(hrt_absolute_time());
	reply.mode_req_angular_velocity = true;
	reply.mode_req_attitude = true;
	reply.mode_req_local_alt = true;
	reply.mode_req_local_position = true;
	_arming_check_reply_pub.publish(reply);
}

// ---------------------------------------------------------------------------------------
// Flight

void StructureHold::engage(const vehicle_local_position_s &lpos)
{
	_setpoint.pos = Vector3f(lpos.x, lpos.y, lpos.z);
	_setpoint.yaw = lpos.heading;
	_state = structure_hold_status_s::STATE_ENGAGED;
	_last_control_us = 0;
	_vel_fd_valid = false;
	_fallback_active = false;

	for (int i = 0; i < 4; i++) { _last_action[i] = 0.f; }

	float hold[3];
	_setpoint.pos.copyTo(hold);
	_policy.engage(hold, _setpoint.yaw);  // structure x = the heading at engage
	PX4_INFO("engaged (%s): holding N %.2f E %.2f D %.2f m, yaw %.0f deg",
		 _param_ctrl.get() == 1 ? "policy" : "cascade",
		 (double)_setpoint.pos(0), (double)_setpoint.pos(1), (double)_setpoint.pos(2),
		 (double)math::degrees(_setpoint.yaw));
}

void StructureHold::disengage(const char *why)
{
	if (_state == structure_hold_status_s::STATE_ENGAGED) {
		PX4_INFO("disengaged: %s", why);
	}

	_state = structure_hold_status_s::STATE_STANDBY;
}

void StructureHold::handBack(Trip trip, bool xy_valid, bool latch)
{
	if (latch) {
		_watchdog.latch(trip);
	}

	_last_trip = trip;

	if (_trip_count < 255) {
		++_trip_count;
	}

	_state = structure_hold_status_s::STATE_HANDBACK;
	_handback_step = 0;
	_handback_xy_valid = xy_valid;
	PX4_WARN("watchdog %s: %s, handing back", latch ? "trip" : "refusal", Watchdog::name(trip));
	requestHandbackMode();
}

void StructureHold::requestHandbackMode()
{
	// Preferred target first; commander may refuse (no manual control for Position mode,
	// no position for Hold), so each step falls through to the next after a short wait.
	uint8_t main_mode = PX4_CUSTOM_MAIN_MODE_POSCTL;
	uint8_t sub_mode = 0;
	const char *name = "Position mode";

	switch (_handback_step) {
	case 0:
		if (!_handback_xy_valid) {
			main_mode = PX4_CUSTOM_MAIN_MODE_ALTCTL;
			name = "Altitude mode";
		}

		break;

	case 1:
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_AUTO_LOITER;
		name = "Hold";
		break;

	default:
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_AUTO_LAND;
		name = "Land";
		break;
	}

	vehicle_command_s cmd{};
	cmd.timestamp = hrt_absolute_time();
	cmd.command = vehicle_command_s::VEHICLE_CMD_DO_SET_MODE;
	cmd.param1 = 1.f;  // custom mode enabled
	cmd.param2 = main_mode;
	cmd.param3 = sub_mode;
	cmd.target_system = _vehicle_status.system_id;
	cmd.target_component = _vehicle_status.component_id;
	cmd.source_system = _vehicle_status.system_id;
	cmd.source_component = _vehicle_status.component_id;
	cmd.confirmation = 0;
	cmd.from_external = false;
	_vehicle_command_pub.publish(cmd);
	_handback_request_us = cmd.timestamp;

	PX4_INFO("handback: requesting %s", name);
}

void StructureHold::handbackStep(const hrt_abstime now)
{
	// Still the active mode: keep the vehicle level at hover thrust while commander
	// switches, and escalate if the requested mode was refused.
	vehicle_rates_setpoint_s sp{};
	sp.timestamp = hrt_absolute_time();
	sp.roll = 0.f;
	sp.pitch = 0.f;
	sp.yaw = 0.f;
	sp.thrust_body[2] = -_hover_thrust;
	_rates_setpoint_pub.publish(sp);

	if (now - _handback_request_us > HANDBACK_RETRY && _handback_step < 2) {
		++_handback_step;
		requestHandbackMode();
	}
}

bool StructureHold::useOdometryFix() const
{
	const int32_t src = _param_fix_src.get();
	return src == 1 || (src == 0 && _odometry_seen);
}

void StructureHold::updateFix(const hrt_abstime now)
{
	// A fix is a new odometry sample with a finite position. The estimate bridge (and a
	// real VIO) publishes nothing during a dropout, so the gap is what we measure.
	vehicle_odometry_s odom;

	while (_visual_odometry_sub.update(&odom)) {
		if (PX4_ISFINITE(odom.position[0]) && PX4_ISFINITE(odom.position[1]) && PX4_ISFINITE(odom.position[2])) {
			_last_fix_us = odom.timestamp > 0 ? odom.timestamp : now;
			_last_fix_sample_us = odom.timestamp_sample > 0 ? odom.timestamp_sample : _last_fix_us;
			_odometry_seen = true;
		}
	}

	if (!useOdometryFix()) {
		_last_fix_us = _local_position.timestamp;
		_last_fix_sample_us = _local_position.timestamp_sample > 0 ? _local_position.timestamp_sample
				      : _local_position.timestamp;
	}

	_stale_s = now > _last_fix_us ? (now - _last_fix_us) * 1e-6f : 0.f;
	_last_estimate_age_s = now > _last_fix_sample_us ? (now - _last_fix_sample_us) * 1e-6f : 0.f;
}

void StructureHold::controlStep(const hrt_abstime now)
{
	perf_begin(_loop_perf);
	const hrt_abstime start = hrt_absolute_time();

	// Hover thrust from the estimator when it has one, else the parameter.
	hover_thrust_estimate_s hte;

	if (_hover_thrust_sub.update(&hte) && hte.valid) {
		_hover_thrust = hte.hover_thrust;
		_hover_thrust_from_estimator = true;

	} else if (!_hover_thrust_from_estimator) {
		_hover_thrust = _param_mpc_thr_hover.get();
	}

	// Watchdog first: it decides whether the controller runs at all this tick. The stale
	// trip counts from the last position fix (or the last local position update, whichever
	// is older), so an EKF2 that stops publishing trips it too.
	const Quatf q(_attitude.q);
	const Dcmf r(q);

	WatchdogInputs in{};
	in.now_us = now;
	in.estimate_timestamp_us = math::min(_last_fix_us, static_cast<hrt_abstime>(_local_position.timestamp));
	in.xy_valid = _local_position.xy_valid;
	in.z_valid = _local_position.z_valid;
	in.v_xy_valid = _local_position.v_xy_valid;
	in.tilt_rad = acosf(math::constrain(r(2, 2), -1.f, 1.f));
	in.position_error_m = (_setpoint.pos - Vector3f(_local_position.x, _local_position.y, _local_position.z)).norm();
	in.test_fault = _param_test_fault.get() != 0;

	const Trip trip = _watchdog.check(in);

	if (trip != Trip::None) {
		if (in.test_fault) {
			_param_test_fault.set(0);  // one-shot test hook
			_param_test_fault.commit_no_notification();
		}

		handBack(trip, in.xy_valid);
		perf_end(_loop_perf);
		return;
	}

	const Vector3f pos(_local_position.x, _local_position.y, _local_position.z);
	Vector3f vel(_local_position.vx, _local_position.vy, _local_position.vz);

	if (_param_vel_src.get() == 1) {
		// Velocity from the differentiated position estimate, low-pass filtered (about
		// 5 Hz). For simulators whose velocity estimate is unusable; see README.
		if (_vel_fd_valid && now > _prev_pos_us) {
			const float dt = math::constrain((now - _prev_pos_us) * 1e-6f, 0.002f, 0.1f);
			const Vector3f raw = (pos - _prev_pos) / dt;
			const float alpha = dt / (0.032f + dt);
			_vel_fd = _vel_fd + (raw - _vel_fd) * alpha;

		} else {
			_vel_fd = vel;
		}

		_prev_pos = pos;
		_prev_pos_us = now;
		_vel_fd_valid = true;
		vel = _vel_fd;
	}

	if (_param_ctrl.get() == 1) {
		// Policy v2 with the brake-and-hover fallback (PolicyController.hpp).
		VehicleSample sample{};
		pos.copyTo(sample.pos_ned);
		vel.copyTo(sample.vel_ned);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				sample.R[i][j] = r(i, j);
			}
		}

		sample.rates_frd[0] = _angular_velocity.xyz[0];
		sample.rates_frd[1] = _angular_velocity.xyz[1];
		sample.rates_frd[2] = _angular_velocity.xyz[2];
		sample.stale = _stale_s > _param_fix_gap.get();
		// The env's age is 0 whenever the estimate is fresh and counts up only during a
		// dropout (envs/estimate/core.py), so the policy never saw a nonzero age with
		// stale = 0. Feeding the time since the last fix instead (a 0 to 24 ms sawtooth at
		// the 50 Hz vision rate in Gazebo; a constant 0.1 s shifts the v5 hold 0.1 m in the
		// mock) is an input it was not trained on. It stays in the status message as
		// estimate_age_s.
		sample.age_s = sample.stale ? _stale_s : 0.f;

		const PolicyOutput out = _policy.step(sample, _stale_s, _hover_thrust);

		if (!out.ok) {
			PX4_WARN("policy produced a non-finite action");
		}

		if (out.fallback_active != _fallback_active) {
			PX4_INFO("%s", out.fallback_active ? "fallback: braking, estimate stale" : "fallback: fresh fix, policy flies");
		}

		_fallback_active = out.fallback_active;

		for (int i = 0; i < 4; i++) { _last_action[i] = out.action[i]; }

		_last_output.rates = Vector3f(out.setpoint.rates_frd);
		_last_output.thrust = out.setpoint.thrust;
		_last_output.pos_err = _setpoint.pos - pos;
		_last_output.vel_sp.zero();

	} else {
		// Placeholder cascade: hold the captured setpoint.
		HoldState st{};
		st.pos = pos;
		st.vel = vel;
		st.q = q;
		_last_output = holdStep(_gains, st, _setpoint, _hover_thrust);
		_fallback_active = false;
	}

	vehicle_rates_setpoint_s sp{};
	sp.timestamp = hrt_absolute_time();
	sp.roll = _last_output.rates(0);
	sp.pitch = _last_output.rates(1);
	sp.yaw = _last_output.rates(2);
	sp.thrust_body[0] = 0.f;
	sp.thrust_body[1] = 0.f;
	sp.thrust_body[2] = -_last_output.thrust;
	sp.reset_integral = false;
	_rates_setpoint_pub.publish(sp);

	_last_loop_time_us = static_cast<uint32_t>(hrt_absolute_time() - start);
	perf_end(_loop_perf);
}

void StructureHold::publishStatus(const hrt_abstime now)
{
	structure_hold_status_s status{};
	status.timestamp = now;
	status.state = _state;
	status.trip_reason = static_cast<uint8_t>(_last_trip);
	status.trip_count = _trip_count;
	status.mode_id = _mode_id;
	status.arming_check_id = _arming_check_id;
	_setpoint.pos.copyTo(status.setpoint_ned);
	status.yaw_setpoint = _setpoint.yaw;
	_last_output.pos_err.copyTo(status.position_error_ned);
	status.estimate_age_s = _last_estimate_age_s;
	status.stale_s = _stale_s;
	status.reset_age_s = _watchdog.resetAge(now);
	status.controller = _param_ctrl.get() == 1 ? structure_hold_status_s::CTRL_POLICY
			    : structure_hold_status_s::CTRL_CASCADE;
	status.fallback_active = _fallback_active;
	status.yaw_guarded = _param_ctrl.get() == 1 && _policy.yawGuard().enabled && !_fallback_active;
	status.odometry_fix = useOdometryFix();

	for (int i = 0; i < 4; i++) { status.action[i] = _last_action[i]; }
	_last_output.rates.copyTo(status.rates_setpoint);
	status.thrust_setpoint = _last_output.thrust;
	status.loop_time_us = _last_loop_time_us;
	_status_pub.publish(status);
	_last_status_us = now;
}

// ---------------------------------------------------------------------------------------

void StructureHold::Run()
{
	if (should_exit()) {
		_angular_velocity_sub.unregisterCallback();

		if (_mode_id != -1) {
			sendUnregistration();
		}

		exit_and_cleanup();
		return;
	}

	// Consume the sample that woke us; the policy observes the body rates from it.
	_angular_velocity_sub.update(&_angular_velocity);

	const hrt_abstime now = hrt_absolute_time();

	if (!_registration_sent) {
		sendRegistration();
		return;
	}

	if (_mode_id == -1) {
		checkRegistrationReply();
		return;
	}

	if (_parameter_update_sub.updated()) {
		parameter_update_s update;
		_parameter_update_sub.copy(&update);
		updateParams();
	}

	// Commander polls registered components every 300 ms and wants a reply within 50 ms.
	if (_arming_check_request_sub.updated()) {
		arming_check_request_s request;

		if (_arming_check_request_sub.copy(&request)) {
			replyToArmingCheck(request.request_id);
		}
	}

	if (_vehicle_status_sub.updated()) {
		_vehicle_status_sub.copy(&_vehicle_status);
		const bool armed = _vehicle_status.arming_state == vehicle_status_s::ARMING_STATE_ARMED;

		if (_armed && !armed && _watchdog.latched() != Trip::None) {
			PX4_INFO("disarmed: clearing latched trip (%s)", Watchdog::name(_watchdog.latched()));
			_watchdog.clearLatch();
		}

		if (_armed && !armed) {
			_last_trip = Trip::None;
		}

		_armed = armed;
	}

	if (_local_position_sub.update(&_local_position)) {
		_watchdog.noteReset(_local_position.xy_reset_counter, now);
	}

	_attitude_sub.update(&_attitude);
	updateFix(now);

	const bool mode_active = _vehicle_status.nav_state == _mode_id;

	switch (_state) {
	case structure_hold_status_s::STATE_STANDBY:
		if (mode_active) {
			if (_watchdog.inResetHoldoff(now)) {
				// Commander normally refuses the switch (replyToArmingCheck), but its poll
				// is 300 ms behind the reset: hand straight back, nothing latched.
				PX4_WARN("engage refused: EKF reset its position %.1f s ago (SH_RST_HOLD %.1f s)",
					 (double)_watchdog.resetAge(now), (double)_param_rst_hold.get());
				handBack(Trip::EstimateReset, _local_position.xy_valid, false);

			} else {
				engage(_local_position);
			}
		}

		break;

	case structure_hold_status_s::STATE_ENGAGED:
		if (!mode_active) {
			disengage("mode changed");

		} else if (now - _last_control_us >= CONTROL_INTERVAL) {
			_last_control_us = now;
			controlStep(now);
		}

		break;

	case structure_hold_status_s::STATE_HANDBACK:
		if (!mode_active) {
			PX4_INFO("handback complete: stock PX4 in charge");
			_state = structure_hold_status_s::STATE_STANDBY;

		} else if (now - _last_control_us >= CONTROL_INTERVAL) {
			_last_control_us = now;
			handbackStep(now);
		}

		break;

	default:
		break;
	}

	const bool controlling = _state == structure_hold_status_s::STATE_ENGAGED
				 || _state == structure_hold_status_s::STATE_HANDBACK;

	if ((controlling && _last_control_us == now) || now - _last_status_us >= STANDBY_STATUS_INTERVAL) {
		publishStatus(now);
	}
}

// ---------------------------------------------------------------------------------------
// Module boilerplate

int StructureHold::task_spawn(int argc, char *argv[])
{
	StructureHold *instance = new StructureHold();

	if (instance) {
		_object.store(instance);
		_task_id = task_id_is_work_queue;

		if (instance->init()) {
			return PX4_OK;
		}

	} else {
		PX4_ERR("alloc failed");
	}

	delete instance;
	_object.store(nullptr);
	_task_id = -1;

	return PX4_ERROR;
}

int StructureHold::custom_command(int argc, char *argv[])
{
	if (argc > 0 && !strcmp(argv[0], "trip")) {
		if (!is_running()) {
			PX4_ERR("not running");
			return 1;
		}

		// Same path as the SH_TEST_FAULT parameter, for use from the shell.
		int32_t one = 1;
		param_set(param_find("SH_TEST_FAULT"), &one);
		return 0;
	}

	return print_usage("unknown command");
}

int StructureHold::print_status()
{
	static const char *state_names[] = {"unregistered", "standby", "engaged", "handback"};
	PX4_INFO("state: %s", state_names[_state < 4 ? _state : 0]);
	PX4_INFO("mode_id %d, arming_check_id %d", _mode_id, _arming_check_id);
	PX4_INFO("latched trip: %s, last: %s, trips since boot: %u, EKF reset %.1f s ago (hold-off %.1f s)",
		 Watchdog::name(_watchdog.latched()), Watchdog::name(_last_trip), _trip_count,
		 (double)_watchdog.resetAge(hrt_absolute_time()), (double)_param_rst_hold.get());
	PX4_INFO("controller: %s (policy %s, history %d x %d), fallback %s, fix source %s, stale %.2f s",
		 _param_ctrl.get() == 1 ? "policy" : "cascade", policy::NAME, policy::HISTORY, policy::NUM_OBS,
		 _fallback_active ? "ACTIVE" : "idle", useOdometryFix() ? "visual odometry" : "local position",
		 (double)_stale_s);
	PX4_INFO("policy defaults: yaw %s, fallback %s, rates %.1f rad/s; flying: yaw guard %s, fallback %s, rate clamp %.1f rad/s",
		 policy::YAW_FROM_POLICY ? "policy" : "module", policy::FALLBACK_DEFAULT_ON ? "on" : "off",
		 (double)policy::MAX_RATE_RAD_S, _policy.yawGuard().enabled ? "on" : "off",
		 _policy.rule().forced ? "forced" : (_policy.rule().enabled ? "on" : "off"),
		 (double)resolveRateMax(_param_rate_max.get()));

	if (_state == structure_hold_status_s::STATE_ENGAGED) {
		PX4_INFO("setpoint NED: %.2f %.2f %.2f m, yaw %.1f deg", (double)_setpoint.pos(0), (double)_setpoint.pos(1),
			 (double)_setpoint.pos(2), (double)math::degrees(_setpoint.yaw));
		PX4_INFO("error NED: %.2f %.2f %.2f m, thrust %.2f", (double)_last_output.pos_err(0),
			 (double)_last_output.pos_err(1), (double)_last_output.pos_err(2), (double)_last_output.thrust);
	}

	perf_print_counter(_loop_perf);
	return 0;
}

int StructureHold::print_usage(const char *reason)
{
	if (reason) {
		PX4_WARN("%s\n", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
Structure Hold flight mode. Registers "Structure Hold" as a named mode with commander and
holds the position and yaw it had when engaged by publishing body-rate and thrust setpoints
to the stock rate controller: policy v2 (compiled-in MLP) with a classical brake-and-hover
fallback once the position fix is stale (SH_CTRL 1), or a placeholder P cascade (SH_CTRL 0).
The watchdog hands control back to Position mode (or Altitude mode) when it trips. Status
is published on structure_hold_status.

Engage from the shell with `commander mode ext1` (first registered external mode) or from
a ground station / MAVLink as the external mode named "Structure Hold".
)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("structure_hold", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_COMMAND_DESCR("trip", "Trip the watchdog now (test hook, same as SH_TEST_FAULT 1)");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();

	return 0;
}

extern "C" __EXPORT int structure_hold_main(int argc, char *argv[])
{
	return StructureHold::main(argc, argv);
}
