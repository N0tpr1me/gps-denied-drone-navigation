#include "StructureHold.hpp"

#include <commander/px4_custom_mode.h>
#include <px4_platform_common/log.h>

#include <cstdlib>
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

	WallGuardLimits wall{};
	wall.min_standoff_m = math::max(_param_wall_min.get(), 0.f);
	_wall_guard.setLimits(wall);
}

// ---------------------------------------------------------------------------------------
// Mode registration (PX4 external-component handshake, driven from inside the firmware)

void StructureHold::sendRegistration(const char *name, uint64_t &request_id)
{
	register_ext_component_request_s request{};
	request.timestamp = hrt_absolute_time();
	request_id = request.timestamp ^ 0x5348u;  // "SH": any value unique among registrations
	request.request_id = request_id;
	strncpy(request.name, name, sizeof(request.name) - 1);
	request.px4_ros2_api_version = register_ext_component_request_s::LATEST_PX4_ROS2_API_VERSION;
	request.register_arming_check = true;  // required for a mode
	request.register_mode = true;
	request.register_mode_executor = false;
	_register_request_pub.publish(request);
}

void StructureHold::sendUnregistration(const char *name, int8_t mode_id, int8_t arming_check_id)
{
	unregister_ext_component_s request{};
	request.timestamp = hrt_absolute_time();
	strncpy(request.name, name, sizeof(request.name) - 1);
	request.arming_check_id = arming_check_id;
	request.mode_id = mode_id;
	request.mode_executor_id = -1;
	_unregister_pub.publish(request);
}

void StructureHold::checkRegistrationReplies()
{
	// Structure Hold registers first (so it stays the first external mode, ext1), then
	// Structure Land (ext2) and Structure Hover (ext3) once Hold's reply is in.
	register_ext_component_reply_s reply;
	int tries = register_ext_component_reply_s::ORB_QUEUE_LENGTH;

	while (_register_reply_sub.update(&reply) && --tries >= 0) {
		if (_mode_id == -1 && reply.request_id == _request_id) {
			if (!reply.success) {
				PX4_ERR("mode registration rejected by commander");
				continue;
			}

			_mode_id = reply.mode_id;
			_arming_check_id = reply.arming_check_id;
			_state = structure_hold_status_s::STATE_STANDBY;
			PX4_INFO("mode registered: mode_id %d, arming_check_id %d", _mode_id, _arming_check_id);
			sendControlModeConfig(_mode_id, false);

		} else if (_land_mode_id == -1 && _land_registration_sent && reply.request_id == _land_request_id) {
			if (!reply.success) {
				PX4_ERR("Structure Land registration rejected by commander");
				continue;
			}

			_land_mode_id = reply.mode_id;
			_land_arming_check_id = reply.arming_check_id;
			PX4_INFO("Structure Land registered: mode_id %d, arming_check_id %d", _land_mode_id,
				 _land_arming_check_id);
			sendControlModeConfig(_land_mode_id, true);

		} else if (_hover_mode_id == -1 && _hover_registration_sent && reply.request_id == _hover_request_id) {
			if (!reply.success) {
				PX4_ERR("Structure Hover registration rejected by commander");
				continue;
			}

			_hover_mode_id = reply.mode_id;
			_hover_arming_check_id = reply.arming_check_id;
			PX4_INFO("Structure Hover registered: mode_id %d, arming_check_id %d", _hover_mode_id,
				 _hover_arming_check_id);
			sendControlModeConfig(_hover_mode_id, true);
		}
	}

	// One request in flight at a time, so the external mode numbers follow this order.
	if (_mode_id == -1 || (_land_registration_sent && _land_mode_id == -1)) {
		return;
	}

	if (!_land_registration_sent && _param_land_en.get() != 0) {
		sendRegistration(LAND_MODE_NAME, _land_request_id);
		_land_registration_sent = true;

	} else if (!_hover_registration_sent && _param_hb_guard.get() != 0) {
		sendRegistration(HOVER_MODE_NAME, _hover_request_id);
		_hover_registration_sent = true;
	}
}

void StructureHold::sendControlModeConfig(int8_t mode_id, bool position_control)
{
	// What commander enables while the mode's nav_state is active. Structure Hold: only
	// the rate loop and control allocation, so our vehicle_rates_setpoint is the one
	// mc_rate_control follows. Structure Land: PX4's whole position-to-rate cascade, fed
	// by our trajectory_setpoint, as in its own position modes (the land detector reads
	// the climb-rate flag and the commanded descent to find touchdown).
	vehicle_control_mode_s config{};
	config.timestamp = hrt_absolute_time();
	config.source_id = mode_id;
	config.flag_control_rates_enabled = true;
	config.flag_control_allocation_enabled = true;
	config.flag_control_attitude_enabled = position_control;
	config.flag_control_position_enabled = position_control;
	config.flag_control_velocity_enabled = position_control;
	config.flag_control_altitude_enabled = position_control;
	config.flag_control_climb_rate_enabled = position_control;
	config.flag_control_acceleration_enabled = position_control;
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
	// With SH_WALL_REQ set, also while no forward range is coming in (WallGuard.hpp).
	const hrt_abstime now = hrt_absolute_time();
	reply.can_arm_and_run = _watchdog.latched() == Trip::None && !_watchdog.inResetHoldoff(now)
				&& (_param_wall_req.get() == 0 || rangeFresh(now));
	reply.mode_req_angular_velocity = true;
	reply.mode_req_attitude = true;
	reply.mode_req_local_alt = true;
	reply.mode_req_local_position = true;
	_arming_check_reply_pub.publish(reply);

	// Structure Land and Structure Hover only in flight. A latched trip blocks neither:
	// they are how the vehicle stays off the wall after a handback, and PX4's own position
	// controller flies them, not the policy.
	reply.can_arm_and_run = _armed;

	if (_land_arming_check_id != -1) {
		reply.registration_id = _land_arming_check_id;
		_arming_check_reply_pub.publish(reply);
	}

	if (_hover_arming_check_id != -1) {
		reply.registration_id = _hover_arming_check_id;
		_arming_check_reply_pub.publish(reply);
	}

	// config_control_setpoints has a queue of one, so a mode's config sent at registration
	// can be overwritten by the next mode's before commander reads it (Structure Land then
	// flew with no position control in SIH). Re-send one mode's config per poll, in turn.
	const int8_t ids[3] = {_mode_id, _land_mode_id, _hover_mode_id};
	_config_resend = (_config_resend + 1) % 3;

	if (ids[_config_resend] != -1) {
		sendControlModeConfig(ids[_config_resend], _config_resend != 0);
	}
}

// ---------------------------------------------------------------------------------------
// Flight

void StructureHold::engage(const vehicle_local_position_s &lpos)
{
	_setpoint.pos = Vector3f(lpos.x, lpos.y, lpos.z);
	_setpoint.yaw = lpos.heading;
	_state = structure_hold_status_s::STATE_ENGAGED;
	_active_mode_id = _mode_id;
	_last_control_us = 0;
	_vel_fd_valid = false;
	_fallback_active = false;

	for (int i = 0; i < 4; i++) { _last_action[i] = 0.f; }

	_wall_guard.engage();
	_wall_out = WallGuardOutput{};

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

void StructureHold::engageGuarded(const vehicle_local_position_s &lpos, bool descend)
{
	// The setpoint is captured on the estimate like stock Land's or Position mode's; what
	// differs is that the wall guard keeps measuring it against the wall.
	_setpoint.pos = Vector3f(lpos.x, lpos.y, lpos.z);
	_setpoint.yaw = lpos.heading;
	_state = descend ? structure_hold_status_s::STATE_LANDING : structure_hold_status_s::STATE_HOVER;
	_active_mode_id = descend ? _land_mode_id : _hover_mode_id;
	_land_xy_reset_counter = lpos.xy_reset_counter;
	_guard_speed = 0.f;
	_last_control_us = 0;
	_fallback_active = false;
	_last_output = HoldOutput{};

	for (int i = 0; i < 4; i++) { _last_action[i] = 0.f; }

	_wall_guard.engage();
	_wall_out = WallGuardOutput{};
	PX4_INFO("%s: N %.2f E %.2f D %.2f m, yaw %.0f deg%s", descend ? LAND_MODE_NAME : HOVER_MODE_NAME,
		 (double)_setpoint.pos(0), (double)_setpoint.pos(1), (double)_setpoint.pos(2),
		 (double)math::degrees(_setpoint.yaw), descend ? ", descending" : "");
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

	// A trip in Structure Hold next to a measured wall goes to Structure Hover first, and a
	// trip in Structure Land to stock Land first, since the landing was already asked for;
	// a trip in Hover, or a refusal at entry, goes straight to Position mode and on.
	_handback_guarded = latch && xy_valid && _active_mode_id == _mode_id
			    && guardedHandbackAvailable(hrt_absolute_time());
	_handback_landing = latch && _land_mode_id != -1 && _active_mode_id == _land_mode_id;
	_state = structure_hold_status_s::STATE_HANDBACK;
	_handback_step = 0;
	_handback_xy_valid = xy_valid;
	PX4_WARN("watchdog %s: %s, handing back", latch ? "trip" : "refusal", Watchdog::name(trip));
	requestHandbackMode();
}

bool StructureHold::guardedHandbackAvailable(const hrt_abstime now) const
{
	return _hover_mode_id != -1 && _wall_guard.enabled() && rangeFresh(now)
	       && _range.distance_m < HOVER_WALL_RANGE_M;
}

void StructureHold::requestHandbackMode()
{
	// Preferred target first; commander may refuse (no manual control for Position mode,
	// no position for Hold), so each step falls through to the next after a short wait.
	uint8_t main_mode = PX4_CUSTOM_MAIN_MODE_POSCTL;
	uint8_t sub_mode = 0;
	const char *name = "Position mode";

	const bool first_target = _handback_guarded || _handback_landing;

	if (_handback_guarded && _handback_step == 0) {
		// External modes' custom sub modes follow their nav_states (px4_custom_mode.h).
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_EXTERNAL1 + (_hover_mode_id - vehicle_status_s::NAVIGATION_STATE_EXTERNAL1);
		name = HOVER_MODE_NAME;

	} else if (_handback_landing && _handback_step == 0) {
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_AUTO_LAND;
		name = "Land";
	}

	switch (_handback_step - (first_target ? 1 : 0)) {
	case -1:
		break;

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
	// switches, and escalate if the requested mode was refused. From Structure Land, whose
	// rate setpoints come from PX4's own cascade, stop the descent there instead.
	if (_active_mode_id == _land_mode_id || _active_mode_id == _hover_mode_id) {
		publishTrajectorySetpoint(NAN, Vector3f(0.f, 0.f, 0.f));

	} else {
		vehicle_rates_setpoint_s sp{};
		sp.timestamp = hrt_absolute_time();
		sp.roll = 0.f;
		sp.pitch = 0.f;
		sp.yaw = 0.f;
		sp.thrust_body[2] = -_hover_thrust;
		_rates_setpoint_pub.publish(sp);
	}

	if (now - _handback_request_us > HANDBACK_RETRY && _handback_step < (_handback_guarded || _handback_landing ? 3 : 2)) {
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

void StructureHold::updateRange()
{
	// Newest forward-facing sample over all distance_sensor instances (a downward
	// rangefinder for EKF2's height is common on the same vehicle and is skipped).
	for (auto &sub : _distance_sensor_subs) {
		distance_sensor_s ds;

		while (sub.update(&ds)) {
			if (ds.orientation != distance_sensor_s::ROTATION_FORWARD_FACING || ds.timestamp < _range.timestamp_us) {
				continue;
			}

			_range.timestamp_us = ds.timestamp;
			_range.distance_m = ds.current_distance;
			_range.min_distance_m = ds.min_distance;
			_range.max_distance_m = ds.max_distance;
			_range.signal_quality = ds.signal_quality;
		}
	}
}

bool StructureHold::rangeFresh(const hrt_abstime now) const
{
	return _wall_guard.usable(_range, now);
}

void StructureHold::applyWallGuard(const hrt_abstime now, const Vector3f &pos, const Dcmf &r)
{
	float pos_ned[3], hold[3];
	pos.copyTo(pos_ned);
	_setpoint.pos.copyTo(hold);
	const float normal[2] = {cosf(_setpoint.yaw), sinf(_setpoint.yaw)};
	const float body_x[3] = {r(0, 0), r(1, 0), r(2, 0)};
	const bool shifted_before = _wall_guard.totalShift() > 0.f;
	_wall_out = _wall_guard.apply(_range, now, pos_ned, normal, body_x, hold);

	if (_wall_out.shift_m > 0.f) {
		_setpoint.pos = Vector3f(hold);

		if (_state == structure_hold_status_s::STATE_ENGAGED) {
			_policy.moveHold(hold);
		}

		if (!shifted_before) {
			PX4_WARN("wall guard: hold point %.2f m from the wall, moving it back to %.2f m",
				 (double)(_wall_out.clearance_m - _wall_out.shift_m), (double)_wall_out.clearance_m);
		}
	}
}

Trip StructureHold::checkWatchdog(const hrt_abstime now, const Dcmf &r, float position_error_m)
{
	// The stale trip counts from the last position fix (or the last local position update,
	// whichever is older), so an EKF2 that stops publishing trips it too. A trip hands
	// back to stock PX4.
	WatchdogInputs in{};
	in.now_us = now;
	in.estimate_timestamp_us = math::min(_last_fix_us, static_cast<hrt_abstime>(_local_position.timestamp));
	in.xy_valid = _local_position.xy_valid;
	in.z_valid = _local_position.z_valid;
	in.v_xy_valid = _local_position.v_xy_valid;
	in.tilt_rad = acosf(math::constrain(r(2, 2), -1.f, 1.f));
	in.position_error_m = position_error_m;
	in.test_fault = _param_test_fault.get() != 0;

	const Trip trip = _watchdog.check(in);

	if (trip != Trip::None) {
		if (in.test_fault) {
			_param_test_fault.set(0);  // one-shot test hook
			_param_test_fault.commit_no_notification();
		}

		handBack(trip, in.xy_valid);
	}

	return trip;
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

	// Watchdog first: it decides whether the controller runs at all this tick.
	const Quatf q(_attitude.q);
	const Dcmf r(q);
	const float position_error_m = (_setpoint.pos - Vector3f(_local_position.x, _local_position.y,
					_local_position.z)).norm();

	if (checkWatchdog(now, r, position_error_m) != Trip::None) {
		perf_end(_loop_perf);
		return;
	}

	const Vector3f pos(_local_position.x, _local_position.y, _local_position.z);

	// Measured floor under the hold point before either controller reads it.
	applyWallGuard(now, pos, r);

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

void StructureHold::guardedStep(const hrt_abstime now)
{
	const Quatf q(_attitude.q);
	const Dcmf r(q);
	const Vector3f pos(_local_position.x, _local_position.y, _local_position.z);

	// An EKF2 xy reset moves the setpoint with the estimate, as PX4's own modes do, so the
	// reset itself does not move the vehicle; the guard's range check is unaffected.
	if (_local_position.xy_reset_counter != _land_xy_reset_counter) {
		_land_xy_reset_counter = _local_position.xy_reset_counter;

		if (PX4_ISFINITE(_local_position.delta_xy[0]) && PX4_ISFINITE(_local_position.delta_xy[1])) {
			_setpoint.pos(0) += _local_position.delta_xy[0];
			_setpoint.pos(1) += _local_position.delta_xy[1];
		}
	}

	// Structure Land: height is the descent's, not a setpoint to hold, so only xy counts as
	// hold error. Structure Hover holds the height it entered at.
	const bool descend = _state == structure_hold_status_s::STATE_LANDING;
	const Vector3f error_ned = _setpoint.pos - pos;
	const float position_error_m = descend ? matrix::Vector2f(error_ned(0), error_ned(1)).norm() : error_ned.norm();

	if (checkWatchdog(now, r, position_error_m) != Trip::None) {
		return;
	}

	applyWallGuard(now, pos, r);

	// A guard that keeps pushing means the wall is closing in at that rate (the estimate
	// recovering after a jump). Position-only setpoints leave PX4's xy loop one P-lag
	// behind: on Gazebo seed 3 the guard pushed 0.3 to 0.44 m and the vehicle still came to
	// 0.37 m. The push rate, low-passed over about 0.3 s, goes in as velocity feed-forward
	// away from the wall.
	const float dt = CONTROL_INTERVAL * 1e-6f;
	_guard_speed += (_wall_out.shift_m / dt - _guard_speed) * (dt / (GUARD_SPEED_TAU_S + dt));
	const float vx = -_guard_speed * cosf(_setpoint.yaw);
	const float vy = -_guard_speed * sinf(_setpoint.yaw);

	if (descend) {
		// Slow the descent while the vehicle is still closer to the wall than the guarded
		// point, so it catches up before it is on the ground: MPC_LAND_SPEED up to 0.15 m
		// behind, falling to LAND_SLOW_MPS at 0.4 m. Never slower, so the land detector
		// still sees a commanded descent (it wants 1.1 x LNDMC_Z_VEL_MAX) and the landing
		// always completes.
		const float lag_m = -(error_ned(0) * cosf(_setpoint.yaw) + error_ned(1) * sinf(_setpoint.yaw));
		const float fast = math::max(_param_mpc_land_speed.get(), LAND_SLOW_MPS);
		const float w = math::constrain((lag_m - 0.15f) / 0.25f, 0.f, 1.f);
		publishTrajectorySetpoint(NAN, Vector3f(vx, vy, fast + (LAND_SLOW_MPS - fast) * w));

	} else {
		publishTrajectorySetpoint(_setpoint.pos(2), Vector3f(vx, vy, NAN));
	}

	_last_output = HoldOutput{};
	_last_output.pos_err = _setpoint.pos - pos;
}

void StructureHold::publishTrajectorySetpoint(float z_ned, const Vector3f &velocity_ned)
{
	// xy on the (guarded) setpoint. Landing: z by velocity only, so mc_pos_control and the
	// land detector see a commanded descent exactly as in stock Land. Hover: z held at
	// z_ned. A zero velocity with no position z is the handback's stop.
	trajectory_setpoint_s sp{};
	sp.timestamp = hrt_absolute_time();
	sp.position[0] = _setpoint.pos(0);
	sp.position[1] = _setpoint.pos(1);
	sp.position[2] = z_ned;
	sp.velocity[0] = velocity_ned(0);
	sp.velocity[1] = velocity_ned(1);
	sp.velocity[2] = velocity_ned(2);

	for (int i = 0; i < 3; i++) {
		sp.acceleration[i] = NAN;
		sp.jerk[i] = NAN;
	}

	sp.yaw = _setpoint.yaw;
	sp.yawspeed = NAN;
	_trajectory_setpoint_pub.publish(sp);
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
	status.wall_distance_m = _wall_out.wall_m;
	status.hold_clearance_m = _wall_out.clearance_m;
	status.standoff_shift_m = _wall_guard.totalShift();

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
			sendUnregistration(MODE_NAME, _mode_id, _arming_check_id);
		}

		if (_land_mode_id != -1) {
			sendUnregistration(LAND_MODE_NAME, _land_mode_id, _land_arming_check_id);
		}

		if (_hover_mode_id != -1) {
			sendUnregistration(HOVER_MODE_NAME, _hover_mode_id, _hover_arming_check_id);
		}

		exit_and_cleanup();
		return;
	}

	// Consume the sample that woke us; the policy observes the body rates from it.
	_angular_velocity_sub.update(&_angular_velocity);

	const hrt_abstime now = hrt_absolute_time();

	if (!_registration_sent) {
		sendRegistration(MODE_NAME, _request_id);
		_registration_sent = true;
		return;
	}

	if (_mode_id == -1 || (_land_registration_sent && _land_mode_id == -1)
	    || (_hover_registration_sent && _hover_mode_id == -1)
	    || (!_land_registration_sent && _param_land_en.get() != 0)
	    || (!_hover_registration_sent && _param_hb_guard.get() != 0)) {
		checkRegistrationReplies();

		if (_mode_id == -1) {
			return;
		}
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
	updateRange();

	const bool mode_active = _vehicle_status.nav_state == _active_mode_id;
	const bool land_selected = _land_mode_id != -1 && _vehicle_status.nav_state == _land_mode_id;
	const bool hover_selected = _hover_mode_id != -1 && _vehicle_status.nav_state == _hover_mode_id;

	switch (_state) {
	case structure_hold_status_s::STATE_STANDBY:
		if (land_selected) {
			if (_armed) {
				engageGuarded(_local_position, true);
			}

		} else if (hover_selected) {
			if (_armed) {
				engageGuarded(_local_position, false);
			}

		} else if (_vehicle_status.nav_state == _mode_id) {
			if (_watchdog.inResetHoldoff(now)) {
				// Commander normally refuses the switch (replyToArmingCheck), but its poll
				// is 300 ms behind the reset: hand straight back, nothing latched.
				PX4_WARN("engage refused: EKF reset its position %.1f s ago (SH_RST_HOLD %.1f s)",
					 (double)_watchdog.resetAge(now), (double)_param_rst_hold.get());
				_active_mode_id = _mode_id;
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

	case structure_hold_status_s::STATE_LANDING:
	case structure_hold_status_s::STATE_HOVER:
		if (!mode_active || !_armed) {
			PX4_INFO("%s ended: %s", _state == structure_hold_status_s::STATE_LANDING ? LAND_MODE_NAME
				 : HOVER_MODE_NAME, _armed ? "mode changed" : "disarmed");
			_state = structure_hold_status_s::STATE_STANDBY;
			_active_mode_id = -1;

		} else if (now - _last_control_us >= CONTROL_INTERVAL) {
			_last_control_us = now;
			guardedStep(now);
		}

		break;

	case structure_hold_status_s::STATE_HANDBACK:
		if (!mode_active) {
			PX4_INFO("handback complete: stock PX4 in charge");
			_state = structure_hold_status_s::STATE_STANDBY;
			_active_mode_id = -1;

		} else if (now - _last_control_us >= CONTROL_INTERVAL) {
			_last_control_us = now;
			handbackStep(now);
		}

		break;

	default:
		break;
	}

	const bool controlling = _state == structure_hold_status_s::STATE_ENGAGED
				 || _state == structure_hold_status_s::STATE_LANDING
				 || _state == structure_hold_status_s::STATE_HOVER
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

// Policy inference timing for the props-off bench test (docs/bench-test.md): a separate
// controller on synthetic samples, so the module's own history is never touched. Static so
// the observation history stays off the shell task's small stack.
static int runPolicyBench(int iterations)
{
	static PolicyController bench{};
	const float hold[3] = {0.f, 0.f, -2.5f};
	bench.engage(hold, 0.f);

	VehicleSample sample{};
	uint64_t total_us = 0;
	uint32_t worst_us = 0;
	float sink = 0.f;

	for (int i = 0; i < iterations; i++) {
		// A slow sweep of position and rate so every tick sees a new observation.
		sample.pos_ned[0] = 0.01f * static_cast<float>(i % 50);
		sample.pos_ned[2] = -2.5f + 0.005f * static_cast<float>(i % 20);
		sample.rates_frd[0] = 0.002f * static_cast<float>(i % 30);
		const hrt_abstime start = hrt_absolute_time();
		const PolicyOutput out = bench.step(sample, 0.f, 0.5f);
		const uint32_t dt = static_cast<uint32_t>(hrt_absolute_time() - start);
		total_us += dt;
		worst_us = math::max(worst_us, dt);
		sink += out.setpoint.thrust;
	}

	const float mean_us = static_cast<float>(total_us) / static_cast<float>(iterations);
	// The budget is the 10 ms control tick; flag anything over a tenth of it. Short lines:
	// PX4 cuts a log line at about 120 characters.
	PX4_INFO("bench %s: %s, %d steps", worst_us < 1000 ? "ok" : "OVER 1 ms", policy::NAME, iterations);
	PX4_INFO("bench: mean %.1f us, worst %u us (check %.1f)", (double)mean_us, worst_us, (double)sink);
	return worst_us < 1000 ? 0 : 1;
}

int StructureHold::custom_command(int argc, char *argv[])
{
	if (argc > 0 && !strcmp(argv[0], "bench")) {
		if (is_running() && (get_instance()->_state == structure_hold_status_s::STATE_ENGAGED
				     || get_instance()->_state == structure_hold_status_s::STATE_LANDING
				     || get_instance()->_state == structure_hold_status_s::STATE_HOVER)) {
			PX4_ERR("refused while engaged");
			return 1;
		}

		const int iterations = argc > 1 ? atoi(argv[1]) : 1000;
		return runPolicyBench(math::constrain(iterations, 1, 100000));
	}

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
	static const char *state_names[] = {"unregistered", "standby", "engaged", "handback", "landing", "hover"};
	PX4_INFO("state: %s", state_names[_state < 6 ? _state : 0]);
	PX4_INFO("mode_id %d, arming_check_id %d; Structure Land mode_id %d, arming_check_id %d", _mode_id,
		 _arming_check_id, _land_mode_id, _land_arming_check_id);
	PX4_INFO("Structure Hover mode_id %d, arming_check_id %d", _hover_mode_id, _hover_arming_check_id);
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
	const hrt_abstime now = hrt_absolute_time();
	PX4_INFO("wall guard: %s (SH_WALL_MIN %.2f m, SH_WALL_REQ %d), forward range %s %.2f m, %.2f s old",
		 _wall_guard.enabled() ? "on" : "off", (double)_param_wall_min.get(), (int)_param_wall_req.get(),
		 rangeFresh(now) ? "fresh" : "none", (double)_range.distance_m,
		 _range.timestamp_us > 0 && now >= _range.timestamp_us ? (double)((now - _range.timestamp_us) * 1e-6f) : -1.0);

	if (_state == structure_hold_status_s::STATE_ENGAGED || _state == structure_hold_status_s::STATE_LANDING
	    || _state == structure_hold_status_s::STATE_HOVER) {
		PX4_INFO("setpoint NED: %.2f %.2f %.2f m, yaw %.1f deg", (double)_setpoint.pos(0), (double)_setpoint.pos(1),
			 (double)_setpoint.pos(2), (double)math::degrees(_setpoint.yaw));
		PX4_INFO("error NED: %.2f %.2f %.2f m, thrust %.2f", (double)_last_output.pos_err(0),
			 (double)_last_output.pos_err(1), (double)_last_output.pos_err(2), (double)_last_output.thrust);
		PX4_INFO("wall %.2f m, hold point clearance %.2f m, moved back %.2f m since engage",
			 (double)_wall_out.wall_m, (double)_wall_out.clearance_m, (double)_wall_guard.totalShift());
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
With a forward rangefinder (distance_sensor, forward facing) the hold point is kept at least
SH_WALL_MIN from the measured wall.
The watchdog hands control back to Position mode (or Altitude mode) when it trips. Status
is published on structure_hold_status.

With SH_LAND_EN a second mode, "Structure Land", lands next to the structure: PX4's
position controller descends at MPC_LAND_SPEED while the wall guard keeps the xy setpoint
SH_WALL_MIN from the measured wall. It stays available after a latched trip. With
SH_HB_GUARD a third mode, "Structure Hover" (ext3), holds the same guarded setpoint
without descending; a trip next to a measured wall hands back into it first.

Engage from the shell with `commander mode ext1` (first registered external mode; Structure
Land is ext2) or from a ground station / MAVLink as the external mode by name.
)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("structure_hold", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_COMMAND_DESCR("trip", "Trip the watchdog now (test hook, same as SH_TEST_FAULT 1)");
	PRINT_MODULE_USAGE_COMMAND_DESCR("bench", "Time the policy on synthetic samples (bench test; refused while engaged)");
	PRINT_MODULE_USAGE_ARG("<iterations>", "Policy steps to time (default 1000)", true);
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();

	return 0;
}

extern "C" __EXPORT int structure_hold_main(int argc, char *argv[])
{
	return StructureHold::main(argc, argv);
}
