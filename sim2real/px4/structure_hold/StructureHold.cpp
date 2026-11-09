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
	_param_cp_dist = param_find("CP_DIST");  // PX4's collision prevention, absent in some builds
	updateParams();
}

StructureHold::~StructureHold()
{
	perf_free(_loop_perf);
	perf_free(_tick_perf);
}

bool StructureHold::init()
{
	// Run() on angular-velocity samples at most every WAKE_INTERVAL; the control step itself
	// is decimated to 100 Hz.
	_angular_velocity_sub.set_interval_us(WAKE_INTERVAL);

	if (!_angular_velocity_sub.registerCallback()) {
		PX4_ERR("callback registration failed");
		return false;
	}

	return true;
}

// A parameter outside its range is replaced in the module's cached copy only (set(), never
// committed), so the stored value stays what the operator wrote and is checked again at the
// next update (ParamLimits.hpp).
template<typename P>
static void sanitizeParam(P &param, const FloatRange &range, const char *name)
{
	const float v = param.get();

	if (!inRange(v, range)) {
		const float s = sanitize(v, range);
		PX4_WARN("%s %.3f outside [%.2f, %.2f], flying %.3f", name, (double)v, (double)range.min,
			 (double)range.max, (double)s);
		param.set(s);
	}
}

template<typename P>
static void sanitizeParam(P &param, const IntRange &range, const char *name)
{
	const int32_t v = param.get();

	if (!inRange(v, range)) {
		PX4_WARN("%s %d outside [%d, %d], flying the default %d", name, (int)v, (int)range.min, (int)range.max,
			 (int)range.def);
		param.set(range.def);
	}
}

void StructureHold::updateParams()
{
	ModuleParams::updateParams();

	sanitizeParam(_param_wd_est_to, param_limits::SH_WD_EST_TO, "SH_WD_EST_TO");
	sanitizeParam(_param_wd_pos_err, param_limits::SH_WD_POS_ERR, "SH_WD_POS_ERR");
	sanitizeParam(_param_wd_tilt, param_limits::SH_WD_TILT, "SH_WD_TILT");
	sanitizeParam(_param_wd_frz_t, param_limits::SH_WD_FRZ_T, "SH_WD_FRZ_T");
	sanitizeParam(_param_wd_vio_dv, param_limits::SH_WD_VIO_DV, "SH_WD_VIO_DV");
	sanitizeParam(_param_rst_hold, param_limits::SH_RST_HOLD, "SH_RST_HOLD");
	sanitizeParam(_param_bench, param_limits::SH_BENCH, "SH_BENCH");
	sanitizeParam(_param_p_pos, param_limits::SH_P_POS, "SH_P_POS");
	sanitizeParam(_param_p_vel, param_limits::SH_P_VEL, "SH_P_VEL");
	sanitizeParam(_param_p_att, param_limits::SH_P_ATT, "SH_P_ATT");
	sanitizeParam(_param_v_max, param_limits::SH_V_MAX, "SH_V_MAX");
	sanitizeParam(_param_vel_src, param_limits::SH_VEL_SRC, "SH_VEL_SRC");
	sanitizeParam(_param_ctrl, param_limits::SH_CTRL, "SH_CTRL");
	sanitizeParam(_param_fb_en, param_limits::SH_FB_EN, "SH_FB_EN");
	sanitizeParam(_param_fb_t, param_limits::SH_FB_T, "SH_FB_T");
	sanitizeParam(_param_fb_k, param_limits::SH_FB_K, "SH_FB_K");
	sanitizeParam(_param_fb_tilt, param_limits::SH_FB_TILT, "SH_FB_TILT");
	sanitizeParam(_param_fix_src, param_limits::SH_FIX_SRC, "SH_FIX_SRC");
	sanitizeParam(_param_fix_gap, param_limits::SH_FIX_GAP, "SH_FIX_GAP");
	sanitizeParam(_param_yaw_guard, param_limits::SH_YAW_GUARD, "SH_YAW_GUARD");
	sanitizeParam(_param_yaw_p, param_limits::SH_YAW_P, "SH_YAW_P");
	sanitizeParam(_param_rate_max, param_limits::SH_RATE_MAX, "SH_RATE_MAX");
	sanitizeParam(_param_wall_min, param_limits::SH_WALL_MIN, "SH_WALL_MIN");
	sanitizeParam(_param_wall_req, param_limits::SH_WALL_REQ, "SH_WALL_REQ");
	sanitizeParam(_param_land_en, param_limits::SH_LAND_EN, "SH_LAND_EN");
	sanitizeParam(_param_hb_guard, param_limits::SH_HB_GUARD, "SH_HB_GUARD");
	sanitizeParam(_param_hb_range, param_limits::SH_HB_RANGE, "SH_HB_RANGE");
	sanitizeParam(_param_ko_dist, param_limits::SH_KO_DIST, "SH_KO_DIST");
	sanitizeParam(_param_shift_max, param_limits::SH_SHIFT_MAX, "SH_SHIFT_MAX");
	sanitizeParam(_param_gcs_per, param_limits::SH_GCS_PER, "SH_GCS_PER");

	WatchdogLimits limits{};
	limits.estimate_timeout_us = static_cast<uint64_t>(_param_wd_est_to.get() * 1e6f);
	limits.max_position_error_m = _param_wd_pos_err.get();
	limits.max_tilt_rad = math::radians(_param_wd_tilt.get());
	limits.reset_holdoff_us = static_cast<uint64_t>(_param_rst_hold.get() * 1e6f);
	limits.max_fix_frozen_s = _param_wd_frz_t.get();
	_watchdog.setLimits(limits);
	_pilot.setLimits(limits);
	_runaway_fix.setLimit(_param_wd_vio_dv.get());

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
	wall.min_standoff_m = _param_wall_min.get();
	wall.spike_gate_m = policy::RANGE_SPIKE_GATE_M;  // short returns off dust (WallGuard.hpp)
	wall.max_total_shift_m = _param_shift_max.get();  // the retreat cap (WallGuard.hpp)
	_wall_guard.setLimits(wall);
	_range_ref.setSpikeGate(policy::RANGE_SPIKE_GATE_M);

	float cp_dist = -1.f;

	if (_param_cp_dist != PARAM_INVALID) {
		param_get(_param_cp_dist, &cp_dist);
	}

	KeepOutLimits keep_out{};
	keep_out.keep_out_m = resolveKeepOut(_param_ko_dist.get(), cp_dist, param_limits::SH_KO_DIST.max);
	keep_out.wall_min_m = _param_wall_min.get();
	keep_out.max_total_shift_m = _param_shift_max.get();
	_keep_out.setLimits(keep_out);
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
	// Never while disarmed, so the vehicle cannot be armed into it on the ground: the hold
	// point would be the ground and the policy never learned a take-off.
	const hrt_abstime now = hrt_absolute_time();
	reply.can_arm_and_run = _armed && _watchdog.latched() == Trip::None && !_watchdog.inResetHoldoff(now)
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
		// SH_HB_RANGE 1: Structure Hover also flies with the horizontal estimate gone, on the
		// rangefinder (RangeHover.hpp), so commander must not refuse it, or fail it over,
		// for want of a local position.
		reply.registration_id = _hover_arming_check_id;
		reply.mode_req_local_position = _param_hb_range.get() == 0;
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
	_engaged_us = hrt_absolute_time();
	_vel_fd_valid = false;
	_hold_heading_reset_counter = lpos.heading_reset_counter;
	_fallback_active = false;

	for (int i = 0; i < 4; i++) { _last_action[i] = 0.f; }

	_wall_guard.engage();
	_wall_out = WallGuardOutput{};
	_range_ref.engage();
	_keep_out.engage();
	_keep_out_out = KeepOutOutput{};
	_keep_out_warned = false;
	_no_room_warned = false;
	_wall_capped_warned = false;
	_keep_out_capped_warned = false;

	if (_keep_out.enabled() && !keepOutDataFresh(hrt_absolute_time())) {
		mavlink_log_warning(&_mavlink_log_pub, "Structure Hold: keep-out %.1f m is on but no obstacle data is coming in",
				    (double)_keep_out.limits().keep_out_m);
	}

	float hold[3];
	_setpoint.pos.copyTo(hold);
	_policy.engage(hold, _setpoint.yaw);  // structure x = the heading at engage

	if (_landed) {
		mavlink_log_warning(&_mavlink_log_pub, "Structure Hold: engaged on the ground (SH_BENCH 1), props off only");
	}
	sayEngaged(_engaged_us, MODE_NAME);
	PX4_INFO("engaged (%s): holding N %.2f E %.2f D %.2f m, yaw %.0f deg",
		 _param_ctrl.get() == 1 ? "policy" : "cascade",
		 (double)_setpoint.pos(0), (double)_setpoint.pos(1), (double)_setpoint.pos(2),
		 (double)math::degrees(_setpoint.yaw));
}

Trip StructureHold::entryRefusal(const hrt_abstime now) const
{
	// Commander's arming check covers the latch, the reset hold-off and SH_WALL_REQ, but it
	// polls every 300 ms and knows nothing of the ground or a broken estimate; these are
	// checked again at the switch itself.
	// SH_BENCH lifts this for the props-off bench test only.
	if (_landed && _param_bench.get() == 0) {
		return Trip::Landed;
	}

	// A NaN heading would make the structure frame NaN, and the policy would then fly on an
	// observation of zeros (buildObservation) until the 2 m hold-error trip.
	if (!estimateFinite() || !PX4_ISFINITE(_local_position.heading)) {
		return Trip::EstimateInvalid;
	}

	if (_watchdog.inResetHoldoff(now)) {
		return Trip::EstimateReset;
	}

	if (fixFrozen()) {
		return Trip::EstimateFrozen;
	}

	if (fixRunaway()) {
		return Trip::EstimateRunaway;
	}

	return Trip::None;
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
	_land_z_reset_counter = lpos.z_reset_counter;
	_land_heading_reset_counter = lpos.heading_reset_counter;
	_guard_speed = 0.f;
	_last_control_us = 0;
	_fallback_active = false;
	_last_output = HoldOutput{};

	for (int i = 0; i < 4; i++) { _last_action[i] = 0.f; }

	_wall_guard.engage();
	_wall_out = WallGuardOutput{};
	_keep_out.engage();
	_keep_out_out = KeepOutOutput{};
	_keep_out_warned = false;
	_no_room_warned = false;
	_wall_capped_warned = false;
	_keep_out_capped_warned = false;
	_range_hover.stop();
	_range_hover_xy_back_us = 0;
	sayEngaged(hrt_absolute_time(), descend ? LAND_MODE_NAME : HOVER_MODE_NAME);
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
	// a trip in Hover, or a refusal at entry, goes straight to Position mode and on
	// (Handback.hpp, decided rows of 2026-10-21 and 2026-10-28).
	HandbackFrom from = HandbackFrom::Hold;

	if (_land_mode_id != -1 && _active_mode_id == _land_mode_id) {
		from = HandbackFrom::Land;

	} else if (_hover_mode_id != -1 && _active_mode_id == _hover_mode_id) {
		from = HandbackFrom::Hover;
	}

	_handback_plan = planHandback(from, latch, xy_valid, guardedHandbackAvailable(hrt_absolute_time()),
				      _param_hb_range.get() != 0);
	_range_hover_handback = _range_hover.active();
	_range_hover.stop();
	_state = structure_hold_status_s::STATE_HANDBACK;
	_handback_step = 0;
	_handback_logged_last = false;
	if (latch) {
		// The code first (logs and eval/field_faults.py match "watchdog trip: <code>"), then
		// the plain reason, then what the pilot can still do: the latch only clears on disarm.
		mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: watchdog trip: %s (%s), handing back",
				     Watchdog::name(trip), Watchdog::reason(trip));
		mavlink_log_info(&_mavlink_log_pub, "Structure Hold: locked out until disarm");

	} else {  // an engage refusal, already sent to the ground station with its reason
		PX4_WARN("watchdog refusal: %s, handing back", Watchdog::name(trip));
	}
	requestHandbackMode();
}

void StructureHold::endHandback(const char *why)
{
	PX4_INFO("handback ended: %s", why);
	_range_hover_handback = false;
	_state = structure_hold_status_s::STATE_STANDBY;
	_active_mode_id = -1;
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
	const HandbackTarget target = _handback_plan.target(_handback_step);
	uint8_t main_mode = PX4_CUSTOM_MAIN_MODE_POSCTL;
	uint8_t sub_mode = 0;

	switch (target) {
	case HandbackTarget::StructureHover:
		// External modes' custom sub modes follow their nav_states (px4_custom_mode.h).
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_EXTERNAL1 + (_hover_mode_id - vehicle_status_s::NAVIGATION_STATE_EXTERNAL1);
		break;

	case HandbackTarget::Position:
		break;

	case HandbackTarget::Altitude:
		main_mode = PX4_CUSTOM_MAIN_MODE_ALTCTL;
		break;

	case HandbackTarget::Hold:
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_AUTO_LOITER;
		break;

	case HandbackTarget::StockLand:
	case HandbackTarget::Land:
		main_mode = PX4_CUSTOM_MAIN_MODE_AUTO;
		sub_mode = PX4_CUSTOM_SUB_MODE_AUTO_LAND;
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

	// The last target is asked for again until it takes; say so once.
	const bool last = _handback_step >= _handback_plan.lastStep();

	if (!last || !_handback_logged_last) {
		mavlink_log_info(&_mavlink_log_pub, "Structure Hold: handing back to %s%s", handbackTargetName(target),
				 last ? " (repeated until it takes)" : "");
		_handback_logged_last = last;
	}
}

void StructureHold::handbackStep(const hrt_abstime now)
{
	// Still the active mode: keep the vehicle level at hover thrust while commander
	// switches, and escalate if the requested mode was refused. From Structure Land, whose
	// rate setpoints come from PX4's own cascade, stop the descent there instead.
	const bool xy_valid = _local_position.xy_valid && PX4_ISFINITE(_local_position.x)
			      && PX4_ISFINITE(_local_position.y) && !fixRunaway();

	if (_active_mode_id == _hover_mode_id && _param_hb_range.get() != 0 && (_range_hover_handback || !xy_valid)) {
		// From Structure Hover with the range or the horizontal estimate gone: no xy
		// setpoint PX4 could fly, so lean into the wind as before the estimate went, height
		// held, until commander takes the next mode.
		publishAccelerationSetpoint(_wind_ff.x(), _wind_ff.y());

	} else if (_active_mode_id == _land_mode_id || _active_mode_id == _hover_mode_id) {
		publishTrajectorySetpoint(NAN, Vector3f(0.f, 0.f, 0.f));

	} else {
		// After a refusal on the ground, idle thrust: hover thrust there would lift off.
		publishRates(Vector3f(0.f, 0.f, 0.f), _last_trip == Trip::Landed ? 0.f : _hover_thrust);
	}

	if (now - _handback_request_us > HANDBACK_RETRY) {
		_handback_step = static_cast<uint8_t>(_handback_plan.next(_handback_step));
		requestHandbackMode();
	}
}

bool StructureHold::estimateFinite() const
{
	return PX4_ISFINITE(_local_position.x) && PX4_ISFINITE(_local_position.y) && PX4_ISFINITE(_local_position.z)
	       && PX4_ISFINITE(_local_position.vx) && PX4_ISFINITE(_local_position.vy)
	       && PX4_ISFINITE(_local_position.vz);
}

void StructureHold::publishRates(const Vector3f &rates, float thrust)
{
	// Every rate setpoint goes through the same last check (PolicyController.hpp).
	float r[3] = {rates(0), rates(1), rates(2)};

	const bool ok = sanitizeSetpoint(r, thrust, _hover_thrust, _gains.thrust_min, _gains.thrust_max);

	if (!ok && !_setpoint_replaced) {
		PX4_WARN("non-finite rate setpoint replaced");  // once per episode, not at 100 Hz
	}

	_setpoint_replaced = !ok;

	vehicle_rates_setpoint_s sp{};
	sp.timestamp = hrt_absolute_time();
	sp.roll = r[0];
	sp.pitch = r[1];
	sp.yaw = r[2];
	sp.thrust_body[0] = 0.f;
	sp.thrust_body[1] = 0.f;
	sp.thrust_body[2] = -thrust;
	sp.reset_integral = false;
	_rates_setpoint_pub.publish(sp);
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
			_frozen_fix.update(_last_fix_sample_us, odom.position[0], odom.position[1], odom.position[2]);

			// The fix's velocity in NED for the runaway check: NED as sent, or the body frame
			// turned by the attitude; any other frame (a VIO's own heading) is not compared.
			Vector3f v(NAN, NAN, NAN);

			if (odom.velocity_frame == vehicle_odometry_s::VELOCITY_FRAME_NED) {
				v = Vector3f(odom.velocity);

			} else if (odom.velocity_frame == vehicle_odometry_s::VELOCITY_FRAME_BODY_FRD) {
				v = Dcmf(Quatf(_attitude.q)) * Vector3f(odom.velocity);
			}

			if (useOdometryFix()) {
				_runaway_fix.fix(_last_fix_sample_us, v(0), v(1), v(2), odom.reset_counter);
			}
		}
	}

	// Only the VIO's own odometry can freeze: EKF2's output moves with the IMU regardless.
	_fix_frozen_s = useOdometryFix() ? _frozen_fix.frozenSeconds() : 0.f;

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
	// Sensors facing another horizontal direction (PX4's 45 deg yaw orientations) feed the
	// keep-out, one reading kept per instance.
	int instance = 0;

	for (auto &sub : _distance_sensor_subs) {
		distance_sensor_s ds;

		while (sub.update(&ds)) {
			if (ds.orientation > distance_sensor_s::ROTATION_FORWARD_FACING && ds.orientation <= 7
			    && instance < SIDE_RANGES) {
				RangeSample &side = _side_range[instance];
				side.timestamp_us = ds.timestamp;
				side.distance_m = ds.current_distance;
				side.min_distance_m = ds.min_distance;
				side.max_distance_m = ds.max_distance;
				side.signal_quality = ds.signal_quality;
				_side_yaw[instance] = ds.orientation * (M_PI_F / 4.f);
			}

			if (ds.orientation != distance_sensor_s::ROTATION_FORWARD_FACING || ds.timestamp < _range.timestamp_us) {
				continue;
			}

			_range.timestamp_us = ds.timestamp;
			_range.distance_m = ds.current_distance;
			_range.min_distance_m = ds.min_distance;
			_range.max_distance_m = ds.max_distance;
			_range.signal_quality = ds.signal_quality;
		}

		++instance;
	}

	obstacle_distance_s od;

	if (_obstacle_distance_sub.update(&od)) {
		_scan.timestamp_us = od.timestamp;
		_scan.frame = od.frame;
		_scan.increment_deg = od.increment;
		_scan.angle_offset_deg = od.angle_offset;
		_scan.min_distance_cm = od.min_distance;
		_scan.max_distance_cm = od.max_distance;
		static_assert(sizeof(od.distances) == sizeof(_scan.distances_cm), "obstacle_distance bins");
		memcpy(_scan.distances_cm, od.distances, sizeof(_scan.distances_cm));
	}
}

bool StructureHold::keepOutDataFresh(const hrt_abstime now) const
{
	const uint64_t max_age = _keep_out.limits().max_age_us;

	if (_scan.timestamp_us > 0 && now >= _scan.timestamp_us && now - _scan.timestamp_us <= max_age) {
		return true;
	}

	for (int i = 0; i < SIDE_RANGES; i++) {
		if (_side_range[i].timestamp_us > 0 && now >= _side_range[i].timestamp_us
		    && now - _side_range[i].timestamp_us <= max_age) {
			return true;
		}
	}

	return false;
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
	_range_ref.update(_wall_out, _range.timestamp_us);  // the wall-range input (WallGuard.hpp)

	if (_wall_out.shift_m > 0.f) {
		_setpoint.pos = Vector3f(hold);

		if (_state == structure_hold_status_s::STATE_ENGAGED) {
			_policy.moveHold(hold);
		}

		if (!shifted_before) {
			mavlink_log_warning(&_mavlink_log_pub, "Structure Hold: wall guard moved the hold point %.2f m back, to %.2f m from the wall",
					    (double)_wall_out.shift_m, (double)_wall_out.clearance_m);
		}
	}

	if (_wall_out.capped && !_wall_capped_warned) {
		_wall_capped_warned = true;
		mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: wall guard has moved the hold point %.1f m back this hold (SH_SHIFT_MAX), not moving it further",
				     (double)_wall_guard.totalShift());
	}

	applyKeepOut(now, pos, r);
}

void StructureHold::applyKeepOut(const hrt_abstime now, const Vector3f &pos, const Dcmf &r)
{
	if (!_keep_out.enabled()) {
		_keep_out_out = KeepOutOutput{};
		return;
	}

	// This tick's obstacle points: the scan, then each side sensor's usable reading on its
	// bearing (the vehicle's yaw plus the sensor's mounting yaw). Ranges are taken as
	// horizontal, as PX4's collision prevention does for obstacle_distance.
	const float yaw = atan2f(r(1, 0), r(0, 0));
	_keep_out.begin();
	_keep_out.addScan(_scan, now, yaw);
	for (int i = 0; i < SIDE_RANGES; i++) {
		const RangeSample &side = _side_range[i];

		if (side.timestamp_us > 0 && now >= side.timestamp_us && now - side.timestamp_us <= _keep_out.limits().max_age_us
		    && side.signal_quality != 0
		    && PX4_ISFINITE(side.distance_m) && side.distance_m > side.min_distance_m
		    && side.distance_m < side.max_distance_m) {
			_keep_out.addPoint(yaw + _side_yaw[i], side.distance_m);
		}
	}

	float pos_ned[3], hold[3];
	pos.copyTo(pos_ned);
	_setpoint.pos.copyTo(hold);
	const float normal[2] = {cosf(_setpoint.yaw), sinf(_setpoint.yaw)};
	const float before[2] = {hold[0], hold[1]};
	_keep_out_out = _keep_out.apply(pos_ned, normal, _wall_out.range_ok ? _wall_out.wall_m : -1.f, hold);

	if (_keep_out_out.shift_m > 0.f) {
		_setpoint.pos = Vector3f(hold);

		if (_state == structure_hold_status_s::STATE_ENGAGED) {
			_policy.moveHold(hold);
		}

		// The wall-range input's reference is the hold point's wall distance: it follows.
		_range_ref.moveBack(-((hold[0] - before[0]) * normal[0] + (hold[1] - before[1]) * normal[1]));

		if (!_keep_out_warned) {
			_keep_out_warned = true;
			mavlink_log_warning(&_mavlink_log_pub, "Structure Hold: keep-out moved the hold point %.2f m, to %.2f m from the nearest obstacle",
					    (double)_keep_out_out.shift_m, (double)_keep_out_out.nearest_m);
		}
	}

	if (_keep_out_out.no_room && !_no_room_warned) {
		_no_room_warned = true;
		mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: keep-out has no room, holding between obstacles");
	}

	if (_keep_out_out.capped && !_keep_out_capped_warned) {
		_keep_out_capped_warned = true;
		mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: keep-out has moved the hold point %.1f m this hold (SH_SHIFT_MAX), not moving it further",
				     (double)_keep_out.totalShift());
	}
}

Trip StructureHold::checkWatchdog(const hrt_abstime now, const Dcmf &r, float position_error_m, bool range_only)
{
	// The stale trip counts from the last position fix (or the last local position update,
	// whichever is older), so an EKF2 that stops publishing trips it too. A trip hands
	// back to stock PX4.
	WatchdogInputs in{};
	in.now_us = now;
	in.estimate_timestamp_us = math::min(_last_fix_us, static_cast<hrt_abstime>(_local_position.timestamp));
	// EKF2 does not publish NaN with a valid flag, but a NaN in any of these would reach
	// the controllers unchecked, so it counts as invalid (vz with the horizontal velocity:
	// the policy reads all three).
	in.xy_valid = _local_position.xy_valid && PX4_ISFINITE(_local_position.x) && PX4_ISFINITE(_local_position.y);
	in.z_valid = _local_position.z_valid && PX4_ISFINITE(_local_position.z);
	in.v_xy_valid = _local_position.v_xy_valid && PX4_ISFINITE(_local_position.vx)
			&& PX4_ISFINITE(_local_position.vy) && PX4_ISFINITE(_local_position.vz);
	in.tilt_rad = acosf(math::constrain(r(2, 2), -1.f, 1.f));
	in.position_error_m = position_error_m;
	in.test_fault = _param_test_fault.get() != 0;
	// A frozen fix stops the policy. In Structure Land and Structure Hover PX4's own
	// controller flies with the wall guard; handing back from there would put Position mode
	// on the same frozen estimate without the guard (a Gazebo freeze flight drifted to
	// 0.64 m from the wall that way), so it is no trip there. With SH_HB_RANGE the hover
	// flies on the range instead (estimateGood()).
	const bool guarded_mode = _state == structure_hold_status_s::STATE_HOVER
				  || _state == structure_hold_status_s::STATE_LANDING;
	in.fix_frozen_s = guarded_mode ? 0.f : _fix_frozen_s;
	// A runaway fix stops the policy and Structure Hover on it (with SH_HB_RANGE 1 the hover
	// flies on the range instead, estimateGood()); the handback then takes the horizontal
	// estimate as gone: Altitude mode, never Position mode chasing the runaway. Structure
	// Land carries on down.
	in.fix_runaway = _state != structure_hold_status_s::STATE_LANDING && fixRunaway();
	const bool xy_valid = in.xy_valid && !fixRunaway();

	if (range_only) {
		// Structure Hover on the range alone flies no horizontal estimate and no position
		// fix: what still counts is the height, EKF2 publishing at all, the tilt and the
		// test hook. The range itself is checked by rangeHoverStep (Trip::RangeLost).
		in.xy_valid = true;
		in.v_xy_valid = true;
		in.estimate_timestamp_us = _local_position.timestamp;
		in.position_error_m = 0.f;
		in.fix_frozen_s = 0.f;
		in.fix_runaway = false;
	}

	const Trip trip = _watchdog.check(in);

	if (trip != Trip::None) {
		if (in.test_fault) {
			_param_test_fault.set(0);  // one-shot test hook
			_param_test_fault.commit_no_notification();
		}

		handBack(trip, xy_valid);
	}

	return trip;
}

void StructureHold::controlStep(const hrt_abstime now)
{
	perf_begin(_loop_perf);
	perf_count(_tick_perf);
	const hrt_abstime start = hrt_absolute_time();

	// Hover thrust from the estimator when it has one, else the parameter; both clamped to
	// [0.1, 0.9] (ParamLimits.hpp), a NaN estimate falls back to the parameter.
	hover_thrust_estimate_s hte;

	if (_hover_thrust_sub.update(&hte) && hte.valid) {
		_hover_thrust = sanitizeHoverThrust(hte.hover_thrust, _param_mpc_thr_hover.get());
		_hover_thrust_from_estimator = true;

	} else if (!_hover_thrust_from_estimator) {
		_hover_thrust = sanitizeHoverThrust(_param_mpc_thr_hover.get(), param_limits::HOVER_THRUST.def);
	}

	// An EKF2 heading reset (a vision or magnetometer yaw alignment) changes the estimate's
	// yaw of the heading the vehicle still has, not the vehicle: the structure yaw, which is
	// the wall's normal, follows it, as the guarded modes' setpoint does and PX4's own flight
	// tasks do. Otherwise the policy (or the yaw guard) would turn the vehicle by the reset
	// to meet the old number and hold the wall axis on a heading that is no longer the wall's.
	// The hold point stays: the position frame did not move.
	if (_local_position.heading_reset_counter != _hold_heading_reset_counter) {
		_hold_heading_reset_counter = _local_position.heading_reset_counter;

		if (PX4_ISFINITE(_local_position.delta_heading) && _param_ctrl.get() == 1) {
			_setpoint.yaw = _policy.rotateFrame(_local_position.delta_heading);
			PX4_INFO("heading reset by %.1f deg: structure yaw now %.1f deg",
				 (double)math::degrees(_local_position.delta_heading), (double)math::degrees(_setpoint.yaw));

		} else if (PX4_ISFINITE(_local_position.delta_heading)) {
			_setpoint.yaw = matrix::wrap_pi(_setpoint.yaw + _local_position.delta_heading);
		}
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
	pilotStep(now, MODE_NAME, r, position_error_m, false);

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
		// Seconds since engage (read only by policies with the engage-time input, v22 on).
		// A wall-guard move of the hold point is not an engage, as in the env.
		sample.engage_s = now > _engaged_us ? (now - _engaged_us) * 1e-6f : 0.f;
		// The wall-range input (read only by policies with it, v23 on): the latched
		// reference minus the measured perpendicular distance, positive toward the wall.
		sample.range_ok = _range_ref.error(_wall_out, sample.range_err_m);

		const PolicyOutput out = _policy.step(sample, _stale_s, _hover_thrust);

		if (!out.ok) {
			mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: policy produced a non-finite action");
		}

		if (out.fallback_active != _fallback_active) {
			mavlink_log_warning(&_mavlink_log_pub, "Structure Hold: %s",
					    out.fallback_active ? "fallback, braking on a stale estimate" : "fresh fix, policy flies again");
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

	publishRates(_last_output.rates, _last_output.thrust);

	_last_loop_time_us = static_cast<uint32_t>(hrt_absolute_time() - start);
	_max_loop_time_us = math::max(_max_loop_time_us, _last_loop_time_us);
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

	// Height and heading resets the same way (FlightTask's EKF reset handlers): Structure
	// Hover's held height would otherwise climb or sink by the reset, and after a heading
	// reset PX4 would turn the vehicle by the reset to meet the old yaw setpoint, with the
	// wall normal (the same angle) left on the old, corrected heading.
	if (_local_position.z_reset_counter != _land_z_reset_counter) {
		_land_z_reset_counter = _local_position.z_reset_counter;

		if (PX4_ISFINITE(_local_position.delta_z)) {
			_setpoint.pos(2) += _local_position.delta_z;
		}
	}

	if (_local_position.heading_reset_counter != _land_heading_reset_counter) {
		_land_heading_reset_counter = _local_position.heading_reset_counter;

		if (PX4_ISFINITE(_local_position.delta_heading)) {
			_setpoint.yaw = matrix::wrap_pi(_setpoint.yaw + _local_position.delta_heading);
		}
	}

	// Structure Hover with the horizontal estimate gone or stale (SH_HB_RANGE 1): the wall
	// distance on the rangefinder alone, until the estimate has been back for a second.
	if (rangeHoverWanted(now)) {
		rangeHoverStep(now, r);
		return;
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
	pilotStep(now, descend ? LAND_MODE_NAME : HOVER_MODE_NAME, r, position_error_m, false);

	// A guard that keeps pushing means the wall is closing in at that rate (the estimate
	// recovering after a jump). Position-only setpoints leave PX4's xy loop one P-lag
	// behind: on Gazebo seed 3 the guard pushed 0.3 to 0.44 m and the vehicle still came to
	// 0.37 m. The push rate, low-passed over about 0.3 s, goes in as velocity feed-forward
	// away from the wall.
	const float dt = CONTROL_INTERVAL * 1e-6f;
	_guard_speed += (_wall_out.shift_m / dt - _guard_speed) * (dt / (GUARD_SPEED_TAU_S + dt));
	// And when PX4's loop carries the vehicle itself inside the standoff anyway (past the
	// guarded point), a back-off on the measured range (WallGuard::backOffSpeed).
	const float away = _guard_speed + _wall_guard.backOffSpeed(_wall_out);
	const float vx = -away * cosf(_setpoint.yaw);
	const float vy = -away * sinf(_setpoint.yaw);

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

bool StructureHold::estimateGood() const
{
	// Valid, finite, fresh, not frozen and not run away: what the range hover and the wind
	// feed-forward trust.
	return _local_position.xy_valid && _local_position.v_xy_valid && PX4_ISFINITE(_local_position.x)
	       && PX4_ISFINITE(_local_position.y) && PX4_ISFINITE(_local_position.vx) && PX4_ISFINITE(_local_position.vy)
	       && _stale_s < RANGE_HOVER_STALE_S && !fixFrozen() && !fixRunaway();
}

bool StructureHold::fixFrozen() const
{
	const float limit = _param_wd_frz_t.get();
	return limit > 0.f && !(_fix_frozen_s < limit);
}

bool StructureHold::rangeHoverWanted(const hrt_abstime now) const
{
	(void)now;
	return _param_hb_range.get() != 0 && _state == structure_hold_status_s::STATE_HOVER
	       && (_range_hover.active() || !estimateGood());
}

float StructureHold::measuredWall(const hrt_abstime now, const Dcmf &r) const
{
	// The perpendicular distance to the wall along the structure normal (the held heading),
	// as the wall guard measures it, or NaN without a usable reading.
	if (!_wall_guard.usable(_range, now)) {
		return NAN;
	}

	const float cos_off = r(0, 0) * cosf(_setpoint.yaw) + r(1, 0) * sinf(_setpoint.yaw);

	if (!(cos_off >= cosf(_wall_guard.limits().max_off_axis_rad))) {
		return NAN;
	}

	return _range.distance_m * cos_off;
}

void StructureHold::rangeHoverStep(const hrt_abstime now, const Dcmf &r)
{
	const float nx = cosf(_setpoint.yaw), ny = sinf(_setpoint.yaw);  // wall normal, NED
	float wall = measuredWall(now, r);

	// The same spike gate the wall guard applies: a lone short return is not a wall.
	if (PX4_ISFINITE(wall) && !_wall_guard.acceptWall(_range.timestamp_us, wall)) {
		wall = NAN;
	}
	const bool good = estimateGood();
	const bool xy_valid = _local_position.xy_valid && PX4_ISFINITE(_local_position.x)
			      && PX4_ISFINITE(_local_position.y) && !fixRunaway();

	if (!_range_hover.active()) {
		if (!PX4_ISFINITE(wall)) {
			handBack(Trip::RangeLost, xy_valid);
			return;
		}

		// The wind the vehicle leaned into while the estimate was good, split along the
		// normal (toward the wall positive) and along the wall (t = n rotated +90 deg).
		const float fx = _wind_ff.valid() ? _wind_ff.x() : 0.f;
		const float fy = _wind_ff.valid() ? _wind_ff.y() : 0.f;
		_range_hover.start(wall, _param_wall_min.get(), fx * nx + fy * ny, -fx * ny + fy * nx);
		_range_hover_seen_us = now;
		_range_hover_xy_back_us = 0;

		if (!PX4_ISFINITE(_setpoint.pos(2))) {
			_setpoint.pos(2) = _local_position.z;
		}

		mavlink_log_critical(&_mavlink_log_pub, "Structure Hover: position lost, holding %.2f m from the wall on the rangefinder",
				     (double)_range_hover.reference());
	}

	// The estimate back and fresh for a second: hold the point on it again, where the
	// vehicle is now (the wall guard keeps that off the wall).
	if (good) {
		if (_range_hover_xy_back_us == 0) {
			_range_hover_xy_back_us = now;

		} else if (now - _range_hover_xy_back_us >= RANGE_HOVER_XY_BACK) {
			_range_hover.stop();
			_setpoint.pos(0) = _local_position.x;
			_setpoint.pos(1) = _local_position.y;
			_land_xy_reset_counter = _local_position.xy_reset_counter;
			_wall_guard.engage();
			_guard_speed = 0.f;
			mavlink_log_info(&_mavlink_log_pub, "Structure Hover: position back, holding on the estimate");
			return;
		}

	} else {
		_range_hover_xy_back_us = 0;
	}

	if (checkWatchdog(now, r, 0.f, true) != Trip::None) {
		return;
	}

	if (PX4_ISFINITE(wall)) {
		_range_hover_seen_us = now;

	} else if (now - _range_hover_seen_us > RANGE_HOVER_LOST) {
		handBack(Trip::RangeLost, xy_valid);
		return;
	}

	const float a_n = _range_hover.step(wall, _range.timestamp_us, CONTROL_INTERVAL * 1e-6f);
	const float a_t = _range_hover.alongWall();
	publishAccelerationSetpoint(a_n * nx - a_t * ny, a_n * ny + a_t * nx);

	// Status: the measured distance and the reference it holds.
	_wall_out = WallGuardOutput{};
	_wall_out.range_ok = PX4_ISFINITE(wall);
	_wall_out.wall_m = PX4_ISFINITE(wall) ? wall : -1.f;
	_wall_out.clearance_m = _range_hover.reference();
	_last_output = HoldOutput{};
	pilotStep(now, HOVER_MODE_NAME, r, 0.f, true);
}

void StructureHold::publishAccelerationSetpoint(float ax_ned, float ay_ned)
{
	// No horizontal position or velocity setpoint: PX4's position controller turns the
	// acceleration into a tilt without any horizontal estimate, as in Altitude mode, and
	// holds the height on z (Structure Hover's own).
	trajectory_setpoint_s sp{};
	sp.timestamp = hrt_absolute_time();
	sp.position[0] = NAN;
	sp.position[1] = NAN;
	sp.position[2] = _setpoint.pos(2);

	for (int i = 0; i < 3; i++) {
		sp.velocity[i] = NAN;
		sp.jerk[i] = NAN;
	}

	sp.acceleration[0] = PX4_ISFINITE(ax_ned) ? ax_ned : 0.f;
	sp.acceleration[1] = PX4_ISFINITE(ay_ned) ? ay_ned : 0.f;
	sp.acceleration[2] = NAN;
	sp.yaw = _setpoint.yaw;
	sp.yawspeed = NAN;
	_trajectory_setpoint_pub.publish(sp);
}

void StructureHold::updateWindFeedForward(const hrt_abstime now)
{
	if (now - _wind_ff_us < CONTROL_INTERVAL) {
		return;
	}

	const float dt = _wind_ff_us == 0 ? 0.f : math::min((now - _wind_ff_us) * 1e-6f, 0.1f);
	_wind_ff_us = now;

	if (!_armed) {
		_wind_ff.reset();
		return;
	}

	// The lean while flying on a good estimate, in any mode (Structure Hold, its hover, or
	// stock Position mode): frozen on the ground, with the estimate gone or stale, and
	// while the range hover itself flies.
	const Dcmf r(Quatf(_attitude.q));
	const float body_z[3] = {r(0, 2), r(1, 2), r(2, 2)};
	_wind_ff.update(body_z, dt, !_landed && estimateGood() && !_range_hover.active());
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
	status.vio_imu_dv = _runaway_fix.disagreement();
	status.vio_runaway = fixRunaway();
	status.wall_distance_m = _wall_out.wall_m;
	status.hold_clearance_m = _wall_out.clearance_m;
	status.standoff_shift_m = _wall_guard.totalShift();
	status.keep_out_m = _keep_out.limits().keep_out_m;
	status.obstacle_clearance_m = _keep_out_out.nearest_m;
	status.keep_out_shift_m = _keep_out.totalShift();
	status.keep_out_points = static_cast<uint8_t>(_keep_out_out.points);
	status.keep_out_no_room = _keep_out_out.no_room;
	const bool flying = _state == structure_hold_status_s::STATE_ENGAGED
			    || _state == structure_hold_status_s::STATE_HOVER
			    || _state == structure_hold_status_s::STATE_LANDING;
	status.watchdog_used = flying ? _margin.used : -1.f;
	status.watchdog_closest = flying ? static_cast<uint8_t>(_margin.limit) : 0;

	for (int i = 0; i < 4; i++) { status.action[i] = _last_action[i]; }
	_last_output.rates.copyTo(status.rates_setpoint);
	status.thrust_setpoint = _last_output.thrust;
	status.loop_time_us = _last_loop_time_us;
	_status_pub.publish(status);
	_last_status_us = now;
}

// ---------------------------------------------------------------------------------------
// The pilot's live view (PilotStatus.hpp, docs/pilot-status.md): the trip messages say what
// a safety function did; these say how close the vehicle is to one before it acts.

float StructureHold::watchdogStaleS(const hrt_abstime now, bool range_only) const
{
	// Counted as checkWatchdog() counts it: from the older of the last fix and the last
	// EKF2 update, or from EKF2 alone on the rangefinder hover, which flies without a fix.
	const hrt_abstime ts = range_only ? _local_position.timestamp
			       : math::min(_last_fix_us, static_cast<hrt_abstime>(_local_position.timestamp));
	return now > ts ? (now - ts) * 1e-6f : 0.f;
}

void StructureHold::wallText(char *buf, size_t len, float wall_m) const
{
	if (!_wall_guard.enabled()) {
		snprintf(buf, len, "wall guard off");

	} else if (PX4_ISFINITE(wall_m) && wall_m >= 0.f) {
		snprintf(buf, len, "%.2f m from the wall", (double)wall_m);

	} else {
		snprintf(buf, len, "no wall range");
	}
}

void StructureHold::sayEngaged(const hrt_abstime now, const char *mode)
{
	// The pilot hears at once that the switch took, and whether the wall guard sees the wall.
	_pilot.start(now);
	_margin = Margin{};
	_hold_error_m = 0.f;
	_wd_stale_s = 0.f;
	char wall[32];
	wallText(wall, sizeof(wall), measuredWall(now, Dcmf(Quatf(_attitude.q))));
	mavlink_log_info(&_mavlink_log_pub, "%s: engaged, %s", mode, wall);
}

void StructureHold::pilotStep(const hrt_abstime now, const char *mode, const Dcmf &r, float position_error_m,
			      bool range_only)
{
	// Called on every flying tick the watchdog let through, with what it judged.
	MarginInputs in{};
	in.stale_s = watchdogStaleS(now, range_only);
	in.fix_frozen_s = _fix_frozen_s;
	in.tilt_rad = acosf(math::constrain(r(2, 2), -1.f, 1.f));
	in.position_error_m = position_error_m;
	in.frozen_checked = _state == structure_hold_status_s::STATE_ENGAGED;  // as checkWatchdog()
	in.position_checked = !range_only;
	_margin = _pilot.closest(in);
	_hold_error_m = range_only ? -1.f : position_error_m;
	_wd_stale_s = in.stale_s;

	const PilotEvent ev = _pilot.update(now, in);

	if (ev.kind == PilotEvent::Warn) {
		switch (ev.limit) {
		case Trip::EstimateStale:
			mavlink_log_warning(&_mavlink_log_pub, "%s: no position fix for %.1f s, watchdog hands back at %.1f s",
					    mode, (double)ev.value, (double)ev.max);
			break;

		case Trip::EstimateFrozen:
			mavlink_log_warning(&_mavlink_log_pub, "%s: position fix has not moved for %.1f s, watchdog trips at %.1f s",
					    mode, (double)ev.value, (double)ev.max);
			break;

		case Trip::Tilt:
			mavlink_log_warning(&_mavlink_log_pub, "%s: tilted %.0f deg, watchdog trips at %.0f deg",
					    mode, (double)math::degrees(ev.value), (double)math::degrees(ev.max));
			break;

		case Trip::PositionError:
			mavlink_log_warning(&_mavlink_log_pub, "%s: %.2f m off the hold point, watchdog trips at %.1f m",
					    mode, (double)ev.value, (double)ev.max);
			break;

		default:
			break;
		}

	} else if (ev.kind == PilotEvent::Clear) {
		mavlink_log_info(&_mavlink_log_pub, "%s: back inside the watchdog limits", mode);
	}

	if (_pilot.summaryDue(now, _param_gcs_per.get())) {
		if (range_only) {
			mavlink_log_info(&_mavlink_log_pub, "Structure Hover: on the rangefinder, %.2f m from the wall, holding %.2f m",
					 (double)_wall_out.wall_m, (double)_wall_out.clearance_m);

		} else {
			char wall[32];
			wallText(wall, sizeof(wall), _wall_out.wall_m);
			mavlink_log_info(&_mavlink_log_pub, "%s: %.2f m off the hold point, %s, fix %.2f s old",
					 mode, (double)position_error_m, wall, (double)_stale_s);
		}
	}
}

void StructureHold::publishLive(const hrt_abstime now)
{
	// The same view as numbers, a few times a second; MAVLink sends the newest at the
	// stream's rate (DEBUG_FLOAT_ARRAY: 1 Hz on a normal ground-station link, 10 Hz onboard).
	if (now - _last_live_us < LIVE_INTERVAL) {
		return;
	}

	_last_live_us = now;
	const bool flying = _state == structure_hold_status_s::STATE_ENGAGED
			    || _state == structure_hold_status_s::STATE_HOVER
			    || _state == structure_hold_status_s::STATE_LANDING;

	debug_array_s live{};
	live.timestamp = now;
	live.id = LIVE_ARRAY_ID;
	static_assert(sizeof("SH_STATUS") <= sizeof(live.name), "array name");
	memcpy(live.name, "SH_STATUS", sizeof("SH_STATUS"));
	static_assert(LIVE_COUNT <= debug_array_s::ARRAY_SIZE, "live array size");
	float *d = live.data;
	d[LIVE_STATE] = _state;
	d[LIVE_HOLD_ERROR] = flying ? _hold_error_m : -1.f;
	// Before engaging, the forward range as it reads (whether the guard will see the wall).
	d[LIVE_WALL] = flying ? _wall_out.wall_m : (rangeFresh(now) ? _range.distance_m : -1.f);
	d[LIVE_CLEARANCE] = flying ? _wall_out.clearance_m : -1.f;
	d[LIVE_GUARD_SHIFT] = _wall_guard.totalShift();
	d[LIVE_FIX_AGE] = _stale_s;
	d[LIVE_WD_SECONDS] = flying ? _pilot.secondsToStaleTrip(_wd_stale_s) : -1.f;
	d[LIVE_WD_USED] = flying ? 100.f * _margin.used : -1.f;
	d[LIVE_WD_CLOSEST] = flying ? static_cast<float>(_margin.limit) : 0.f;
	d[LIVE_TRIP] = static_cast<float>(_last_trip);
	d[LIVE_TRIP_COUNT] = _trip_count;
	d[LIVE_TILT] = math::degrees(acosf(math::constrain(Dcmf(Quatf(_attitude.q))(2, 2), -1.f, 1.f)));
	d[LIVE_FALLBACK] = _fallback_active ? 1.f : 0.f;
	_live_pub.publish(live);
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
			_runaway_fix.reset();  // a VIO that ran away is trusted again after a disarm
		}

		_armed = armed;
	}

	if (_local_position_sub.update(&_local_position)) {
		_watchdog.noteReset(_local_position.xy_reset_counter, now);
		_runaway_fix.inertial(_local_position.timestamp_sample, _local_position.ax, _local_position.ay,
				      _local_position.az);
	}

	vehicle_land_detected_s land_detected;

	if (_land_detected_sub.update(&land_detected)) {
		_landed = land_detected.landed;
	}

	_attitude_sub.update(&_attitude);
	updateFix(now);
	updateRange();
	updateWindFeedForward(now);

	// Said once, in any mode: in Position mode too the pilot is flying on that estimate.
	if (fixRunaway() != _runaway_warned) {
		_runaway_warned = fixRunaway();

		if (_runaway_warned) {
			mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: VIO velocity ran %.1f m/s from the IMU's (SH_WD_VIO_DV), VIO not trusted until it restarts",
					     (double)_runaway_fix.disagreement());
		}
	}

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

		} else if (_vehicle_status.nav_state == _mode_id && _armed) {
			// Disarmed in the mode (a kill switch in flight, or selected on the ground):
			// nothing to fly, and the arming check keeps it from being armed into.
			const Trip refusal = entryRefusal(now);

			if (refusal == Trip::EstimateReset) {
				// Commander normally refuses the switch (replyToArmingCheck), but its poll
				// is 300 ms behind the reset: hand straight back, nothing latched.
				mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: engage refused, EKF reset its position %.1f s ago (SH_RST_HOLD %.1f s)",
					 (double)_watchdog.resetAge(now), (double)_param_rst_hold.get());

			} else if (refusal != Trip::None) {
				mavlink_log_critical(&_mavlink_log_pub, "Structure Hold: engage refused: %s (%s)",
						     Watchdog::name(refusal), Watchdog::reason(refusal));
			}

			if (refusal != Trip::None) {
				_active_mode_id = _mode_id;
				handBack(refusal, _local_position.xy_valid && PX4_ISFINITE(_local_position.x)
					 && PX4_ISFINITE(_local_position.y) && !fixRunaway(), false);

			} else {
				engage(_local_position);
			}
		}

		break;

	case structure_hold_status_s::STATE_ENGAGED:
		if (!mode_active) {
			disengage("mode changed");

		} else if (!_armed) {
			// Otherwise the controller kept running on the ground, and a trip there stayed
			// latched into the next flight (the latch clears on the disarm edge only).
			disengage("disarmed");

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
			endHandback("stock PX4 in charge");

		} else if (!_armed) {
			endHandback("disarmed");

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

	publishLive(now);
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
	PX4_INFO("bench: mean %.1f us, worst %u us (check %.1f)", (double)mean_us,
		 static_cast<unsigned>(worst_us), (double)sink);
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
	PX4_INFO("controller: %s (policy %s, history %d x %d), fallback %s, fix source %s, stale %.2f s, unmoved %.2f s (SH_WD_FRZ_T %.1f)",
		 _param_ctrl.get() == 1 ? "policy" : "cascade", policy::NAME, policy::HISTORY, policy::NUM_OBS,
		 _fallback_active ? "ACTIVE" : "idle", useOdometryFix() ? "visual odometry" : "local position",
		 (double)_stale_s, (double)_fix_frozen_s, (double)_param_wd_frz_t.get());
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

	PX4_INFO("keep-out: %.2f m (SH_KO_DIST %.2f, -1 follows CP_DIST), obstacle data %s; retreat cap %.1f m per hold (SH_SHIFT_MAX, 0 = none)",
		 (double)_keep_out.limits().keep_out_m, (double)_param_ko_dist.get(), keepOutDataFresh(now) ? "fresh" : "none",
		 (double)_param_shift_max.get());

	if (_state == structure_hold_status_s::STATE_ENGAGED || _state == structure_hold_status_s::STATE_LANDING
	    || _state == structure_hold_status_s::STATE_HOVER) {
		PX4_INFO("setpoint NED: %.2f %.2f %.2f m, yaw %.1f deg", (double)_setpoint.pos(0), (double)_setpoint.pos(1),
			 (double)_setpoint.pos(2), (double)math::degrees(_setpoint.yaw));
		PX4_INFO("error NED: %.2f %.2f %.2f m, thrust %.2f", (double)_last_output.pos_err(0),
			 (double)_last_output.pos_err(1), (double)_last_output.pos_err(2), (double)_last_output.thrust);
		PX4_INFO("wall %.2f m, hold point clearance %.2f m, moved back %.2f m since engage%s",
			 (double)_wall_out.wall_m, (double)_wall_out.clearance_m, (double)_wall_guard.totalShift(),
			 _wall_guard.capped() ? ", AT THE CAP" : "");
		PX4_INFO("keep-out: %d points, nearest obstacle %.2f m from the hold point, moved %.2f m since engage%s%s",
			 _keep_out_out.points, (double)_keep_out_out.nearest_m, (double)_keep_out.totalShift(),
			 _keep_out_out.no_room ? ", NO ROOM" : "", _keep_out.capped() ? ", AT THE CAP" : "");
	}

	PX4_INFO("range hover (SH_HB_RANGE %d): %s, reference %.2f m, error %.2f m; wind lean N %.2f E %.2f m/s^2%s",
		 (int)_param_hb_range.get(), _range_hover.active() ? "FLYING" : "idle", (double)_range_hover.reference(),
		 (double)_range_hover.error(), (double)_wind_ff.x(), (double)_wind_ff.y(), _wind_ff.valid() ? "" : " (none yet)");

	if (_param_bench.get() != 0) {
		PX4_WARN("SH_BENCH 1: engages on the ground (bench test, props off only)");
	}

	PX4_INFO("worst control step since boot: %u us (budget: the %u us tick)", (unsigned)_max_loop_time_us,
		 (unsigned)CONTROL_INTERVAL);
	perf_print_counter(_loop_perf);
	perf_print_counter(_tick_perf);
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
without descending; a trip next to a measured wall hands back into it first. With
SH_HB_RANGE (on by default) it also flies with the horizontal estimate gone, holding the
measured wall distance on the rangefinder alone.

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
