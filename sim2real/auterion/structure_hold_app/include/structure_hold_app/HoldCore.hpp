/**
 * Structure Hold as an app: the flight logic of the AuterionOS / external-mode variant.
 *
 * The onboard module (sim2real/px4/structure_hold) runs inside PX4. On an Auterion vehicle
 * third-party code runs as an app on the companion side and reaches PX4 over ROS 2 through
 * the PX4 ROS 2 Interface Library (px4_ros2), which the Auterion SDK is built on
 * (docs/auterion-app.md). This header is that app's brain, kept free of ROS and PX4 so
 * tests/test_hold_core.cpp runs it in a bare g++ build, exactly as the module's own tests do.
 *
 * It reuses the module's PX4-free parts unchanged, so both variants fly the same code:
 * PolicyController.hpp (observation, policy, brake-and-hover fallback, yaw guard, setpoint
 * mapping), Watchdog.hpp (trips, frozen fix, latch), Handback.hpp (which stock mode to ask
 * for after a trip) and ParamLimits.hpp (parameter ranges).
 *
 * What the app does not do yet, and the module does: the wall guard and the wall-range
 * input, Structure Hover and Structure Land, the estimator-reset hold-off and the bench
 * mode. docs/auterion-app.md lists them as the next steps; v22 needs none of them.
 *
 * Differences forced by running outside PX4, each one the conservative choice:
 *   - an engage on the ground, or while a trip is latched, is refused before the mode
 *     starts (arming check), instead of starting and handing back;
 *   - a trip asks for Position mode, then Hold, then Land, like a trip with no measured
 *     wall in the module; while commander switches, the vehicle is held level at hover
 *     thrust, as the module does;
 *   - if the app stops (crash, link to PX4 lost) commander's own external-mode timeout
 *     takes over and switches to its failsafe mode; the app cannot prevent that and does
 *     not try to.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include "Handback.hpp"
#include "ParamLimits.hpp"
#include "PolicyController.hpp"
#include "Watchdog.hpp"

namespace structure_hold_app
{

using structure_hold::clampf;

/** The app's settings, named and ranged like the module's SH_ parameters. */
struct AppParams {
	float wd_est_to_s{structure_hold::param_limits::SH_WD_EST_TO.def};
	float wd_pos_err_m{structure_hold::param_limits::SH_WD_POS_ERR.def};
	float wd_tilt_deg{structure_hold::param_limits::SH_WD_TILT.def};
	float wd_frz_t_s{structure_hold::param_limits::SH_WD_FRZ_T.def};
	int32_t vel_src{structure_hold::param_limits::SH_VEL_SRC.def};  // 1: velocity from the position
	int32_t fb_en{structure_hold::param_limits::SH_FB_EN.def};
	float fb_t_s{structure_hold::param_limits::SH_FB_T.def};
	int32_t fix_src{structure_hold::param_limits::SH_FIX_SRC.def};  // 0 auto, 1 odometry, 2 local position
	float fix_gap_s{structure_hold::param_limits::SH_FIX_GAP.def};
	int32_t yaw_guard{structure_hold::param_limits::SH_YAW_GUARD.def};
	float yaw_p{structure_hold::param_limits::SH_YAW_P.def};
	float rate_max{structure_hold::param_limits::SH_RATE_MAX.def};
	float fb_k{structure_hold::param_limits::SH_FB_K.def};
	float fb_tilt{structure_hold::param_limits::SH_FB_TILT.def};
	float hover_thrust{structure_hold::param_limits::HOVER_THRUST.def};  // MPC_THR_HOVER stand-in

	/** Every value through the module's ranges (ParamLimits.hpp). Returns how many changed. */
	int sanitize()
	{
		using namespace structure_hold;
		int changed = 0;
		auto f = [&changed](float &v, const FloatRange &r) {
			if (!inRange(v, r)) { ++changed; }

			v = structure_hold::sanitize(v, r);
		};
		auto i = [&changed](int32_t &v, const IntRange &r) {
			if (!inRange(v, r)) { ++changed; }

			v = structure_hold::sanitize(v, r);
		};
		f(wd_est_to_s, param_limits::SH_WD_EST_TO);
		f(wd_pos_err_m, param_limits::SH_WD_POS_ERR);
		f(wd_tilt_deg, param_limits::SH_WD_TILT);
		f(wd_frz_t_s, param_limits::SH_WD_FRZ_T);
		i(vel_src, param_limits::SH_VEL_SRC);
		i(fb_en, param_limits::SH_FB_EN);
		f(fb_t_s, param_limits::SH_FB_T);
		i(fix_src, param_limits::SH_FIX_SRC);
		f(fix_gap_s, param_limits::SH_FIX_GAP);
		i(yaw_guard, param_limits::SH_YAW_GUARD);
		f(yaw_p, param_limits::SH_YAW_P);
		f(rate_max, param_limits::SH_RATE_MAX);
		f(fb_k, param_limits::SH_FB_K);
		f(fb_tilt, param_limits::SH_FB_TILT);
		f(hover_thrust, param_limits::HOVER_THRUST);
		return changed;
	}
};

/** What the app reads from PX4 each tick, already in PX4's NED / FRD frames. */
struct AppInputs {
	uint64_t now_us{0};
	// EKF2's local position (vehicle_local_position)
	uint64_t lpos_us{0};  // when it was received
	float pos_ned[3] {0.f, 0.f, 0.f};
	float vel_ned[3] {0.f, 0.f, 0.f};
	bool xy_valid{false};
	bool z_valid{false};
	bool v_xy_valid{false};
	float q[4] {1.f, 0.f, 0.f, 0.f};  // body FRD -> NED, (w, x, y, z)
	float rates_frd[3] {0.f, 0.f, 0.f};
	float heading{0.f};  // rad, NED yaw
	bool armed{false};
	bool landed{true};
	float hover_thrust_estimate{NAN};  // hover_thrust_estimate when valid, else NaN
	bool test_fault{false};
};

enum class AppState : uint8_t {
	Standby = 0,   // not the active mode
	Engaged = 1,   // the policy flies
	Handback = 2,  // tripped, asking commander for the next stock mode
};

struct AppOutput {
	AppState state{AppState::Standby};
	structure_hold::Setpoint setpoint{};       // FRD rates [rad/s], normalised thrust
	structure_hold::Trip trip{structure_hold::Trip::None};  // set on the tick it tripped
	bool request_mode{false};                  // send a mode request for `target` this tick
	structure_hold::HandbackTarget target{structure_hold::HandbackTarget::Position};
	bool last_target{false};                   // target is the last step (Land, repeated)
	bool fallback_active{false};
	bool fallback_changed{false};              // fallback started or ended this tick
	bool policy_ok{true};
	float stale_s{0.f};
	float hold_error_m{0.f};
};

class HoldCore
{
public:
	static constexpr uint64_t HANDBACK_RETRY_US = 300000;  // as the module's HANDBACK_RETRY
	static constexpr float GRAVITY = structure_hold::GRAVITY_MPS2;

	HoldCore() { configure(AppParams{}); }

	/** Apply settings (sanitised first). Returns how many values were out of range. */
	int configure(AppParams p)
	{
		const int changed = p.sanitize();
		_params = p;

		structure_hold::WatchdogLimits lim{};
		lim.estimate_timeout_us = static_cast<uint64_t>(p.wd_est_to_s * 1e6f);
		lim.max_tilt_rad = p.wd_tilt_deg * 0.017453292f;
		lim.max_position_error_m = p.wd_pos_err_m;
		lim.max_fix_frozen_s = p.wd_frz_t_s;
		_watchdog.setLimits(lim);

		structure_hold::FallbackRule rule{};
		rule.enabled = structure_hold::resolveFallback(p.fb_en);
		rule.forced = p.fb_en == 2;
		rule.threshold_s = p.fb_t_s;
		rule.brake_vel = p.fb_k;
		rule.tilt = p.fb_tilt;
		_policy.setRule(rule);

		structure_hold::YawGuard guard{};
		guard.enabled = structure_hold::resolveYawGuard(p.yaw_guard);
		guard.gain = p.yaw_p;
		_policy.setYawGuard(guard);

		structure_hold::SetpointLimits sl{};
		sl.rate_max = structure_hold::resolveRateMax(p.rate_max);
		_policy.setLimits(sl);
		return changed;
	}

	const AppParams &params() const { return _params; }

	/** Every vehicle_visual_odometry sample (the VIO's own fix), as the module's updateFix. */
	void noteOdometry(uint64_t recv_us, uint64_t sample_us, const float pos_ned[3])
	{
		if (!std::isfinite(pos_ned[0]) || !std::isfinite(pos_ned[1]) || !std::isfinite(pos_ned[2])) {
			return;
		}

		_odom_us = recv_us;
		_odom_sample_us = sample_us > 0 ? sample_us : recv_us;
		_odometry_seen = true;
		_frozen.update(_odom_sample_us, pos_ned[0], pos_ned[1], pos_ned[2]);
	}

	bool useOdometryFix() const { return _params.fix_src == 1 || (_params.fix_src == 0 && _odometry_seen); }

	/**
	 * Whether the mode may start now, and if not why (the arming check reports it, so
	 * commander refuses the switch with a reason the pilot sees). Not while engaged: once
	 * flying, update() is the only judge.
	 */
	structure_hold::Trip entryRefusal(const AppInputs &in) const
	{
		using structure_hold::Trip;

		if (_watchdog.latched() != Trip::None) { return _watchdog.latched(); }

		if (in.landed || !in.armed) { return Trip::Landed; }

		structure_hold::WatchdogInputs w = watchdogInputs(in, 0.f);

		if (!_watchdog.mayEnter(w)) {
			const Trip t = _watchdog.check(w);
			return t == Trip::None ? Trip::EstimateReset : t;
		}

		if (!std::isfinite(in.heading)) { return Trip::EstimateInvalid; }

		return Trip::None;
	}

	/** The mode became active: hold where the vehicle is, facing the way it faces. */
	void activate(const AppInputs &in)
	{
		_policy.engage(in.pos_ned, in.heading);

		for (int i = 0; i < 3; i++) { _hold[i] = in.pos_ned[i]; }

		_engaged_us = in.now_us;
		_vel_fd_valid = false;
		_fallback_active = false;
		_state = AppState::Engaged;
	}

	/** Commander switched to another mode (the pilot, a failsafe, or our own handback). */
	void deactivate() { _state = AppState::Standby; }

	/** Disarm ends everything and clears the latch, as in the module. */
	void disarmed()
	{
		_watchdog.clearLatch();
		_state = AppState::Standby;
		_frozen.reset();
	}

	AppState state() const { return _state; }
	structure_hold::Trip latched() const { return _watchdog.latched(); }
	const float *hold() const { return _hold; }

	/** One control tick while the mode is active. */
	AppOutput update(const AppInputs &in)
	{
		AppOutput out{};
		updateHoverThrust(in);
		out.stale_s = staleSeconds(in);

		if (_state == AppState::Standby) {
			out.state = _state;
			return out;
		}

		if (!in.armed) {
			disarmed();
			out.state = _state;
			return out;
		}

		float d[3];

		for (int i = 0; i < 3; i++) { d[i] = in.pos_ned[i] - _hold[i]; }

		out.hold_error_m = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

		if (_state == AppState::Engaged) {
			const structure_hold::WatchdogInputs w = watchdogInputs(in, out.hold_error_m);
			const structure_hold::Trip trip = _watchdog.check(w);

			if (trip != structure_hold::Trip::None) {
				_watchdog.latch(trip);
				out.trip = trip;
				_plan = structure_hold::planHandback(structure_hold::HandbackFrom::Hold, true, w.xy_valid, false);
				_step = 0;
				_state = AppState::Handback;
				requestMode(in.now_us, out);

			} else {
				fly(in, out);
			}
		}

		if (_state == AppState::Handback) {
			if (out.trip == structure_hold::Trip::None && in.now_us - _request_us > HANDBACK_RETRY_US) {
				_step = _plan.next(_step);
				requestMode(in.now_us, out);
			}

			levelHover(in, out);
		}

		out.state = _state;
		return out;
	}

private:
	float staleSeconds(const AppInputs &in) const
	{
		const uint64_t fix_us = useOdometryFix() ? _odom_us : in.lpos_us;
		return in.now_us > fix_us ? static_cast<float>(in.now_us - fix_us) * 1e-6f : 0.f;
	}

	static void rotation(const float q[4], float R[3][3])
	{
		const float w = q[0], x = q[1], y = q[2], z = q[3];
		R[0][0] = 1.f - 2.f * (y * y + z * z);
		R[0][1] = 2.f * (x * y - w * z);
		R[0][2] = 2.f * (x * z + w * y);
		R[1][0] = 2.f * (x * y + w * z);
		R[1][1] = 1.f - 2.f * (x * x + z * z);
		R[1][2] = 2.f * (y * z - w * x);
		R[2][0] = 2.f * (x * z - w * y);
		R[2][1] = 2.f * (y * z + w * x);
		R[2][2] = 1.f - 2.f * (x * x + y * y);
	}

	structure_hold::WatchdogInputs watchdogInputs(const AppInputs &in, float hold_error_m) const
	{
		structure_hold::WatchdogInputs w{};
		w.now_us = in.now_us;
		// From the last fix or the last local position, whichever is older, as the module.
		const uint64_t fix_us = useOdometryFix() ? _odom_us : in.lpos_us;
		w.estimate_timestamp_us = fix_us < in.lpos_us ? fix_us : in.lpos_us;
		w.xy_valid = in.xy_valid && std::isfinite(in.pos_ned[0]) && std::isfinite(in.pos_ned[1]);
		w.z_valid = in.z_valid && std::isfinite(in.pos_ned[2]);
		w.v_xy_valid = in.v_xy_valid && std::isfinite(in.vel_ned[0]) && std::isfinite(in.vel_ned[1])
			       && std::isfinite(in.vel_ned[2]);
		float R[3][3];
		rotation(in.q, R);
		w.tilt_rad = acosf(clampf(R[2][2], -1.f, 1.f));
		w.position_error_m = hold_error_m;
		w.test_fault = in.test_fault;
		w.fix_frozen_s = useOdometryFix() ? _frozen.frozenSeconds() : 0.f;
		return w;
	}

	void updateHoverThrust(const AppInputs &in)
	{
		_hover_thrust = structure_hold::sanitizeHoverThrust(in.hover_thrust_estimate, _params.hover_thrust);
	}

	void fly(const AppInputs &in, AppOutput &out)
	{
		structure_hold::VehicleSample s{};
		float vel[3] = {in.vel_ned[0], in.vel_ned[1], in.vel_ned[2]};

		if (_params.vel_src == 1) {
			// Velocity from the differentiated position, low-passed (about 5 Hz), as the
			// module's SH_VEL_SRC 1: for simulators whose velocity estimate is unusable.
			if (_vel_fd_valid && in.now_us > _prev_pos_us) {
				const float dt = clampf(static_cast<float>(in.now_us - _prev_pos_us) * 1e-6f, 0.002f, 0.1f);
				const float alpha = dt / (0.032f + dt);

				for (int i = 0; i < 3; i++) {
					const float raw = (in.pos_ned[i] - _prev_pos[i]) / dt;
					_vel_fd[i] += (raw - _vel_fd[i]) * alpha;
				}

			} else {
				for (int i = 0; i < 3; i++) { _vel_fd[i] = vel[i]; }
			}

			for (int i = 0; i < 3; i++) {
				_prev_pos[i] = in.pos_ned[i];
				vel[i] = _vel_fd[i];
			}

			_prev_pos_us = in.now_us;
			_vel_fd_valid = true;
		}

		for (int i = 0; i < 3; i++) {
			s.pos_ned[i] = in.pos_ned[i];
			s.vel_ned[i] = vel[i];
			s.rates_frd[i] = in.rates_frd[i];
		}

		rotation(in.q, s.R);
		s.stale = out.stale_s > _params.fix_gap_s;
		s.age_s = s.stale ? out.stale_s : 0.f;
		s.engage_s = in.now_us > _engaged_us ? static_cast<float>(in.now_us - _engaged_us) * 1e-6f : 0.f;

		const structure_hold::PolicyOutput po = _policy.step(s, out.stale_s, _hover_thrust);
		out.setpoint = po.setpoint;
		out.policy_ok = po.ok;
		out.fallback_active = po.fallback_active;
		out.fallback_changed = po.fallback_active != _fallback_active;
		_fallback_active = po.fallback_active;
	}

	void levelHover(const AppInputs &in, AppOutput &out)
	{
		// Level at hover thrust while commander switches: P on the tilt, no yaw rate.
		float R[3][3];
		rotation(in.q, R);
		// Body-frame down axis of NED down is the third row of R; level means (0, 0, 1).
		const float k = _params.fb_tilt;
		// Rolled right puts gravity on +y (R[2][1] > 0): roll back left. Nose up puts it on
		// -x (R[2][0] < 0): pitch down.
		out.setpoint.rates_frd[0] = clampf(-k * R[2][1], -1.f, 1.f);
		out.setpoint.rates_frd[1] = clampf(k * R[2][0], -1.f, 1.f);
		out.setpoint.rates_frd[2] = 0.f;
		out.setpoint.thrust = _hover_thrust;
		structure_hold::sanitizeSetpoint(out.setpoint.rates_frd, out.setpoint.thrust, _hover_thrust, 0.08f, 0.9f);
	}

	void requestMode(uint64_t now_us, AppOutput &out)
	{
		out.request_mode = true;
		out.target = _plan.target(_step);
		out.last_target = _step >= _plan.lastStep();
		_request_us = now_us;
	}

	AppParams _params{};
	structure_hold::PolicyController _policy{};
	structure_hold::Watchdog _watchdog{};
	structure_hold::FrozenFix _frozen{};
	structure_hold::HandbackPlan _plan{};
	int _step{0};
	uint64_t _request_us{0};
	AppState _state{AppState::Standby};
	float _hold[3] {0.f, 0.f, 0.f};
	uint64_t _engaged_us{0};
	float _hover_thrust{0.5f};
	bool _fallback_active{false};
	bool _odometry_seen{false};
	uint64_t _odom_us{0};
	uint64_t _odom_sample_us{0};
	bool _vel_fd_valid{false};
	float _vel_fd[3] {0.f, 0.f, 0.f};
	float _prev_pos[3] {0.f, 0.f, 0.f};
	uint64_t _prev_pos_us{0};
};

} // namespace structure_hold_app
