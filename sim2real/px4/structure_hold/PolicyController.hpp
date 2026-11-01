/**
 * Policy v2 + brake-and-hover fallback: the Structure Hold controller.
 *
 * Replaces the placeholder P cascade (HoldController.hpp, kept for SH_CTRL 0). Every control
 * tick it
 *
 *   1. builds the observation the Python env writes (envs/structure_hold/env.py, 20 values
 *      in the structure frame / FLU body, ENU-style) from PX4's NED / FRD state,
 *   2. pushes it on the observation history and evaluates the MLP (PolicyWeights.hpp,
 *      generated from training/policies/structure_hold_v2.onnx + .json),
 *   3. if the estimate has been stale for T_FB or longer, replaces the policy's action by
 *      the classical brake-and-hover action of docs/fallback.md (the policy is still
 *      evaluated so its history stays current for the hand-back),
 *   4. turns the env-convention action (FLU rates / max_rate, thrust in [-1, 1]) into FRD
 *      body rates and a normalised collective thrust for vehicle_rates_setpoint.
 *
 * Frames (docs/conventions.md): PX4 boundary is NED world and FRD body; the policy sees the
 * structure frame (x toward the surface = the heading at engage, z up, y = z cross x) and
 * FLU body axes, exactly as in the mock where the structure frame is the ENU world. The
 * conversion lives here and nowhere else. Plain C++ (no PX4 headers) so
 * tests/test_policy.cpp can check every step against values recorded from the Python env.
 */

#pragma once

#include <cmath>

#include "PolicyWeights.hpp"

namespace structure_hold
{

static constexpr float GRAVITY_MPS2 = 9.80665f;

/** The brake-and-hover rule of docs/fallback.md (constants are the mock airframe's). */
struct FallbackRule {
	bool enabled{policy::FALLBACK_DEFAULT_ON};  // SH_FB_EN; -1 defers to policy.json output.fallback
	bool forced{false};       // test hook: the fallback flies every tick
	float threshold_s{0.5f};  // T_FB: take over once the estimate has been stale this long
	float brake_vel{1.5f};    // K_BRAKE [1/s]: acceleration per m/s of dead-reckoned velocity
	float max_acc{6.0f};      // A_MAX [m/s^2], per axis of the structure frame
	float tilt{4.0f};         // K_TILT [rad/s per rad]
	float yaw{3.0f};          // K_YAW [rad/s per rad]
};

/**
 * Yaw guard: the module holds the wall-facing heading with its own P loop (the PD
 * yardstick's and the fallback's gain) and ignores the policy's yaw output. Policy v2 was
 * trained without a heading term and yaws freely, which the stock rate loop of a real
 * airframe does not tolerate (docs/decisions.md). The flown yaw replaces the policy's in
 * last_action, as the env does for the fallback.
 */
struct YawGuard {
	bool enabled{!policy::YAW_FROM_POLICY};  // SH_YAW_GUARD; -1 defers to policy.json output.yaw
	float gain{3.0f};  // rad/s per rad of heading error
};

/**
 * The per-policy defaults come from policy.json through PolicyWeights.hpp; the PX4
 * parameters can override them in the field. SH_YAW_GUARD and SH_FB_EN: -1 (auto) takes
 * the policy's own default, 0 off, 1 on (SH_FB_EN 2: forced, test hook). SH_RATE_MAX: 0
 * (auto) is the policy's action scale, otherwise the clamp in rad/s.
 */
static constexpr int PARAM_AUTO = -1;

inline bool resolveYawGuard(int sh_yaw_guard)
{
	return sh_yaw_guard < 0 ? !policy::YAW_FROM_POLICY : sh_yaw_guard != 0;
}

inline bool resolveFallback(int sh_fb_en)
{
	return sh_fb_en < 0 ? policy::FALLBACK_DEFAULT_ON : sh_fb_en != 0;
}

inline float resolveRateMax(float sh_rate_max)
{
	return sh_rate_max > 0.f ? sh_rate_max : policy::MAX_RATE_RAD_S;
}

/** How an env-convention action becomes a PX4 rates setpoint. */
struct SetpointLimits {
	float thrust_min{0.08f};  // normalised collective thrust
	float thrust_max{0.9f};
	float rate_max{policy::MAX_RATE_RAD_S};  // [rad/s] clamp on the commanded body rates
};

/** Structure frame: origin at the hold point, x along the heading at engage, z up. */
struct StructureFrame {
	float hold_ned[3] {0.f, 0.f, 0.f};
	float cos_yaw{1.f};  // of the NED heading at engage
	float sin_yaw{0.f};

	void set(const float hold[3], float yaw)
	{
		for (int i = 0; i < 3; i++) { hold_ned[i] = hold[i]; }

		cos_yaw = cosf(yaw);
		sin_yaw = sinf(yaw);
	}

	/** NED vector -> structure frame (x toward the surface, y left, z up). */
	void toStructure(const float v[3], float out[3]) const
	{
		out[0] = v[0] * cos_yaw + v[1] * sin_yaw;
		out[1] = v[0] * sin_yaw - v[1] * cos_yaw;
		out[2] = -v[2];
	}

	/** Structure-frame vector -> NED. */
	void toNed(const float u[3], float out[3]) const
	{
		out[0] = u[0] * cos_yaw + u[1] * sin_yaw;
		out[1] = u[0] * sin_yaw - u[1] * cos_yaw;
		out[2] = -u[2];
	}
};

/** What the module reads from PX4 each tick. */
struct VehicleSample {
	float pos_ned[3] {0.f, 0.f, 0.f};
	float vel_ned[3] {0.f, 0.f, 0.f};
	float R[3][3] {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};  // body FRD -> NED
	float rates_frd[3] {0.f, 0.f, 0.f};
	bool stale{false};  // no fresh position fix within the expected interval
	float age_s{0.f};   // 0 while fresh, time since the last fix while stale (the env's age)
	float engage_s{0.f};  // time since engage (the env's engage_time), s
	// Forward rangefinder (the env's wall-range input): the hold point's latched wall
	// distance minus the measured one, m, positive toward the wall; and whether a usable
	// reading backs it this tick.
	bool range_ok{false};
	float range_err_m{0.f};
};

struct Setpoint {
	float rates_frd[3] {0.f, 0.f, 0.f};  // rad/s
	float thrust{0.f};                   // normalised collective, 0..1
};

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline float finiteOrZero(float v) { return std::isfinite(v) ? v : 0.f; }

/** One NUM_OBS slice of the policy input, as envs/structure_hold/env.py _observe writes it. */
inline void buildObservation(const StructureFrame &f, const VehicleSample &s, const float last_action[4],
			     float obs[policy::NUM_OBS])
{
	using namespace policy;
	float d_ned[3], d_s[3], v_s[3];

	for (int i = 0; i < 3; i++) { d_ned[i] = s.pos_ned[i] - f.hold_ned[i]; }

	f.toStructure(d_ned, d_s);
	f.toStructure(s.vel_ned, v_s);

	for (int i = 0; i < 3; i++) {
		obs[OBS_POS_ERR + i] = clampf(d_s[i], -OBS_POS_CLIP_M, OBS_POS_CLIP_M);
		obs[OBS_VEL + i] = v_s[i];
	}

	// Gravity direction in the FLU body frame: NED down is R^T (0,0,1) = third row of R in
	// FRD; FLU flips y and z.
	obs[OBS_GRAVITY_BODY + 0] = s.R[2][0];
	obs[OBS_GRAVITY_BODY + 1] = -s.R[2][1];
	obs[OBS_GRAVITY_BODY + 2] = -s.R[2][2];

	// Body x axis (first column of R, NED) projected on the structure x and y axes.
	const float fx[3] = {s.R[0][0], s.R[1][0], s.R[2][0]};
	float h[3];
	f.toStructure(fx, h);
	obs[OBS_HEADING + 0] = h[0];
	obs[OBS_HEADING + 1] = h[1];

	// Body rates FRD -> FLU, normalised.
	obs[OBS_ANG_VEL + 0] = s.rates_frd[0] / MAX_RATE_RAD_S;
	obs[OBS_ANG_VEL + 1] = -s.rates_frd[1] / MAX_RATE_RAD_S;
	obs[OBS_ANG_VEL + 2] = -s.rates_frd[2] / MAX_RATE_RAD_S;

	for (int i = 0; i < 4; i++) { obs[OBS_LAST_ACTION + i] = last_action[i]; }

	obs[OBS_STALE] = s.stale ? 1.f : 0.f;
	obs[OBS_AGE] = clampf(s.age_s, 0.f, 1.f);

	if (HAS_ENGAGE_TIME) { obs[OBS_ENGAGE_TIME] = clampf(s.engage_s / ENGAGE_OBS_CAP_S, 0.f, 1.f); }

	if (HAS_WALL_RANGE) {
		obs[OBS_WALL_RANGE] = s.range_ok ? clampf(finiteOrZero(s.range_err_m), -RANGE_OBS_CLIP_M, RANGE_OBS_CLIP_M) : 0.f;
		obs[OBS_RANGE_OK] = s.range_ok ? 1.f : 0.f;
	}

	for (int i = 0; i < NUM_OBS; i++) { obs[i] = finiteOrZero(obs[i]); }
}

/**
 * The brake-and-hover action (docs/fallback.md), in the env's action convention, computed
 * as envs/structure_hold/baseline.py acceleration_to_action does from the observation and
 * the attitude: acceleration setpoint -K_BRAKE * v_est (no position term), thrust vector,
 * P attitude loop on the thrust axis, heading held on the structure.
 */
inline void fallbackAction(const FallbackRule &r, const StructureFrame &f, const VehicleSample &s,
			   const float obs[policy::NUM_OBS], float action[4])
{
	using namespace policy;
	float acc_s[3], f_s[3], f_ned[3], z_des_frd[3];

	for (int i = 0; i < 3; i++) {
		acc_s[i] = clampf(-r.brake_vel * obs[OBS_VEL + i], -r.max_acc, r.max_acc);
		f_s[i] = acc_s[i] * MASS_KG;
	}

	f_s[2] += MASS_KG * GRAVITY_MPS2;  // structure z is up
	float f_norm = sqrtf(f_s[0] * f_s[0] + f_s[1] * f_s[1] + f_s[2] * f_s[2]);

	if (f_norm < 1e-6f) { f_norm = 1e-6f; }

	f.toNed(f_s, f_ned);

	// Desired thrust axis in the body frame: R^T f / |f| (FRD), then FLU.
	for (int i = 0; i < 3; i++) {
		z_des_frd[i] = (s.R[0][i] * f_ned[0] + s.R[1][i] * f_ned[1] + s.R[2][i] * f_ned[2]) / f_norm;
	}

	const float z_des_flu[3] = {z_des_frd[0], -z_des_frd[1], -z_des_frd[2]};

	// tilt_err = z_body x z_des_body with z_body = (0, 0, 1): (-z_y, z_x, 0).
	float rates_flu[3];
	rates_flu[0] = r.tilt * (-z_des_flu[1]);
	rates_flu[1] = r.tilt * z_des_flu[0];
	rates_flu[2] = -r.yaw * atan2f(obs[OBS_HEADING + 1], obs[OBS_HEADING + 0]);

	// Thrust: component of the demanded force along the current body z (up in FLU).
	const float up_flu[3] = {-obs[OBS_GRAVITY_BODY + 0], -obs[OBS_GRAVITY_BODY + 1], -obs[OBS_GRAVITY_BODY + 2]};
	float thrust = f_norm * (z_des_flu[0] * up_flu[0] + z_des_flu[1] * up_flu[1] + z_des_flu[2] * up_flu[2]);

	if (thrust < 0.f) { thrust = 0.f; }

	for (int i = 0; i < 3; i++) { action[i] = clampf(rates_flu[i] / MAX_RATE_RAD_S, -1.f, 1.f); }

	action[3] = clampf(2.f * thrust / MAX_THRUST_N - 1.f, -1.f, 1.f);
}

/**
 * Env-convention action -> PX4 setpoint: FLU rates in units of max_rate become FRD rad/s;
 * the thrust in newtons for the mock airframe is scaled through the hover point, so the
 * policy's hover thrust lands on the vehicle's hover thrust whatever its mass.
 */
inline Setpoint actionToSetpoint(const float action[4], float hover_thrust, const SetpointLimits &lim)
{
	using namespace policy;
	Setpoint sp{};
	sp.rates_frd[0] = clampf(action[0] * MAX_RATE_RAD_S, -lim.rate_max, lim.rate_max);
	sp.rates_frd[1] = clampf(-action[1] * MAX_RATE_RAD_S, -lim.rate_max, lim.rate_max);
	sp.rates_frd[2] = clampf(-action[2] * MAX_RATE_RAD_S, -lim.rate_max, lim.rate_max);
	const float thrust_n = 0.5f * (clampf(action[3], -1.f, 1.f) + 1.f) * MAX_THRUST_N;
	sp.thrust = clampf(thrust_n / (MASS_KG * GRAVITY_MPS2) * hover_thrust, lim.thrust_min, lim.thrust_max);
	return sp;
}

struct PolicyOutput {
	Setpoint setpoint{};
	float action[4] {0.f, 0.f, 0.f, 0.f};         // what flew (policy or fallback), env convention
	float policy_action[4] {0.f, 0.f, 0.f, 0.f};  // the policy's own output this tick
	bool fallback_active{false};
	bool yaw_guarded{false};  // the yaw action flown is the module's heading hold
	bool ok{true};  // false if the network produced a non-finite value (zeros flown instead)
};

class PolicyController
{
public:
	static constexpr int HISTORY = policy::HISTORY;
	static constexpr int NUM_OBS = policy::NUM_OBS;

	void setRule(const FallbackRule &r) { _rule = r; }
	void setYawGuard(const YawGuard &g) { _yaw_guard = g; }
	void setLimits(const SetpointLimits &l) { _limits = l; }
	const FallbackRule &rule() const { return _rule; }
	const YawGuard &yawGuard() const { return _yaw_guard; }
	const StructureFrame &frame() const { return _frame; }

	/** New hold: structure frame from the current position and heading, history cleared. */
	void engage(const float hold_ned[3], float yaw)
	{
		_frame.set(hold_ned, yaw);
		_history.clear();

		for (int i = 0; i < 4; i++) { _last_action[i] = 0.f; }
	}

	/** Move the hold point (the wall guard's floor) keeping the heading and the history: the
	 * policy sees a step in its position error, as it does after an estimate jump. */
	void moveHold(const float hold_ned[3])
	{
		for (int i = 0; i < 3; i++) { _frame.hold_ned[i] = hold_ned[i]; }
	}

	/** One control tick. ``stale_s`` is the time since the last fresh position fix. */
	PolicyOutput step(const VehicleSample &s, float stale_s, float hover_thrust)
	{
		PolicyOutput out{};
		buildObservation(_frame, s, _last_action, _obs);
		_history.push(_obs);

		float y[policy::OUTPUT_SIZE] = {};
		policy::forward(policy::NET, _history.stacked(), y, _scratch, policy::MAX_WIDTH);

		for (int i = 0; i < 4; i++) {
			if (!std::isfinite(y[i])) {
				out.ok = false;
				y[i] = 0.f;
			}

			out.policy_action[i] = clampf(y[i], -1.f, 1.f);
			out.action[i] = out.policy_action[i];
		}

		out.fallback_active = _rule.forced || (_rule.enabled && stale_s >= _rule.threshold_s - 1e-6f);

		if (out.fallback_active) {
			fallbackAction(_rule, _frame, s, _obs, out.action);

		} else if (_yaw_guard.enabled) {
			const float yaw_err = atan2f(_obs[policy::OBS_HEADING + 1], _obs[policy::OBS_HEADING + 0]);
			out.action[2] = clampf(-_yaw_guard.gain * yaw_err / policy::MAX_RATE_RAD_S, -1.f, 1.f);
			out.yaw_guarded = true;
		}

		for (int i = 0; i < 4; i++) { _last_action[i] = out.action[i]; }

		out.setpoint = actionToSetpoint(out.action, hover_thrust, _limits);
		return out;
	}

	const float *lastObservation() const { return _obs; }

private:
	FallbackRule _rule{};
	YawGuard _yaw_guard{};
	SetpointLimits _limits{};
	StructureFrame _frame{};
	policy::ObsHistory<HISTORY, NUM_OBS> _history{};
	float _last_action[4] {0.f, 0.f, 0.f, 0.f};
	float _obs[NUM_OBS] {};
	float _scratch[2 * policy::MAX_WIDTH] {};
};

} // namespace structure_hold
