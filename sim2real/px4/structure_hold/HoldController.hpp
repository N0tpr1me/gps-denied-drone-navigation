/**
 * Placeholder hold controller for the Structure Hold skeleton.
 *
 * Stands in for the learned policy until one exists: a plain P cascade
 * (position -> velocity -> acceleration -> attitude -> body rates) that holds one fixed
 * NED setpoint and yaw. It produces exactly what the policy will produce, body rates
 * (FRD, rad/s) and a normalised collective thrust, so the module plumbing around it does
 * not change when the policy replaces it.
 *
 * Frames: PX4 boundary, so NED world and FRD body (docs/conventions.md). The quaternion
 * rotates body (FRD) to world (NED), as in vehicle_attitude.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <mathlib/mathlib.h>

namespace structure_hold
{

struct HoldGains {
	float p_pos{0.95f};      // [1/s]   position error -> velocity setpoint
	float p_vel{1.8f};       // [1/s]   velocity error -> acceleration setpoint
	float p_att{6.5f};       // [1/s]   roll/pitch attitude error -> rate setpoint
	float p_yaw{2.8f};       // [1/s]   yaw error -> yaw rate setpoint
	float v_max{1.0f};       // [m/s]   velocity setpoint limit (each of horizontal, vertical)
	float tilt_max{0.5f};    // [rad]   attitude setpoint tilt limit
	float rate_max{3.5f};    // [rad/s] roll/pitch rate setpoint limit
	float yaw_rate_max{1.5f}; // [rad/s]
	float thrust_min{0.08f}; // normalised collective thrust limits
	float thrust_max{0.9f};
};

struct HoldState {
	matrix::Vector3f pos;    // NED, m
	matrix::Vector3f vel;    // NED, m/s
	matrix::Quatf q;         // body FRD -> world NED
};

struct HoldSetpoint {
	matrix::Vector3f pos;    // NED, m
	float yaw{0.f};          // rad
};

struct HoldOutput {
	matrix::Vector3f rates;  // FRD body rates, rad/s
	float thrust{0.f};       // normalised collective thrust, 0..1
	matrix::Vector3f pos_err; // setpoint - position, NED, m
	matrix::Vector3f vel_sp;  // NED, m/s
};

inline HoldOutput holdStep(const HoldGains &g, const HoldState &st, const HoldSetpoint &sp, float hover_thrust)
{
	using matrix::Vector3f;
	using matrix::Vector2f;
	using matrix::Dcmf;
	using matrix::Quatf;

	constexpr float gravity = 9.80665f;

	HoldOutput out{};
	out.pos_err = sp.pos - st.pos;

	// Position -> velocity, limited separately in the horizontal plane and vertically.
	Vector3f vel_sp = out.pos_err * g.p_pos;
	Vector2f vel_sp_xy(vel_sp(0), vel_sp(1));

	if (vel_sp_xy.norm() > g.v_max) {
		vel_sp_xy = vel_sp_xy.normalized() * g.v_max;
	}

	vel_sp(0) = vel_sp_xy(0);
	vel_sp(1) = vel_sp_xy(1);
	vel_sp(2) = math::constrain(vel_sp(2), -g.v_max, g.v_max);
	out.vel_sp = vel_sp;

	// Velocity -> acceleration -> specific thrust vector (NED). Hover needs -gravity on z.
	Vector3f acc_sp = (vel_sp - st.vel) * g.p_vel;
	Vector3f thr = acc_sp;
	thr(2) -= gravity;

	// Never command less than half of hover vertically, so the tilt limit below has a floor.
	thr(2) = math::min(thr(2), -0.5f * gravity);

	// Tilt limit: horizontal thrust no more than tan(tilt_max) of the vertical part.
	Vector2f thr_xy(thr(0), thr(1));
	const float max_xy = fabsf(thr(2)) * tanf(g.tilt_max);

	if (thr_xy.norm() > max_xy) {
		thr_xy = thr_xy.normalized() * max_xy;
		thr(0) = thr_xy(0);
		thr(1) = thr_xy(1);
	}

	// Attitude setpoint from the thrust direction and the yaw setpoint (PX4's bodyzToAttitude).
	const Vector3f body_z = -thr.normalized();
	const Vector3f y_c(-sinf(sp.yaw), cosf(sp.yaw), 0.f);
	Vector3f body_x = y_c.cross(body_z);

	if (body_z(2) < 0.f) {
		// Thrust pointing down (inverted): keep yaw continuous by flipping x.
		body_x = -body_x;
	}

	if (body_x.norm() < 1e-4f) {
		body_x = Vector3f(1.f, 0.f, 0.f);
	}

	body_x.normalize();
	const Vector3f body_y = body_z.cross(body_x);

	Dcmf r_sp;

	for (int i = 0; i < 3; i++) {
		r_sp(i, 0) = body_x(i);
		r_sp(i, 1) = body_y(i);
		r_sp(i, 2) = body_z(i);
	}

	const Quatf q_sp(r_sp);

	// Attitude error in the body frame -> rates. Shortest rotation (q_e(0) >= 0).
	Quatf q_e = st.q.inversed() * q_sp;

	if (q_e(0) < 0.f) {
		q_e = -q_e;
	}

	const Vector3f e_att = 2.f * q_e.imag();
	out.rates(0) = math::constrain(g.p_att * e_att(0), -g.rate_max, g.rate_max);
	out.rates(1) = math::constrain(g.p_att * e_att(1), -g.rate_max, g.rate_max);
	out.rates(2) = math::constrain(g.p_yaw * e_att(2), -g.yaw_rate_max, g.yaw_rate_max);

	// Collective thrust: scale the specific thrust by the hover point.
	out.thrust = math::constrain(thr.norm() / gravity * hover_thrust, g.thrust_min, g.thrust_max);

	return out;
}

} // namespace structure_hold
