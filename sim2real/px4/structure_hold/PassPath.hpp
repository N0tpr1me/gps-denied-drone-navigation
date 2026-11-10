/**
 * Structure Hold surface pass: the hold point slides along the wall at the same standoff.
 *
 * The product's Decided row is "hold and 5 m passes"; the mock flies them since November 2026
 * (envs/structure_hold/passes.py, docs/surface-passes.md). This is the same path for the
 * module: after the engage the hold point waits start_s (the settle), then moves length_m in a
 * plane parallel to the wall, sideways (to the right of the engage heading for a positive
 * length, to the left for a negative one) or up and down it, on a trapezoidal speed profile:
 * accel_mps2 up to speed_mps, cruise, accel_mps2 down to a stop. Then it holds where it
 * stopped. The standoff never changes, because the engage heading is the wall normal and the
 * pass direction is at right angles to it.
 *
 * The policy needs no new input. Its position error is to the moving hold point, and its
 * velocity input is the estimate's velocity minus the hold point's (PolicyController.hpp,
 * VehicleSample::hold_vel_ned), which is zero on a plain hold, so every released policy reads
 * what it read before. That is the contract the mock scored v22 on.
 *
 * The path is stepped, not a function of time, so it can be held back and let go again:
 * each tick the speed moves by at most accel * dt toward
 *
 *     min(speed, sqrt(2 * accel * distance left))        (0 while held back)
 *
 * which is the trapezoid when nothing interferes (the square root is the speed from which
 * accel still stops in the distance left). hold = true brakes the point to a stop at accel
 * and keeps it there; the module sets it once the position estimate has been stale too long
 * (passWaits below), so the vehicle is not sent after a point it cannot see. On
 * release the pass picks up where it stopped. stop() ends the pass for the rest of the
 * engage (an obstacle in the way): the point brakes the same way and stays where it comes
 * to rest, brakeDistance() further on, which is how far ahead the module looks for the
 * obstacle. A point stopped dead from 0.5 m/s left the vehicle 0.3 m past it (SIH).
 *
 * Plain C++, no PX4 headers, so tests/test_pass_path.cpp runs it in a bare g++ build.
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace structure_hold
{

struct PassLimits {
	float length_m{0.f};     // SH_PASS_LEN, signed; 0 = no pass
	bool vertical{false};    // SH_PASS_DIR 1: up (positive length) or down the wall
	float speed_mps{0.5f};   // SH_PASS_SPD, the cruise speed
	float start_s{3.f};      // SH_PASS_T0, the settle after the engage
	float accel_mps2{0.5f};  // as the mock's SurfacePass.accel_mps2
};

enum class PassState : uint8_t {
	Off = 0,      // no pass this engage
	Waiting = 1,  // the settle before it starts
	Moving = 2,
	Done = 3,     // the whole length flown; holding at its end
	Stopped = 4,  // ended early (stop()) and at rest
};

struct PassStep {
	float delta_ned[3] {0.f, 0.f, 0.f};  // how far the hold point moves this tick, m
	float vel_ned[3] {0.f, 0.f, 0.f};    // the hold point's velocity, m/s
	bool started{false};                 // this tick is the first one that moves
	bool finished{false};                // this tick reached the end
	bool stopped{false};                 // this tick an early stop (stop()) came to rest
};

class PassPath
{
public:
	static constexpr float MIN_LENGTH_M = 0.01f;  // a shorter pass is no pass
	static constexpr float SNAP_M = 0.001f;       // the last millimetre is taken in one step
	static constexpr float MAX_DT_S = 0.05f;      // a late tick moves the point no further than this

	/** Read at the next engage: a parameter changed mid-pass does not bend the path. */
	void setLimits(const PassLimits &limits) { _limits = limits; }
	const PassLimits &limits() const { return _limits; }
	const PassLimits &plan() const { return _plan; }  // what this engage flies

	/**
	 * New hold. ``yaw_ned`` is the heading at engage (the wall normal); ``enable`` false
	 * flies a plain hold whatever the parameters say.
	 */
	void engage(float yaw_ned, bool enable)
	{
		_plan = _limits;
		_travelled_m = 0.f;
		_speed = 0.f;
		_wait_s = 0.f;
		_ending = false;
		const float length = std::isfinite(_plan.length_m) ? std::fabs(_plan.length_m) : 0.f;
		const bool usable = enable && length >= MIN_LENGTH_M && _plan.speed_mps > 0.f && _plan.accel_mps2 > 0.f;
		_state = usable ? PassState::Waiting : PassState::Off;
		_length_m = usable ? length : 0.f;
		const float sign = _plan.length_m < 0.f ? -1.f : 1.f;

		if (_plan.vertical) {
			// NED z is down: a positive length climbs.
			_dir[0] = 0.f;
			_dir[1] = 0.f;
			_dir[2] = -sign;

		} else {
			// The body's right at engage (FRD y), level: NED (-sin yaw, cos yaw, 0).
			_dir[0] = -sign * sinf(yaw_ned);
			_dir[1] = sign * cosf(yaw_ned);
			_dir[2] = 0.f;
		}
	}

	/**
	 * EKF2 reset its heading by ``delta_yaw``: the wall normal follows it
	 * (PolicyController::rotateFrame), and so does a sideways pass.
	 */
	void rotate(float delta_yaw)
	{
		const float c = cosf(delta_yaw), s = sinf(delta_yaw);
		const float n = _dir[0] * c - _dir[1] * s;
		_dir[1] = _dir[0] * s + _dir[1] * c;
		_dir[0] = n;
	}

	/**
	 * End the pass for the rest of this engage: the point brakes to a stop at accel (the
	 * next steps report it, PassStep::stopped) and stays there. A pass still waiting never
	 * starts.
	 */
	void stop()
	{
		if (_state == PassState::Waiting) {
			_state = PassState::Stopped;

		} else if (_state == PassState::Moving) {
			_ending = true;
		}
	}

	/** True between stop() and the point coming to rest. */
	bool ending() const { return _ending && _state == PassState::Moving; }

	/** How much further the point goes if stop() is called now, m. */
	float brakeDistance() const
	{
		return _state == PassState::Moving ? _speed * _speed / (2.f * _plan.accel_mps2) : 0.f;
	}

	/** One control tick of ``dt`` seconds. ``hold`` brakes the point and keeps it still. */
	PassStep step(float dt, bool hold)
	{
		PassStep out{};

		if (!(dt > 0.f)) { return out; }  // also a NaN

		if (dt > MAX_DT_S) { dt = MAX_DT_S; }

		if (_state == PassState::Waiting) {
			// The settle counts only on a fresh estimate, so a pass never starts blind.
			if (!hold) { _wait_s += dt; }

			if (_wait_s < _plan.start_s) { return out; }

			_state = PassState::Moving;
			out.started = true;
		}

		if (_state != PassState::Moving) { return out; }

		const float left = _length_m - _travelled_m;
		const float a = _plan.accel_mps2;
		float target = 0.f;

		if (!hold && !_ending) {
			target = sqrtf(2.f * a * left);

			if (target > _plan.speed_mps) { target = _plan.speed_mps; }
		}

		const float dv = a * dt;
		_speed += target > _speed ? (target - _speed < dv ? target - _speed : dv)
			  : -(_speed - target < dv ? _speed - target : dv);
		float ds = _speed * dt;

		// The end, held back or not: braking from the last millimetres must not carry the
		// point past the length.
		if (left - ds <= SNAP_M) {
			ds = left;
			_speed = 0.f;
			_state = PassState::Done;
			out.finished = true;

		} else if (_ending && !(_speed > 0.f)) {
			_state = PassState::Stopped;
			out.stopped = true;
		}

		_travelled_m += ds;

		for (int i = 0; i < 3; i++) {
			out.delta_ned[i] = _dir[i] * ds;
			out.vel_ned[i] = _dir[i] * _speed;
		}

		return out;
	}

	PassState state() const { return _state; }
	bool moving() const { return _state == PassState::Moving; }
	float travelled() const { return _travelled_m; }  // m along the pass, never negative
	float length() const { return _length_m; }        // m, this engage's pass; 0 without one
	float speed() const { return _speed; }
	const float *direction() const { return _dir; }   // unit vector, NED

private:
	PassLimits _limits{};
	PassLimits _plan{};  // the limits latched at engage
	PassState _state{PassState::Off};
	float _dir[3] {0.f, 1.f, 0.f};
	float _length_m{0.f};
	float _travelled_m{0.f};
	float _speed{0.f};
	float _wait_s{0.f};
	bool _ending{false};  // stop() was called: braking to rest
};

/** The longest a pass moves on a position fix that has gone stale, s. */
static constexpr float PASS_BLIND_S = 2.f;

/**
 * Whether the pass waits for the estimate this tick (``PassPath::step``'s ``hold``).
 *
 * A short dropout does not hold the pass back: EKF2 dead-reckons through it and the vehicle
 * follows the point as before. Braking the point at the fallback's 0.5 s was tried first: two
 * 1.5 s dropouts in the cruise then braked and restarted it twice, and v22 surged 0.5 m ahead
 * of it along the wall (Gazebo, docs/surface-passes.md "In Gazebo"). The pass waits when
 * the vehicle no longer follows the point: the brake-and-hover fallback has taken over
 * (``fallback_on`` and ``fallback_t_s``), or the fix has been gone ``PASS_BLIND_S``, past
 * which dead reckoning along a wall is not worth the metres.
 */
inline bool passWaits(float stale_s, bool fallback_on, float fallback_t_s)
{
	if (!std::isfinite(stale_s)) { return true; }

	return stale_s >= PASS_BLIND_S || (fallback_on && stale_s >= fallback_t_s - 1e-6f);
}

/** Time a pass of ``length_m`` takes from its start to its stop (the mock's profile), s. */
inline float passDuration(float length_m, float speed_mps, float accel_mps2)
{
	const float d = std::fabs(length_m);

	if (!(d > 0.f) || !(speed_mps > 0.f) || !(accel_mps2 > 0.f)) { return 0.f; }

	const float peak = sqrtf(d * accel_mps2) < speed_mps ? sqrtf(d * accel_mps2) : speed_mps;
	return 2.f * peak / accel_mps2 + (d - peak * peak / accel_mps2) / peak;
}

} // namespace structure_hold
