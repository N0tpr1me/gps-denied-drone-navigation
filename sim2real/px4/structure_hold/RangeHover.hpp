/**
 * Structure Hover on the rangefinder alone (SH_HB_RANGE): what keeps the vehicle off the
 * wall once the horizontal position estimate is gone.
 *
 * The decided handback order (Handback.hpp) goes to Altitude mode when EKF2 has no
 * horizontal position: height held, attitude level, nothing holding the vehicle off the
 * wall. On the 8 s link-loss field fault next to a wall in a 4 m/s wind blowing into it
 * (docs/field-faults.md) the wind then pushed the vehicle into the wall. Two things still
 * work when the position is gone: the forward rangefinder measures the wall, and the tilt
 * the vehicle needed to hold station while it still had an estimate is the wind it was
 * leaning into. This controller turns both into a horizontal acceleration setpoint for
 * PX4's own position controller, which holds the height as it does in Altitude mode and
 * needs no horizontal estimate for an acceleration-only xy setpoint:
 *
 *     along the wall normal n:  a_n = ff_n + clamp(kp e + kd de/dt + I)
 *     along the wall:           a_t = ff_t
 *
 * where e = wall - ref is positive when the vehicle is farther from the wall than the
 * reference (so a positive a_n accelerates it toward the wall), I is the integral of
 * ki e, de/dt comes from successive range readings, low-passed, and ref is the distance
 * when the range hover began, never under the minimum standoff (SH_WALL_MIN). Nothing
 * measures drift along the wall, so that axis keeps the wind feed-forward alone: the
 * vehicle drifts along the wall as fast as the wind there changes, which is no worse than
 * Altitude mode and a lot better than being pushed onto the face.
 *
 * The feed-forward (WindFeedForward) is the horizontal specific force of the thrust,
 * -g (R02, R12) / R22 in NED, low-passed while the estimate is good and frozen once it is
 * not, so the lean of a vehicle braking on a dead-reckoned estimate does not count as wind.
 *
 * Plain C++, no PX4 headers, so tests/test_range_hover.cpp runs it in a bare g++ build.
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace structure_hold
{

/** Low-passed horizontal specific force of the thrust, NED, m/s^2: the wind the vehicle
 * leans into while it holds station. */
class WindFeedForward
{
public:
	static constexpr float GRAVITY = 9.80665f;
	static constexpr float TAU_S = 2.f;  // long against a gust, short against a wind shift
	static constexpr float MAX_ACCEL = 5.f;  // about 27 deg of lean; more is not wind

	/**
	 * One tick. ``body_z_ned`` is the third column of the body-to-NED rotation, ``good``
	 * whether the position estimate is valid and fresh (frozen otherwise).
	 */
	void update(const float body_z_ned[3], float dt, bool good)
	{
		if (!good || !(dt > 0.f) || !(body_z_ned[2] > 0.5f)) {  // frozen, or tilted past 60 deg
			return;
		}

		const float ax = clamp(-GRAVITY * body_z_ned[0] / body_z_ned[2], MAX_ACCEL);
		const float ay = clamp(-GRAVITY * body_z_ned[1] / body_z_ned[2], MAX_ACCEL);

		if (!std::isfinite(ax) || !std::isfinite(ay)) {
			return;
		}

		if (!_have) {
			_a[0] = ax;
			_a[1] = ay;
			_have = true;
			return;
		}

		const float k = dt / (TAU_S + dt);
		_a[0] += (ax - _a[0]) * k;
		_a[1] += (ay - _a[1]) * k;
	}

	void reset() { _a[0] = _a[1] = 0.f; _have = false; }
	bool valid() const { return _have; }
	float x() const { return _a[0]; }
	float y() const { return _a[1]; }

private:
	static float clamp(float v, float lim) { return v < -lim ? -lim : (v > lim ? lim : v); }

	float _a[2] {0.f, 0.f};
	bool _have{false};
};

struct RangeHoverGains {
	// Sized on tests/test_range_hover.cpp's point mass (0.15 s tilt lag, 20 Hz range): a
	// 4 m/s wind into the wall with no feed-forward at all comes no closer than 0.58 m from
	// a 1 m reference, and 3 cm of range noise moves the vehicle about 3 cm.
	float kp{2.5f};              // m/s^2 per m of distance error
	float kd{3.0f};              // m/s^2 per m/s of range rate
	float ki{1.0f};              // m/s^2 per m s
	float max_correction{3.f};   // m/s^2, on top of the feed-forward
	float max_integral{1.5f};    // m/s^2
	float rate_tau_s{0.15f};     // low-pass on the range rate
	float max_sample_gap_s{0.5f};  // a longer gap between readings restarts the rate
};

class RangeHover
{
public:
	void setGains(const RangeHoverGains &gains) { _gains = gains; }
	const RangeHoverGains &gains() const { return _gains; }

	/**
	 * Begin at the measured perpendicular distance ``wall_m``; the reference is that, but
	 * never under ``min_standoff_m``. ``ff_n`` / ``ff_t`` are the wind feed-forward along
	 * the wall normal (toward the wall) and along the wall.
	 */
	void start(float wall_m, float min_standoff_m, float ff_n, float ff_t)
	{
		_ref_m = wall_m > min_standoff_m ? wall_m : min_standoff_m;
		_ff_n = std::isfinite(ff_n) ? ff_n : 0.f;
		_ff_t = std::isfinite(ff_t) ? ff_t : 0.f;
		_integral = 0.f;
		_rate = 0.f;
		_last_wall_m = wall_m;
		_last_sample_us = 0;
		_have_rate = false;
		_err_m = wall_m - _ref_m;
		_active = true;
	}

	void stop() { _active = false; }
	bool active() const { return _active; }

	/**
	 * One control tick of ``dt`` seconds. ``wall_m`` is this tick's measured perpendicular
	 * distance, or NaN without a usable reading; ``sample_us`` the reading's own timestamp
	 * (the rate updates once per reading). Without a reading the last error is held and the
	 * integral does not move. Returns the acceleration along the wall normal, toward the
	 * wall positive.
	 */
	float step(float wall_m, uint64_t sample_us, float dt)
	{
		if (!_active) {
			return 0.f;
		}

		const bool reading = std::isfinite(wall_m);

		if (reading && sample_us != _last_sample_us) {
			const float gap_s = _last_sample_us == 0 || sample_us < _last_sample_us ? -1.f
					    : static_cast<float>(sample_us - _last_sample_us) * 1e-6f;

			if (gap_s > 0.f && gap_s <= _gains.max_sample_gap_s) {
				const float raw = (wall_m - _last_wall_m) / gap_s;
				_rate = _have_rate ? _rate + (raw - _rate) * (gap_s / (_gains.rate_tau_s + gap_s)) : raw;
				_have_rate = true;

			} else {
				_rate = 0.f;
				_have_rate = false;
			}

			_last_wall_m = wall_m;
			_last_sample_us = sample_us;
			_err_m = wall_m - _ref_m;
		}

		if (reading && dt > 0.f) {
			_integral = clamp(_integral + _gains.ki * _err_m * dt, _gains.max_integral);
		}

		const float correction = clamp(_gains.kp * _err_m + _gains.kd * _rate + _integral, _gains.max_correction);
		return _ff_n + correction;
	}

	float reference() const { return _ref_m; }
	float error() const { return _err_m; }
	float rate() const { return _rate; }
	float integral() const { return _integral; }
	float alongWall() const { return _ff_t; }

private:
	static float clamp(float v, float lim) { return v < -lim ? -lim : (v > lim ? lim : v); }

	RangeHoverGains _gains{};
	bool _active{false};
	float _ref_m{0.f};
	float _ff_n{0.f};
	float _ff_t{0.f};
	float _integral{0.f};
	float _rate{0.f};
	bool _have_rate{false};
	float _last_wall_m{0.f};
	uint64_t _last_sample_us{0};
	float _err_m{0.f};
};

} // namespace structure_hold
