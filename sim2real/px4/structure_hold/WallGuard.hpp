/**
 * Structure Hold wall guard: a measured floor under the hold point.
 *
 * The hold point is the engage point, captured on the position estimate. When the estimate
 * is off toward the wall at that moment (a VIO jump that EKF2 reset onto, still decaying),
 * the hold point lies on the wall face in the real world, and any controller that holds it
 * flies into the wall: v11 struck the wall that way on Gazebo seed 3 and v10 met the same
 * sequence in stock Position mode (docs/baseline.md, v11 section; docs/real-flight-gap.md).
 * The policy only sees estimate minus hold point, so no retrain can see it coming.
 *
 * A forward rangefinder (PX4 distance_sensor, orientation forward) measures the wall
 * directly. Each tick the guard turns the range into the distance from the hold point to
 * the wall along the structure x axis (the heading at engage, which points at the surface):
 *
 *     clearance = range * (body x . n) - (hold - pos) . n
 *
 * The first term is the measured perpendicular distance from the vehicle to the wall, the
 * second how far the hold point sits ahead of the vehicle along the normal n. Both are in
 * the estimate's own frame, so an offset in the estimate cancels: the clearance is where
 * the vehicle will end up once it reaches the hold point. When it is below the minimum
 * standoff, the hold point moves straight back along -n by the shortfall. It only ever
 * moves away from the wall (a ratchet): a longer range later never pulls the point in, so
 * a spurious long reading or a lost sensor cannot walk the vehicle toward the surface.
 *
 * A sample counts only when it is fresh, inside the sensor's own min/max range, not
 * flagged bad (signal_quality 0), and the body x axis points within max_off_axis_rad of
 * the wall normal (otherwise the beam may be hitting something else). With no usable
 * sample the guard does nothing and the hold flies as before; whether the mode may engage
 * without a sensor is the module's choice (SH_WALL_REQ).
 *
 * Spike gate (spike_gate_m, 0 off): a new reading whose perpendicular distance differs from
 * the last accepted one by more than the gate is not used until the next reading agrees
 * with it, so a real step passes at its second reading and a lone short return does not
 * move the point. Without it, short returns off dust, fog or spray in front of the sensor
 * read as a close wall and the ratchet marched the hold point a metre back in 20 s of
 * simulated dust (docs/field-faults.md, degrade). Each reading (its timestamp) is decided
 * once; the gate belongs to the sensor, so it carries across engages. It is the same rule
 * as RangeReference's gate below (the env's RangeSensor.spike_gate_m), applied first.
 *
 * Plain C++, no PX4 headers, so tests/test_wall_guard.cpp runs it in a bare g++ build.
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace structure_hold
{

struct WallGuardLimits {
	float min_standoff_m{0.f};            // SH_WALL_MIN; 0 = guard off
	uint64_t max_age_us{200000};          // older range samples are ignored
	float max_off_axis_rad{0.6109f};      // 35 deg between body x and the wall normal
	float spike_gate_m{0.f};              // a reading this far from the last accepted waits for a second
};

/** One forward rangefinder reading, as PX4's distance_sensor carries it. */
struct RangeSample {
	uint64_t timestamp_us{0};
	float distance_m{NAN};
	float min_distance_m{0.f};
	float max_distance_m{0.f};
	int8_t signal_quality{-1};  // -1 unknown, 0 bad, 1..100
};

struct WallGuardOutput {
	bool range_ok{false};      // a usable sample this tick
	bool spike{false};         // a usable sample the gate is holding back (range_ok false)
	float wall_m{-1.f};        // perpendicular distance vehicle -> wall, -1 without a sample
	float clearance_m{-1.f};   // hold point -> wall along n, -1 without a sample
	float shift_m{0.f};        // hold point moved back this tick
};

class WallGuard
{
public:
	void setLimits(const WallGuardLimits &limits) { _limits = limits; }
	const WallGuardLimits &limits() const { return _limits; }
	bool enabled() const { return _limits.min_standoff_m > 0.f; }

	/** New hold: forget the shift of the previous one. */
	void engage() { _total_shift_m = 0.f; }

	/** Total distance the hold point has moved back since engage. */
	float totalShift() const { return _total_shift_m; }

	/** True when the sample may be used at now_us (freshness and the sensor's own flags). */
	bool usable(const RangeSample &r, uint64_t now_us) const
	{
		if (r.timestamp_us == 0 || now_us < r.timestamp_us || now_us - r.timestamp_us > _limits.max_age_us) {
			return false;
		}

		if (r.signal_quality == 0 || !std::isfinite(r.distance_m)) {
			return false;
		}

		return r.distance_m >= r.min_distance_m && r.distance_m <= r.max_distance_m && r.distance_m > 0.f;
	}

	/**
	 * The spike gate on one reading's perpendicular wall distance (see above). True when the
	 * reading may be used. Decided once per ``sample_us``; a repeat returns the same answer.
	 */
	bool acceptWall(uint64_t sample_us, float wall_m)
	{
		if (!(_limits.spike_gate_m > 0.f)) {
			return true;
		}

		if (sample_us != _gate_sample_us) {
			_gate_sample_us = sample_us;
			const bool first = !_have_accepted;
			const bool near_accepted = _have_accepted && std::fabs(wall_m - _accepted_m) <= _limits.spike_gate_m;
			const bool near_pending = _have_pending && std::fabs(wall_m - _pending_m) <= _limits.spike_gate_m;
			_gate_ok = first || near_accepted || near_pending;

			if (_gate_ok) {
				_accepted_m = wall_m;
				_have_accepted = true;
				_have_pending = false;

			} else {
				_pending_m = wall_m;
				_have_pending = true;
			}
		}

		return _gate_ok;
	}

	/**
	 * One tick. ``normal_ned`` is the horizontal unit wall normal (structure x, NED),
	 * ``body_x_ned`` the vehicle's body x axis in NED (first column of R). Moves
	 * ``hold_ned`` back along the normal when its clearance is below the minimum.
	 */
	WallGuardOutput apply(const RangeSample &r, uint64_t now_us, const float pos_ned[3], const float normal_ned[2],
			      const float body_x_ned[3], float hold_ned[3])
	{
		WallGuardOutput out{};

		if (!usable(r, now_us)) {
			return out;
		}

		const float cos_off = body_x_ned[0] * normal_ned[0] + body_x_ned[1] * normal_ned[1];

		if (!(cos_off >= cosf(_limits.max_off_axis_rad))) {
			return out;
		}

		out.wall_m = r.distance_m * cos_off;

		if (!acceptWall(r.timestamp_us, out.wall_m)) {
			out.spike = true;
			return out;
		}

		out.range_ok = true;
		const float ahead = (hold_ned[0] - pos_ned[0]) * normal_ned[0] + (hold_ned[1] - pos_ned[1]) * normal_ned[1];
		out.clearance_m = out.wall_m - ahead;

		if (enabled() && out.clearance_m < _limits.min_standoff_m) {
			out.shift_m = _limits.min_standoff_m - out.clearance_m;
			hold_ned[0] -= out.shift_m * normal_ned[0];
			hold_ned[1] -= out.shift_m * normal_ned[1];
			out.clearance_m = _limits.min_standoff_m;
			_total_shift_m += out.shift_m;
		}

		return out;
	}

private:
	WallGuardLimits _limits{};
	float _total_shift_m{0.f};
	uint64_t _gate_sample_us{0};
	bool _gate_ok{false};
	float _accepted_m{0.f};
	bool _have_accepted{false};
	float _pending_m{0.f};
	bool _have_pending{false};
};

/**
 * The wall-range policy input (policies with HAS_WALL_RANGE, v23 on; envs/structure_hold
 * env.range_input): the hold point's wall distance, latched at the first usable range after
 * engage, minus the measured perpendicular distance each tick. Positive toward the wall, like
 * the estimated x error, but measured on the wall: once latched, an estimate jump or drift
 * does not move it. At an engage on the estimate the latch is the vehicle's own measured
 * distance (the clearance with the hold point where the vehicle is); the guard's own shifts
 * move the reference back with the hold point.
 *
 * Spike rejection (the env's RangeSensor.spike_gate_m): a new reading whose perpendicular
 * distance is more than the gate from the last accepted one is not used, and the input goes
 * unusable until the next accepted reading, unless the reading before it, also rejected,
 * agrees with it within the gate (a real step is accepted at its second reading). The filter
 * belongs to the sensor, so it carries across engages. Gate 0 turns it off.
 */
class RangeReference
{
public:
	void setSpikeGate(float gate_m) { _gate_m = gate_m; }

	void engage() { _valid = false; }

	/** One tick, after WallGuard::apply; ``sample_us`` is the reading's own timestamp. */
	void update(const WallGuardOutput &w, uint64_t sample_us)
	{
		if (!w.range_ok) { return; }

		if (sample_us != _sample_us) {  // a new reading: through the gate once
			_sample_us = sample_us;
			const bool first = !_have_accepted;
			const bool near_accepted = _have_accepted && std::fabs(w.wall_m - _accepted_m) <= _gate_m;
			const bool near_pending = _have_pending && std::fabs(w.wall_m - _pending_m) <= _gate_m;
			_sample_ok = !(_gate_m > 0.f) || first || near_accepted || near_pending;

			if (_sample_ok) {
				_accepted_m = w.wall_m;
				_have_accepted = true;
				_have_pending = false;

			} else {
				_pending_m = w.wall_m;
				_have_pending = true;
			}
		}

		if (!_valid) {
			if (_sample_ok) {
				_ref_m = w.clearance_m;  // after this tick's shift, if any
				_valid = true;
			}

		} else {
			_ref_m += w.shift_m;  // the guard moved the hold point, whatever the gate said
		}
	}

	/** True with ``err_m`` set when an accepted, usable reading backs the input this tick. */
	bool error(const WallGuardOutput &w, float &err_m) const
	{
		err_m = 0.f;

		if (!_valid || !w.range_ok || !_sample_ok) { return false; }

		err_m = _ref_m - w.wall_m;
		return true;
	}

	bool valid() const { return _valid; }
	float reference() const { return _ref_m; }

private:
	float _gate_m{0.f};
	float _ref_m{0.f};
	bool _valid{false};
	uint64_t _sample_us{0};
	bool _sample_ok{false};
	float _accepted_m{0.f};
	bool _have_accepted{false};
	float _pending_m{0.f};
	bool _have_pending{false};
};

} // namespace structure_hold
