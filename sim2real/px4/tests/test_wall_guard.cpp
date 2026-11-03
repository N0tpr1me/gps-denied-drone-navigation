// Unit tests for the Structure Hold wall guard. No PX4 tree needed:
//   g++ -std=c++14 -Wall -Wextra -Werror -I../structure_hold test_wall_guard.cpp && ./a.out
// (CI adds PX4's stricter warnings: -Wshadow -Wdouble-promotion -Wfloat-equal.)

#include "WallGuard.hpp"

#include <cmath>
#include <cstdio>

using namespace structure_hold;

static int failures = 0;

#define CHECK(cond)                                                                 \
	do {                                                                        \
		if (!(cond)) {                                                      \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			++failures;                                                 \
		}                                                                   \
	} while (0)

static bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }

static const uint64_t NOW = 10000000;

static RangeSample range(float d)
{
	RangeSample r{};
	r.timestamp_us = NOW - 20000;  // 20 ms old
	r.distance_m = d;
	r.min_distance_m = 0.1f;
	r.max_distance_m = 12.f;
	r.signal_quality = -1;
	return r;
}

static WallGuard guard(float min_standoff)
{
	WallGuardLimits l{};
	l.min_standoff_m = min_standoff;
	WallGuard g{};
	g.setLimits(l);
	return g;
}

int main()
{
	// Wall to the north (heading 0): normal = +N, body x = +N.
	const float n_north[2] = {1.f, 0.f};
	const float bx_north[3] = {1.f, 0.f, 0.f};
	const float origin[3] = {0.f, 0.f, -2.5f};

	// Hold point at the vehicle, wall 1 m ahead: clearance 1 m, no shift.
	{
		WallGuard g = guard(0.7f);
		float hold[3] = {0.f, 0.f, -2.5f};
		const WallGuardOutput o = g.apply(range(1.0f), NOW, origin, n_north, bx_north, hold);
		CHECK(o.range_ok);
		CHECK(near(o.wall_m, 1.0f));
		CHECK(near(o.clearance_m, 1.0f));
		CHECK(near(o.shift_m, 0.f));
		CHECK(near(hold[0], 0.f) && near(hold[1], 0.f) && near(hold[2], -2.5f));
	}

	// The seed 3 case: the vehicle is 0.55 m off the wall and the hold point lies 0.5 m
	// ahead of it (on the wall face in the real world). The point moves back to 0.7 m
	// clearance, i.e. 0.15 m behind the vehicle; altitude is untouched.
	{
		WallGuard g = guard(0.7f);
		float hold[3] = {0.5f, 0.f, -2.5f};
		const WallGuardOutput o = g.apply(range(0.55f), NOW, origin, n_north, bx_north, hold);
		CHECK(o.range_ok);
		CHECK(near(o.shift_m, 0.65f));
		CHECK(near(o.clearance_m, 0.7f));
		CHECK(near(hold[0], -0.15f));
		CHECK(near(hold[2], -2.5f));
		CHECK(near(g.totalShift(), 0.65f));
	}

	// An estimate offset cancels: the same physical situation seen through an estimate
	// that sits 1.4 m toward the wall (pos and hold both shifted) gives the same shift.
	{
		WallGuard g = guard(0.7f);
		const float pos[3] = {1.4f, 0.f, -2.5f};
		float hold[3] = {1.9f, 0.f, -2.5f};
		const WallGuardOutput o = g.apply(range(0.55f), NOW, pos, n_north, bx_north, hold);
		CHECK(near(o.shift_m, 0.65f));
		CHECK(near(hold[0], 1.25f));
	}

	// Ratchet: a longer reading afterwards never pulls the point back toward the wall.
	{
		WallGuard g = guard(0.7f);
		float hold[3] = {0.5f, 0.f, -2.5f};
		g.apply(range(0.55f), NOW, origin, n_north, bx_north, hold);
		const float after = hold[0];
		const WallGuardOutput o = g.apply(range(3.0f), NOW, origin, n_north, bx_north, hold);
		CHECK(near(o.shift_m, 0.f));
		CHECK(near(hold[0], after));
		g.engage();
		CHECK(near(g.totalShift(), 0.f));
	}

	// Wall to the east (heading 90 deg), vehicle yawed 20 deg off the normal: the range is
	// projected onto the normal and the point moves west only.
	{
		WallGuard g = guard(0.7f);
		const float n_east[2] = {0.f, 1.f};
		const float a = 20.f * 3.14159265f / 180.f;
		const float bx[3] = {std::sin(a), std::cos(a), 0.f};
		float hold[3] = {0.3f, 0.2f, -2.5f};
		const WallGuardOutput o = g.apply(range(0.8f), NOW, origin, n_east, bx, hold);
		CHECK(o.range_ok);
		CHECK(near(o.wall_m, 0.8f * std::cos(a)));
		CHECK(near(o.clearance_m, 0.7f));
		CHECK(near(hold[0], 0.3f));  // along the wall: untouched
		CHECK(near(hold[1], 0.2f - (0.7f - (0.8f * std::cos(a) - 0.2f))));
	}

	// Too far off axis (40 deg > 35 deg): the beam may miss the wall, no action.
	{
		WallGuard g = guard(0.7f);
		const float a = 40.f * 3.14159265f / 180.f;
		const float bx[3] = {std::cos(a), std::sin(a), 0.f};
		float hold[3] = {0.5f, 0.f, -2.5f};
		const WallGuardOutput o = g.apply(range(0.3f), NOW, origin, n_north, bx, hold);
		CHECK(!o.range_ok);
		CHECK(near(hold[0], 0.5f));
		CHECK(o.wall_m < 0.f);
	}

	// Rejected samples: stale, from the future, bad quality, NaN, outside the sensor range.
	{
		WallGuard g = guard(0.7f);
		RangeSample stale = range(0.3f);
		stale.timestamp_us = NOW - 300000;
		RangeSample future = range(0.3f);
		future.timestamp_us = NOW + 1000;
		RangeSample bad = range(0.3f);
		bad.signal_quality = 0;
		RangeSample nan = range(NAN);
		RangeSample below = range(0.05f);
		RangeSample above = range(13.f);
		RangeSample never{};
		const RangeSample rejected[] = {stale, future, bad, nan, below, above, never};

		for (const RangeSample &r : rejected) {
			float hold[3] = {0.5f, 0.f, -2.5f};
			const WallGuardOutput o = g.apply(r, NOW, origin, n_north, bx_north, hold);
			CHECK(!o.range_ok);
			CHECK(near(hold[0], 0.5f));
			CHECK(near(o.shift_m, 0.f));
		}
	}

	// More of what real drivers and links send (MAVLink DISTANCE_SENSOR is centimetres in a
	// uint16, so 0 and 65535 are common "no reading" values): infinities, zero, negative,
	// a sensor that left its own min/max unset (0) or NaN, 0 cm. None may move the point.
	{
		WallGuard g = guard(0.7f);
		RangeSample pinf = range(INFINITY);
		RangeSample ninf = range(-INFINITY);
		RangeSample zero = range(0.f);
		RangeSample negative = range(-0.4f);
		RangeSample max655 = range(655.35f);
		RangeSample unset_limits = range(0.3f);
		unset_limits.min_distance_m = 0.f;
		unset_limits.max_distance_m = 0.f;
		RangeSample nan_max = range(0.3f);
		nan_max.max_distance_m = NAN;
		RangeSample nan_min = range(0.3f);
		nan_min.min_distance_m = NAN;
		const RangeSample rejected[] = {pinf, ninf, zero, negative, max655, unset_limits, nan_max, nan_min};

		for (const RangeSample &r : rejected) {
			float hold[3] = {0.5f, 0.f, -2.5f};
			const WallGuardOutput o = g.apply(r, NOW, origin, n_north, bx_north, hold);
			CHECK(!o.range_ok);
			CHECK(near(hold[0], 0.5f) && near(hold[1], 0.f));
			CHECK(near(g.totalShift(), 0.f));
		}

		// Signal quality: -1 (unknown) and 1..100 are used, only 0 (bad) is refused.
		const int8_t qualities[] = {-1, 1, 50, 100};

		for (int8_t q : qualities) {
			RangeSample r = range(1.0f);
			r.signal_quality = q;
			CHECK(g.usable(r, NOW));
		}

		// Exactly at the sensor's limits counts (inclusive), just outside does not.
		CHECK(g.usable(range(0.1f), NOW) && g.usable(range(12.f), NOW));
		CHECK(!g.usable(range(0.099f), NOW) && !g.usable(range(12.01f), NOW));
		// Exactly max_age old counts, one microsecond more does not.
		RangeSample edge = range(1.f);
		edge.timestamp_us = NOW - 200000;
		CHECK(g.usable(edge, NOW));
		edge.timestamp_us -= 1;
		CHECK(!g.usable(edge, NOW));
	}

	// A broken estimate or attitude with a good range: no shift, and no NaN written into the
	// hold point (the watchdog trips on the estimate itself).
	{
		const float nan_pos[3] = {NAN, 0.f, -2.5f};
		const float nan_bx[3] = {NAN, 0.f, 0.f};
		const float nan_n[2] = {NAN, 0.f};

		for (int which = 0; which < 3; which++) {
			WallGuard g = guard(0.7f);
			float hold[3] = {0.5f, 0.f, -2.5f};
			const WallGuardOutput o = g.apply(range(0.4f), NOW, which == 0 ? nan_pos : origin,
							  which == 2 ? nan_n : n_north, which == 1 ? nan_bx : bx_north, hold);
			CHECK(near(o.shift_m, 0.f));
			CHECK(std::isfinite(hold[0]) && std::isfinite(hold[1]) && near(hold[0], 0.5f));
			CHECK(near(g.totalShift(), 0.f));
		}
	}

	// A burst of short glitches (a bird, spray) after a real back-off: the ratchet keeps the
	// worst one only, it does not add them up while the vehicle stays where it is.
	{
		WallGuard g = guard(0.7f);
		float hold[3] = {0.f, 0.f, -2.5f};
		g.apply(range(0.4f), NOW, origin, n_north, bx_north, hold);  // 0.3 m back
		const float after_first = hold[0];

		for (int i = 0; i < 20; i++) {
			g.apply(range(0.4f), NOW, origin, n_north, bx_north, hold);
		}

		CHECK(near(hold[0], after_first));
		CHECK(near(g.totalShift(), 0.3f));
	}

	// Guard off (min standoff 0): distances are still reported, the point never moves.
	{
		WallGuard g = guard(0.f);
		CHECK(!g.enabled());
		float hold[3] = {0.5f, 0.f, -2.5f};
		const WallGuardOutput o = g.apply(range(0.55f), NOW, origin, n_north, bx_north, hold);
		CHECK(o.range_ok);
		CHECK(near(o.clearance_m, 0.05f));
		CHECK(near(hold[0], 0.5f));
	}

	// Wall-range input reference: latched at the first usable range after engage, moved by
	// the guard's shifts, and the error is measured on the wall, not on the estimate.
	{
		WallGuard g = guard(0.7f);
		RangeReference ref{};
		float err = 99.f;
		CHECK(!ref.error(WallGuardOutput{}, err) && near(err, 0.f));  // nothing latched yet

		// Engage at the vehicle, wall 1.5 m ahead: reference 1.5 m, error 0.
		float hold[3] = {0.f, 0.f, -2.5f};
		WallGuardOutput o = g.apply(range(1.5f), NOW, origin, n_north, bx_north, hold);
		ref.update(o, 1u);
		CHECK(ref.valid() && near(ref.reference(), 1.5f));
		CHECK(ref.error(o, err) && near(err, 0.f));

		// The estimate jumps 0.4 m toward the wall (the vehicle has not moved): the guard's
		// clearance follows the estimate, the reference and the error do not.
		const float jumped[3] = {0.4f, 0.f, -2.5f};
		o = g.apply(range(1.5f), NOW, jumped, n_north, bx_north, hold);
		ref.update(o, 2u);
		CHECK(near(o.clearance_m, 1.9f));
		CHECK(near(ref.reference(), 1.5f));
		CHECK(ref.error(o, err) && near(err, 0.f));

		// The vehicle really moves 0.3 m toward the wall: error +0.3 m.
		o = g.apply(range(1.2f), NOW, origin, n_north, bx_north, hold);
		ref.update(o, 3u);
		CHECK(ref.error(o, err) && near(err, 0.3f));

		// The guard shifts the hold point back 0.4 m: the reference moves with it.
		float near_hold[3] = {1.f, 0.f, -2.5f};  // hold point 1 m ahead, wall 1.3 m ahead
		o = g.apply(range(1.3f), NOW, origin, n_north, bx_north, near_hold);
		ref.update(o, 4u);
		CHECK(near(o.shift_m, 0.4f));
		CHECK(near(ref.reference(), 1.9f));

		// No usable range: no input; a new engage forgets the reference.
		RangeSample stale = range(1.5f);
		stale.timestamp_us = NOW - 500000;
		o = g.apply(stale, NOW, origin, n_north, bx_north, hold);
		ref.update(o, 5u);
		CHECK(!ref.error(o, err) && near(err, 0.f));
		ref.engage();
		CHECK(!ref.valid());

		// Pitched 20 deg: the perpendicular distance is range x cos, as in the env.
		const float a = 0.349066f;
		const float bx_pitch[3] = {std::cos(a), 0.f, -std::sin(a)};
		float hold2[3] = {0.f, 0.f, -2.5f};
		o = g.apply(range(2.f), NOW, origin, n_north, bx_pitch, hold2);
		ref.update(o, 6u);
		CHECK(near(ref.reference(), 2.f * std::cos(a)));
	}

	// Spike rejection: one outlier is not used and does not move the reference; a real step
	// is accepted at its second reading; repeats of the same reading are decided once.
	{
		WallGuard g = guard(0.f);
		RangeReference ref{};
		ref.setSpikeGate(0.25f);
		float err = 0.f;
		float hold[3] = {0.f, 0.f, -2.5f};
		WallGuardOutput o = g.apply(range(1.5f), NOW, origin, n_north, bx_north, hold);
		ref.update(o, 1u);
		CHECK(ref.error(o, err) && near(err, 0.f));

		o = g.apply(range(6.0f), NOW, origin, n_north, bx_north, hold);  // an outlier
		ref.update(o, 2u);
		CHECK(!ref.error(o, err));
		ref.update(o, 2u);  // the same reading on the next tick: still rejected
		CHECK(!ref.error(o, err));

		o = g.apply(range(1.52f), NOW, origin, n_north, bx_north, hold);  // back to normal
		ref.update(o, 3u);
		CHECK(ref.error(o, err) && near(err, -0.02f));

		o = g.apply(range(1.0f), NOW, origin, n_north, bx_north, hold);  // a real 0.5 m step
		ref.update(o, 4u);
		CHECK(!ref.error(o, err));
		o = g.apply(range(1.01f), NOW, origin, n_north, bx_north, hold);
		ref.update(o, 5u);
		CHECK(ref.error(o, err) && near(err, 0.49f));
		CHECK(near(ref.reference(), 1.5f));

		// A latch is never taken on a rejected reading.
		ref.engage();
		o = g.apply(range(8.0f), NOW, origin, n_north, bx_north, hold);
		ref.update(o, 6u);
		CHECK(!ref.valid());
		o = g.apply(range(1.02f), NOW, origin, n_north, bx_north, hold);
		ref.update(o, 7u);
		CHECK(ref.valid() && near(ref.reference(), 1.02f));
	}

	// The guard's own spike gate (dust, fog: short returns off the particles). A lone short
	// reading is held back and moves nothing; a short reading the next one agrees with is a
	// real step and moves the point; a repeat of the same reading is decided once; long
	// outliers are held back too; the gate is off by default.
	{
		WallGuardLimits l{};
		l.min_standoff_m = 0.7f;
		l.spike_gate_m = 0.25f;
		WallGuard g{};
		g.setLimits(l);
		float hold[3] = {0.f, 0.f, -2.5f};
		RangeSample r = range(1.0f);
		r.timestamp_us = NOW - 20000;
		WallGuardOutput o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(o.range_ok && !o.spike && near(o.shift_m, 0.f));

		r = range(0.15f);  // a short return off the dust
		r.timestamp_us = NOW - 15000;
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(!o.range_ok && o.spike && near(o.wall_m, 0.15f) && near(o.shift_m, 0.f));
		CHECK(near(hold[0], 0.f) && near(g.totalShift(), 0.f));
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);  // the same reading, next tick
		CHECK(!o.range_ok && o.spike && near(hold[0], 0.f));

		r = range(0.98f);  // the wall again: used at once
		r.timestamp_us = NOW - 10000;
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(o.range_ok && !o.spike && near(o.shift_m, 0.f));

		r = range(0.4f);  // a real step toward the wall: held once...
		r.timestamp_us = NOW - 5000;
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(!o.range_ok && o.spike && near(hold[0], 0.f));
		r = range(0.42f);  // ...and acted on when the next reading agrees
		r.timestamp_us = NOW - 1000;
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(o.range_ok && near(o.shift_m, 0.28f) && near(hold[0], -0.28f));

		r = range(6.0f);  // a long outlier: held back as well
		r.timestamp_us = NOW - 500;
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(!o.range_ok && o.spike);
		CHECK(!g.acceptWall(NOW - 500, 6.0f) && g.acceptWall(NOW - 400, 0.45f));

		// A new engage keeps the gate's memory (it belongs to the sensor): the first reading
		// after it is judged against the last accepted one.
		g.engage();
		r = range(0.1f);
		r.timestamp_us = NOW - 100;
		o = g.apply(r, NOW, origin, n_north, bx_north, hold);
		CHECK(!o.range_ok && o.spike);

		// Gate 0 (the default): every usable reading is used.
		WallGuard off = guard(0.7f);
		float hold2[3] = {0.f, 0.f, -2.5f};
		off.apply(range(1.0f), NOW, origin, n_north, bx_north, hold2);
		o = off.apply(range(0.15f), NOW, origin, n_north, bx_north, hold2);
		CHECK(o.range_ok && near(o.shift_m, 0.55f));
	}

	// Back-off on the vehicle's own measured distance (Structure Hover and Land): nothing
	// outside the standoff, gain x shortfall inside it, capped; nothing without a usable
	// reading or with the guard off.
	{
		WallGuard g = guard(0.7f);
		WallGuardOutput o{};
		o.range_ok = true;
		o.wall_m = 0.9f;
		CHECK(near(g.backOffSpeed(o), 0.f));
		o.wall_m = 0.7f;
		CHECK(near(g.backOffSpeed(o), 0.f));
		o.wall_m = 0.6f;
		CHECK(near(g.backOffSpeed(o), 0.2f));
		o.wall_m = 0.2f;
		CHECK(near(g.backOffSpeed(o), 0.5f));
		o.range_ok = false;
		CHECK(near(g.backOffSpeed(o), 0.f));
		WallGuard off = guard(0.f);
		o.range_ok = true;
		CHECK(near(off.backOffSpeed(o), 0.f));
	}

	// Retreat cap (SH_SHIFT_MAX): a sensor stuck reading 0.3 m while the vehicle follows the
	// hold point back would ratchet it away for ever; with a 1 m cap it moves 1 m and stops,
	// telling the module once per tick that it is capped. Cap 0 is no cap.
	{
		WallGuardLimits l{};
		l.min_standoff_m = 0.7f;
		l.max_total_shift_m = 1.0f;
		WallGuard g{};
		g.setLimits(l);
		float pos[3] = {0.f, 0.f, -2.5f};
		float hold[3] = {0.f, 0.f, -2.5f};
		WallGuardOutput o = g.apply(range(0.3f), NOW, pos, n_north, bx_north, hold);
		CHECK(near(o.shift_m, 0.4f) && !o.capped && !g.capped());
		pos[0] = hold[0];  // the vehicle reaches the point; the reading is still 0.3 m
		o = g.apply(range(0.3f), NOW, pos, n_north, bx_north, hold);
		CHECK(near(o.shift_m, 0.4f) && !o.capped && near(g.totalShift(), 0.8f));
		pos[0] = hold[0];
		o = g.apply(range(0.3f), NOW, pos, n_north, bx_north, hold);
		CHECK(near(o.shift_m, 0.2f) && o.capped && g.capped());
		CHECK(near(o.clearance_m, 0.5f));  // short of the standoff: the cap held it
		CHECK(near(g.totalShift(), 1.0f) && near(hold[0], -1.0f));
		pos[0] = hold[0];
		o = g.apply(range(0.3f), NOW, pos, n_north, bx_north, hold);
		CHECK(near(o.shift_m, 0.f) && o.capped && o.range_ok && near(hold[0], -1.0f));
		g.engage();
		CHECK(!g.capped());  // a new hold starts the count again

		WallGuard free = guard(0.7f);
		float pos2[3] = {0.f, 0.f, -2.5f};
		float hold2[3] = {0.f, 0.f, -2.5f};

		for (int i = 0; i < 10; i++) {
			const WallGuardOutput f = free.apply(range(0.3f), NOW, pos2, n_north, bx_north, hold2);
			CHECK(near(f.shift_m, 0.4f) && !f.capped);
			pos2[0] = hold2[0];
		}

		CHECK(near(free.totalShift(), 4.0f) && !free.capped());
	}

	if (failures == 0) {
		std::printf("wall guard tests: all passed\n");
		return 0;
	}

	std::printf("wall guard tests: %d failure(s)\n", failures);
	return 1;
}
