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

	if (failures == 0) {
		std::printf("wall guard tests: all passed\n");
		return 0;
	}

	std::printf("wall guard tests: %d failure(s)\n", failures);
	return 1;
}
