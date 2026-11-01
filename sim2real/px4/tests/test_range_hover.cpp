// Unit tests for Structure Hover on the rangefinder alone (RangeHover.hpp). No PX4 tree:
//   g++ -std=c++14 -O2 -Wall -Wextra -Werror -I../structure_hold test_range_hover.cpp && ./a.out
// (CI adds PX4's stricter warnings: -Wshadow -Wdouble-promotion -Wfloat-equal.)
//
// The closed-loop cases fly a 1D point mass along the wall normal: the commanded
// acceleration reaches the vehicle through a 0.15 s lag (PX4's tilt response), air drag
// pulls it toward the wind speed, and the range comes at 20 Hz. They check the property the
// controller exists for: the vehicle stays off the wall and settles back to the reference.

#include "RangeHover.hpp"

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

struct FlightResult {
	float min_wall_m;
	float max_wall_m;
	float final_wall_m;
};

/**
 * Fly ``seconds`` at 100 Hz from ``wall0_m`` with velocity ``v0`` (toward the wall
 * positive). ``wind_toward_mps`` is the wind speed along the normal; drag ``drag`` 1/s pulls
 * the vehicle's velocity toward it, so holding station takes an acceleration of
 * -drag * wind away from the wall. ``ff_n`` is the feed-forward the controller is given.
 */
static FlightResult fly(float wall0_m, float v0, float wind_toward_mps, float ff_n, float seconds,
			float min_standoff_m = 0.7f, float drag = 0.35f)
{
	RangeHover h;
	h.start(wall0_m, min_standoff_m, ff_n, 0.f);
	const float dt = 0.01f;
	float wall = wall0_m, v = v0, a_lag = ff_n;
	uint64_t t_us = 1000000, last_reading_us = 0;
	FlightResult r{wall, wall, wall};

	for (int i = 0; i < static_cast<int>(seconds / dt); i++) {
		t_us += 10000;

		if (t_us - last_reading_us >= 50000) {  // 20 Hz rangefinder
			last_reading_us = t_us;
		}

		const float a_cmd = h.step(wall, last_reading_us, dt);
		a_lag += (a_cmd - a_lag) * (dt / (0.15f + dt));
		const float acc = a_lag + drag * (wind_toward_mps - v);
		v += acc * dt;
		wall -= v * dt;  // moving toward the wall shortens the distance
		r.min_wall_m = std::fmin(r.min_wall_m, wall);
		r.max_wall_m = std::fmax(r.max_wall_m, wall);
	}

	r.final_wall_m = wall;
	return r;
}

int main()
{
	// Reference: the distance at the start, never under the minimum standoff.
	{
		RangeHover h;
		h.start(1.2f, 0.7f, 0.f, 0.f);
		CHECK(h.active() && near(h.reference(), 1.2f));
		h.start(0.5f, 0.7f, 0.f, 0.f);
		CHECK(near(h.reference(), 0.7f) && near(h.error(), -0.2f));
		h.stop();
		CHECK(!h.active() && near(h.step(1.f, 1, 0.01f), 0.f));
	}

	// At the reference and still: the output is the feed-forward alone.
	{
		RangeHover h;
		h.start(1.f, 0.7f, -1.2f, 0.4f);
		CHECK(near(h.step(1.f, 1000, 0.01f), -1.2f) && near(h.alongWall(), 0.4f));
		CHECK(near(h.step(1.f, 51000, 0.01f), -1.2f));
	}

	// Too close: accelerate away (negative); too far: toward (positive); clamped.
	{
		RangeHover h;
		h.start(1.f, 0.7f, 0.f, 0.f);
		CHECK(h.step(0.8f, 1000, 0.01f) < 0.f);
		h.start(1.f, 0.7f, 0.f, 0.f);
		CHECK(h.step(1.3f, 1000, 0.01f) > 0.f);
		h.start(1.f, 0.7f, 0.f, 0.f);
		CHECK(near(h.step(9.f, 1000, 0.01f), h.gains().max_correction));
	}

	// The rate comes from successive readings: closing in on the wall pushes away even
	// at zero error, and a reading repeated on later ticks does not move the rate.
	{
		RangeHover h;
		h.start(1.f, 0.7f, 0.f, 0.f);
		h.step(1.05f, 1000, 0.01f);
		const float a = h.step(1.f, 51000, 0.01f);  // 1 m/s toward the wall, at the reference
		CHECK(h.rate() < -0.5f && a < 0.f);
		const float rate = h.rate();
		h.step(1.f, 51000, 0.01f);
		CHECK(near(h.rate(), rate));
		// A long gap between readings restarts the rate instead of differencing across it.
		h.step(0.9f, 51000 + 2000000, 0.01f);
		CHECK(near(h.rate(), 0.f));
	}

	// Without a reading the error is held and the integral does not move.
	{
		RangeHover h;
		h.start(1.f, 0.7f, 0.f, 0.f);
		h.step(1.2f, 1000, 0.1f);
		const float i0 = h.integral();
		CHECK(i0 > 0.f);
		h.step(NAN, 1000, 0.1f);
		CHECK(near(h.integral(), i0) && near(h.error(), 0.2f));
		// And the integral is clamped.
		for (int k = 0; k < 10000; k++) { h.step(5.f, 1000, 0.1f); }
		CHECK(near(h.integral(), h.gains().max_integral));
	}

	// Closed loop, the feed-forward right: a 4 m/s wind into the wall, held in place.
	{
		const float wind = 4.f, drag = 0.35f;
		const FlightResult r = fly(1.f, 0.f, wind, -drag * wind, 30.f);
		CHECK(r.min_wall_m > 0.9f && r.max_wall_m < 1.1f && near(r.final_wall_m, 1.f, 0.05f));
	}

	// Closed loop, no feed-forward (none was learned): the wind pushes it in, the range
	// and the integral bring it back; it never gets near prop contact (0.40 m).
	{
		const FlightResult r = fly(1.f, 0.f, 4.f, 0.f, 40.f);
		std::printf("no feed-forward: closest %.2f m, end %.2f m\n", static_cast<double>(r.min_wall_m),
			    static_cast<double>(r.final_wall_m));
		CHECK(r.min_wall_m > 0.5f && near(r.final_wall_m, 1.f, 0.05f));
	}

	// Closed loop, entering while drifting toward the wall at 0.5 m/s (the vehicle had
	// been braking on a dead-reckoned estimate) with half the right feed-forward.
	{
		const float wind = 4.f, drag = 0.35f;
		const FlightResult r = fly(0.9f, 0.5f, wind, -0.5f * drag * wind, 40.f);
		std::printf("drifting entry: closest %.2f m, end %.2f m\n", static_cast<double>(r.min_wall_m),
			    static_cast<double>(r.final_wall_m));
		CHECK(r.min_wall_m > 0.6f && near(r.final_wall_m, 0.9f, 0.05f));
	}

	// Entered already inside the standoff: it backs out to SH_WALL_MIN.
	{
		const FlightResult r = fly(0.5f, 0.f, 0.f, 0.f, 30.f);
		CHECK(r.min_wall_m > 0.45f && near(r.final_wall_m, 0.7f, 0.05f));
	}

	// Wind feed-forward: the thrust's horizontal specific force, low-passed, frozen when
	// the estimate is not good, ignoring a vehicle tilted past 60 deg.
	{
		WindFeedForward w;
		CHECK(!w.valid());
		const float tilt = 0.1f;  // leaning north: body z tips south
		const float bz[3] = {-std::sin(tilt), 0.f, std::cos(tilt)};
		w.update(bz, 0.01f, true);
		CHECK(w.valid() && near(w.x(), WindFeedForward::GRAVITY * std::tan(tilt), 1e-3f) && near(w.y(), 0.f));
		const float level[3] = {0.f, 0.f, 1.f};
		const float before = w.x();
		w.update(level, 0.01f, false);  // estimate bad: frozen
		CHECK(near(w.x(), before));
		w.update(level, 0.01f, true);  // good: moves toward level slowly
		CHECK(w.x() < before && w.x() > 0.99f * before);
		const float flipped[3] = {0.9f, 0.f, 0.4f};
		const float x = w.x();
		w.update(flipped, 0.01f, true);
		CHECK(near(w.x(), x));
		w.reset();
		CHECK(!w.valid() && near(w.x(), 0.f));
	}

	if (failures == 0) {
		std::printf("test_range_hover: all checks passed\n");
		return 0;
	}

	std::printf("test_range_hover: %d failure(s)\n", failures);
	return 1;
}
