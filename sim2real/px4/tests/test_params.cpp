// Unit tests for the module's own parameter limits. No PX4 tree needed:
//   g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal
//     -I../structure_hold test_params.cpp -o test_params && ./test_params
// tests/test_param_limits.py checks the numbers against structure_hold_params.c.

#include "ParamLimits.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace structure_hold;

static int failures = 0;

#define CHECK(cond)                                                                 \
	do {                                                                        \
		if (!(cond)) {                                                      \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			++failures;                                                 \
		}                                                                   \
	} while (0)

static bool same(float a, float b) { return !(a < b) && !(a > b); }

static const FloatRange FLOATS[] = {
	param_limits::SH_WD_EST_TO, param_limits::SH_WD_POS_ERR, param_limits::SH_WD_TILT, param_limits::SH_WD_FRZ_T,
	param_limits::SH_RST_HOLD,
	param_limits::SH_P_POS, param_limits::SH_P_VEL, param_limits::SH_P_ATT, param_limits::SH_V_MAX,
	param_limits::SH_FB_T, param_limits::SH_FIX_GAP, param_limits::SH_YAW_P, param_limits::SH_RATE_MAX,
	param_limits::SH_FB_K, param_limits::SH_FB_TILT, param_limits::SH_WALL_MIN, param_limits::HOVER_THRUST,
};

static const IntRange INTS[] = {
	param_limits::SH_VEL_SRC, param_limits::SH_CTRL, param_limits::SH_FB_EN, param_limits::SH_FIX_SRC,
	param_limits::SH_YAW_GUARD, param_limits::SH_WALL_REQ, param_limits::SH_LAND_EN, param_limits::SH_HB_GUARD,
	param_limits::SH_HB_RANGE,
};

int main()
{
	const float inf = std::numeric_limits<float>::infinity();

	// Every range is well formed and its default lies inside it.
	for (const FloatRange &r : FLOATS) {
		CHECK(r.min < r.max);
		CHECK(inRange(r.def, r));
		CHECK(same(sanitize(r.def, r), r.def));
	}

	for (const IntRange &r : INTS) {
		CHECK(r.min < r.max);
		CHECK(inRange(r.def, r));
		CHECK(sanitize(r.def, r) == r.def);
	}

	// Floats: clamped to the nearest limit, non-finite takes the default, edges kept.
	for (const FloatRange &r : FLOATS) {
		CHECK(same(sanitize(r.min, r), r.min));
		CHECK(same(sanitize(r.max, r), r.max));
		CHECK(same(sanitize(r.min - 1.f, r), r.min));
		CHECK(same(sanitize(-1e30f, r), r.min));
		CHECK(same(sanitize(r.max + 1.f, r), r.max));
		CHECK(same(sanitize(NAN, r), r.def));
		CHECK(same(sanitize(inf, r), r.def));
		CHECK(same(sanitize(-inf, r), r.def));
		CHECK(!inRange(NAN, r));
		CHECK(!inRange(r.max + 1.f, r));
	}

	// The unsafe cases that motivated the limits (ParamLimits.hpp).
	CHECK(same(sanitize(-3.f, param_limits::SH_YAW_P), 0.5f));        // no positive yaw feedback
	CHECK(same(sanitize(-1.5f, param_limits::SH_FB_K), 0.2f));        // the brake always brakes
	CHECK(same(sanitize(0.f, param_limits::SH_WD_EST_TO), 0.05f));    // a timeout, not undefined behaviour
	CHECK(same(sanitize(-2.f, param_limits::SH_WD_EST_TO), 0.05f));
	CHECK(same(sanitize(NAN, param_limits::SH_FIX_GAP), 0.1f));       // dropouts stay visible
	CHECK(same(sanitize(50.f, param_limits::SH_WALL_MIN), 5.f));      // no 50 m retreat
	CHECK(same(sanitize(1e9f, param_limits::SH_WD_EST_TO), 10.f));    // stale trip cannot be switched off
	CHECK(same(sanitize(90.f, param_limits::SH_WD_TILT), 80.f));

	// Integers: anything outside the range takes the default, never a neighbouring value.
	CHECK(sanitize(5, param_limits::SH_CTRL) == 1);      // was: the placeholder cascade flew
	CHECK(sanitize(-1, param_limits::SH_CTRL) == 1);
	CHECK(sanitize(3, param_limits::SH_FB_EN) == -1);    // not 2, the forced test hook
	CHECK(sanitize(-2, param_limits::SH_FB_EN) == -1);
	CHECK(sanitize(2, param_limits::SH_FB_EN) == 2);
	CHECK(sanitize(7, param_limits::SH_FIX_SRC) == 0);
	CHECK(sanitize(2, param_limits::SH_YAW_GUARD) == -1);
	CHECK(sanitize(2, param_limits::SH_HB_GUARD) == 1);
	CHECK(sanitize(0, param_limits::SH_HB_GUARD) == 0);

	// Hover thrust: clamped; NaN or inf falls back to the (clamped) parameter, then 0.5.
	CHECK(same(sanitizeHoverThrust(0.42f, 0.6f), 0.42f));
	CHECK(same(sanitizeHoverThrust(0.f, 0.6f), 0.1f));
	CHECK(same(sanitizeHoverThrust(1.5f, 0.6f), 0.9f));
	CHECK(same(sanitizeHoverThrust(NAN, 0.6f), 0.6f));
	CHECK(same(sanitizeHoverThrust(-inf, 2.f), 0.9f));
	CHECK(same(sanitizeHoverThrust(NAN, NAN), 0.5f));

	if (failures == 0) {
		std::printf("parameter limit tests: all passed\n");
		return 0;
	}

	std::printf("parameter limit tests: %d failure(s)\n", failures);
	return 1;
}
