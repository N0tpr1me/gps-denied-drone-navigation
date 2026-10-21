// Unit tests for the Structure Hold watchdog. No PX4 tree needed:
//   g++ -std=c++17 -Wall -Wextra -Werror -I../structure_hold test_watchdog.cpp && ./a.out
// One test per watchdog rule (build plan, "the watchdog table, one test per row").

#include "Watchdog.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string_view>

using namespace structure_hold;

static int failures = 0;

#define CHECK(cond)                                                                 \
	do {                                                                        \
		if (!(cond)) {                                                      \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			++failures;                                                 \
		}                                                                   \
	} while (0)

static WatchdogInputs healthy()
{
	WatchdogInputs in{};
	in.now_us = 10'000'000;
	in.estimate_timestamp_us = 9'990'000;  // 10 ms old
	in.xy_valid = true;
	in.z_valid = true;
	in.v_xy_valid = true;
	in.tilt_rad = 0.1f;
	in.position_error_m = 0.2f;
	in.test_fault = false;
	return in;
}

int main()
{
	WatchdogLimits limits{};
	limits.estimate_timeout_us = 500'000;
	limits.max_tilt_rad = 0.785f;
	limits.max_position_error_m = 2.0f;
	Watchdog wd(limits);

	// Healthy inputs: may enter, keeps flying.
	CHECK(wd.check(healthy()) == Trip::None);
	CHECK(wd.mayEnter(healthy()));

	// Test fault wins over everything.
	{
		auto in = healthy();
		in.test_fault = true;
		in.xy_valid = false;
		CHECK(wd.check(in) == Trip::TestFault);
	}

	// Invalid estimate flags.
	for (int which = 0; which < 3; ++which) {
		auto in = healthy();
		if (which == 0) { in.xy_valid = false; }
		if (which == 1) { in.z_valid = false; }
		if (which == 2) { in.v_xy_valid = false; }
		CHECK(wd.check(in) == Trip::EstimateInvalid);
		CHECK(!wd.mayEnter(in));
	}

	// Stale estimate: exactly at the timeout is fine, one microsecond past is not.
	{
		auto in = healthy();
		in.estimate_timestamp_us = in.now_us - limits.estimate_timeout_us;
		CHECK(wd.check(in) == Trip::None);
		in.estimate_timestamp_us -= 1;
		CHECK(wd.check(in) == Trip::EstimateStale);
		// A timestamp from the future (clock reset) also counts as stale.
		in.estimate_timestamp_us = in.now_us + 1;
		CHECK(wd.check(in) == Trip::EstimateStale);
	}

	// Tilt, including NaN.
	{
		auto in = healthy();
		in.tilt_rad = 0.8f;
		CHECK(wd.check(in) == Trip::Tilt);
		in.tilt_rad = NAN;
		CHECK(wd.check(in) == Trip::Tilt);
	}

	// Position error, including NaN; ignored for entry (no setpoint yet).
	{
		auto in = healthy();
		in.position_error_m = 2.5f;
		CHECK(wd.check(in) == Trip::PositionError);
		CHECK(wd.mayEnter(in));
		in.position_error_m = NAN;
		CHECK(wd.check(in) == Trip::PositionError);
	}

	// Priority order when several rules fail at once.
	{
		auto in = healthy();
		in.tilt_rad = 1.0f;
		in.position_error_m = 5.0f;
		CHECK(wd.check(in) == Trip::Tilt);
		in.estimate_timestamp_us = 0;
		CHECK(wd.check(in) == Trip::EstimateStale);
		in.z_valid = false;
		CHECK(wd.check(in) == Trip::EstimateInvalid);
	}

	// Latching: a trip blocks entry until cleared; latching None changes nothing.
	{
		wd.latch(Trip::None);
		CHECK(wd.latched() == Trip::None);
		wd.latch(Trip::Tilt);
		CHECK(wd.latched() == Trip::Tilt);
		CHECK(!wd.mayEnter(healthy()));
		CHECK(wd.check(healthy()) == Trip::None);  // check() itself is stateless
		wd.clearLatch();
		CHECK(wd.mayEnter(healthy()));
	}

	// Names are stable (they end up in logs and the status topic).
	CHECK(std::string_view(Watchdog::name(Trip::None)) == "none");
	CHECK(std::string_view(Watchdog::name(Trip::EstimateStale)) == "estimate_stale");
	CHECK(std::string_view(Watchdog::name(Trip::PositionError)) == "position_error");

	if (failures == 0) {
		std::printf("watchdog tests: all passed\n");
		return 0;
	}

	std::printf("watchdog tests: %d failure(s)\n", failures);
	return 1;
}
