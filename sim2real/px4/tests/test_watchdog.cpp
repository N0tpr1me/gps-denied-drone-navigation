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

// RunawayFix driven like the module: the IMU (local position ax, ay, az) at 100 Hz and the
// VIO's velocity at 50 Hz, both from a vehicle swinging in gusts. ``err(t)`` is the VIO's
// velocity error along north, ``noise`` its white velocity noise.
template<typename Err>
static float flyRunaway(RunawayFix &f, float seconds, Err err, float noise = 0.f, float imu_bias = 0.f,
			uint64_t t0 = 1'000'000, uint8_t counter = 0)
{
	unsigned rng = 12345u;
	auto gauss = [&rng]() {
		float sum = 0.f;
		for (int i = 0; i < 12; i++) {  // Irwin-Hall: near enough to a unit normal
			rng = rng * 1664525u + 1013904223u;
			sum += static_cast<float>(rng >> 8) / 16777216.f;
		}
		return sum - 6.f;
	};
	float max_dv = 0.f;
	const int steps = static_cast<int>(seconds * 100.f);

	for (int k = 0; k < steps; k++) {
		const float t = 0.01f * k;
		const uint64_t t_us = t0 + 10'000ULL * k;
		// True motion: a 0.5 Hz swing of 0.4 m/s and 1.2 m/s^2 peak, on all three axes.
		const float w = 3.14159265f;
		const float a = 0.4f * w * std::cos(w * t);
		const float v = 0.4f * std::sin(w * t);
		f.inertial(t_us, a + imu_bias, -a, 0.5f * a);

		if (k % 2 == 0) {
			f.fix(t_us, v + err(t) + noise * gauss(), -v + noise * gauss(), 0.5f * v + noise * gauss(), counter);
			max_dv = std::fmax(max_dv, f.disagreement());
		}
	}

	return max_dv;
}

static void testRunawayFix()
{
	// Off by default: never runaway, whatever the VIO does.
	{
		RunawayFix f;
		flyRunaway(f, 6.f, [](float t) { return 3.f * t; });
		CHECK(!f.runaway());
	}

	// A healthy VIO in gusts, with dust-level noise (0.2 m/s) and a 0.2 m/s^2 IMU bias the
	// EKF has not learnt: the vehicle's own motion cancels, nothing trips.
	{
		RunawayFix f;
		f.setLimit(1.f);
		const float max_dv = flyRunaway(f, 60.f, [](float) { return 0.f; }, 0.2f, 0.2f);
		CHECK(!f.runaway());
		CHECK(max_dv < 0.95f);  // about 0.4 of it is the bias over the 2 s window
	}

	// A relocalisation jump moves the position, not the velocity: nothing to see here (the
	// check reads velocities only). A constant velocity offset from the start never changes.
	{
		RunawayFix f;
		f.setLimit(1.f);
		flyRunaway(f, 20.f, [](float) { return 2.f; });
		CHECK(!f.runaway());
	}

	// A runaway at 0.5 m/s^2 from 5 s: the change over 2 s passes 1 m/s just after 7 s, and
	// three slots later it trips; it stays tripped after the VIO stops accelerating.
	{
		RunawayFix f;
		f.setLimit(1.f);
		flyRunaway(f, 7.f, [](float t) { return t > 5.f ? 0.5f * (t - 5.f) : 0.f; }, 0.05f);
		CHECK(!f.runaway());
		RunawayFix g;
		g.setLimit(1.f);
		flyRunaway(g, 8.f, [](float t) { return t > 5.f ? 0.5f * (t - 5.f) : 0.f; }, 0.05f);
		CHECK(g.runaway());
		CHECK(g.disagreement() > 1.f);
		flyRunaway(g, 10.f, [](float) { return 1.5f; }, 0.05f, 0.f, 9'000'000);  // stopped accelerating
		CHECK(g.runaway());
		CHECK(g.disagreement() < 0.5f);
		// The VIO says it restarted: trusted again, and a fresh window starts.
		flyRunaway(g, 1.f, [](float) { return 0.f; }, 0.f, 0.f, 20'000'000, 1);
		CHECK(!g.runaway());
		// So does a disarm (reset()).
		RunawayFix h;
		h.setLimit(1.f);
		flyRunaway(h, 10.f, [](float t) { return 2.f * t; });
		CHECK(h.runaway());
		h.reset();
		CHECK(!h.runaway() && h.disagreement() <= 0.f);
	}

	// A short glitch (a 2 m/s velocity spike for 0.1 s) is not a runaway: it needs three
	// slots in a row over the limit and leaves the window again.
	{
		RunawayFix f;
		f.setLimit(1.f);
		flyRunaway(f, 10.f, [](float t) { return (t > 5.f && t < 5.1f) ? 2.f : 0.f; });
		CHECK(!f.runaway());
	}

	// No velocity from the VIO (NaN), or a gap: nothing is compared across it.
	{
		RunawayFix f;
		f.setLimit(1.f);
		for (int k = 0; k < 1000; k++) {
			f.inertial(1'000'000 + 10'000ULL * k, 5.f, 0.f, 0.f);
			f.fix(1'000'000 + 10'000ULL * k, NAN, 0.f, 0.f, 0);
		}
		CHECK(!f.runaway() && f.disagreement() <= 0.f);
		// Fixes 0.6 s apart: every one starts over, the window never fills.
		RunawayFix g;
		g.setLimit(1.f);
		for (int k = 0; k < 100; k++) {
			for (int j = 0; j < 60; j++) {
				g.inertial(1'000'000 + 600'000ULL * k + 10'000ULL * j, 0.f, 0.f, 0.f);
			}
			g.fix(1'000'000 + 600'000ULL * k, 10.f * k, 0.f, 0.f, 0);
		}
		CHECK(!g.runaway() && g.disagreement() <= 0.f);
		// The IMU stops for longer than the gap: the window starts over too.
		RunawayFix h;
		h.setLimit(1.f);
		flyRunaway(h, 3.f, [](float) { return 0.f; });
		h.inertial(10'000'000, 0.f, 0.f, 0.f);  // 6 s after its last sample
		h.fix(10'000'000, 0.f, 0.f, 0.f, 0);
		CHECK(h.disagreement() <= 0.f);
	}
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

	// Entry hold-off after an estimator reset (xy_reset_counter changed): refused for
	// reset_holdoff_us, not latched, and check() (the in-flight rules) ignores it.
	{
		WatchdogLimits lim = limits;
		lim.reset_holdoff_us = 10'000'000;
		Watchdog w(lim);
		const uint64_t t0 = 100'000'000;
		auto at = [&](uint64_t t) { auto in = healthy(); in.now_us = t; in.estimate_timestamp_us = t - 10'000; return in; };
		// The first counter value is only remembered, never a reset.
		w.noteReset(3, t0);
		CHECK(!w.inResetHoldoff(t0));
		CHECK(w.resetAge(t0) < 0.f);
		CHECK(w.mayEnter(at(t0)));
		// Same counter again: nothing.
		w.noteReset(3, t0 + 1'000'000);
		CHECK(w.mayEnter(at(t0 + 1'000'000)));
		// Counter changes: refused for 10 s, allowed from 10 s on; check() unaffected.
		w.noteReset(4, t0 + 2'000'000);
		CHECK(w.inResetHoldoff(t0 + 2'000'000));
		CHECK(!w.mayEnter(at(t0 + 2'000'000)));
		CHECK(w.check(at(t0 + 2'000'000)) == Trip::None);
		CHECK(std::fabs(w.resetAge(t0 + 5'000'000) - 3.f) < 1e-6f);
		CHECK(!w.mayEnter(at(t0 + 11'999'999)));
		CHECK(w.mayEnter(at(t0 + 12'000'000)));
		CHECK(w.latched() == Trip::None);
		// A second reset restarts the hold-off; a wrapped counter (255 -> 0) counts too.
		w.noteReset(255, t0 + 20'000'000);
		w.noteReset(0, t0 + 21'000'000);
		CHECK(!w.mayEnter(at(t0 + 30'000'000)));
		CHECK(w.mayEnter(at(t0 + 31'000'000)));
		// Hold-off 0 disables the rule.
		lim.reset_holdoff_us = 0;
		w.setLimits(lim);
		w.noteReset(1, t0 + 40'000'000);
		CHECK(!w.inResetHoldoff(t0 + 40'000'000));
		CHECK(w.mayEnter(at(t0 + 40'000'000)));
	}

	// Frozen fix (SH_WD_FRZ_T): off by default, so a fix that never moves trips nothing.
	{
		Watchdog off{};
		auto in = healthy();
		in.fix_frozen_s = 100.f;
		CHECK(off.check(in) == Trip::None);
	}

	// On: trips once the fix has not moved for the limit, refuses entry while it is so,
	// and sits after the stale trip (a fix that stopped arriving is stale, not frozen).
	{
		WatchdogLimits frz{};
		frz.max_fix_frozen_s = 1.f;
		Watchdog wdf(frz);
		auto in = healthy();
		in.fix_frozen_s = 0.9f;
		CHECK(wdf.check(in) == Trip::None);
		in.fix_frozen_s = 1.f;
		CHECK(wdf.check(in) == Trip::EstimateFrozen);
		CHECK(!wdf.mayEnter(in));
		in.fix_frozen_s = NAN;
		CHECK(wdf.check(in) == Trip::EstimateFrozen);
		in.fix_frozen_s = 5.f;
		in.estimate_timestamp_us = in.now_us - 2'000'000;
		CHECK(wdf.check(in) == Trip::EstimateStale);
	}

	// FrozenFix: timed by the fixes' own sample times; any axis moving more than 0.1 mm
	// restarts it; a sample time going backwards (a VIO restart) restarts it too.
	{
		FrozenFix f;
		CHECK(f.frozenSeconds() <= 0.f);
		f.update(1'000'000, 1.f, 2.f, -2.5f);
		CHECK(f.frozenSeconds() <= 0.f);
		f.update(1'500'000, 1.f, 2.f, -2.5f);
		f.update(2'000'000, 1.00005f, 2.f, -2.5f);  // inside the tolerance
		CHECK(std::fabs(f.frozenSeconds() - 1.f) < 1e-6f);
		f.update(2'020'000, 1.f, 2.f, -2.5003f);  // z moved 0.3 mm
		CHECK(f.frozenSeconds() <= 0.f);
		f.update(4'020'000, 1.f, 2.f, -2.5003f);
		CHECK(std::fabs(f.frozenSeconds() - 2.f) < 1e-6f);
		f.update(3'000'000, 1.f, 2.f, -2.5003f);  // time went backwards
		CHECK(f.frozenSeconds() <= 0.f);
		// A real VIO's noise: a few mm per sample never freezes.
		FrozenFix noisy;
		for (int k = 0; k < 500; k++) {
			noisy.update(1'000'000 + 20'000ULL * k, 0.003f * std::sin(0.7f * k), 0.f, -2.5f);
		}
		CHECK(noisy.frozenSeconds() < 0.05f);
		f.reset();
		CHECK(f.frozenSeconds() <= 0.f);
	}

	// A runaway fix trips after the watchdog's other estimate checks, before tilt.
	{
		Watchdog wd(limits);
		WatchdogInputs in = healthy();
		in.fix_runaway = true;
		CHECK(wd.check(in) == Trip::EstimateRunaway);
		CHECK(!wd.mayEnter(in));
		in.tilt_rad = 2.f;
		CHECK(wd.check(in) == Trip::EstimateRunaway);
		in.xy_valid = false;
		CHECK(wd.check(in) == Trip::EstimateInvalid);
	}

	testRunawayFix();

	// Names are stable (they end up in logs and the status topic).
	CHECK(std::string_view(Watchdog::name(Trip::None)) == "none");
	CHECK(std::string_view(Watchdog::name(Trip::EstimateStale)) == "estimate_stale");
	CHECK(std::string_view(Watchdog::name(Trip::PositionError)) == "position_error");
	CHECK(std::string_view(Watchdog::name(Trip::EstimateReset)) == "estimate_reset");
	CHECK(std::string_view(Watchdog::name(Trip::Landed)) == "landed");
	CHECK(static_cast<int>(Trip::Landed) == 7);  // structure_hold_status TRIP_LANDED
	CHECK(std::string_view(Watchdog::name(Trip::RangeLost)) == "range_lost");
	CHECK(static_cast<int>(Trip::RangeLost) == 8);  // structure_hold_status TRIP_RANGE_LOST
	CHECK(std::string_view(Watchdog::name(Trip::EstimateFrozen)) == "estimate_frozen");
	CHECK(static_cast<int>(Trip::EstimateFrozen) == 9);  // structure_hold_status TRIP_ESTIMATE_FROZEN
	CHECK(std::string_view(Watchdog::name(Trip::EstimateRunaway)) == "estimate_runaway");
	CHECK(static_cast<int>(Trip::EstimateRunaway) == 10);  // structure_hold_status TRIP_ESTIMATE_RUNAWAY

	// Every trip has its own plain-words reason for the pilot, and the longest ground-station
	// message built from it ("Structure Hold: engage refused: " or the longer "watchdog trip"
	// form, code, reason) fits PX4's mavlink_log text, char[127] with its terminator.
	{
		const std::string_view trip_prefix = "Structure Hold: watchdog trip: ";
		const std::string_view trip_suffix = ", handing back";
		const std::string_view unknown = Watchdog::reason(static_cast<Trip>(200));
		CHECK(unknown == "unknown");

		for (int t = 0; t <= static_cast<int>(Trip::EstimateRunaway); t++) {
			const Trip trip = static_cast<Trip>(t);
			const std::string_view reason = Watchdog::reason(trip);
			const std::string_view code = Watchdog::name(trip);
			CHECK(!reason.empty() && reason != unknown && reason != code);
			const size_t length = trip_prefix.size() + code.size() + 3 + reason.size() + trip_suffix.size();

			if (length > 126) {
				std::printf("trip %d message is %zu characters\n", t, length);
			}

			CHECK(length <= 126);

			for (int u = 0; u < t; u++) {
				CHECK(reason != std::string_view(Watchdog::reason(static_cast<Trip>(u))));
			}
		}

		CHECK(std::string_view(Watchdog::reason(Trip::EstimateFrozen)).find("SH_WD_FRZ_T")
		      != std::string_view::npos);
	}

	if (failures == 0) {
		std::printf("watchdog tests: all passed\n");
		return 0;
	}

	std::printf("watchdog tests: %d failure(s)\n", failures);
	return 1;
}
