// Unit tests for the pilot's live status (PilotStatus.hpp). No PX4 tree:
//   g++ -std=c++14 -O2 -Wall -Wextra -Werror -I../structure_hold test_pilot_status.cpp && ./a.out
// (CI adds PX4's stricter warnings: -Wshadow -Wdouble-promotion -Wfloat-equal.)

#include "PilotStatus.hpp"

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

static const uint64_t S = 1000000;  // one second in microseconds

// The module's defaults: 5 s no-fix, 1 s frozen, 45 deg tilt, 2 m hold error.
static PilotStatus defaults()
{
	WatchdogLimits limits{};
	limits.estimate_timeout_us = 5 * S;
	limits.max_fix_frozen_s = 1.f;
	limits.max_tilt_rad = 45.f * 3.14159265f / 180.f;
	limits.max_position_error_m = 2.f;
	PilotStatus p;
	p.setLimits(limits);
	p.start(100 * S);
	return p;
}

static MarginInputs calm()
{
	MarginInputs in{};
	in.stale_s = 0.02f;
	in.fix_frozen_s = 0.f;
	in.tilt_rad = 0.1f;
	in.position_error_m = 0.1f;
	return in;
}

int main()
{
	// A normal hold warns about nothing; the closest limit is the tilt (13 %), not the hold error (5 %).
	{
		PilotStatus p = defaults();
		const MarginInputs in = calm();

		for (uint64_t t = 100 * S; t < 160 * S; t += S / 100) {
			CHECK(p.update(t, in).kind == PilotEvent::None);
		}

		const Margin m = p.closest(in);
		CHECK(m.limit == Trip::Tilt);  // 0.1 rad of 0.785 = 13 % beats 0.1 m of 2 m = 5 %
		CHECK(near(m.used, 0.1f / (45.f * 3.14159265f / 180.f), 1e-3f));
		CHECK(near(p.secondsToStaleTrip(in.stale_s), 4.98f));
	}

	// A dropout: one warning at 2.5 s of a 5 s timeout, carrying the values for the text,
	// then the note once the fix is back. Nothing else in between.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		int warns = 0, clears = 0;
		PilotEvent first{};

		for (int k = 0; k <= 400; k++) {  // 4 s without a fix at 100 Hz
			in.stale_s = k * 0.01f;
			const PilotEvent ev = p.update(100 * S + static_cast<uint64_t>(k) * S / 100, in);

			if (ev.kind == PilotEvent::Warn) {
				if (warns == 0) { first = ev; }

				++warns;
			}

			if (ev.kind == PilotEvent::Clear) { ++clears; }
		}

		CHECK(warns == 1 && clears == 0);
		CHECK(first.limit == Trip::EstimateStale && near(first.value, 2.5f, 0.011f) && near(first.max, 5.f));
		CHECK(near(p.secondsToStaleTrip(4.f), 1.f));
		CHECK(near(p.secondsToStaleTrip(7.f), 0.f));
		in.stale_s = 0.02f;
		CHECK(p.update(105 * S, in).kind == PilotEvent::Clear);
		CHECK(p.update(105 * S + S / 100, in).kind == PilotEvent::None);
	}

	// Tilt warns at 75 % (33.75 deg), not at 50 %: gusty flight sits near half the limit.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		in.tilt_rad = 30.f * 3.14159265f / 180.f;
		CHECK(p.update(100 * S, in).kind == PilotEvent::None);
		in.tilt_rad = 34.f * 3.14159265f / 180.f;
		const PilotEvent ev = p.update(101 * S, in);
		CHECK(ev.kind == PilotEvent::Warn && ev.limit == Trip::Tilt);
	}

	// A gusty tilt bouncing across the threshold: one warning per 10 s, and none while it
	// stays above the clear level.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		int warns = 0;

		for (int k = 0; k < 3000; k++) {  // 30 s at 100 Hz, a 2 s gust cycle
			const float deg = (k % 200) < 100 ? 36.f : 10.f;
			in.tilt_rad = deg * 3.14159265f / 180.f;

			if (p.update(100 * S + static_cast<uint64_t>(k) * S / 100, in).kind == PilotEvent::Warn) { ++warns; }
		}

		CHECK(warns == 3);  // at 0, 10 and 20 s
	}

	// Held above the warning level: warned once, not again while it stays there.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		in.position_error_m = 1.2f;
		int warns = 0;

		for (int k = 0; k < 3000; k++) {
			if (p.update(100 * S + static_cast<uint64_t>(k) * S / 100, in).kind == PilotEvent::Warn) { ++warns; }
		}

		CHECK(warns == 1);
	}

	// Two limits at once: one event per tick, the watchdog's order first (stale before hold
	// error), the other on the next tick.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		in.stale_s = 3.f;
		in.position_error_m = 1.5f;
		const PilotEvent a = p.update(100 * S, in);
		const PilotEvent b = p.update(100 * S + S / 100, in);
		const PilotEvent c = p.update(100 * S + S / 50, in);
		CHECK(a.kind == PilotEvent::Warn && a.limit == Trip::EstimateStale);
		CHECK(b.kind == PilotEvent::Warn && b.limit == Trip::PositionError);
		CHECK(c.kind == PilotEvent::None);
		CHECK(p.closest(in).limit == Trip::PositionError && near(p.closest(in).used, 0.75f));
	}

	// Checks the mode does not run are not measured: the frozen fix in Structure Hover and
	// Land, the hold error on the rangefinder alone; nor the frozen fix with SH_WD_FRZ_T 0.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		in.fix_frozen_s = 0.9f;
		in.position_error_m = 1.9f;
		in.frozen_checked = false;
		in.position_checked = false;
		CHECK(p.update(100 * S, in).kind == PilotEvent::None);
		CHECK(p.closest(in).limit == Trip::Tilt);

		WatchdogLimits off{};
		off.max_fix_frozen_s = 0.f;
		PilotStatus q;
		q.setLimits(off);
		q.start(0);
		MarginInputs fr = calm();
		fr.fix_frozen_s = 9.f;
		float v = 0.f, mx = 0.f;
		CHECK(!q.limit(1, fr, v, mx));
	}

	// A NaN counts as at the limit (the watchdog trips on it), and is warned about.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		in.position_error_m = NAN;
		const PilotEvent ev = p.update(100 * S, in);
		CHECK(ev.kind == PilotEvent::Warn && ev.limit == Trip::PositionError);
		CHECK(near(p.closest(in).used, 1.f));
		CHECK(near(PilotStatus::used(1.f, 0.f), 0.f));
		CHECK(near(PilotStatus::used(-1.f, 2.f), 0.f));
	}

	// start() forgets earlier warnings: a new engage warns afresh.
	{
		PilotStatus p = defaults();
		MarginInputs in = calm();
		in.position_error_m = 1.2f;
		CHECK(p.update(100 * S, in).kind == PilotEvent::Warn);
		p.start(101 * S);
		CHECK(p.update(101 * S, in).kind == PilotEvent::Warn);
	}

	// The status line: once per period from the start, never with a period of 0.
	{
		PilotStatus p = defaults();  // started at 100 s
		int due = 0;

		for (uint64_t t = 100 * S; t <= 130 * S; t += S / 100) {
			if (p.summaryDue(t, 10.f)) { ++due; }
		}

		CHECK(due == 3);  // 110, 120, 130 s
		CHECK(!p.summaryDue(200 * S, 0.f));
		CHECK(!p.summaryDue(200 * S, NAN));
		CHECK(p.summaryDue(200 * S, 10.f));
	}

	if (failures == 0) {
		std::printf("test_pilot_status: all checks passed\n");
		return 0;
	}

	std::printf("test_pilot_status: %d failure(s)\n", failures);
	return 1;
}
