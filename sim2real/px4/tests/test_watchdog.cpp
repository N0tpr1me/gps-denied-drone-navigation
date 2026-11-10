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

// StillFix driven like the module on the ground: the VIO's velocity at 30 Hz from ``t0``,
// with white noise, ``speed(t)`` along its x axis, and the land detector's at-rest flag.
template<typename Speed, typename Rest>
static uint64_t sitStill(StillFix &f, float seconds, Speed speed, Rest at_rest, float noise = 0.05f,
			 uint64_t t0 = 1'000'000)
{
	unsigned rng = 777u;
	auto uniform = [&rng]() {
		rng = rng * 1664525u + 1013904223u;
		return static_cast<float>(rng >> 8) / 16777216.f - 0.5f;
	};
	const int steps = static_cast<int>(seconds * 30.f);
	uint64_t t_us = t0;

	for (int k = 0; k < steps; k++) {
		const float t = static_cast<float>(k) / 30.f;
		t_us = t0 + 33'333ULL * k;
		f.fix(t_us, speed(t) + noise * uniform(), noise * uniform(), noise * uniform(), at_rest(t));
		f.tick(t_us);
	}

	return t_us + 33'333ULL;
}

static void testStillFix()
{
	auto rest = [](float) { return true; };
	auto zero = [](float) { return 0.f; };

	// Off (limit 0): never moving, whatever the VIO says.
	{
		StillFix f;
		sitStill(f, 20.f, [](float t) { return 3.f * t; }, rest);
		CHECK(!f.moving());
	}

	// A healthy VIO on the ground, noisy as in dust (0.35 m/s peak to peak around zero): the
	// slot mean stays far under 0.3 m/s.
	{
		StillFix f;
		f.setLimit(0.3f);
		sitStill(f, 120.f, zero, rest, 0.7f);
		CHECK(!f.moving() && f.speed() < 0.15f);
	}

	// A VIO running away on the floor at 0.5 m/s^2 from 5 s: blocked within 3 s of its
	// start, and it stays blocked while the VIO keeps its 2.8 m/s.
	{
		StillFix f;
		f.setLimit(0.3f);
		auto run = [](float t) { return t < 5.f ? 0.f : std::fmin(0.5f * (t - 5.f), 2.8f); };
		sitStill(f, 5.f, run, rest);
		CHECK(!f.moving());
		StillFix g;
		g.setLimit(0.3f);
		sitStill(g, 8.f, run, rest);
		CHECK(g.moving());
		sitStill(g, 60.f, [](float) { return 2.8f; }, rest, 0.05f, 9'000'000);
		CHECK(g.moving());
	}

	// One slot over the limit (a person walks past the camera for a second) does not block.
	{
		StillFix f;
		f.setLimit(0.3f);
		sitStill(f, 20.f, [](float t) { return (t > 5.1f && t < 5.9f) ? 1.f : 0.f; }, rest);
		CHECK(!f.moving());
	}

	// Still again: the block lifts after ten slots under the limit, not sooner, and a slot
	// over it starts the count again.
	{
		StillFix f;
		f.setLimit(0.3f);
		uint64_t t = sitStill(f, 5.f, [](float) { return 1.f; }, rest);
		CHECK(f.moving());
		t = sitStill(f, 8.f, zero, rest, 0.05f, t);
		CHECK(f.moving());
		t = sitStill(f, 4.f, zero, rest, 0.05f, t);
		CHECK(!f.moving());
		t = sitStill(f, 5.f, [](float) { return 1.f; }, rest, 0.05f, t);
		CHECK(f.moving());
		t = sitStill(f, 8.f, zero, rest, 0.05f, t);
		t = sitStill(f, 1.5f, [](float) { return 1.f; }, rest, 0.05f, t);
		t = sitStill(f, 8.f, zero, rest, 0.05f, t);
		CHECK(f.moving());
		sitStill(f, 4.f, zero, rest, 0.05f, t);
		CHECK(!f.moving());
	}

	// Carried by hand (not at rest): a real speed proves nothing, and a blocked VIO is not
	// cleared by slots the vehicle was not at rest in.
	{
		StillFix f;
		f.setLimit(0.3f);
		sitStill(f, 30.f, [](float) { return 1.5f; }, [](float) { return false; });
		CHECK(!f.moving());
		// At rest only part of each slot: dropped too.
		StillFix g;
		g.setLimit(0.3f);
		sitStill(g, 30.f, [](float) { return 1.5f; }, [](float t) { return std::fmod(t, 1.f) < 0.5f; });
		CHECK(!g.moving());
		StillFix h;
		h.setLimit(0.3f);
		uint64_t t = sitStill(h, 5.f, [](float) { return 1.f; }, rest);
		CHECK(h.moving());
		t = sitStill(h, 30.f, zero, [](float) { return false; }, 0.05f, t);
		CHECK(h.moving());
		sitStill(h, 12.f, zero, rest, 0.05f, t);
		CHECK(!h.moving());
	}

	// A runaway caught another way (RunawayFix while carried): distrusted until seen still.
	{
		StillFix f;
		f.setLimit(0.3f);
		f.distrust();
		CHECK(f.moving());
		uint64_t t = sitStill(f, 8.f, zero, rest);
		CHECK(f.moving());
		sitStill(f, 4.f, zero, rest, 0.05f, t);
		CHECK(!f.moving());
	}

	// The VIO goes silent for 5 s: EKF2 no longer fuses it, the block lifts. A VIO that
	// sends no velocity (NaN) is never judged moving.
	{
		StillFix f;
		f.setLimit(0.3f);
		const uint64_t t = sitStill(f, 5.f, [](float) { return 1.f; }, rest);
		CHECK(f.moving());
		f.tick(t + 4'000'000);
		CHECK(f.moving());
		f.tick(t + 5'200'000);
		CHECK(!f.moving());
		StillFix g;
		g.setLimit(0.3f);

		for (int k = 0; k < 600; k++) {
			g.fix(1'000'000 + 33'333ULL * k, NAN, 5.f, 0.f, true);
		}

		CHECK(!g.moving());
	}

	// reset() (the module calls it on arming) clears the block and the counts.
	{
		StillFix f;
		f.setLimit(0.3f);
		sitStill(f, 5.f, [](float) { return 1.f; }, rest);
		CHECK(f.moving());
		f.reset();
		CHECK(!f.moving() && f.speed() <= 0.f);
	}
}

static void testRunawayFix()
{
	// Off by default: never runaway, whatever the VIO does.
	{
		RunawayFix f;
		flyRunaway(f, 6.f, [](float t) { return 3.f * t; });
		CHECK(!f.runaway());
	}

	// A healthy VIO in gusts, with dust-level noise (0.2 m/s) and a 0.1 m/s^2 IMU bias the
	// EKF has not learnt: the vehicle's own motion cancels, nothing trips.
	{
		RunawayFix f;
		f.setLimit(0.8f);
		const float max_dv = flyRunaway(f, 60.f, [](float) { return 0.f; }, 0.2f, 0.1f);
		CHECK(!f.runaway());
		CHECK(max_dv < 0.65f);  // 0.3 of it is the bias over the 3 s window
	}

	// A relocalisation jump moves the position, not the velocity: nothing to see here (the
	// check reads velocities only). A constant velocity offset from the start never changes.
	{
		RunawayFix f;
		f.setLimit(1.f);
		flyRunaway(f, 20.f, [](float) { return 2.f; });
		CHECK(!f.runaway());
	}

	// A runaway at 0.5 m/s^2 from 5 s, at the default 0.8 m/s: the change over 3 s passes it
	// 1.6 s in, and two 0.25 s slots later it trips; it stays tripped after the VIO stops
	// accelerating.
	{
		RunawayFix f;
		f.setLimit(0.8f);
		flyRunaway(f, 6.5f, [](float t) { return t > 5.f ? 0.5f * (t - 5.f) : 0.f; }, 0.05f);
		CHECK(!f.runaway());
		RunawayFix g;
		g.setLimit(0.8f);
		flyRunaway(g, 8.f, [](float t) { return t > 5.f ? 0.5f * (t - 5.f) : 0.f; }, 0.05f);
		CHECK(g.runaway());
		CHECK(g.disagreement() > 0.8f);
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

	// Dust: 0.5 s tracking losses every 2 s do not blind it. The IMU integral runs through the
	// gaps and the change is scaled back to the 3 s window, so a healthy VIO stays quiet and
	// a runaway still trips.
	{
		auto lossy = [](RunawayFix &f, float accel) {
			for (int k = 0; k < 3000; k++) {
				const uint64_t t_us = 1'000'000 + 10'000ULL * k;
				const float t = 0.01f * k;
				f.inertial(t_us, 0.1f, 0.f, 0.f);  // a 0.1 m/s^2 bias, the vehicle still
				const bool lost = std::fmod(t, 2.f) > 1.5f;

				if (k % 2 == 0 && !lost) {
					f.fix(t_us, t > 10.f ? accel * (t - 10.f) : 0.f, 0.f, 0.f, 0);
				}
			}
		};
		RunawayFix healthy_dust;
		healthy_dust.setLimit(1.f);
		lossy(healthy_dust, 0.f);
		CHECK(!healthy_dust.runaway());
		RunawayFix runaway_dust;
		runaway_dust.setLimit(1.f);
		lossy(runaway_dust, 0.8f);
		CHECK(runaway_dust.runaway());
	}

	// A short glitch (a 2 m/s velocity spike for 0.1 s) is not a runaway: it needs two
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
		// Fixes 2.5 s apart (the VIO lost): every one starts over, the window never fills.
		RunawayFix g;
		g.setLimit(1.f);
		for (int k = 0; k < 40; k++) {
			for (int j = 0; j < 250; j++) {
				g.inertial(1'000'000 + 2'500'000ULL * k + 10'000ULL * j, 0.f, 0.f, 0.f);
			}
			g.fix(1'000'000 + 2'500'000ULL * k, 10.f * k, 0.f, 0.f, 0);
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
	testStillFix();

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
