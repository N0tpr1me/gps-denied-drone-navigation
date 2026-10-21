/**
 * Structure Hold watchdog (stub, v0).
 *
 * The one part of the module a customer's safety engineer reads line by line, so it is
 * plain C++ with fixed thresholds: no learning, no heap, no PX4 headers, one logged
 * reason per trip. It decides two things:
 *
 *   - mayEnter(): whether the mode may be entered this tick, and
 *   - check():    whether the policy may keep flying this tick (Trip::None) or must stop,
 *                 and why.
 *
 * A trip latches until clearLatch() (the module clears it on disarm), so a mode that
 * handed back cannot be re-engaged in the same flight without the pilot noticing.
 *
 * This header is deliberately free of PX4 dependencies so it compiles and runs in a plain
 * g++ unit test (tests/test_watchdog.cpp) without a PX4 tree.
 */

#pragma once

#include <cstdint>

namespace structure_hold
{

enum class Trip : uint8_t {
	None = 0,
	TestFault = 1,        // SH_TEST_FAULT set by a test or the pilot
	EstimateInvalid = 2,  // local position / velocity flagged invalid
	EstimateStale = 3,    // no local position update within the timeout
	Tilt = 4,             // vehicle tilt beyond the limit
	PositionError = 5,    // hold error beyond the limit
};

struct WatchdogLimits {
	uint64_t estimate_timeout_us{500000};  // 0.5 s
	float max_tilt_rad{0.785f};            // 45 deg
	float max_position_error_m{2.0f};
};

struct WatchdogInputs {
	uint64_t now_us{0};
	uint64_t estimate_timestamp_us{0};
	bool xy_valid{false};
	bool z_valid{false};
	bool v_xy_valid{false};
	float tilt_rad{0.f};
	float position_error_m{0.f};
	bool test_fault{false};
};

class Watchdog
{
public:
	Watchdog() = default;
	explicit Watchdog(const WatchdogLimits &limits) : _limits(limits) {}

	void setLimits(const WatchdogLimits &limits) { _limits = limits; }
	const WatchdogLimits &limits() const { return _limits; }

	/** First reason the policy must stop this tick, in fixed priority order. */
	Trip check(const WatchdogInputs &in) const
	{
		if (in.test_fault) {
			return Trip::TestFault;
		}

		if (!in.xy_valid || !in.z_valid || !in.v_xy_valid) {
			return Trip::EstimateInvalid;
		}

		if (in.now_us < in.estimate_timestamp_us
		    || in.now_us - in.estimate_timestamp_us > _limits.estimate_timeout_us) {
			return Trip::EstimateStale;
		}

		if (!(in.tilt_rad <= _limits.max_tilt_rad)) {  // also catches NaN
			return Trip::Tilt;
		}

		if (!(in.position_error_m <= _limits.max_position_error_m)) {
			return Trip::PositionError;
		}

		return Trip::None;
	}

	/** Entry is allowed when nothing is latched and the same checks pass with zero hold error. */
	bool mayEnter(WatchdogInputs in) const
	{
		if (_latched != Trip::None) {
			return false;
		}

		in.position_error_m = 0.f;
		return check(in) == Trip::None;
	}

	void latch(Trip trip) { if (trip != Trip::None) { _latched = trip; } }
	void clearLatch() { _latched = Trip::None; }
	Trip latched() const { return _latched; }

	static const char *name(Trip trip)
	{
		switch (trip) {
		case Trip::None: return "none";

		case Trip::TestFault: return "test_fault";

		case Trip::EstimateInvalid: return "estimate_invalid";

		case Trip::EstimateStale: return "estimate_stale";

		case Trip::Tilt: return "tilt";

		case Trip::PositionError: return "position_error";
		}

		return "unknown";
	}

private:
	WatchdogLimits _limits{};
	Trip _latched{Trip::None};
};

} // namespace structure_hold
