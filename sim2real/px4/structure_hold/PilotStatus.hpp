/**
 * Structure Hold: what the pilot is told while the module flies (docs/pilot-status.md).
 *
 * The trip messages say what happened after a safety function acted. This header decides
 * what the pilot hears before that: how close the vehicle is to each watchdog limit, a
 * warning when it gets near one, a note when it is back inside them all, and when the next
 * one-line status is due. The module turns these into ground-station text (STATUSTEXT) and
 * a live number array (DEBUG_FLOAT_ARRAY "SH_STATUS", indices in LiveIndex below).
 *
 * Plain C++, no PX4 headers, so tests/test_pilot_status.cpp runs it without a PX4 tree.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include "Watchdog.hpp"

namespace structure_hold
{

/** Index of each number in the live array; docs/pilot-status.md lists the same table and
 * tests/test_pilot_status_doc.py keeps the two in step. */
enum LiveIndex : uint8_t {
	LIVE_STATE = 0,          // structure_hold_status STATE_*
	LIVE_HOLD_ERROR = 1,     // [m] distance from the hold point; -1 where none is held
	LIVE_WALL = 2,           // [m] forward rangefinder, perpendicular to the wall; -1 without one
	LIVE_CLEARANCE = 3,      // [m] hold point to the measured wall; -1 without a range
	LIVE_GUARD_SHIFT = 4,    // [m] how far the wall guard moved the hold point back
	LIVE_FIX_AGE = 5,        // [s] since the last position fix
	LIVE_WD_SECONDS = 6,     // [s] left before the no-fix trip (SH_WD_EST_TO); -1 when not flying
	LIVE_WD_USED = 7,        // [%] of the closest watchdog limit used; -1 when not flying
	LIVE_WD_CLOSEST = 8,     // trip code of that limit (3 stale, 4 tilt, 5 hold error, 9 frozen)
	LIVE_TRIP = 9,           // last trip or refusal code; 0 none, cleared on disarm
	LIVE_TRIP_COUNT = 10,    // trips since boot
	LIVE_TILT = 11,          // [deg]
	LIVE_FALLBACK = 12,      // 1 while the brake-and-hover fallback flies
	LIVE_COUNT = 13
};

/** The watchdog's inputs this tick, and which of its checks apply in the mode flying now
 * (the module's checkWatchdog() decides the same). */
struct MarginInputs {
	float stale_s{0.f};             // as the watchdog counts it: since the older of fix and EKF2 update
	float fix_frozen_s{0.f};
	float tilt_rad{0.f};
	float position_error_m{0.f};
	bool frozen_checked{true};      // off in Structure Hover and Structure Land
	bool position_checked{true};    // off in Structure Hover on the rangefinder alone
};

/** The limit closest to tripping, as a fraction of the limit (1 = trips). */
struct Margin {
	Trip limit{Trip::None};
	float used{0.f};
};

struct PilotEvent {
	enum Kind : uint8_t { None, Warn, Clear };
	Kind kind{None};
	Trip limit{Trip::None};
	float value{0.f};  // SI: s, rad or m
	float max{0.f};    // the limit it is measured against, same unit
};

class PilotStatus
{
public:
	// The four checks a flying vehicle can creep toward, in the watchdog's priority order.
	static constexpr int LIMITS = 4;
	static constexpr float CLEAR_USED = 0.4f;          // below this a warned limit re-arms
	static constexpr uint64_t REWARN_US = 10000000;    // the same limit at most once per 10 s

	/** Fraction of a limit at which the pilot is warned. Tilt runs higher in normal gusty
	 * flight than the rest (about half the limit at 8 m/s), so it warns later. */
	static float warnAt(int i) { return i == 2 ? 0.75f : 0.5f; }

	static Trip tripOf(int i)
	{
		static const Trip trips[LIMITS] = {Trip::EstimateStale, Trip::EstimateFrozen, Trip::Tilt, Trip::PositionError};
		return trips[i];
	}

	void setLimits(const WatchdogLimits &limits) { _limits = limits; }

	/** A mode starts flying: nothing warned yet, the first status line one period from now. */
	void start(uint64_t now_us)
	{
		for (int i = 0; i < LIMITS; i++) {
			_warned[i] = false;
			_last_warn_us[i] = 0;
		}

		_outstanding = false;
		_last_summary_us = now_us;
	}

	/** Each limit's value and maximum this tick, or false when it does not apply. */
	bool limit(int i, const MarginInputs &in, float &value, float &max) const
	{
		switch (i) {
		case 0:
			value = in.stale_s;
			max = static_cast<float>(_limits.estimate_timeout_us) * 1e-6f;
			return true;

		case 1:
			value = in.fix_frozen_s;
			max = _limits.max_fix_frozen_s;
			return in.frozen_checked && max > 0.f;

		case 2:
			value = in.tilt_rad;
			max = _limits.max_tilt_rad;
			return true;

		case 3:
			value = in.position_error_m;
			max = _limits.max_position_error_m;
			return in.position_checked;
		}

		return false;
	}

	/** How much of a limit is used; a NaN value counts as at the limit, as the watchdog
	 * trips on it. */
	static float used(float value, float max)
	{
		if (!(max > 0.f)) {
			return 0.f;
		}

		const float u = value / max;
		return std::isfinite(u) ? (u > 0.f ? u : 0.f) : 1.f;
	}

	Margin closest(const MarginInputs &in) const
	{
		Margin m{};

		for (int i = 0; i < LIMITS; i++) {
			float value = 0.f, max = 0.f;

			if (limit(i, in, value, max)) {
				const float u = used(value, max);

				if (m.limit == Trip::None || u > m.used) {
					m.limit = tripOf(i);
					m.used = u;
				}
			}
		}

		return m;
	}

	/** Seconds left before the no-fix trip. */
	float secondsToStaleTrip(float stale_s) const
	{
		const float left = static_cast<float>(_limits.estimate_timeout_us) * 1e-6f - stale_s;
		return std::isfinite(left) && left > 0.f ? left : 0.f;
	}

	/**
	 * One tick of the warnings: at most one event, a warning before a note. A limit warns
	 * once when it passes warnAt() and again only after it has dropped below CLEAR_USED and
	 * REWARN_US has passed, so a gusty tilt does not flood the link. Clear comes once every
	 * limit is back below CLEAR_USED after any warning.
	 */
	PilotEvent update(uint64_t now_us, const MarginInputs &in)
	{
		PilotEvent ev{};
		bool all_clear = true;

		for (int i = 0; i < LIMITS; i++) {
			float value = 0.f, max = 0.f;
			const bool applies = limit(i, in, value, max);
			const float u = applies ? used(value, max) : 0.f;

			if (u < CLEAR_USED) {
				_warned[i] = false;
				continue;
			}

			all_clear = false;
			const bool quiet = _last_warn_us[i] != 0 && now_us - _last_warn_us[i] < REWARN_US;

			if (ev.kind == PilotEvent::None && !_warned[i] && !quiet && !(u < warnAt(i))) {
				_warned[i] = true;
				_last_warn_us[i] = now_us;
				_outstanding = true;
				ev.kind = PilotEvent::Warn;
				ev.limit = tripOf(i);
				ev.value = value;
				ev.max = max;
			}
		}

		if (ev.kind == PilotEvent::None && all_clear && _outstanding) {
			_outstanding = false;
			ev.kind = PilotEvent::Clear;
		}

		return ev;
	}

	/** True once per period while a mode flies; a period of 0 or less turns it off. */
	bool summaryDue(uint64_t now_us, float period_s)
	{
		if (!(period_s > 0.f)) {
			return false;
		}

		if (now_us >= _last_summary_us && now_us - _last_summary_us >= static_cast<uint64_t>(period_s * 1e6f)) {
			_last_summary_us = now_us;
			return true;
		}

		return false;
	}

private:
	WatchdogLimits _limits{};
	bool _warned[LIMITS] {false, false, false, false};
	uint64_t _last_warn_us[LIMITS] {0, 0, 0, 0};
	bool _outstanding{false};
	uint64_t _last_summary_us{0};
};

} // namespace structure_hold
