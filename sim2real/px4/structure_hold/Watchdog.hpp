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
 * Entry is also refused, without latching, on the ground (Trip::Landed: a hold point on
 * the ground is no hold, and the policy was never trained on a take-off; the module checks
 * the land detector), and for reset_holdoff_us after the estimator reset its horizontal
 * position (vehicle_local_position.xy_reset_counter changed; noteReset()). A hold point
 * captured while the estimate is still settling after a reset is not where the vehicle
 * is: policy v11 re-engaged 0.1 to 0.4 s after a 1.1 m vision reset, captured a point
 * on the wall face and struck it (docs/baseline.md, v11 section). Off by default (reset_holdoff_us 0, PX4 param SH_RST_HOLD): stock PX4 captures the same
 * settling estimate, so refusing the engage does not prevent that strike, it only chooses
 * who flies into it. Once the mode is flying, a reset is the policy's to handle (it is
 * trained on them), so check() ignores it.
 *
 * A frozen fix (PX4 param SH_WD_FRZ_T, 1 s by default since 2026-11-02; this struct's
 * max_fix_frozen_s stays 0 = off so the plain watchdog tests start without it): a
 * stuck camera pipeline or VIO process can keep republishing its last pose on time, so the
 * stale trip never sees it and EKF2 keeps it valid (docs/field-faults.md, the freeze fault).
 * A real VIO on a flying vehicle never repeats its position to a tenth of a millimetre for a
 * second: noise and the vehicle's own motion move it. FrozenFix measures how long the fix
 * has stayed inside that tolerance, timed by the fixes' own sample times, so a fix that
 * stops arriving is the stale trip's, not this one's.
 *
 * A runaway fix (PX4 param SH_WD_VIO_DV, docs/field-faults.md "Heavy dust"): a VIO that locks
 * onto something moving (rotor-blown dust: Basalt in heavy dust tracked the drifting grains
 * and ran away by kilometres without reporting a tracking loss, docs/vio-render.md) reports a
 * position and a velocity that agree with each other and keep arriving on time, so neither
 * the stale nor the frozen check sees it. What it cannot fake is the IMU: RunawayFix compares
 * how much the VIO's velocity changed over the last WINDOW_S with how much the IMU says the
 * vehicle's velocity changed over the same time. The vehicle's own motion cancels out (it is
 * in both), so what is left is the change in the VIO's velocity error. A relocalisation jump
 * moves the position, not the velocity, so it does not count. A VIO that ran away is not
 * trusted again until it reports a restart (its reset counter) or the vehicle disarms: one
 * that stopped accelerating can still be metres per second off, and a fresh comparison
 * would not see that.
 *
 * This header is deliberately free of PX4 dependencies so it compiles and runs in a plain
 * g++ unit test (tests/test_watchdog.cpp) without a PX4 tree.
 */

#pragma once

#include <cmath>
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
	EstimateReset = 6,    // entry refused: estimator reset its position too recently
	Landed = 7,           // entry refused: the land detector says the vehicle is on the ground
	RangeLost = 8,        // Structure Hover on the range alone (SH_HB_RANGE) lost its range
	EstimateFrozen = 9,   // the position fix arrives on time but has not moved (SH_WD_FRZ_T)
	EstimateRunaway = 10, // the fix's velocity ran away from the IMU's (SH_WD_VIO_DV)
};

struct WatchdogLimits {
	uint64_t estimate_timeout_us{500000};  // 0.5 s
	float max_tilt_rad{0.785f};            // 45 deg
	float max_position_error_m{2.0f};
	uint64_t reset_holdoff_us{0};          // entry hold-off after an estimator reset; 0 = off
	float max_fix_frozen_s{0.f};           // frozen-fix trip after this long; 0 = off
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
	float fix_frozen_s{0.f};  // how long the fix has not moved (FrozenFix)
	bool fix_runaway{false};  // the fix's velocity ran away from the IMU's (RunawayFix)
};

/** How long the position fix has stayed put: see the header comment. */
class FrozenFix
{
public:
	static constexpr float TOLERANCE_M = 1e-4f;

	/** Every fix sample: its own sample time and position. */
	void update(uint64_t sample_us, float x, float y, float z)
	{
		const bool moved = !_have || !(std::fabs(x - _ref[0]) <= TOLERANCE_M)
				   || !(std::fabs(y - _ref[1]) <= TOLERANCE_M) || !(std::fabs(z - _ref[2]) <= TOLERANCE_M)
				   || sample_us < _last_us;

		if (moved) {
			_ref[0] = x;
			_ref[1] = y;
			_ref[2] = z;
			_since_us = sample_us;
			_have = true;
		}

		_last_us = sample_us;
	}

	/** Seconds between the first and the latest fix at the same position. */
	float frozenSeconds() const
	{
		return _have && _last_us >= _since_us ? static_cast<float>(_last_us - _since_us) * 1e-6f : 0.f;
	}

	void reset() { _have = false; _since_us = _last_us = 0; }

private:
	float _ref[3] {0.f, 0.f, 0.f};
	uint64_t _since_us{0};
	uint64_t _last_us{0};
	bool _have{false};
};

/** Whether the fix's velocity has run away from the IMU's: see the header comment. */
class RunawayFix
{
public:
	static constexpr uint64_t SLOT_US = 100000;  // the difference is averaged over 0.1 s slots
	static constexpr int SLOTS = 20;             // compared over WINDOW_S = 2 s
	static constexpr uint64_t GAP_US = 500000;   // a longer gap in either stream starts over
	static constexpr int TRIP_SLOTS = 3;         // over the limit this many slots in a row

	/** The limit in m/s over the window; 0 = off (never runaway). */
	void setLimit(float limit_mps) { _limit = limit_mps; }
	float limit() const { return _limit; }

	/** Every local position sample: its sample time and the mean earth-frame acceleration,
	 * gravity removed, since the previous one (vehicle_local_position ax, ay, az: the IMU
	 * alone, EKF2's bias estimate subtracted). */
	void inertial(uint64_t t_us, float ax, float ay, float az)
	{
		if (!(std::isfinite(ax) && std::isfinite(ay) && std::isfinite(az))) {
			return;
		}

		if (_imu_t > 0 && t_us > _imu_t && t_us - _imu_t < GAP_US) {
			const float dt = static_cast<float>(t_us - _imu_t) * 1e-6f;
			_v_imu[0] += ax * dt;
			_v_imu[1] += ay * dt;
			_v_imu[2] += az * dt;

		} else if (_imu_t > 0) {
			restart();  // a hole in the IMU integral: the window would span it
		}

		_imu_t = t_us;
	}

	/** Every fix sample: its sample time, its velocity in NED (NaN: none) and the VIO's
	 * reset counter. */
	void fix(uint64_t t_us, float vn, float ve, float vd, uint8_t reset_counter)
	{
		if (_counter_seen && reset_counter != _counter) {
			restart();
			_runaway = false;  // the VIO says it restarted
		}

		_counter = reset_counter;
		_counter_seen = true;

		if (!(std::isfinite(vn) && std::isfinite(ve) && std::isfinite(vd)) || _imu_t == 0) {
			return;
		}

		if (_fix_t > 0 && (t_us <= _fix_t || t_us - _fix_t >= GAP_US)) {
			restart();  // a dropout or a clock step: compare only across unbroken data
		}

		_fix_t = t_us;

		if (_slot_n == 0) {
			_slot_t = t_us;
		}

		_slot[0] += vn - _v_imu[0];
		_slot[1] += ve - _v_imu[1];
		_slot[2] += vd - _v_imu[2];
		++_slot_n;

		if (t_us - _slot_t >= SLOT_US) {
			closeSlot();
		}
	}

	bool runaway() const { return _runaway; }

	/** The latest change in the VIO's velocity error over the window, m/s (0 until a
	 * window is full). */
	float disagreement() const { return _disagreement; }

	void reset()
	{
		restart();
		_runaway = false;
		_counter_seen = false;
		_imu_t = 0;
		_v_imu[0] = _v_imu[1] = _v_imu[2] = 0.f;
	}

private:
	void restart()
	{
		_count = 0;
		_over = 0;
		_slot_n = 0;
		_slot[0] = _slot[1] = _slot[2] = 0.f;
		_fix_t = 0;
		_disagreement = 0.f;
	}

	void closeSlot()
	{
		float *e = _ring[_head];

		for (int i = 0; i < 3; i++) {
			e[i] = _slot[i] / static_cast<float>(_slot_n);
			_slot[i] = 0.f;
		}

		_slot_n = 0;
		const int oldest = (_head + 1) % (SLOTS + 1);
		_head = oldest;

		if (_count < SLOTS + 1) {
			++_count;
		}

		if (_count <= SLOTS) {
			return;  // the window is not full yet
		}

		const float *o = _ring[oldest];
		_disagreement = std::sqrt((e[0] - o[0]) * (e[0] - o[0]) + (e[1] - o[1]) * (e[1] - o[1])
					  + (e[2] - o[2]) * (e[2] - o[2]));
		_over = (_limit > 0.f && _disagreement > _limit) ? _over + 1 : 0;

		if (_over >= TRIP_SLOTS) {
			_runaway = true;
		}
	}

	float _limit{0.f};
	float _v_imu[3] {0.f, 0.f, 0.f};  // the IMU's velocity integral, arbitrary start
	uint64_t _imu_t{0};
	uint64_t _fix_t{0};
	float _slot[3] {0.f, 0.f, 0.f};   // running sum of (fix velocity - IMU integral)
	int _slot_n{0};
	uint64_t _slot_t{0};
	float _ring[SLOTS + 1][3] {};     // slot means, the newest at _head - 1
	int _head{0};
	int _count{0};
	int _over{0};
	float _disagreement{0.f};
	bool _runaway{false};
	uint8_t _counter{0};
	bool _counter_seen{false};
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

		if (_limits.max_fix_frozen_s > 0.f && !(in.fix_frozen_s < _limits.max_fix_frozen_s)) {
			return Trip::EstimateFrozen;
		}

		if (in.fix_runaway) {
			return Trip::EstimateRunaway;
		}

		if (!(in.tilt_rad <= _limits.max_tilt_rad)) {  // also catches NaN
			return Trip::Tilt;
		}

		if (!(in.position_error_m <= _limits.max_position_error_m)) {
			return Trip::PositionError;
		}

		return Trip::None;
	}

	/** Feed every local-position sample: a changed xy reset counter starts the hold-off. */
	void noteReset(uint8_t xy_reset_counter, uint64_t now_us)
	{
		if (_reset_counter_seen && xy_reset_counter != _reset_counter) {
			_last_reset_us = now_us;
			_reset_seen = true;
		}

		_reset_counter = xy_reset_counter;
		_reset_counter_seen = true;
	}

	/** True while entry is refused because of a recent estimator reset. */
	bool inResetHoldoff(uint64_t now_us) const
	{
		return _reset_seen && _limits.reset_holdoff_us > 0 && now_us >= _last_reset_us
		       && now_us - _last_reset_us < _limits.reset_holdoff_us;
	}

	/** Seconds since the last estimator reset, or a negative value when none was seen. */
	float resetAge(uint64_t now_us) const
	{
		if (!_reset_seen || now_us < _last_reset_us) {
			return -1.f;
		}

		return static_cast<float>(now_us - _last_reset_us) * 1e-6f;
	}

	/** Entry is allowed when nothing is latched, no reset hold-off is running, and the same
	 * checks pass with zero hold error. */
	bool mayEnter(WatchdogInputs in) const
	{
		if (_latched != Trip::None || inResetHoldoff(in.now_us)) {
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

		case Trip::EstimateReset: return "estimate_reset";

		case Trip::Landed: return "landed";

		case Trip::RangeLost: return "range_lost";

		case Trip::EstimateFrozen: return "estimate_frozen";

		case Trip::EstimateRunaway: return "estimate_runaway";
		}

		return "unknown";
	}

	/** The same reason in plain words for the pilot, with the parameter that sets the limit.
	 * The ground-station message puts it after the code ("watchdog trip: tilt (tilted past
	 * SH_WD_TILT)"), so scripts that match the code keep working. PX4 caps a mavlink_log text
	 * at 126 characters; test_watchdog.cpp keeps every message inside that. */
	static const char *reason(Trip trip)
	{
		switch (trip) {
		case Trip::None: return "no trip";

		case Trip::TestFault: return "test fault set by SH_TEST_FAULT";

		case Trip::EstimateInvalid: return "PX4 marked the position estimate invalid";

		case Trip::EstimateStale: return "no position estimate for SH_WD_EST_TO";

		case Trip::Tilt: return "tilted past SH_WD_TILT";

		case Trip::PositionError: return "drifted past SH_WD_POS_ERR from the hold point";

		case Trip::EstimateReset: return "position estimate just reset, see SH_RST_HOLD";

		case Trip::Landed: return "on the ground";

		case Trip::RangeLost: return "rangefinder lost while holding on it alone";

		case Trip::EstimateFrozen: return "position fix stopped moving for SH_WD_FRZ_T";

		case Trip::EstimateRunaway: return "VIO velocity ran away from the IMU's, past SH_WD_VIO_DV";
		}

		return "unknown";
	}

private:
	WatchdogLimits _limits{};
	Trip _latched{Trip::None};
	uint8_t _reset_counter{0};
	bool _reset_counter_seen{false};
	bool _reset_seen{false};
	uint64_t _last_reset_us{0};
};

} // namespace structure_hold
