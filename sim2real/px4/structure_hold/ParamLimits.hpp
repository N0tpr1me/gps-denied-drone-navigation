/**
 * Structure Hold parameter limits, applied by the module itself.
 *
 * The @min / @max in structure_hold_params.c are metadata for ground stations: PX4's
 * param_set does not enforce them, so a script, a typo in a params file or a MAVLink
 * PARAM_SET can store any value, NaN included. Several of ours are unsafe outside their
 * range: a negative SH_YAW_P or SH_FB_K turns a stabilising loop into a destabilising
 * one, a negative or NaN SH_WD_EST_TO is undefined behaviour when it becomes a timeout in
 * microseconds, a NaN SH_FIX_GAP hides every dropout from the policy, and an SH_CTRL of 5
 * silently flew the placeholder cascade.
 *
 * So every parameter the module reads goes through sanitize() once per parameter update:
 * a float outside its range is clamped to the nearest limit (as ground stations do), a
 * non-finite float takes the default, and an integer outside its range takes the default
 * (clamping an enum would pick a neighbour that may mean something else, such as SH_FB_EN
 * 2, the forced test hook). The module warns when it changes a value and flies the result.
 *
 * The numbers here must match structure_hold_params.c; tests/test_param_limits.py fails
 * when they drift apart. Plain C++, no PX4 headers (tests/test_params.cpp).
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace structure_hold
{

struct FloatRange {
	float min;
	float max;
	float def;
};

struct IntRange {
	int32_t min;
	int32_t max;
	int32_t def;
};

namespace param_limits
{
// Watchdog
static constexpr FloatRange SH_WD_EST_TO{0.05f, 10.0f, 5.0f};
static constexpr FloatRange SH_WD_POS_ERR{0.2f, 10.0f, 2.0f};
static constexpr FloatRange SH_WD_TILT{10.0f, 80.0f, 45.0f};
static constexpr FloatRange SH_WD_FRZ_T{0.0f, 10.0f, 1.0f};
static constexpr FloatRange SH_WD_VIO_DV{0.0f, 10.0f, 1.0f};
static constexpr FloatRange SH_RST_HOLD{0.0f, 60.0f, 0.0f};
static constexpr IntRange SH_BENCH{0, 1, 0};
// SH_TEST_FAULT is left alone: any nonzero value trips, as documented. SH_EN is read by
// the start script, not the module.

// Placeholder cascade
static constexpr FloatRange SH_P_POS{0.1f, 3.0f, 0.95f};
static constexpr FloatRange SH_P_VEL{0.1f, 6.0f, 1.8f};
static constexpr FloatRange SH_P_ATT{1.0f, 12.0f, 6.5f};
static constexpr FloatRange SH_V_MAX{0.2f, 5.0f, 1.0f};
static constexpr IntRange SH_VEL_SRC{0, 1, 0};

// Controller, fallback, fix tracking, yaw and rate limits
static constexpr IntRange SH_CTRL{0, 1, 1};
static constexpr IntRange SH_FB_EN{-1, 2, -1};
static constexpr FloatRange SH_FB_T{0.05f, 5.0f, 0.5f};
static constexpr IntRange SH_FIX_SRC{0, 2, 0};
static constexpr FloatRange SH_FIX_GAP{0.01f, 1.0f, 0.1f};
static constexpr IntRange SH_YAW_GUARD{-1, 1, -1};
static constexpr FloatRange SH_YAW_P{0.5f, 10.0f, 3.0f};
static constexpr FloatRange SH_RATE_MAX{0.0f, 6.0f, 0.0f};
static constexpr FloatRange SH_FB_K{0.2f, 5.0f, 1.5f};
static constexpr FloatRange SH_FB_TILT{1.0f, 12.0f, 4.0f};

// Wall guard and the guarded modes
static constexpr FloatRange SH_WALL_MIN{0.0f, 5.0f, 0.7f};
static constexpr IntRange SH_WALL_REQ{0, 1, 0};
static constexpr IntRange SH_LAND_EN{0, 1, 1};
static constexpr IntRange SH_HB_GUARD{0, 1, 1};
static constexpr IntRange SH_HB_RANGE{0, 1, 1};
// Keep-out from PX4's collision-prevention data; -1 follows CP_DIST
static constexpr FloatRange SH_KO_DIST{-1.0f, 5.0f, -1.0f};
// Retreat cap on both guards per hold; 0 = no cap
static constexpr FloatRange SH_SHIFT_MAX{0.0f, 10.0f, 3.0f};

// Ground station (PilotStatus.hpp)
static constexpr FloatRange SH_GCS_PER{0.0f, 60.0f, 10.0f};

// Not ours: PX4's hover thrust (MPC_THR_HOVER, or hover_thrust_estimate). Outside this the
// thrust scaling of every controller here is wrong by more than any airframe needs.
static constexpr FloatRange HOVER_THRUST{0.1f, 0.9f, 0.5f};
} // namespace param_limits

/** Clamp into [min, max]; a non-finite value takes the default. */
inline float sanitize(float v, const FloatRange &r)
{
	if (!std::isfinite(v)) {
		return r.def;
	}

	return v < r.min ? r.min : (v > r.max ? r.max : v);
}

/** An integer (enum or boolean) outside [min, max] takes the default. */
inline int32_t sanitize(int32_t v, const IntRange &r)
{
	return (v < r.min || v > r.max) ? r.def : v;
}

/** True when sanitize() leaves the value as it is (bit for bit, without -Wfloat-equal). */
inline bool inRange(float v, const FloatRange &r) { return std::isfinite(v) && v >= r.min && v <= r.max; }
inline bool inRange(int32_t v, const IntRange &r) { return v >= r.min && v <= r.max; }

/** Hover thrust for the thrust scaling: clamped, and a non-finite estimate falls back to
 * the (clamped) parameter. */
inline float sanitizeHoverThrust(float estimate, float fallback)
{
	if (std::isfinite(estimate)) {
		return sanitize(estimate, param_limits::HOVER_THRUST);
	}

	return sanitize(fallback, param_limits::HOVER_THRUST);
}

} // namespace structure_hold
