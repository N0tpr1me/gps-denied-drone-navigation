/**
 * Structure Hold handback sequence: which stock or guarded mode the module asks commander
 * for after a watchdog trip or an entry refusal, in order.
 *
 * Decided rows (docs/decisions.md, 2026-10-21 and 2026-10-28):
 *
 *   - a trip in Structure Hold next to a measured wall (Structure Hover registered, wall
 *     guard on, a usable forward range under 5 m, xy estimate valid) asks for Structure
 *     Hover first;
 *   - a trip in Structure Land asks for stock Land first, since the landing was already
 *     asked for;
 *   - then, and for every other trip or refusal: Position mode (Altitude mode when the
 *     horizontal estimate is invalid), Hold, Land.
 *
 * Proposed, off by default (SH_HB_RANGE 1, docs/decisions.md 2026-11-01): a trip in
 * Structure Hold next to a measured wall with the horizontal estimate gone also asks for
 * Structure Hover first, which then holds the wall distance on the rangefinder alone
 * (RangeHover.hpp) instead of Altitude mode's level drift; Altitude mode, Hold and Land
 * follow as before if commander refuses it.
 *
 * Commander may refuse a target (no manual control source for Position mode, no position
 * for Hold), so the module moves to the next one when the current one has not taken
 * after a short wait. Land is the last target and is asked for again until commander
 * takes it or the pilot picks a mode: before 2026-11 the module asked once and then kept
 * the vehicle level at hover thrust for as long as commander stayed in the mode.
 *
 * Plain C++, no PX4 headers, so tests/test_handback.cpp pins the sequence in a bare g++
 * build; the module maps the targets to commander's custom modes.
 */

#pragma once

#include <cstdint>

namespace structure_hold
{

enum class HandbackTarget : uint8_t {
	StructureHover = 0,  // the module's guarded hover (ext3)
	StockLand = 1,       // first target after a trip in Structure Land
	Position = 2,
	Altitude = 3,        // instead of Position when the xy estimate is invalid
	Hold = 4,            // AUTO_LOITER
	Land = 5,
};

/** Where the trip happened. */
enum class HandbackFrom : uint8_t {
	Hold = 0,   // Structure Hold, or an entry refusal
	Land = 1,   // Structure Land
	Hover = 2,  // Structure Hover
};

struct HandbackPlan {
	bool guarded{false};   // Structure Hover first
	bool landing{false};   // stock Land first
	bool xy_valid{true};   // Position mode, else Altitude mode

	/** Index of the last target (Land). */
	int lastStep() const { return (guarded || landing) ? 3 : 2; }

	/** Target for step 0 .. lastStep(); later steps repeat Land. */
	HandbackTarget target(int step) const
	{
		const int first = (guarded || landing) ? 1 : 0;

		if (step < first) {
			return guarded ? HandbackTarget::StructureHover : HandbackTarget::StockLand;
		}

		switch (step - first) {
		case 0: return xy_valid ? HandbackTarget::Position : HandbackTarget::Altitude;

		case 1: return HandbackTarget::Hold;

		default: return HandbackTarget::Land;
		}
	}

	/** Step to ask for next when the current one has not taken: never past Land. */
	int next(int step) const { return step < lastStep() ? step + 1 : lastStep(); }
};

/**
 * The plan for one trip. ``latched`` is false for an entry refusal (nothing was flying, so
 * no guarded first target); ``wall_measured`` is the module's check that Structure Hover
 * is registered, the guard is on and a usable forward range sees the wall within 5 m;
 * ``range_hover`` is SH_HB_RANGE (Structure Hover may fly on the range alone).
 */
inline HandbackPlan planHandback(HandbackFrom from, bool latched, bool xy_valid, bool wall_measured,
				 bool range_hover = false)
{
	HandbackPlan p{};
	p.xy_valid = xy_valid;
	p.guarded = latched && (xy_valid || range_hover) && from == HandbackFrom::Hold && wall_measured;
	p.landing = latched && from == HandbackFrom::Land;
	return p;
}

inline const char *handbackTargetName(HandbackTarget t)
{
	switch (t) {
	case HandbackTarget::StructureHover: return "Structure Hover";

	case HandbackTarget::StockLand: return "Land";

	case HandbackTarget::Position: return "Position mode";

	case HandbackTarget::Altitude: return "Altitude mode";

	case HandbackTarget::Hold: return "Hold";

	case HandbackTarget::Land: return "Land";
	}

	return "unknown";
}

} // namespace structure_hold
