// Unit tests for the handback sequence: pins the decided rows (docs/decisions.md,
// 2026-10-21 and 2026-10-28) step by step. No PX4 tree needed:
//   g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal
//     -I../structure_hold test_handback.cpp -o test_handback && ./test_handback

#include "Handback.hpp"

#include <cstdio>
#include <cstring>

using namespace structure_hold;
using T = HandbackTarget;

static int failures = 0;

#define CHECK(cond)                                                                 \
	do {                                                                        \
		if (!(cond)) {                                                      \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			++failures;                                                 \
		}                                                                   \
	} while (0)

// The targets asked for over ``n`` retries, as the module walks them (next() each retry).
static bool sequenceIs(const HandbackPlan &p, const T *expected, int n)
{
	int step = 0;

	for (int i = 0; i < n; i++) {
		if (p.target(step) != expected[i]) {
			std::printf("  step %d: got %s, expected %s\n", i, handbackTargetName(p.target(step)),
				    handbackTargetName(expected[i]));
			return false;
		}

		step = p.next(step);
	}

	return true;
}

int main()
{
	// Trip in Structure Hold next to a measured wall: Structure Hover, then Position mode,
	// Hold, Land, and Land again for as long as commander stays in the mode.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Hold, true, true, true);
		CHECK(p.guarded && !p.landing);
		const T seq[] = {T::StructureHover, T::Position, T::Hold, T::Land, T::Land, T::Land};
		CHECK(sequenceIs(p, seq, 6));
		CHECK(p.lastStep() == 3);
	}

	// Trip in Structure Hold with no wall measured: Position mode, Hold, Land.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Hold, true, true, false);
		CHECK(!p.guarded && !p.landing);
		const T seq[] = {T::Position, T::Hold, T::Land, T::Land};
		CHECK(sequenceIs(p, seq, 4));
		CHECK(p.lastStep() == 2);
	}

	// Horizontal estimate invalid: Altitude mode instead of Position mode, and never
	// Structure Hover (it holds a position), even with a wall measured.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Hold, true, false, true);
		CHECK(!p.guarded);
		const T seq[] = {T::Altitude, T::Hold, T::Land};
		CHECK(sequenceIs(p, seq, 3));
	}

	// SH_HB_RANGE 1 (the default since 2026-11-02): with the horizontal estimate gone next to a
	// measured wall, Structure Hover first (it holds the wall distance on the range alone),
	// then Altitude mode, Hold, Land. With an estimate, or without a wall, nothing changes.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Hold, true, false, true, true);
		CHECK(p.guarded && !p.xy_valid);
		const T seq[] = {T::StructureHover, T::Altitude, T::Hold, T::Land};
		CHECK(sequenceIs(p, seq, 4));
		const HandbackPlan no_wall = planHandback(HandbackFrom::Hold, true, false, false, true);
		CHECK(!no_wall.guarded && no_wall.target(0) == T::Altitude);
		const HandbackPlan with_xy = planHandback(HandbackFrom::Hold, true, true, true, true);
		const HandbackPlan decided = planHandback(HandbackFrom::Hold, true, true, true);
		CHECK(with_xy.guarded == decided.guarded && with_xy.target(0) == decided.target(0)
		      && with_xy.target(1) == decided.target(1));
		const HandbackPlan refusal = planHandback(HandbackFrom::Hold, false, false, true, true);
		CHECK(!refusal.guarded);
		const HandbackPlan hover = planHandback(HandbackFrom::Hover, true, false, true, true);
		CHECK(!hover.guarded && hover.target(0) == T::Altitude);
	}

	// Trip in Structure Land: stock Land first (the landing was already asked for), then
	// Position mode, Hold, Land; with or without a wall.
	for (int wall = 0; wall < 2; wall++) {
		const HandbackPlan p = planHandback(HandbackFrom::Land, true, true, wall != 0);
		CHECK(p.landing && !p.guarded);
		const T seq[] = {T::StockLand, T::Position, T::Hold, T::Land};
		CHECK(sequenceIs(p, seq, 4));
	}

	// Trip in Structure Land with the xy estimate gone: stock Land, then Altitude mode.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Land, true, false, false);
		const T seq[] = {T::StockLand, T::Altitude, T::Hold, T::Land};
		CHECK(sequenceIs(p, seq, 4));
	}

	// Trip in Structure Hover: straight to Position mode (not back into Hover).
	for (int wall = 0; wall < 2; wall++) {
		const HandbackPlan p = planHandback(HandbackFrom::Hover, true, true, wall != 0);
		CHECK(!p.guarded && !p.landing);
		const T seq[] = {T::Position, T::Hold, T::Land};
		CHECK(sequenceIs(p, seq, 3));
	}

	// Entry refusal (reset hold-off, on the ground, broken estimate): nothing was flying,
	// so no guarded first target, even next to a wall.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Hold, false, true, true);
		CHECK(!p.guarded && !p.landing);
		const T seq[] = {T::Position, T::Hold, T::Land};
		CHECK(sequenceIs(p, seq, 3));
	}

	// next() never runs past the last step.
	{
		const HandbackPlan p = planHandback(HandbackFrom::Hold, true, true, true);
		CHECK(p.next(3) == 3);
		CHECK(p.next(250) == 3);
		CHECK(p.target(250) == T::Land);
	}

	// Names end up in the log; StockLand and Land are the same PX4 mode.
	CHECK(std::strcmp(handbackTargetName(T::StructureHover), "Structure Hover") == 0);
	CHECK(std::strcmp(handbackTargetName(T::StockLand), handbackTargetName(T::Land)) == 0);

	if (failures == 0) {
		std::printf("handback tests: all passed\n");
		return 0;
	}

	std::printf("handback tests: %d failure(s)\n", failures);
	return 1;
}
