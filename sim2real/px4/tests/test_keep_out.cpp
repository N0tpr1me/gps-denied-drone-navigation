// Unit tests for the Structure Hold keep-out (PX4 collision-prevention data). No PX4 tree:
//   g++ -std=c++14 -Wall -Wextra -Werror -I../structure_hold test_keep_out.cpp && ./a.out
// (CI adds PX4's stricter warnings: -Wshadow -Wdouble-promotion -Wfloat-equal.)

#include "KeepOut.hpp"

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

static bool near(float a, float b, float tol = 1e-3f) { return std::fabs(a - b) <= tol; }

static const uint64_t NOW = 10000000;
static const float PI = 3.14159265f;
static const float DEG = PI / 180.f;

static KeepOut keepOut(float keep_out, float wall_min = 0.7f)
{
	KeepOutLimits l{};
	l.keep_out_m = keep_out;
	l.wall_min_m = wall_min;
	KeepOut k{};
	k.setLimits(l);
	return k;
}

// An empty 5 deg scan: every bin "nothing there" (max + 1).
static ObstacleScan emptyScan(uint8_t frame)
{
	ObstacleScan s{};
	s.timestamp_us = NOW - 50000;
	s.frame = frame;
	s.increment_deg = 5.f;
	s.angle_offset_deg = 0.f;
	s.min_distance_cm = 20;
	s.max_distance_cm = 1000;

	for (int i = 0; i < ObstacleScan::BINS; i++) { s.distances_cm[i] = 1001; }

	return s;
}

// A flat wall ``d`` m ahead along bearing ``b`` as a 5 deg lidar ring sees it.
static void addWallToScan(ObstacleScan &s, float bearing_deg, float d)
{
	for (int j = 0; j < ObstacleScan::BINS; j++) {
		const float off = (s.angle_offset_deg + 5.f * j - bearing_deg) * DEG;
		const float c = cosf(off);

		if (c > 0.15f) {
			const float r = d / c;

			if (r < 9.9f) { s.distances_cm[j] = static_cast<uint16_t>(r * 100.f + 0.5f); }
		}
	}
}

int main()
{
	const float pos[3] = {0.f, 0.f, -2.5f};
	const float n_north[2] = {1.f, 0.f};  // wall to the north

	// No points, nothing moves.
	{
		KeepOut k = keepOut(1.0f);
		k.begin();
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(o.points == 0);
		CHECK(near(o.shift_m, 0.f));
		CHECK(near(o.nearest_m, -1.f));
		CHECK(near(hold[0], 0.f) && near(hold[1], 0.f));
	}

	// A pole 0.6 m to the east (bearing 90 deg), keep-out 1 m: the hold point moves 0.4 m west.
	{
		KeepOut k = keepOut(1.0f);
		k.begin();
		k.addPoint(90.f * DEG, 0.6f);
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(near(hold[0], 0.f));
		CHECK(near(hold[1], -0.4f));
		CHECK(near(o.shift_m, 0.4f));
		CHECK(near(o.nearest_m, 1.0f));
		CHECK(!o.no_room);
		CHECK(near(k.totalShift(), 0.4f));
		CHECK(near(hold[2], -2.5f));  // height untouched
	}

	// The same pole beyond the keep-out: nothing moves, the nearest distance is reported.
	{
		KeepOut k = keepOut(1.0f);
		k.begin();
		k.addPoint(90.f * DEG, 1.5f);
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(near(o.shift_m, 0.f));
		CHECK(near(o.nearest_m, 1.5f));
	}

	// Measured from the hold point, not the vehicle: the vehicle is 0.5 m west of a hold
	// point that sits 0.3 m from the pole.
	{
		KeepOut k = keepOut(1.0f);
		const float p[3] = {0.f, -0.5f, -2.5f};
		k.begin();
		k.addPoint(90.f * DEG, 0.8f);  // pole at E 0.3
		float hold[3] = {0.f, 0.f, -2.5f};
		k.apply(p, n_north, -1.f, hold);
		CHECK(near(hold[1], -0.7f));
	}

	// Ratchet: a pole that goes away does not pull the hold point back.
	{
		KeepOut k = keepOut(1.0f);
		float hold[3] = {0.f, 0.f, -2.5f};
		k.begin();
		k.addPoint(90.f * DEG, 0.6f);
		k.apply(pos, n_north, -1.f, hold);
		k.begin();
		k.addPoint(90.f * DEG, 5.f);
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(near(o.shift_m, 0.f));
		CHECK(near(hold[1], -0.4f));
		k.engage();
		CHECK(near(k.totalShift(), 0.f));
	}

	// The wall ahead is the structure, not an obstacle: a 360 deg lidar sees the wall 1 m
	// ahead in many bins (up to 1.4 m off to the sides); keep-out 1.5 m leaves the hold alone.
	{
		KeepOut k = keepOut(1.5f);
		ObstacleScan s = emptyScan(ObstacleScan::FRAME_GLOBAL);
		addWallToScan(s, 0.f, 1.0f);
		k.begin();
		CHECK(k.addScan(s, NOW, 0.f));
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(o.points > 20);
		CHECK(near(o.shift_m, 0.f));
		CHECK(near(o.nearest_m, -1.f));  // no obstacle points at all
	}

	// Lidar only, no forward rangefinder: the wall 0.4 m from the hold point is pushed back
	// to SH_WALL_MIN 0.7 m along the normal.
	{
		KeepOut k = keepOut(1.5f, 0.7f);
		ObstacleScan s = emptyScan(ObstacleScan::FRAME_GLOBAL);
		addWallToScan(s, 0.f, 0.4f);
		k.begin();
		k.addScan(s, NOW, 0.f);
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(near(hold[0], -0.3f, 0.01f));
		CHECK(near(hold[1], 0.f, 0.01f));
		CHECK(!o.no_room);
	}

	// A pipe standing 0.5 m off the wall face ahead-right is an obstacle: wall 1.5 m ahead,
	// the pipe at N 1.0, E 0.5 from the vehicle (1.12 m). Keep-out 1.3 m pushes the hold away.
	{
		KeepOut k = keepOut(1.3f, 0.7f);
		ObstacleScan s = emptyScan(ObstacleScan::FRAME_GLOBAL);
		addWallToScan(s, 0.f, 1.5f);
		const float b = atan2f(0.5f, 1.0f) / DEG;  // 26.6 deg
		s.distances_cm[static_cast<int>(b / 5.f + 0.5f)] = static_cast<uint16_t>(hypotf(1.0f, 0.5f) * 100.f);
		k.begin();
		k.addScan(s, NOW, 0.f);
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
		CHECK(o.shift_m > 0.05f);
		CHECK(o.nearest_m > 1.25f);
		CHECK(hold[0] < 0.f && hold[1] < 0.f);  // back and to the left, away from it
	}

	// Body-frame scan: the obstacle in bin 18 (90 deg, the vehicle's right) with the vehicle
	// heading east (90 deg) is to the south.
	{
		KeepOut k = keepOut(1.0f, 0.f);
		ObstacleScan s = emptyScan(ObstacleScan::FRAME_BODY_FRD);
		s.distances_cm[18] = 60;
		k.begin();
		CHECK(k.addScan(s, NOW, 90.f * DEG));
		const float n_east[2] = {0.f, 1.f};
		float hold[3] = {0.f, 0.f, -2.5f};
		k.apply(pos, n_east, -1.f, hold);
		CHECK(near(hold[0], 0.4f));  // moved north, away from the obstacle in the south
		CHECK(near(hold[1], 0.f));
	}

	// Angle offset and increment: a 10 deg scan starting at -90 deg; bin 9 is at 0 deg (north).
	// With no wall estimate (rangefinder absent, the point is in the cone so it *is* the wall)
	// it becomes the plane floor: hold pushed south to 0.7 m.
	{
		KeepOut k = keepOut(1.0f, 0.7f);
		ObstacleScan s = emptyScan(ObstacleScan::FRAME_LOCAL_NED);
		s.increment_deg = 10.f;
		s.angle_offset_deg = -90.f;
		s.distances_cm[9] = 50;
		k.begin();
		k.addScan(s, NOW, 0.f);
		float hold[3] = {0.f, 0.f, -2.5f};
		k.apply(pos, n_north, -1.f, hold);
		CHECK(near(hold[0], -0.2f));
	}

	// Bins PX4 does not count: at or below the minimum, at or above the maximum, UINT16_MAX;
	// a stale scan; an unknown frame; a zero increment.
	{
		KeepOut k = keepOut(1.0f);
		ObstacleScan s = emptyScan(ObstacleScan::FRAME_GLOBAL);
		s.distances_cm[18] = 20;      // = min
		s.distances_cm[19] = 10;      // below min
		s.distances_cm[20] = 1000;    // = max
		s.distances_cm[21] = 65535;   // unknown
		k.begin();
		CHECK(k.addScan(s, NOW, 0.f));
		float hold[3] = {0.f, 0.f, -2.5f};
		CHECK(k.apply(pos, n_north, -1.f, hold).points == 0);

		ObstacleScan stale = emptyScan(ObstacleScan::FRAME_GLOBAL);
		stale.distances_cm[18] = 60;
		stale.timestamp_us = NOW - 600000;
		k.begin();
		CHECK(!k.addScan(stale, NOW, 0.f));
		ObstacleScan future = stale;
		future.timestamp_us = NOW + 1000;
		CHECK(!k.addScan(future, NOW, 0.f));
		ObstacleScan bad_frame = stale;
		bad_frame.timestamp_us = NOW;
		bad_frame.frame = 7;
		CHECK(!k.addScan(bad_frame, NOW, 0.f));
		ObstacleScan no_inc = bad_frame;
		no_inc.frame = ObstacleScan::FRAME_GLOBAL;
		no_inc.increment_deg = 0.f;
		CHECK(!k.addScan(no_inc, NOW, 0.f));
		CHECK(k.apply(pos, n_north, -1.f, hold).points == 0);
	}

	// Two poles 1.2 m apart with the hold point between them leave room: the point slides
	// out of the line between them until it is 1 m from both (0.8 m off the line).
	{
		KeepOut k = keepOut(1.0f, 0.f);
		float hold[3] = {0.f, 0.f, -2.5f};
		KeepOutOutput o{};

		for (int tick = 0; tick < 200; tick++) {
			k.begin();
			k.addPoint(90.f * DEG, 0.6f);
			k.addPoint(-90.f * DEG, 0.6f);
			o = k.apply(pos, n_north, -1.f, hold);
		}

		CHECK(near(hold[1], 0.f, 0.01f));
		CHECK(o.nearest_m > 0.95f);
	}

	// No room: walls 0.6 m east and 0.6 m west (lines of points), keep-out 1 m. The hold
	// point stays between them (not flicked to one side) and no_room is set.
	{
		KeepOut k = keepOut(1.0f, 0.f);
		float hold[3] = {0.f, 0.f, -2.5f};

		for (int tick = 0; tick < 50; tick++) {
			k.begin();

			for (float x = -2.5f; x < 2.51f; x += 0.25f) {
				k.addPoint(atan2f(0.6f, x), hypotf(x, 0.6f));
				k.addPoint(atan2f(-0.6f, x), hypotf(x, 0.6f));
			}

			const KeepOutOutput o = k.apply(pos, n_north, -1.f, hold);
			CHECK(o.no_room);
		}

		CHECK(near(hold[1], 0.f, 0.01f));
		CHECK(near(hold[0], 0.f, 0.05f));
	}

	// A corridor seen down its length is not a wall ahead: side walls 0.5 m either side,
	// nose along the corridor. No point lies within 10 deg of the nose closer than 2.8 m,
	// so nothing walks the hold point backwards tick after tick.
	{
		KeepOut k = keepOut(0.3f, 0.7f);
		float hold[3] = {0.f, 0.f, -2.5f};

		for (int tick = 0; tick < 50; tick++) {
			k.begin();

			for (float x = -2.5f; x < 2.51f; x += 0.25f) {
				k.addPoint(atan2f(0.5f, x), hypotf(x, 0.5f));
				k.addPoint(atan2f(-0.5f, x), hypotf(x, 0.5f));
			}

			k.apply(pos, n_north, -1.f, hold);
		}

		CHECK(near(hold[0], 0.f, 0.01f) && near(hold[1], 0.f, 0.01f));
	}

	// Corridor wide enough: walls 0.9 m east and 1.5 m west, keep-out 1 m. The point settles
	// with both satisfied (0.1 m west).
	{
		KeepOut k = keepOut(1.0f, 0.f);
		float hold[3] = {0.f, 0.f, -2.5f};
		KeepOutOutput o{};

		for (int tick = 0; tick < 20; tick++) {
			k.begin();
			k.addPoint(90.f * DEG, 0.9f);
			k.addPoint(-90.f * DEG, 1.5f);
			o = k.apply(pos, n_north, -1.f, hold);
		}

		CHECK(!o.no_room);
		CHECK(hold[1] <= -0.099f && hold[1] >= -0.5f);
	}

	// Keep-out against the wall floor: a pole behind-right pushes the hold toward the wall,
	// the floor stops it at 0.7 m; the sideways part of the push still happens.
	{
		KeepOut k = keepOut(1.0f, 0.7f);
		float hold[3] = {0.f, 0.f, -2.5f};

		for (int tick = 0; tick < 20; tick++) {
			k.begin();
			k.addPoint(0.f, 0.8f);                 // the wall, 0.8 m ahead (in the cone)
			k.addPoint(135.f * DEG, 0.5f);         // pole behind-right (south-east)
			k.apply(pos, n_north, -1.f, hold);
		}

		CHECK(hold[0] <= 0.1f + 0.02f);  // never closer than 0.7 m to the wall at N 0.8
		CHECK(hold[1] < -0.1f);          // moved west, away from the pole
	}

	// The same with only the forward rangefinder seeing the wall (no lidar point ahead): the
	// keep-out must not push the hold point inside the floor the wall guard keeps, as it did
	// in SIH (`sitl_test.py --keep-out --wall`: Structure Hover 0.50 m from a creeping wall).
	{
		KeepOut k = keepOut(1.0f, 0.7f);
		float hold[3] = {0.f, 0.f, -2.5f};

		for (int tick = 0; tick < 20; tick++) {
			k.begin();
			k.addPoint(135.f * DEG, 0.5f);         // pole behind-right (south-east)
			k.apply(pos, n_north, 0.8f, hold);     // rangefinder: wall 0.8 m ahead
		}

		CHECK(hold[0] <= 0.1f + 0.02f);
		CHECK(hold[1] < -0.1f);
	}

	// The rangefinder's wall distance classifies points too: with the range at 1 m, a point
	// 1.2 m ahead and 0.9 m to the side (on the face, outside the cone) is the structure.
	{
		KeepOut k = keepOut(1.5f, 0.7f);
		k.begin();
		k.addPoint(atan2f(0.9f, 1.0f), hypotf(1.0f, 0.9f));
		float hold[3] = {0.f, 0.f, -2.5f};
		const KeepOutOutput o = k.apply(pos, n_north, 1.0f, hold);
		CHECK(near(o.shift_m, 0.f));
		// Without the range it is an obstacle (outside the cone, no wall known): pushed.
		KeepOut k2 = keepOut(1.5f, 0.7f);
		k2.begin();
		k2.addPoint(atan2f(0.9f, 1.0f), hypotf(1.0f, 0.9f));
		float hold2[3] = {0.f, 0.f, -2.5f};
		CHECK(k2.apply(pos, n_north, -1.f, hold2).shift_m > 0.05f);
	}

	// Non-finite input is dropped; the point list is capped.
	{
		KeepOut k = keepOut(1.0f);
		k.begin();
		k.addPoint(NAN, 0.5f);
		k.addPoint(0.f, NAN);
		k.addPoint(0.f, -1.f);

		for (int i = 0; i < 200; i++) { k.addPoint(1.f, 5.f); }

		float hold[3] = {0.f, 0.f, -2.5f};
		CHECK(k.apply(pos, n_north, -1.f, hold).points == KeepOut::MAX_POINTS);
	}

	// SH_KO_DIST -1 follows PX4's CP_DIST; 0 is off whatever CP_DIST says; capped at 5 m.
	CHECK(near(resolveKeepOut(-1.f, -1.f, 5.f), 0.f));  // both PX4 defaults: off
	CHECK(near(resolveKeepOut(-1.f, 1.2f, 5.f), 1.2f));
	CHECK(near(resolveKeepOut(-1.f, NAN, 5.f), 0.f));
	CHECK(near(resolveKeepOut(-1.f, 30.f, 5.f), 5.f));
	CHECK(near(resolveKeepOut(0.f, 1.2f, 5.f), 0.f));
	CHECK(near(resolveKeepOut(0.8f, 1.2f, 5.f), 0.8f));
	CHECK(near(resolveKeepOut(0.8f, -1.f, 5.f), 0.8f));

	if (failures == 0) {
		std::printf("keep-out tests passed\n");
		return 0;
	}

	std::printf("%d keep-out check(s) failed\n", failures);
	return 1;
}
