/**
 * Structure Hold keep-out: PX4's collision-prevention data as a hard limit on the hold point.
 *
 * PX4's own collision prevention (CP_DIST, src/lib/collision_prevention) only acts in
 * Position mode: it bends the pilot's stick setpoint. Structure Hold commands body rates, so
 * a drone maker's obstacle sensors were ignored inside the hold. This reads the same data
 * PX4's collision prevention reads (docs/decisions.md, 2026-11-02 keep-out row):
 *
 *   - obstacle_distance: a horizontal ring of distances (a 360 deg lidar such as the
 *     LightWare SF45, or OBSTACLE_DISTANCE from a companion computer), north-aligned
 *     (MAV_FRAME_GLOBAL / LOCAL_NED) or body-aligned (MAV_FRAME_BODY_FRD);
 *   - distance_sensor instances facing any horizontal yaw orientation (45 deg steps).
 *
 * Every reading becomes an obstacle point in the horizontal plane, relative to the vehicle.
 * The hold point is then kept at least keep_out_m from every point, by moving it away from
 * the points that are too close, and only ever away (a ratchet, as the wall guard): a point
 * that later goes away never pulls the hold back. It is a keep-out, not avoidance: the
 * module plans nothing and never moves the hold point toward free space it cannot see.
 *
 * The structure itself is not an obstacle. The hold sits 1 to 3 m from it on purpose, with
 * its own floor, SH_WALL_MIN. So points on the wall face are told apart from obstacles:
 * the wall distance W is the shortest perpendicular distance along the wall normal n
 * (the heading at engage) among the forward rangefinder and the points within
 * wall_cone_rad (10 deg) of n, a narrow beam like the rangefinder's (a wider cone would
 * take a corridor's side walls for a wall ahead and walk the hold point backwards); a point whose distance along n is within structure_tol_m of W, or
 * beyond it, is on (or behind) the structure. Those points keep the hold point SH_WALL_MIN
 * off the wall plane, as the wall guard does with the rangefinder, so a drone with a lidar
 * and no forward rangefinder gets a wall floor too. Everything nearer than the wall is an
 * obstacle: a pole, a pipe standing off the face, a side wall, a person.
 *
 * Point constraints are solved together with the wall plane by a few averaged projection
 * passes per tick (each pass moves the hold point by the mean of the pushes it needs), so
 * obstacles on opposite sides settle the point between them instead of flicking it from
 * one to the other. When there is no room (two obstacles nearer together than twice the
 * keep-out), the point ends between them and no_room is set: the module tells the pilot.
 *
 * Plain C++, no PX4 headers, so tests/test_keep_out.cpp runs it in a bare g++ build.
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace structure_hold
{

/**
 * The keep-out distance flown: SH_KO_DIST when it is set (0 = off, > 0 metres), else, at its
 * default -1, PX4's own CP_DIST, so a drone maker who set up collision prevention for
 * Position mode gets the same distance inside the hold without touching our parameters.
 * CP_DIST is -1 (off) by default in PX4; a missing, NaN or non-positive one means off. Both
 * are capped at max_m.
 */
inline float resolveKeepOut(float sh_ko_dist, float cp_dist, float max_m)
{
	const float v = sh_ko_dist < 0.f ? cp_dist : sh_ko_dist;

	if (!std::isfinite(v) || !(v > 0.f)) {
		return 0.f;
	}

	return v > max_m ? max_m : v;
}

struct KeepOutLimits {
	float keep_out_m{0.f};             // SH_KO_DIST (or PX4's CP_DIST); 0 = no obstacle keep-out
	float wall_min_m{0.f};             // SH_WALL_MIN, the floor toward the structure; 0 = none
	uint64_t max_age_us{500000};       // PX4 collision prevention's own timeout (RANGE_STREAM_TIMEOUT_US)
	float wall_cone_rad{0.1745f};      // 10 deg: points this close to the normal measure the wall
	float structure_tol_m{0.3f};       // a point this close to the wall plane is the structure
};

/** PX4 obstacle_distance, as the uORB message carries it. */
struct ObstacleScan {
	static constexpr int BINS = 72;
	static constexpr uint8_t FRAME_GLOBAL = 0;
	static constexpr uint8_t FRAME_LOCAL_NED = 1;
	static constexpr uint8_t FRAME_BODY_FRD = 12;

	uint64_t timestamp_us{0};
	uint8_t frame{FRAME_GLOBAL};
	uint16_t distances_cm[BINS] {};
	float increment_deg{0.f};
	float angle_offset_deg{0.f};
	uint16_t min_distance_cm{0};
	uint16_t max_distance_cm{0};
};

struct KeepOutOutput {
	int points{0};               // obstacle and structure points used this tick
	float nearest_m{-1.f};       // hold point to the nearest obstacle point, -1 without one
	float shift_m{0.f};          // how far the hold point moved this tick
	bool no_room{false};         // the keep-out could not be met everywhere
};

class KeepOut
{
public:
	static constexpr int MAX_POINTS = ObstacleScan::BINS + 8;
	static constexpr int PASSES = 8;

	void setLimits(const KeepOutLimits &limits) { _limits = limits; }
	const KeepOutLimits &limits() const { return _limits; }
	bool enabled() const { return _limits.keep_out_m > 0.f; }

	/** New hold: forget the shift of the previous one. */
	void engage() { _total_shift_m = 0.f; }
	float totalShift() const { return _total_shift_m; }

	/** Start collecting this tick's points. */
	void begin() { _n = 0; }

	/** One obstacle at ``distance_m`` on NED bearing ``bearing_rad`` from the vehicle. */
	void addPoint(float bearing_rad, float distance_m)
	{
		if (_n < MAX_POINTS && std::isfinite(bearing_rad) && std::isfinite(distance_m) && distance_m >= 0.f) {
			_px[_n] = distance_m * cosf(bearing_rad);
			_py[_n] = distance_m * sinf(bearing_rad);
			++_n;
		}
	}

	/**
	 * Every bin of a fresh scan that holds an obstacle, as PX4's collision prevention reads
	 * it: a bin counts when it is above the sensor's minimum and below its maximum
	 * (max + 1 means nothing there, UINT16_MAX unknown). Bin j is centred on
	 * angle_offset + j * increment, clockwise; ``heading_rad`` turns a body-frame scan to NED.
	 * Returns false (nothing added) for a stale scan or an unknown frame.
	 */
	bool addScan(const ObstacleScan &s, uint64_t now_us, float heading_rad)
	{
		if (s.timestamp_us == 0 || now_us < s.timestamp_us || now_us - s.timestamp_us > _limits.max_age_us
		    || !(s.increment_deg > 0.f)) {
			return false;
		}

		float base = s.angle_offset_deg * DEG;

		if (s.frame == ObstacleScan::FRAME_BODY_FRD) {
			base += heading_rad;

		} else if (s.frame != ObstacleScan::FRAME_GLOBAL && s.frame != ObstacleScan::FRAME_LOCAL_NED) {
			return false;
		}

		const int bins = static_cast<int>(360.f / s.increment_deg);

		for (int j = 0; j < bins && j < ObstacleScan::BINS; j++) {
			const uint16_t d = s.distances_cm[j];

			if (d > s.min_distance_cm && d < s.max_distance_cm) {
				addPoint(base + static_cast<float>(j) * s.increment_deg * DEG, d * 0.01f);
			}
		}

		return true;
	}

	/**
	 * One tick, after the points are in. ``pos_ned`` is the vehicle, ``normal_ned`` the
	 * horizontal unit wall normal (structure x), ``wall_m`` the forward rangefinder's
	 * perpendicular wall distance or negative without one. Moves ``hold_ned`` (xy only).
	 */
	KeepOutOutput apply(const float pos_ned[3], const float normal_ned[2], float wall_m, float hold_ned[3])
	{
		KeepOutOutput out{};
		out.points = _n;

		if (_n == 0) {
			return out;
		}

		const float nx = normal_ned[0], ny = normal_ned[1];
		const float cos_max = cosf(_limits.wall_cone_rad);

		// The wall: the nearest perpendicular distance among the forward range and the points
		// inside the wall cone. Only the cone's own points set the plane floor here; the
		// rangefinder's floor is the wall guard's.
		float wall_cone = INFINITY;

		for (int i = 0; i < _n; i++) {
			const float along = _px[i] * nx + _py[i] * ny;
			const float d = sqrtf(_px[i] * _px[i] + _py[i] * _py[i]);

			if (along > 0.f && along >= cos_max * d) {
				wall_cone = fminf(wall_cone, along);
			}
		}

		const float wall = (wall_m > 0.f) ? fminf(wall_m, wall_cone) : wall_cone;
		const bool have_wall = std::isfinite(wall);
		const bool plane = have_wall && _limits.wall_min_m > 0.f && std::isfinite(wall_cone);
		const float k = _limits.keep_out_m;

		// Obstacle points in NED, and the wall plane's offset along n (from the vehicle).
		float hx = hold_ned[0], hy = hold_ned[1];
		const float x0 = hx, y0 = hy;

		for (int pass = 0; pass < PASSES; pass++) {
			float sx = 0.f, sy = 0.f;
			int count = 0;

			if (plane) {
				const float clearance = wall_cone - ((hx - pos_ned[0]) * nx + (hy - pos_ned[1]) * ny);

				if (clearance < _limits.wall_min_m) {
					const float s = _limits.wall_min_m - clearance;
					sx -= s * nx;
					sy -= s * ny;
					++count;
				}
			}

			if (k > 0.f) {
				for (int i = 0; i < _n; i++) {
					if (isStructure(i, nx, ny, wall, have_wall)) { continue; }

					const float dx = hx - (pos_ned[0] + _px[i]);
					const float dy = hy - (pos_ned[1] + _py[i]);
					const float c = sqrtf(dx * dx + dy * dy);

					if (c < k) {
						// Right on the point: push straight back along the bearing to it.
						const float ux = c > 1e-3f ? dx / c : -_px[i] / fmaxf(hypotf(_px[i], _py[i]), 1e-3f);
						const float uy = c > 1e-3f ? dy / c : -_py[i] / fmaxf(hypotf(_px[i], _py[i]), 1e-3f);
						sx += (k - c) * ux;
						sy += (k - c) * uy;
						++count;
					}
				}
			}

			if (count == 0) { break; }

			hx += sx / static_cast<float>(count);
			hy += sy / static_cast<float>(count);
		}

		// What is left: the nearest obstacle and whether anything is still inside.
		float worst = 0.f;

		for (int i = 0; i < _n; i++) {
			if (isStructure(i, nx, ny, wall, have_wall)) { continue; }

			const float c = hypotf(hx - (pos_ned[0] + _px[i]), hy - (pos_ned[1] + _py[i]));
			out.nearest_m = out.nearest_m < 0.f ? c : fminf(out.nearest_m, c);
			worst = fmaxf(worst, k - c);
		}

		if (plane) {
			worst = fmaxf(worst, _limits.wall_min_m - (wall_cone - ((hx - pos_ned[0]) * nx + (hy - pos_ned[1]) * ny)));
		}

		out.no_room = worst > NO_ROOM_M;
		out.shift_m = hypotf(hx - x0, hy - y0);

		if (out.shift_m > 0.f) {
			hold_ned[0] = hx;
			hold_ned[1] = hy;
			_total_shift_m += out.shift_m;
		}

		return out;
	}

private:
	static constexpr float DEG = 0.017453292f;
	static constexpr float NO_ROOM_M = 0.05f;

	bool isStructure(int i, float nx, float ny, float wall, bool have_wall) const
	{
		return have_wall && _px[i] * nx + _py[i] * ny >= wall - _limits.structure_tol_m;
	}

	KeepOutLimits _limits{};
	float _px[MAX_POINTS] {};
	float _py[MAX_POINTS] {};
	int _n{0};
	float _total_shift_m{0.f};
};

} // namespace structure_hold
