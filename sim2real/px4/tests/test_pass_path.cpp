// Unit tests for the Structure Hold surface pass (PassPath.hpp). No PX4 tree needed:
//   g++ -std=c++14 -O2 -Wall -Wextra -Werror -I../structure_hold test_pass_path.cpp && ./a.out
// (CI adds PX4's stricter warnings: -Wshadow -Wdouble-promotion -Wfloat-equal.)

#include "PassPath.hpp"
#include "PolicyController.hpp"

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

static const float DT = 0.01f;  // the module's control tick
static const float PI = 3.14159265358979f;

static PassPath path(float length, bool vertical = false, float speed = 0.5f, float start = 0.f)
{
	PassLimits l{};
	l.length_m = length;
	l.vertical = vertical;
	l.speed_mps = speed;
	l.start_s = start;
	PassPath p{};
	p.setLimits(l);
	return p;
}

// The mock's closed-form trapezoid (envs/structure_hold/passes.py PassPaths.at): distance
// flown t seconds after the pass starts.
static float mockDistance(float t, float length, float speed, float accel)
{
	const float vp = std::fmin(speed, std::sqrt(length * accel));
	const float ta = vp / accel;
	const float tc = (length - vp * ta) / vp;
	const float dur = 2.f * ta + tc;
	const float tau = std::fmin(std::fmax(t, 0.f), dur);

	if (tau < ta) { return 0.5f * accel * tau * tau; }

	if (tau < ta + tc) { return 0.5f * accel * ta * ta + vp * (tau - ta); }

	return length - 0.5f * accel * (dur - tau) * (dur - tau);
}

struct Flown {
	float pos[3] {0.f, 0.f, 0.f};
	float seconds{0.f};
	float max_speed{0.f};
	float max_accel{0.f};
	float max_mock_gap{0.f};  // against the mock's profile, m
	int starts{0};
	int finishes{0};
};

// Step the path until it is no longer waiting or moving (or ``max_s`` runs out).
static Flown fly(PassPath &p, float max_s = 60.f)
{
	Flown f{};
	float prev_speed = 0.f, moving_s = 0.f;
	const float length = p.length();

	for (int i = 0; i < static_cast<int>(max_s / DT); i++) {
		const PassState before = p.state();

		if (before != PassState::Waiting && before != PassState::Moving) { break; }

		const PassStep s = p.step(DT, false);
		f.seconds += DT;
		f.starts += s.started;
		f.finishes += s.finished;

		for (int k = 0; k < 3; k++) { f.pos[k] += s.delta_ned[k]; }

		const float speed = std::sqrt(s.vel_ned[0] * s.vel_ned[0] + s.vel_ned[1] * s.vel_ned[1]
					      + s.vel_ned[2] * s.vel_ned[2]);
		f.max_speed = std::fmax(f.max_speed, speed);

		if (p.state() == PassState::Moving) {  // the last step snaps the final millimetre
			f.max_accel = std::fmax(f.max_accel, std::fabs(speed - prev_speed) / DT);
		}

		prev_speed = speed;

		if (before == PassState::Moving || s.started) {
			moving_s += DT;
			const float gap = std::fabs(p.travelled() - mockDistance(moving_s, length, p.plan().speed_mps,
						    p.plan().accel_mps2));
			f.max_mock_gap = std::fmax(f.max_mock_gap, gap);
		}
	}

	return f;
}

int main()
{
	// Off by default: a plain hold never moves.
	{
		PassPath p{};
		p.engage(0.f, true);
		CHECK(p.state() == PassState::Off);
		const PassStep s = p.step(DT, false);
		CHECK(near(s.delta_ned[0], 0.f, 0.f) && near(s.delta_ned[1], 0.f, 0.f) && near(s.delta_ned[2], 0.f, 0.f));
		CHECK(!s.started && !s.finished);
		CHECK(near(p.length(), 0.f, 0.f));
	}

	// Refused: the bench (enable false), a NaN length, a length under a centimetre.
	{
		PassPath p = path(5.f);
		p.engage(0.f, false);
		CHECK(p.state() == PassState::Off);
		p = path(NAN);
		p.engage(0.f, true);
		CHECK(p.state() == PassState::Off);
		p = path(0.005f);
		p.engage(0.f, true);
		CHECK(p.state() == PassState::Off);
	}

	// 5 m to the right, facing north: the point ends 5 m east, at the same north (the
	// standoff) and height, on the mock's trapezoid.
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);
		CHECK(p.state() == PassState::Waiting);
		CHECK(near(p.length(), 5.f));
		const Flown f = fly(p);
		CHECK(p.state() == PassState::Done);
		CHECK(f.starts == 1 && f.finishes == 1);
		CHECK(near(f.pos[0], 0.f, 1e-5f));
		CHECK(near(f.pos[1], 5.f, 1e-3f));
		CHECK(near(f.pos[2], 0.f, 0.f));
		CHECK(near(p.travelled(), 5.f, 1e-4f));
		CHECK(f.max_speed <= 0.5f + 1e-5f);
		CHECK(f.max_accel <= 0.5f + 1e-3f);
		CHECK(f.max_mock_gap < 0.01f);  // within a centimetre of passes.py all the way
		CHECK(near(f.seconds, passDuration(5.f, 0.5f, 0.5f), 0.15f));
		CHECK(near(passDuration(5.f, 0.5f, 0.5f), 11.f, 1e-4f));
		CHECK(near(passDuration(5.f, 1.f, 0.5f), 7.f, 1e-4f));
		// Done: nothing moves any more.
		const PassStep after = p.step(DT, false);
		CHECK(near(after.delta_ned[1], 0.f, 0.f) && near(after.vel_ned[1], 0.f, 0.f));
	}

	// A negative length goes left; facing east, right is south.
	{
		PassPath p = path(-5.f);
		p.engage(0.f, true);
		Flown f = fly(p);
		CHECK(near(f.pos[1], -5.f, 1e-3f) && near(f.pos[0], 0.f, 1e-5f));
		p = path(2.f);
		p.engage(0.5f * PI, true);
		f = fly(p);
		CHECK(near(f.pos[0], -2.f, 1e-3f) && near(f.pos[1], 0.f, 1e-3f));
	}

	// The pass never changes the standoff: no motion along the wall normal, any heading.
	for (float yaw = -3.f; yaw < 3.1f; yaw += 0.7f) {
		PassPath p = path(5.f);
		p.engage(yaw, true);
		const Flown f = fly(p);
		CHECK(near(f.pos[0] * std::cos(yaw) + f.pos[1] * std::sin(yaw), 0.f, 1e-4f));
		CHECK(near(std::sqrt(f.pos[0] * f.pos[0] + f.pos[1] * f.pos[1]), 5.f, 1e-3f));
	}

	// Up the wall: NED z falls; down with a negative length.
	{
		PassPath p = path(3.f, true);
		p.engage(1.f, true);
		Flown f = fly(p);
		CHECK(near(f.pos[2], -3.f, 1e-3f) && near(f.pos[0], 0.f, 0.f) && near(f.pos[1], 0.f, 0.f));
		p = path(-1.5f, true);
		p.engage(1.f, true);
		CHECK(p.direction()[2] > 0.f);
		f = fly(p);
		CHECK(near(f.pos[2], 1.5f, 1e-3f));
	}

	// A short pass never reaches the cruise speed and still ends on its length.
	{
		PassPath p = path(0.3f, false, 1.f);
		p.engage(0.f, true);
		const Flown f = fly(p);
		CHECK(p.state() == PassState::Done);
		CHECK(near(f.pos[1], 0.3f, 1e-4f));
		CHECK(f.max_speed < 0.42f && f.max_speed > 0.34f);  // sqrt(0.3 * 0.5) = 0.387
		CHECK(f.max_mock_gap < 0.01f);
	}

	// The settle: nothing moves for SH_PASS_T0, and it counts only on a fresh estimate.
	{
		PassPath p = path(5.f, false, 0.5f, 3.f);
		p.engage(0.f, true);

		for (int i = 0; i < 299; i++) {
			const PassStep s = p.step(DT, false);
			CHECK(!s.started && near(s.delta_ned[1], 0.f, 0.f));
		}

		CHECK(p.state() == PassState::Waiting);

		for (int i = 0; i < 200; i++) { p.step(DT, true); }  // 2 s stale: the clock stands

		CHECK(p.state() == PassState::Waiting);
		int started = 0;

		for (int i = 0; i < 5; i++) { started += p.step(DT, false).started; }

		CHECK(started == 1 && p.state() == PassState::Moving);
	}

	// Held back mid-pass (a stale estimate): the point brakes at 0.5 m/s^2 to a stop, stays,
	// and on release flies the rest and ends on the length.
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);

		for (int i = 0; i < 400; i++) { p.step(DT, false); }  // 4 s in: cruising

		CHECK(near(p.speed(), 0.5f, 1e-4f));
		const float at_hold = p.travelled();
		float prev = p.speed();

		for (int i = 0; i < 300; i++) {
			const PassStep s = p.step(DT, true);
			CHECK(p.speed() <= prev + 1e-6f && prev - p.speed() <= 0.5f * DT + 1e-5f);
			CHECK(!s.finished);
			prev = p.speed();
		}

		CHECK(near(p.speed(), 0.f, 0.f));
		CHECK(p.state() == PassState::Moving);
		CHECK(near(p.travelled() - at_hold, 0.25f, 0.01f));  // v^2 / 2a
		const float stopped_at = p.travelled();
		p.step(DT, true);
		CHECK(near(p.travelled(), stopped_at, 0.f));
		const Flown f = fly(p);
		CHECK(p.state() == PassState::Done && f.finishes == 1 && f.starts == 0);
		CHECK(near(p.travelled(), 5.f, 1e-4f));
		CHECK(f.max_speed <= 0.5f + 1e-5f && f.max_accel <= 0.5f + 1e-3f);
	}

	// Held back at any point of the braking at the end: the point never passes the length.
	for (float left = 0.3f; left > 0.0005f; left *= 0.5f) {
		PassPath p = path(1.f);
		p.engage(0.f, true);

		while (p.length() - p.travelled() > left) { p.step(DT, false); }

		for (int i = 0; i < 200; i++) {
			p.step(DT, true);
			CHECK(p.travelled() <= 1.f + 1e-6f);
		}

		fly(p);
		CHECK(p.state() == PassState::Done && near(p.travelled(), 1.f, 1e-6f));
	}

	// stop(): the point brakes to rest at accel, brakeDistance() further on, and stays
	// there for the rest of the engage. (Stopped dead from 0.5 m/s it left the vehicle
	// 0.3 m past the point in SIH.)
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);
		CHECK(near(p.brakeDistance(), 0.f, 0.f));  // nothing to brake before it moves

		for (int i = 0; i < 400; i++) { p.step(DT, false); }

		const float at_stop = p.travelled();
		const float reach = p.brakeDistance();
		CHECK(near(p.speed(), 0.5f) && near(reach, 0.25f, 1e-3f));  // v^2 / (2 a)
		p.stop();
		CHECK(p.moving() && p.ending());
		int ticks = 0, stopped = 0;
		float last_speed = p.speed();
		bool smooth = true;

		while (p.moving() && ticks < 1000) {
			const PassStep s = p.step(DT, false);
			smooth = smooth && s.vel_ned[1] <= last_speed + 1e-6f && last_speed - s.vel_ned[1] <= 0.5f * DT + 1e-6f;
			last_speed = s.vel_ned[1];
			stopped += s.stopped ? 1 : 0;
			CHECK(!s.finished);
			ticks++;
		}

		CHECK(smooth);
		CHECK(stopped == 1);
		CHECK(ticks >= 99 && ticks <= 102);  // 0.5 m/s at 0.5 m/s^2: one second
		CHECK(p.state() == PassState::Stopped && !p.moving() && !p.ending());
		CHECK(near(p.travelled() - at_stop, reach, 0.01f));
		const float rest = p.travelled();
		const PassStep s = p.step(DT, false);
		CHECK(near(s.delta_ned[1], 0.f, 0.f) && near(s.vel_ned[1], 0.f, 0.f) && !s.stopped);
		CHECK(near(p.travelled(), rest, 0.f) && near(p.brakeDistance(), 0.f, 0.f));
		p.stop();  // again: no change
		CHECK(p.state() == PassState::Stopped);
		// The next engage flies a new, whole pass.
		p.engage(0.f, true);
		CHECK(p.state() == PassState::Waiting && near(p.travelled(), 0.f, 0.f) && !p.ending());

		for (int i = 0; i < 2000; i++) { p.step(DT, false); }

		CHECK(p.state() == PassState::Done && near(p.travelled(), 5.f, 1e-4f));
	}

	// stop() before the pass has started: it never starts. On a finished pass: no change.
	{
		PassPath p = path(5.f, false, 0.5f, 3.f);
		p.engage(0.f, true);
		p.step(DT, false);
		p.stop();
		CHECK(p.state() == PassState::Stopped);

		for (int i = 0; i < 1000; i++) { CHECK(!p.step(DT, false).started); }

		CHECK(near(p.travelled(), 0.f, 0.f));
		p = path(1.f);
		p.engage(0.f, true);

		for (int i = 0; i < 1000; i++) { p.step(DT, false); }

		p.stop();
		CHECK(p.state() == PassState::Done && !p.ending());
	}

	// stop() while a stale estimate already holds the point back: it is at rest at once.
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);

		for (int i = 0; i < 300; i++) { p.step(DT, false); }

		for (int i = 0; i < 200; i++) { p.step(DT, true); }

		CHECK(p.moving() && near(p.speed(), 0.f, 0.f));
		const float rest = p.travelled();
		p.stop();
		CHECK(p.step(DT, true).stopped && p.state() == PassState::Stopped);
		CHECK(near(p.travelled(), rest, 0.f));
	}

	// stop() just before the end: the pass ends at its length, never past it.
	{
		PassPath p = path(1.f);
		p.engage(0.f, true);

		while (p.moving() && p.travelled() < 0.97f) { p.step(DT, false); }

		p.stop();

		for (int i = 0; i < 500; i++) { p.step(DT, false); }

		CHECK(!p.moving() && p.travelled() <= 1.f + 1e-6f);
	}

	// A heading reset turns a sideways pass with the wall normal; a vertical one stays up.
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);
		p.rotate(0.5f * PI);
		CHECK(near(p.direction()[0], -1.f) && near(p.direction()[1], 0.f));
		p = path(5.f, true);
		p.engage(0.f, true);
		p.rotate(1.f);
		CHECK(near(p.direction()[2], -1.f) && near(p.direction()[0], 0.f) && near(p.direction()[1], 0.f));
	}

	// Parameters are read at the engage: a change mid-pass does not bend the path.
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);

		for (int i = 0; i < 100; i++) { p.step(DT, false); }

		PassLimits other{};
		other.length_m = -2.f;
		other.vertical = true;
		other.speed_mps = 1.f;
		p.setLimits(other);
		const Flown f = fly(p);
		CHECK(near(p.travelled(), 5.f, 1e-4f));
		CHECK(f.max_speed <= 0.5f + 1e-5f);
		CHECK(near(p.plan().length_m, 5.f, 0.f));
	}

	// A bad tick: zero, negative and NaN dt move nothing; a late tick counts as 50 ms.
	{
		PassPath p = path(5.f);
		p.engage(0.f, true);

		for (int i = 0; i < 300; i++) { p.step(DT, false); }

		const float before = p.travelled();
		p.step(0.f, false);
		p.step(-1.f, false);
		p.step(NAN, false);
		CHECK(near(p.travelled(), before, 0.f));
		p.step(10.f, false);
		CHECK(near(p.travelled() - before, 0.5f * PassPath::MAX_DT_S, 1e-5f));
	}

	// The policy's velocity input is relative to the hold point (PolicyController.hpp): a
	// vehicle keeping up with the pass reads as standing still, and a plain hold reads as
	// before.
	{
		StructureFrame frame{};
		const float hold[3] = {0.f, 0.f, -2.5f};
		frame.set(hold, 0.3f);
		const float last_action[4] = {0.f, 0.f, 0.f, 0.f};
		VehicleSample s{};
		s.vel_ned[0] = 0.2f;
		s.vel_ned[1] = -0.4f;
		s.vel_ned[2] = 0.1f;
		float plain[policy::NUM_OBS], keeping_up[policy::NUM_OBS], lagging[policy::NUM_OBS];
		buildObservation(frame, s, last_action, plain);

		for (int i = 0; i < 3; i++) { s.hold_vel_ned[i] = s.vel_ned[i]; }

		buildObservation(frame, s, last_action, keeping_up);
		CHECK(near(keeping_up[policy::OBS_VEL + 0], 0.f, 1e-6f) && near(keeping_up[policy::OBS_VEL + 1], 0.f, 1e-6f)
		      && near(keeping_up[policy::OBS_VEL + 2], 0.f, 1e-6f));
		// The hold point twice as fast: the input is minus the plain one.
		for (int i = 0; i < 3; i++) { s.hold_vel_ned[i] = 2.f * s.vel_ned[i]; }

		buildObservation(frame, s, last_action, lagging);

		for (int i = 0; i < 3; i++) { CHECK(near(lagging[policy::OBS_VEL + i], -plain[policy::OBS_VEL + i], 1e-6f)); }

		for (int i = 0; i < policy::NUM_OBS; i++) {
			if (i < policy::OBS_VEL || i >= policy::OBS_VEL + 3) { CHECK(near(lagging[i], plain[i], 0.f)); }
		}
	}

	if (failures == 0) {
		std::printf("test_pass_path: all checks passed\n");
	}

	return failures == 0 ? 0 : 1;
}
