// Unit tests for the app variant's flight logic (HoldCore.hpp). No ROS, no PX4:
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal
//     -I../include -I../../../px4/structure_hold test_hold_core.cpp -o test_hold_core
//   ./test_hold_core
//
// The point of the app variant is that it flies the module's own controller, watchdog and
// handback order, so the first test checks the app's setpoints against a bare
// PolicyController fed the same samples, value for value.

#include "structure_hold_app/HoldCore.hpp"

#include <cmath>
#include <cstdio>

using namespace structure_hold_app;
using structure_hold::HandbackTarget;
using structure_hold::Trip;

static int failures = 0;

#define CHECK(cond)                                                                 \
	do {                                                                        \
		if (!(cond)) {                                                      \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			++failures;                                                 \
		}                                                                   \
	} while (0)

static constexpr uint64_t TICK_US = 10000;  // 100 Hz, as the module

/** Airborne, armed, level, fresh estimate at the origin. */
static AppInputs flying(uint64_t now_us)
{
	AppInputs in{};
	in.now_us = now_us;
	in.lpos_us = now_us;
	in.xy_valid = in.z_valid = in.v_xy_valid = true;
	in.armed = true;
	in.landed = false;
	in.pos_ned[2] = -2.f;
	return in;
}

static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

static void testSameSetpointsAsModuleController()
{
	HoldCore core;
	structure_hold::PolicyController ref;
	const AppParams p = core.params();
	structure_hold::FallbackRule rule{};
	rule.enabled = structure_hold::resolveFallback(p.fb_en);
	ref.setRule(rule);
	structure_hold::YawGuard guard{};
	guard.enabled = structure_hold::resolveYawGuard(p.yaw_guard);
	ref.setYawGuard(guard);

	uint64_t t = 1000000;
	AppInputs in = flying(t);
	in.heading = 0.3f;
	CHECK(core.entryRefusal(in) == Trip::None);
	core.activate(in);
	ref.engage(in.pos_ned, in.heading);

	bool all_equal = true;

	for (int k = 0; k < 200; k++) {
		t += TICK_US;
		in = flying(t);
		// Drift, sway and a little attitude so every observation slot moves.
		in.pos_ned[0] = 0.2f * sinf(0.05f * static_cast<float>(k));
		in.pos_ned[1] = -0.1f * cosf(0.03f * static_cast<float>(k));
		in.vel_ned[0] = 0.01f * cosf(0.05f * static_cast<float>(k));
		const float roll = 0.05f * sinf(0.1f * static_cast<float>(k));
		in.q[0] = cosf(roll / 2.f);
		in.q[1] = sinf(roll / 2.f);
		in.rates_frd[0] = 0.1f * cosf(0.1f * static_cast<float>(k));

		const AppOutput out = core.update(in);

		structure_hold::VehicleSample s{};

		for (int i = 0; i < 3; i++) {
			s.pos_ned[i] = in.pos_ned[i];
			s.vel_ned[i] = in.vel_ned[i];
			s.rates_frd[i] = in.rates_frd[i];
		}

		const float c = cosf(roll), sn = sinf(roll);
		const float R[3][3] = {{1.f, 0.f, 0.f}, {0.f, c, -sn}, {0.f, sn, c}};

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) { s.R[i][j] = R[i][j]; }
		}

		s.engage_s = static_cast<float>(k + 1) * 0.01f;
		const structure_hold::PolicyOutput po = ref.step(s, 0.f, 0.5f);

		CHECK(out.state == AppState::Engaged);

		for (int i = 0; i < 3; i++) {
			if (!near(out.setpoint.rates_frd[i], po.setpoint.rates_frd[i], 1e-5f)) { all_equal = false; }
		}

		if (!near(out.setpoint.thrust, po.setpoint.thrust, 1e-5f)) { all_equal = false; }
	}

	CHECK(all_equal);
}

static void testEntryRefusals()
{
	HoldCore core;
	AppInputs in = flying(1000000);
	in.landed = true;
	CHECK(core.entryRefusal(in) == Trip::Landed);
	in = flying(1000000);
	in.armed = false;
	CHECK(core.entryRefusal(in) == Trip::Landed);
	in = flying(1000000);
	in.xy_valid = false;
	CHECK(core.entryRefusal(in) == Trip::EstimateInvalid);
	in = flying(1000000);
	in.heading = NAN;
	CHECK(core.entryRefusal(in) == Trip::EstimateInvalid);
	in = flying(1000000);
	CHECK(core.entryRefusal(in) == Trip::None);
}

static void testStaleTripHandbackOrderAndLatch()
{
	// v22 flies dropouts itself (policy.json output.fallback off); turn the brake-and-hover
	// fallback on (SH_FB_EN 1) to see it take over before the trip.
	HoldCore core;
	AppParams fb{};
	fb.fb_en = 1;
	CHECK(core.configure(fb) == 0);
	uint64_t t = 1000000;
	AppInputs in = flying(t);
	core.activate(in);
	const uint64_t last_fix = t;

	// Estimate stops: the fallback flies after SH_FB_T, the watchdog trips after SH_WD_EST_TO.
	bool saw_fallback = false;
	AppOutput out{};

	for (;;) {
		t += TICK_US;
		in = flying(t);
		in.lpos_us = last_fix;
		out = core.update(in);

		if (out.fallback_active) { saw_fallback = true; }

		if (out.trip != Trip::None) { break; }

		if (t - last_fix > 6000000) { break; }
	}

	CHECK(saw_fallback);
	CHECK(out.trip == Trip::EstimateStale);
	CHECK(near(out.stale_s, 5.0f, 0.02f));
	CHECK(out.state == AppState::Handback);
	CHECK(out.request_mode && out.target == HandbackTarget::Position);
	CHECK(near(out.setpoint.thrust, 0.5f, 1e-6f));  // level at hover thrust

	// Commander does not switch: Hold after the retry, then Land, repeated.
	HandbackTarget seen[4];
	int n = 0;

	for (int k = 0; k < 200 && n < 4; k++) {
		t += TICK_US;
		in = flying(t);
		in.lpos_us = last_fix;
		out = core.update(in);

		if (out.request_mode) { seen[n++] = out.target; }
	}

	CHECK(n == 4);
	CHECK(seen[0] == HandbackTarget::Hold);
	CHECK(seen[1] == HandbackTarget::Land);
	CHECK(seen[2] == HandbackTarget::Land);
	CHECK(seen[3] == HandbackTarget::Land);

	// Latched until disarm, even with a good estimate again.
	core.deactivate();
	in = flying(t);
	CHECK(core.entryRefusal(in) == Trip::EstimateStale);
	core.disarmed();
	CHECK(core.entryRefusal(in) == Trip::None);
}

static void testAltitudeFirstWithoutXy()
{
	HoldCore core;
	AppInputs in = flying(1000000);
	core.activate(in);
	in = flying(1010000);
	in.xy_valid = false;
	const AppOutput out = core.update(in);
	CHECK(out.trip == Trip::EstimateInvalid);
	CHECK(out.request_mode && out.target == HandbackTarget::Altitude);
}

static void testTestFaultAndTilt()
{
	HoldCore core;
	AppInputs in = flying(1000000);
	core.activate(in);
	in = flying(1010000);
	in.test_fault = true;
	CHECK(core.update(in).trip == Trip::TestFault);

	HoldCore c2;
	in = flying(1000000);
	c2.activate(in);
	in = flying(1010000);
	const float roll = 1.0f;  // 57 deg, past the 45 deg default
	in.q[0] = cosf(roll / 2.f);
	in.q[1] = sinf(roll / 2.f);
	const AppOutput out = c2.update(in);
	CHECK(out.trip == Trip::Tilt);
	// Level-hover command rolls back left.
	CHECK(out.setpoint.rates_frd[0] < 0.f);
}

static void testFrozenOdometry()
{
	HoldCore core;
	uint64_t t = 1000000;
	const float p[3] = {0.f, 0.f, -2.f};
	core.noteOdometry(t, t, p);
	AppInputs in = flying(t);
	core.activate(in);
	AppOutput out{};

	for (int k = 0; k < 150; k++) {
		t += TICK_US;
		core.noteOdometry(t, t, p);  // on time, never moving
		in = flying(t);
		out = core.update(in);

		if (out.trip != Trip::None) { break; }
	}

	CHECK(out.trip == Trip::EstimateFrozen);
}

static void testDisarmStops()
{
	HoldCore core;
	AppInputs in = flying(1000000);
	core.activate(in);
	in = flying(1010000);
	in.armed = false;
	const AppOutput out = core.update(in);
	CHECK(out.state == AppState::Standby);
	CHECK(core.state() == AppState::Standby);
}

static void testParamsSanitised()
{
	AppParams p{};
	p.wd_est_to_s = NAN;
	p.yaw_p = -3.f;
	p.fb_en = 7;
	p.hover_thrust = 0.95f;
	HoldCore core;
	CHECK(core.configure(p) == 4);
	CHECK(near(core.params().wd_est_to_s, 5.0f, 1e-6f));
	CHECK(near(core.params().yaw_p, 0.5f, 1e-6f));
	CHECK(core.params().fb_en == -1);
	CHECK(near(core.params().hover_thrust, 0.9f, 1e-6f));
	CHECK(core.configure(AppParams{}) == 0);
}

static void testHoverAtHoldPoint()
{
	// At the hold point, level and still: thrust near hover, rates small.
	HoldCore core;
	uint64_t t = 1000000;
	AppInputs in = flying(t);
	core.activate(in);
	AppOutput out{};

	for (int k = 0; k < 50; k++) {
		t += TICK_US;
		in = flying(t);
		out = core.update(in);
	}

	CHECK(out.policy_ok);
	CHECK(out.setpoint.thrust > 0.3f && out.setpoint.thrust < 0.7f);

	for (int i = 0; i < 3; i++) { CHECK(std::fabs(out.setpoint.rates_frd[i]) < 1.0f); }
}

int main()
{
	testSameSetpointsAsModuleController();
	testEntryRefusals();
	testStaleTripHandbackOrderAndLatch();
	testAltitudeFirstWithoutXy();
	testTestFaultAndTilt();
	testFrozenOdometry();
	testDisarmStops();
	testParamsSanitised();
	testHoverAtHoldPoint();

	if (failures == 0) {
		std::printf("test_hold_core: all passed\n");
		return 0;
	}

	std::printf("test_hold_core: %d failure(s)\n", failures);
	return 1;
}
