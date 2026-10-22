// Unit tests for the policy controller of the Structure Hold module. No PX4 tree needed:
//   g++ -std=c++14 -O2 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -Wfloat-equal -I../structure_hold test_policy.cpp && ./a.out
// (c++14 and those warnings mirror PX4's own build flags)
//
// Every expected value comes from the Python side (tests/PolicyTestVectors.hpp, written by
// tools/make_test_vectors.py): the observation StructureHoldEnv writes, the fallback
// action acceleration_to_action produces, and onnxruntime's output of the released policy.

#include "PolicyController.hpp"
#include "PolicyTestVectors.hpp"

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

static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }
static bool same(float a, float b) { return !(a < b) && !(a > b); }  // bit-for-bit, without -Wfloat-equal

static VehicleSample sampleOf(int k)
{
	VehicleSample s{};

	for (int i = 0; i < 3; i++) {
		s.pos_ned[i] = testvec::POS_NED[k * 3 + i];
		s.vel_ned[i] = testvec::VEL_NED[k * 3 + i];
		s.rates_frd[i] = testvec::RATES_FRD[k * 3 + i];

		for (int j = 0; j < 3; j++) {
			s.R[i][j] = testvec::R_FRD_NED[k * 9 + i * 3 + j];
		}
	}

	s.stale = testvec::STALE[k];
	s.age_s = testvec::AGE_S[k];
	return s;
}

static StructureFrame frameOf(int k)
{
	StructureFrame f{};
	f.set(&testvec::HOLD_NED[k * 3], testvec::STRUCTURE_YAW_NED);
	return f;
}

int main()
{
	static_assert(testvec::HISTORY == policy::HISTORY, "test vectors and policy header disagree on history");
	static_assert(testvec::NUM_OBS == policy::NUM_OBS, "test vectors and policy header disagree on NUM_OBS");

	// 1. The MLP matches onnxruntime on stacked inputs.
	{
		float scratch[2 * policy::MAX_WIDTH];
		float y[policy::OUTPUT_SIZE];
		float worst = 0.f;

		for (int k = 0; k < testvec::N; k++) {
			CHECK(policy::forward(policy::NET, &testvec::MLP_IN[k * policy::INPUT_SIZE], y, scratch, policy::MAX_WIDTH));

			for (int i = 0; i < 4; i++) {
				const float d = std::fabs(y[i] - testvec::MLP_OUT[k * 4 + i]);
				worst = d > worst ? d : worst;
			}
		}

		std::printf("mlp: max abs diff vs onnxruntime %.3g over %d vectors\n", (double)worst, testvec::N);
		CHECK(worst < 1e-4f);
		CHECK(!policy::forward(policy::NET, testvec::MLP_IN, y, scratch, policy::INPUT_SIZE - 1));  // too small
	}

	// 2. The observation built from NED / FRD matches the env's ENU / FLU observation.
	{
		float worst = 0.f;

		for (int k = 0; k < testvec::N; k++) {
			float obs[policy::NUM_OBS];
			buildObservation(frameOf(k), sampleOf(k), &testvec::LAST_ACTION[k * 4], obs);

			for (int i = 0; i < policy::NUM_OBS; i++) {
				const float d = std::fabs(obs[i] - testvec::OBS[k * policy::NUM_OBS + i]);
				worst = d > worst ? d : worst;
			}
		}

		std::printf("observation: max abs diff vs Python env %.3g\n", (double)worst);
		CHECK(worst < 2e-5f);

		// Vector 0 sits beyond the 5 m clip with a long dropout: clipped, stale, age capped.
		float obs[policy::NUM_OBS];
		buildObservation(frameOf(0), sampleOf(0), &testvec::LAST_ACTION[0], obs);
		CHECK(same(obs[policy::OBS_POS_ERR], -policy::OBS_POS_CLIP_M));
		CHECK(same(obs[policy::OBS_STALE], 1.f));
		CHECK(same(obs[policy::OBS_AGE], 1.f));
	}

	// 3. The fallback action matches acceleration_to_action, with the gains Python used.
	{
		float worst = 0.f;
		FallbackRule rule{};
		rule.brake_vel = testvec::FALLBACK_BRAKE_VEL;
		rule.max_acc = testvec::FALLBACK_MAX_ACC;
		rule.tilt = testvec::FALLBACK_TILT;
		rule.yaw = testvec::FALLBACK_YAW;

		for (int k = 0; k < testvec::N; k++) {
			float action[4];
			fallbackAction(rule, frameOf(k), sampleOf(k), &testvec::OBS[k * policy::NUM_OBS], action);

			for (int i = 0; i < 4; i++) {
				const float d = std::fabs(action[i] - testvec::FALLBACK_ACTION[k * 4 + i]);
				worst = d > worst ? d : worst;
			}
		}

		std::printf("fallback: max abs diff vs Python %.3g\n", (double)worst);
		CHECK(worst < 2e-5f);
	}

	// 4. Structure frame round trip and a rotated frame.
	{
		StructureFrame f{};
		const float hold[3] = {1.f, 2.f, -3.f};
		f.set(hold, 0.7f);
		const float v[3] = {0.3f, -1.2f, 0.5f};
		float u[3], back[3];
		f.toStructure(v, u);
		f.toNed(u, back);

		for (int i = 0; i < 3; i++) { CHECK(near(back[i], v[i], 1e-6f)); }

		// Heading at engage faces +x of the structure frame.
		const float fwd[3] = {cosf(0.7f), sinf(0.7f), 0.f};
		f.toStructure(fwd, u);
		CHECK(near(u[0], 1.f, 1e-6f) && near(u[1], 0.f, 1e-6f) && near(u[2], 0.f, 1e-6f));
		// Up in NED is -z.
		const float down[3] = {0.f, 0.f, 1.f};
		f.toStructure(down, u);
		CHECK(near(u[2], -1.f, 1e-6f));
	}

	// 5. Action -> PX4 setpoint: FLU -> FRD signs, thrust scaled through the hover point.
	{
		SetpointLimits lim{};
		const float a[4] = {0.5f, 0.25f, -0.1f, 2.f * policy::MASS_KG * GRAVITY_MPS2 / policy::MAX_THRUST_N - 1.f};
		const Setpoint sp = actionToSetpoint(a, 0.42f, lim);
		CHECK(near(sp.rates_frd[0], 0.5f * policy::MAX_RATE_RAD_S, 1e-6f));
		CHECK(near(sp.rates_frd[1], -0.25f * policy::MAX_RATE_RAD_S, 1e-6f));
		CHECK(near(sp.rates_frd[2], 0.1f * policy::MAX_RATE_RAD_S, 1e-6f));
		CHECK(near(sp.thrust, 0.42f, 1e-5f));  // the mock's hover thrust lands on the vehicle's
		const float lo[4] = {0.f, 0.f, 0.f, -1.f};
		CHECK(same(actionToSetpoint(lo, 0.42f, lim).thrust, lim.thrust_min));
		const float hi[4] = {3.f, 0.f, 0.f, 1.f};
		CHECK(near(actionToSetpoint(hi, 0.42f, lim).rates_frd[0], policy::MAX_RATE_RAD_S, 1e-6f));
		// Full collective = the policy's thrust-to-weight ratio times the hover thrust.
		const float full = policy::MAX_THRUST_N / (policy::MASS_KG * GRAVITY_MPS2);
		CHECK(near(actionToSetpoint(hi, 0.42f, lim).thrust, clampf(full * 0.42f, lim.thrust_min, lim.thrust_max), 1e-5f));
		CHECK(same(actionToSetpoint(hi, 0.9f, lim).thrust, lim.thrust_max));  // above 0.9, clamped
		lim.rate_max = 0.3f * policy::MAX_RATE_RAD_S;
		CHECK(near(actionToSetpoint(hi, 0.5f, lim).rates_frd[0], lim.rate_max, 1e-6f));
		CHECK(near(actionToSetpoint(a, 0.5f, lim).rates_frd[1], -0.25f * policy::MAX_RATE_RAD_S, 1e-6f));  // under it
	}

	// 6. The controller: engage fills the history, the fallback takes over at the threshold
	//    and hands back the moment the estimate is fresh, the policy keeps running meanwhile.
	{
		PolicyController ctl{};
		FallbackRule rule{};
		rule.enabled = true;  // the rule itself is under test, whatever this policy's default
		rule.threshold_s = testvec::FALLBACK_THRESHOLD_S;
		ctl.setRule(rule);
		YawGuard no_guard{};
		no_guard.enabled = false;  // the policy's own yaw here; the guard has its own test
		ctl.setYawGuard(no_guard);
		const int k = 3;
		ctl.engage(&testvec::HOLD_NED[k * 3], testvec::STRUCTURE_YAW_NED);
		VehicleSample s = sampleOf(k);
		s.stale = false;
		s.age_s = 0.f;  // the env's age: 0 while fresh

		// Fresh: policy flies, last_action starts at zero.
		PolicyOutput o = ctl.step(s, 0.f, 0.5f);
		CHECK(!o.fallback_active && o.ok);
		CHECK(same(ctl.lastObservation()[policy::OBS_LAST_ACTION], 0.f));

		for (int i = 0; i < 4; i++) { CHECK(same(o.action[i], o.policy_action[i])); }  // guard off: all the policy's

		// History filled with the first observation: a repeated step gives the same output
		// as a fresh engage with the same observation would, within float noise.
		PolicyController ctl2{};
		ctl2.setRule(rule);
		ctl2.setYawGuard(no_guard);
		ctl2.engage(&testvec::HOLD_NED[k * 3], testvec::STRUCTURE_YAW_NED);
		PolicyOutput o2 = ctl2.step(s, 0.f, 0.5f);

		for (int i = 0; i < 4; i++) { CHECK(near(o.action[i], o2.action[i], 1e-6f)); }

		// Just under the threshold: still the policy. At it: the fallback, and the flown
		// action is what the pure rule gives from the same observation.
		s.stale = true;
		s.age_s = 0.6f;
		o = ctl.step(s, rule.threshold_s - 0.01f, 0.5f);
		CHECK(!o.fallback_active);
		o = ctl.step(s, rule.threshold_s, 0.5f);
		CHECK(o.fallback_active);
		float expect[4];
		fallbackAction(rule, ctl.frame(), s, ctl.lastObservation(), expect);

		for (int i = 0; i < 4; i++) { CHECK(same(o.action[i], expect[i])); }

		// The flown (fallback) action is what the policy sees as last_action next tick.
		o2 = ctl.step(s, rule.threshold_s + 0.01f, 0.5f);
		CHECK(o2.fallback_active);

		for (int i = 0; i < 4; i++) { CHECK(same(ctl.lastObservation()[policy::OBS_LAST_ACTION + i], expect[i])); }

		// Fresh fix: hand back on the first tick.
		s.stale = false;
		s.age_s = 0.f;
		o = ctl.step(s, 0.f, 0.5f);
		CHECK(!o.fallback_active);

		for (int i = 0; i < 4; i++) { CHECK(same(o.action[i], o.policy_action[i])); }

		// Disabled rule: never active. Forced (test hook): always.
		rule.enabled = false;
		ctl.setRule(rule);
		o = ctl.step(s, 10.f, 0.5f);
		CHECK(!o.fallback_active);
		rule.forced = true;
		ctl.setRule(rule);
		o = ctl.step(s, 0.f, 0.5f);
		CHECK(o.fallback_active);
	}

	// 8. Yaw guard: the policy's yaw output is replaced by the heading hold, the flown
	//    action is what the policy sees next, roll/pitch/thrust stay the policy's, and the
	//    fallback (which holds heading itself) is unaffected.
	{
		PolicyController ctl{};
		FallbackRule rule{};
		rule.enabled = true;
		ctl.setRule(rule);
		const int k = 5;
		ctl.engage(&testvec::HOLD_NED[k * 3], testvec::STRUCTURE_YAW_NED);
		VehicleSample s = sampleOf(k);
		s.stale = false;
		YawGuard guard{};
		guard.enabled = true;
		ctl.setYawGuard(guard);
		PolicyOutput o = ctl.step(s, 0.f, 0.5f);
		const float *obs = ctl.lastObservation();
		const float expect = clampf(-guard.gain * atan2f(obs[policy::OBS_HEADING + 1], obs[policy::OBS_HEADING])
					    / policy::MAX_RATE_RAD_S, -1.f, 1.f);
		CHECK(o.yaw_guarded && same(o.action[2], expect));
		CHECK(same(o.action[0], o.policy_action[0]) && same(o.action[1], o.policy_action[1])
		      && same(o.action[3], o.policy_action[3]));
		ctl.step(s, 0.f, 0.5f);
		CHECK(same(ctl.lastObservation()[policy::OBS_LAST_ACTION + 2], expect));
		o = ctl.step(s, 10.f, 0.5f);  // fallback: its own heading hold, not flagged as guarded
		CHECK(o.fallback_active && !o.yaw_guarded);
		guard.enabled = false;
		ctl.setYawGuard(guard);
		o = ctl.step(s, 0.f, 0.5f);
		CHECK(!o.yaw_guarded && same(o.action[2], o.policy_action[2]));  // explicit off: the policy's yaw flies
	}

	// 9. Parameter resolution: -1 / 0 take the defaults policy.json declared for this policy
	//    (PolicyWeights.hpp), anything else overrides them in the field.
	{
		CHECK(resolveYawGuard(PARAM_AUTO) == !policy::YAW_FROM_POLICY);
		CHECK(!resolveYawGuard(0) && resolveYawGuard(1));
		CHECK(resolveFallback(PARAM_AUTO) == policy::FALLBACK_DEFAULT_ON);
		CHECK(!resolveFallback(0) && resolveFallback(1) && resolveFallback(2));
		CHECK(same(resolveRateMax(0.f), policy::MAX_RATE_RAD_S));
		CHECK(same(resolveRateMax(1.5f), 1.5f));
		const YawGuard g{};
		const FallbackRule r{};
		CHECK(g.enabled == !policy::YAW_FROM_POLICY && r.enabled == policy::FALLBACK_DEFAULT_ON);
		// v3 ships with its own heading hold and without the brake-and-hover rule.
		CHECK(policy::YAW_FROM_POLICY && !policy::FALLBACK_DEFAULT_ON);
		CHECK(same(SetpointLimits{}.rate_max, policy::MAX_RATE_RAD_S));
	}

	// 7. Fallback physics: moving along the wall at 1.5 m/s, level, yawed 10 deg right.
	{
		FallbackRule rule{};
		StructureFrame f{};
		const float hold[3] = {0.f, 0.f, -2.f};
		f.set(hold, 0.f);  // structure x = north
		VehicleSample s{};
		s.vel_ned[1] = 1.5f;  // east = structure -y (right)
		const float yaw = 0.1745f;
		s.R[0][0] = cosf(yaw); s.R[0][1] = -sinf(yaw);
		s.R[1][0] = sinf(yaw); s.R[1][1] = cosf(yaw);
		float obs[policy::NUM_OBS], a[4];
		const float zero[4] = {0.f, 0.f, 0.f, 0.f};
		buildObservation(f, s, zero, obs);
		CHECK(near(obs[policy::OBS_VEL + 1], -1.5f, 1e-6f));
		fallbackAction(rule, f, s, obs, a);
		// Brake: accelerate toward +y (left), so roll left: negative roll rate in FLU, with a
		// small pitch share from the 10 deg yaw.
		CHECK(a[0] < -0.05f && std::fabs(a[1]) < std::fabs(a[0]));
		// Yawed right in NED = negative yaw in FLU terms; the rule turns back left (positive).
		CHECK(a[2] > 0.f);
		// Thrust: the vehicle is level, so the demanded vertical force is exactly the weight.
		CHECK(near(a[3], 2.f * policy::MASS_KG * GRAVITY_MPS2 / policy::MAX_THRUST_N - 1.f, 1e-5f));
	}

	if (failures == 0) {
		std::printf("test_policy: all checks passed\n");
	}

	return failures == 0 ? 0 : 1;
}
