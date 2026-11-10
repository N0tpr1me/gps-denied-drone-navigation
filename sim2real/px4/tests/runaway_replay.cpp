// Replays a stream of IMU and fix samples through Watchdog.hpp's RunawayFix, for the parity
// test with eval/vio_runaway.py (eval/tests/test_vio_runaway.py builds and runs it). One
// sample per input line:
//   i <t_us> <ax> <ay> <az>                 an IMU sample (local position ax, ay, az)
//   f <t_us> <vn> <ve> <vd> <reset_counter> a fix sample; prints "<disagreement> <runaway>"
// The first line is the limit in m/s.

#include "Watchdog.hpp"

#include <cstdio>

int main()
{
	structure_hold::RunawayFix f;
	double limit = 0.0;

	if (std::scanf("%lf", &limit) != 1) { return 1; }

	f.setLimit(static_cast<float>(limit));
	char kind = 0;
	unsigned long long t = 0;
	double a = 0.0, b = 0.0, c = 0.0;

	while (std::scanf(" %c %llu %lf %lf %lf", &kind, &t, &a, &b, &c) == 5) {
		if (kind == 'i') {
			f.inertial(t, static_cast<float>(a), static_cast<float>(b), static_cast<float>(c));

		} else {
			unsigned counter = 0;

			if (std::scanf("%u", &counter) != 1) { return 1; }

			f.fix(t, static_cast<float>(a), static_cast<float>(b), static_cast<float>(c),
			      static_cast<uint8_t>(counter));
			std::printf("%.6f %d\n", static_cast<double>(f.disagreement()), f.runaway() ? 1 : 0);
		}
	}

	return 0;
}
