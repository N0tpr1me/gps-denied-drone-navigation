// Evaluates Disturbance.hh on samples from stdin, for the Python parity test.
//
//   disturbance_parity <params file>  <  "t x y z vx vy vz" per line
//
// Prints per line: wind xyz, shelter, full force xyz, applied force xyz (still-air drag
// subtracted), with 17 significant digits. No Gazebo dependency, so plain g++ builds it.
#include <cstdio>
#include <iostream>

#include "Disturbance.hh"

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::fprintf(stderr, "usage: %s <params file>\n", argv[0]);
    return 2;
  }
  structure_hold::Params p;
  try
  {
    p = structure_hold::LoadParams(argv[1]);
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  double t;
  structure_hold::Vec3 pos, vel;
  while (std::cin >> t >> pos[0] >> pos[1] >> pos[2] >> vel[0] >> vel[1] >> vel[2])
  {
    const auto w = structure_hold::Wind(p, t);
    const double g = structure_hold::Shelter(p, pos);
    const auto f = structure_hold::Force(p, t, pos, vel);
    const auto a = structure_hold::AppliedForce(p, t, pos, vel, true);
    std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                w[0], w[1], w[2], g, f[0], f[1], f[2], a[0], a[1], a[2]);
  }
  return 0;
}
