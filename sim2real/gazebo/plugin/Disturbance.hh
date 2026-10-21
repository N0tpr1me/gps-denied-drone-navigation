// Wind and near-wall disturbance model, C++ copy of envs/disturbance/core.py.
//
// The Python module is the source of truth. This header exists only so the Gazebo plugin
// can evaluate the force inside every physics step without a Python round trip. It must
// stay a line-for-line translation of ``_wind``, ``_shelter``, ``_drag`` and ``_force`` in
// core.py; sim2real/gazebo/parity.py (run by pytest wherever a C++ compiler exists, and
// inside the SITL image) checks the two agree to 1e-9 N on random inputs.
//
// Parameters come from the text file written by sim2real/gazebo/params_io.py for one env:
//   structure-hold-wind-params v1
//   <name> <count> <value> ...        one field per line, row-major for 2-D fields
//
// Frames and units follow docs/conventions.md: world ENU, metres, seconds, newtons.
#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace structure_hold
{

using Vec3 = std::array<double, 3>;

inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 operator*(double s, const Vec3 &a) { return {s * a[0], s * a[1], s * a[2]}; }
inline double dot(const Vec3 &a, const Vec3 &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

struct Params
{
  Vec3 mean_wind{0.0, 0.0, 0.0};
  std::vector<double> gust_start;     // K, s
  std::vector<double> gust_duration;  // K, s
  std::vector<Vec3> gust_wind;        // K, m/s
  std::array<std::vector<double>, 3> turb_amp;    // 3 x M, m/s
  std::vector<double> turb_omega;                 // M, rad/s
  std::array<std::vector<double>, 3> turb_phase;  // 3 x M, rad
  double k_lin{0.0};
  double k_quad{0.0};
  Vec3 wall_point{0.0, 0.0, 0.0};
  Vec3 wall_normal{1.0, 0.0, 0.0};  // unit, from the surface into the free air
  bool has_wall{false};
  double wall_decay{1.0};
  double wall_floor{0.0};
};

namespace detail
{
inline std::map<std::string, std::vector<double>> ReadFields(const std::string &path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open wind params file: " + path);
  std::string line;
  if (!std::getline(in, line) || line != "structure-hold-wind-params v1")
    throw std::runtime_error("unexpected wind params header in " + path + ": '" + line + "'");
  std::map<std::string, std::vector<double>> fields;
  while (std::getline(in, line))
  {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    std::string name;
    std::size_t count = 0;
    ss >> name >> count;
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i)
    {
      if (!(ss >> values[i]))
        throw std::runtime_error("short field '" + name + "' in " + path);
    }
    fields[name] = std::move(values);
  }
  return fields;
}

inline const std::vector<double> &Field(
    const std::map<std::string, std::vector<double>> &fields, const std::string &name,
    std::size_t expected = static_cast<std::size_t>(-1))
{
  auto it = fields.find(name);
  if (it == fields.end())
    throw std::runtime_error("wind params missing field '" + name + "'");
  if (expected != static_cast<std::size_t>(-1) && it->second.size() != expected)
    throw std::runtime_error("wind params field '" + name + "' has the wrong length");
  return it->second;
}

inline Vec3 ToVec3(const std::vector<double> &v, std::size_t offset = 0)
{
  return {v.at(offset), v.at(offset + 1), v.at(offset + 2)};
}
}  // namespace detail

/// Load one env's parameters from the params_io text format.
inline Params LoadParams(const std::string &path)
{
  const auto fields = detail::ReadFields(path);
  Params p;
  p.mean_wind = detail::ToVec3(detail::Field(fields, "mean_wind", 3));
  p.gust_start = detail::Field(fields, "gust_start");
  const std::size_t k = p.gust_start.size();
  p.gust_duration = detail::Field(fields, "gust_duration", k);
  const auto &gw = detail::Field(fields, "gust_wind", 3 * k);
  for (std::size_t i = 0; i < k; ++i)
    p.gust_wind.push_back(detail::ToVec3(gw, 3 * i));
  p.turb_omega = detail::Field(fields, "turb_omega");
  const std::size_t m = p.turb_omega.size();
  const auto &amp = detail::Field(fields, "turb_amp", 3 * m);
  const auto &phase = detail::Field(fields, "turb_phase", 3 * m);
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    p.turb_amp[axis].assign(amp.begin() + axis * m, amp.begin() + (axis + 1) * m);
    p.turb_phase[axis].assign(phase.begin() + axis * m, phase.begin() + (axis + 1) * m);
  }
  p.k_lin = detail::Field(fields, "k_lin", 1)[0];
  p.k_quad = detail::Field(fields, "k_quad", 1)[0];
  p.wall_point = detail::ToVec3(detail::Field(fields, "wall_point", 3));
  p.wall_normal = detail::ToVec3(detail::Field(fields, "wall_normal", 3));
  p.has_wall = detail::Field(fields, "has_wall", 1)[0] != 0.0;
  p.wall_decay = detail::Field(fields, "wall_decay", 1)[0];
  p.wall_floor = detail::Field(fields, "wall_floor", 1)[0];
  return p;
}

/// Free-stream wind (m/s) at time t (s since episode start): mean + gusts + turbulence.
inline Vec3 Wind(const Params &p, double t)
{
  Vec3 w = p.mean_wind;
  for (std::size_t i = 0; i < p.gust_start.size(); ++i)
  {
    if (t >= p.gust_start[i] && t < p.gust_start[i] + p.gust_duration[i])
      w = w + p.gust_wind[i];
  }
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    double turb = 0.0;
    for (std::size_t m = 0; m < p.turb_omega.size(); ++m)
      turb += p.turb_amp[axis][m] * std::cos(p.turb_omega[m] * t + p.turb_phase[axis][m]);
    w[axis] += turb;
  }
  return w;
}

/// Fraction of the free-stream wind that reaches ``pos``: 1 in free air.
inline double Shelter(const Params &p, const Vec3 &pos)
{
  if (!p.has_wall)
    return 1.0;
  const double d = std::max(dot(pos - p.wall_point, p.wall_normal), 0.0);
  return 1.0 - (1.0 - p.wall_floor) * std::exp(-d / p.wall_decay);
}

/// Drag (N) on air velocity ``v_rel`` relative to the drone.
inline Vec3 Drag(const Vec3 &v_rel, double k_lin, double k_quad)
{
  const double speed = std::sqrt(dot(v_rel, v_rel));
  return (k_lin + k_quad * speed) * v_rel;
}

/// Aerodynamic force (N, world ENU) on a drone at ``pos`` moving at ``vel`` at time ``t``.
/// Includes the drag of the drone's own motion through still air, like the Python model.
inline Vec3 Force(const Params &p, double t, const Vec3 &pos, const Vec3 &vel)
{
  const Vec3 air = Shelter(p, pos) * Wind(p, t);
  return Drag(air - vel, p.k_lin, p.k_quad);
}

/// The part of ``Force`` a simulator with its own body drag should add: the full force minus
/// what the model would give in still air, so the drone's own drag is not counted twice.
inline Vec3 AppliedForce(const Params &p, double t, const Vec3 &pos, const Vec3 &vel,
                         bool subtract_still_air_drag)
{
  Vec3 f = Force(p, t, pos, vel);
  if (subtract_still_air_drag)
    f = f - Drag(-1.0 * vel, p.k_lin, p.k_quad);
  return f;
}

}  // namespace structure_hold
