// Run OpenVINS offline on one flight in EuRoC's ASL layout and write what eval/vio_fit.py
// reads (docs/vio-fit.md, "Third fit"):
//
//   ov_run_euroc <estimator_config.yaml> <folder holding mav0/> <out dir>
//
//   <out>/stamped_traj_estimate.txt   t x y z qx qy qz qw vx vy vz, seconds, IMU in the
//                                     VIO's world frame, one line per stereo frame once
//                                     OpenVINS has initialised
//   <out>/frame_time_ms.txt           wall time of each feed_measurement_camera call (ms)
//
// Measurements go in in time order, the IMU first when stamps tie, exactly as OpenVINS's
// ROS serial runner does with a bag. Images are read as 8-bit grayscale; the masks are
// the config's (TUM-VI) or empty. No ROS, no visualisation.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace {

struct Frame {
  long long ns;
  std::string name;
};

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) out.push_back(cell);
  return out;
}

std::vector<Frame> read_frames(const std::string &csv) {
  std::vector<Frame> frames;
  std::ifstream in(csv);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto cells = split_csv(line);
    if (cells.size() < 2) continue;
    std::string name = cells[1];
    while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
    frames.push_back({std::stoll(cells[0]), name});
  }
  return frames;
}

std::vector<ov_core::ImuData> read_imu(const std::string &csv) {
  std::vector<ov_core::ImuData> imu;
  std::ifstream in(csv);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto c = split_csv(line);
    if (c.size() < 7) continue;
    ov_core::ImuData m;
    m.timestamp = std::stoll(c[0]) * 1e-9;
    m.wm << std::stod(c[1]), std::stod(c[2]), std::stod(c[3]);
    m.am << std::stod(c[4]), std::stod(c[5]), std::stod(c[6]);
    imu.push_back(m);
  }
  return imu;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: ov_run_euroc <estimator_config.yaml> <folder holding mav0/> <out dir>\n";
    return 2;
  }
  const std::string config = argv[1], data = std::string(argv[2]) + "/mav0", out = argv[3];

  auto parser = std::make_shared<ov_core::YamlParser>(config);
  std::string verbosity = "WARNING";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);
  ov_msckf::VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = false;  // feed in order, one measurement at a time
  if (!parser->successful()) {
    std::cerr << "unable to parse all parameters in " << config << "\n";
    return 1;
  }
  auto sys = std::make_shared<ov_msckf::VioManager>(params);

  auto left = read_frames(data + "/cam0/data.csv");
  auto right = read_frames(data + "/cam1/data.csv");
  auto imu = read_imu(data + "/imu0/data.csv");
  if (left.empty() || right.size() != left.size() || imu.empty()) {
    std::cerr << "expected matching cam0/cam1 lists and an IMU list under " << data << "\n";
    return 1;
  }

  std::FILE *traj = std::fopen((out + "/stamped_traj_estimate.txt").c_str(), "w");
  std::FILE *timing = std::fopen((out + "/frame_time_ms.txt").c_str(), "w");
  if (!traj || !timing) {
    std::cerr << "cannot write to " << out << "\n";
    return 1;
  }
  std::fprintf(traj, "# t x y z qx qy qz qw vx vy vz\n");

  size_t k = 0;
  int written = 0;
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i].ns != right[i].ns) continue;  // only synchronised stereo pairs
    const double t = left[i].ns * 1e-9;
    // IMU up to a little past the image, so propagation to the image time has both ends.
    while (k < imu.size() && imu[k].timestamp <= t + 0.05) sys->feed_measurement_imu(imu[k++]);

    ov_core::CameraData msg;
    msg.timestamp = t;
    msg.sensor_ids = {0, 1};
    msg.images = {cv::imread(data + "/cam0/data/" + left[i].name, cv::IMREAD_GRAYSCALE),
                  cv::imread(data + "/cam1/data/" + right[i].name, cv::IMREAD_GRAYSCALE)};
    if (msg.images[0].empty() || msg.images[1].empty()) {
      std::cerr << "unreadable image at " << left[i].ns << ", skipped\n";
      continue;
    }
    for (int c = 0; c < 2; ++c) {
      msg.masks.push_back(params.use_mask
                              ? params.masks.at(c)
                              : cv::Mat::zeros(msg.images[c].rows, msg.images[c].cols, CV_8UC1));
    }
    auto t0 = std::chrono::steady_clock::now();
    sys->feed_measurement_camera(msg);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
    std::fprintf(timing, "%.3f\n", ms);

    if (!sys->initialized()) continue;
    auto state = sys->get_state();
    // OpenVINS keeps JPL q_GtoI, which has the same components as Hamilton q_ItoG (x y z w).
    Eigen::Vector4d q = state->_imu->quat();
    Eigen::Vector3d p = state->_imu->pos(), v = state->_imu->vel();
    // The state is stamped on the camera clock; the IMU clock (and the truth's) is
    // dt_CAMtoIMU later, as OpenVINS's own ROS publisher does.
    const double t_imu = state->_timestamp + state->_calib_dt_CAMtoIMU->value()(0);
    std::fprintf(traj, "%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f %.6f %.6f %.6f\n",
                 t_imu, p(0), p(1), p(2), q(0), q(1), q(2), q(3), v(0), v(1), v(2));
    ++written;
  }
  std::fclose(traj);
  std::fclose(timing);
  std::printf("%d poses from %zu frames\n", written, left.size());
  return 0;
}
