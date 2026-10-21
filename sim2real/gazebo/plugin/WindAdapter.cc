// Gazebo world system that applies the shared disturbance force to one link.
//
// Every physics step (PreUpdate) it reads the link's world pose and velocity, evaluates
// Disturbance.hh at the current simulation time and adds the result with
// Link::AddWorldForce, so the force acts in the same step as the state it was computed
// from: zero steps of lag between "commanded" and "applied". Gazebo has its own body drag
// (the x500 rotor model), so by default the model's still-air drag is subtracted.
//
// SDF configuration (all optional):
//   <model_name>x500_0</model_name>       model to push (PX4 spawns gz_x500 as x500_0)
//   <link_name>base_link</link_name>
//   <params>/out/wind_params.txt</params> params_io file; missing file = wind off
//   <trace>/out/wind_trace.csv</trace>    per-step CSV of state, wind and applied force
//   <trace_every>1</trace_every>          log every n-th step
//   <subtract_still_air_drag>true</subtract_still_air_drag>
//   <start_time>0.0</start_time>          episode t = 0 at this sim time, s
// Environment variables STRUCTURE_HOLD_WIND_PARAMS and STRUCTURE_HOLD_WIND_TRACE override
// <params> and <trace>, so one world file serves every scenario.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/common/Console.hh>
#include <gz/math/Vector3.hh>
#include <sdf/Element.hh>

#include "Disturbance.hh"

namespace structure_hold
{

class WindAdapter : public gz::sim::System,
                    public gz::sim::ISystemConfigure,
                    public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override
  {
    auto get = [&](const std::string &key, const std::string &fallback) {
      return _sdf->HasElement(key) ? _sdf->Get<std::string>(key) : fallback;
    };
    this->modelName = get("model_name", "x500_0");
    this->linkName = get("link_name", "base_link");
    this->traceEvery = _sdf->HasElement("trace_every") ? _sdf->Get<int>("trace_every") : 1;
    this->subtractDrag = _sdf->HasElement("subtract_still_air_drag")
                             ? _sdf->Get<bool>("subtract_still_air_drag")
                             : true;
    this->startTime = _sdf->HasElement("start_time") ? _sdf->Get<double>("start_time") : 0.0;
    std::string paramsPath = get("params", "");
    std::string tracePath = get("trace", "");
    if (const char *env = std::getenv("STRUCTURE_HOLD_WIND_PARAMS"))
      paramsPath = env;
    if (const char *env = std::getenv("STRUCTURE_HOLD_WIND_TRACE"))
      tracePath = env;

    if (paramsPath.empty())
    {
      gzmsg << "[WindAdapter] no params file given: wind off" << std::endl;
      return;
    }
    try
    {
      this->params = LoadParams(paramsPath);
    }
    catch (const std::exception &e)
    {
      gzwarn << "[WindAdapter] " << e.what() << ": wind off" << std::endl;
      return;
    }
    this->enabled = true;
    gzmsg << "[WindAdapter] loaded " << paramsPath << " for " << this->modelName << "/"
          << this->linkName << " (mean wind " << this->params.mean_wind[0] << ", "
          << this->params.mean_wind[1] << ", " << this->params.mean_wind[2] << " m/s, "
          << this->params.gust_start.size() << " gusts, " << this->params.turb_omega.size()
          << " turbulence components, wall " << (this->params.has_wall ? "on" : "off")
          << ", still-air drag " << (this->subtractDrag ? "subtracted" : "kept") << ")"
          << std::endl;

    if (!tracePath.empty())
    {
      this->trace.open(tracePath);
      if (this->trace)
      {
        this->trace << "step,t,x,y,z,vx,vy,vz,wx,wy,wz,shelter,fx,fy,fz\n";
        gzmsg << "[WindAdapter] tracing every " << this->traceEvery << " step(s) to "
              << tracePath << std::endl;
      }
      else
      {
        gzwarn << "[WindAdapter] cannot open trace file " << tracePath << std::endl;
      }
    }
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (!this->enabled || _info.paused)
      return;

    if (!this->link.Valid(_ecm))
    {
      // PX4 spawns the vehicle after the world loads: look for it until it appears.
      const auto model = _ecm.EntityByComponents(gz::sim::components::Model(),
                                                 gz::sim::components::Name(this->modelName));
      if (model == gz::sim::kNullEntity)
        return;
      const auto linkEntity = gz::sim::Model(model).LinkByName(_ecm, this->linkName);
      if (linkEntity == gz::sim::kNullEntity)
      {
        gzwarn << "[WindAdapter] model " << this->modelName << " has no link "
               << this->linkName << std::endl;
        this->enabled = false;
        return;
      }
      this->link = gz::sim::Link(linkEntity);
      this->link.EnableVelocityChecks(_ecm);
      gzmsg << "[WindAdapter] attached to " << this->modelName << "/" << this->linkName
            << std::endl;
      return;  // velocity is available from the next step
    }

    const auto pose = this->link.WorldPose(_ecm);
    const auto velocity = this->link.WorldLinearVelocity(_ecm);
    if (!pose || !velocity)
      return;

    const double t = std::chrono::duration<double>(_info.simTime).count() - this->startTime;
    const Vec3 pos{pose->Pos().X(), pose->Pos().Y(), pose->Pos().Z()};
    const Vec3 vel{velocity->X(), velocity->Y(), velocity->Z()};
    const Vec3 force = AppliedForce(this->params, t, pos, vel, this->subtractDrag);
    this->link.AddWorldForce(_ecm, gz::math::Vector3d(force[0], force[1], force[2]));

    if (this->trace && this->step % this->traceEvery == 0)
    {
      const Vec3 wind = Wind(this->params, t);
      const double shelter = Shelter(this->params, pos);
      this->trace.precision(17);
      this->trace << this->step << ',' << t << ',' << pos[0] << ',' << pos[1] << ','
                  << pos[2] << ',' << vel[0] << ',' << vel[1] << ',' << vel[2] << ','
                  << wind[0] << ',' << wind[1] << ',' << wind[2] << ',' << shelter << ','
                  << force[0] << ',' << force[1] << ',' << force[2] << '\n';
    }
    ++this->step;
  }

private:
  std::string modelName;
  std::string linkName;
  Params params;
  bool enabled{false};
  bool subtractDrag{true};
  double startTime{0.0};
  int traceEvery{1};
  long step{0};
  gz::sim::Link link{gz::sim::kNullEntity};
  std::ofstream trace;
};

}  // namespace structure_hold

GZ_ADD_PLUGIN(structure_hold::WindAdapter, gz::sim::System,
              structure_hold::WindAdapter::ISystemConfigure,
              structure_hold::WindAdapter::ISystemPreUpdate)
