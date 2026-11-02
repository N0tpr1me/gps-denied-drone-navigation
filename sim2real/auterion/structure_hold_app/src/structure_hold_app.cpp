/**
 * Structure Hold as an external flight mode: the ROS 2 node of the AuterionOS app variant.
 *
 * Registers "Structure Hold" with PX4's commander through the PX4 ROS 2 Interface Library
 * (px4_ros2, the library the Auterion SDK's Flight Mode API is built on), so on an
 * Auterion vehicle it shows up in Auterion Mission Control like a built-in mode, and on
 * stock PX4 in QGroundControl. PX4's failsafes and arming checks apply to it, and if this
 * process stops, commander's external-mode timeout hands the vehicle back by itself.
 *
 * Every control decision is HoldCore's (include/structure_hold_app/HoldCore.hpp), which
 * runs the onboard module's own policy, watchdog and handback order. This file only moves
 * data between ROS 2 and HoldCore: PX4 state in, body rates + thrust and mode requests out.
 *
 * Settings are ROS 2 parameters named after the module's SH_ parameters (lower case, no
 * prefix), ranged by the same ParamLimits.hpp: `ros2 param set /structure_hold wd_est_to 5.0`.
 */

#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/rates.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/utils/message_version.hpp>
#include <px4_ros2/vehicle_state/land_detected.hpp>
#include <px4_ros2/vehicle_state/vehicle_status.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <memory>
#include <string>

#include "structure_hold_app/HoldCore.hpp"

using structure_hold::HandbackTarget;
using structure_hold::Trip;
using structure_hold::Watchdog;
using structure_hold_app::AppInputs;
using structure_hold_app::AppOutput;
using structure_hold_app::AppParams;
using structure_hold_app::AppState;
using structure_hold_app::HoldCore;

static const std::string kModeName = "Structure Hold";
static const std::string kNodeName = "structure_hold";

// PX4 custom modes for VEHICLE_CMD_DO_SET_MODE (src/modules/commander/px4_custom_mode.h).
static constexpr float MAIN_MODE_ALTCTL = 2.f;
static constexpr float MAIN_MODE_POSCTL = 3.f;
static constexpr float MAIN_MODE_AUTO = 4.f;
static constexpr float SUB_MODE_AUTO_LOITER = 3.f;
static constexpr float SUB_MODE_AUTO_LAND = 6.f;

class StructureHoldMode : public px4_ros2::ModeBase
{
public:
	explicit StructureHoldMode(rclcpp::Node &node)
		: ModeBase(node, Settings{kModeName}.preventArming(true)), _node(node)
	{
		_rates_sp = std::make_shared<px4_ros2::RatesSetpointType>(*this);
		_local_position = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
		_attitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);
		_vehicle_status = std::make_shared<px4_ros2::VehicleStatus>(*this);
		_land_detected = std::make_shared<px4_ros2::LandDetected>(*this);

		// The VIO's own fix, where the VIO publishes it over DDS: lets the stale and frozen
		// checks see the camera pipeline, not only EKF2's output (fix_src, as SH_FIX_SRC).
		_odometry_sub = node.create_subscription<px4_msgs::msg::VehicleOdometry>(
			"fmu/in/vehicle_visual_odometry" + px4_ros2::getMessageNameVersion<px4_msgs::msg::VehicleOdometry>(),
			rclcpp::QoS(1).best_effort(), [this](px4_msgs::msg::VehicleOdometry::UniquePtr msg) {
				const float p[3] = {msg->position[0], msg->position[1], msg->position[2]};
				_core.noteOdometry(nowUs(), msg->timestamp_sample, p);
			});

		// Body rates from EKF2's odometry: PX4 v1.17.0's default DDS topic list exports
		// vehicle_odometry but not vehicle_angular_velocity (commented out in dds_topics.yaml).
		_vehicle_odometry_sub = node.create_subscription<px4_msgs::msg::VehicleOdometry>(
			"fmu/out/vehicle_odometry" + px4_ros2::getMessageNameVersion<px4_msgs::msg::VehicleOdometry>(),
			rclcpp::QoS(1).best_effort(), [this](px4_msgs::msg::VehicleOdometry::UniquePtr msg) {
				for (int i = 0; i < 3; i++) { _rates_frd[i] = msg->angular_velocity[i]; }

				_rates_us = nowUs();
			});

		_command_pub = node.create_publisher<px4_msgs::msg::VehicleCommand>(
			"fmu/in/vehicle_command" + px4_ros2::getMessageNameVersion<px4_msgs::msg::VehicleCommand>(), 1);

		_local_position->onUpdate([this](const px4_msgs::msg::VehicleLocalPosition &) { _lpos_us = nowUs(); });

		// A disarm clears the trip latch, as in the module, whichever mode the vehicle
		// landed in (the handback usually ends in Land, not in this mode).
		_vehicle_status->onUpdate([this](const px4_msgs::msg::VehicleStatus &st) {
			const bool armed = st.arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED;

			if (_was_armed && !armed) {
				if (_core.latched() != Trip::None) {
					RCLCPP_INFO(_node.get_logger(), "disarmed: trip latch cleared");
				}

				_core.disarmed();
			}

			_was_armed = armed;
		});

		declareParameters();
		_param_cb = node.add_on_set_parameters_callback(
		[this](const std::vector<rclcpp::Parameter> &) {
			rcl_interfaces::msg::SetParametersResult r;
			r.successful = true;
			_params_dirty = true;  // applied on the next tick, after the new values are stored
			return r;
		});
		applyParameters();

		// The policy was trained at 100 Hz (envs/structure_hold), as the module's tick.
		setSetpointUpdateRate(100.f);
	}

	void checkArmingAndRunConditions(px4_ros2::HealthAndArmingCheckReporter &reporter) override
	{
		if (_core.state() != AppState::Standby) {
			return;
		}

		if (!haveState()) {
			reporter.armingCheckFailureExt(px4_ros2::events::ID("structure_hold_no_state"),
						       px4_ros2::events::Log::Error, "Structure Hold: no vehicle state");
			return;
		}

		const Trip why = _core.entryRefusal(inputs());

		if (why != Trip::None) {
			reporter.armingCheckFailureExt(px4_ros2::events::ID("structure_hold_refused"),
						       px4_ros2::events::Log::Error, "Structure Hold unavailable");
			logRefusal(why);
		}
	}

	void onActivate() override
	{
		if (!haveState()) {
			// The arming check refuses entry without state, so this is a race at most;
			// stay in Standby and let commander's setpoint timeout hand back.
			RCLCPP_ERROR(_node.get_logger(), "engaged without vehicle state, not flying");
			return;
		}

		const AppInputs in = inputs();
		_core.activate(in);
		RCLCPP_INFO(_node.get_logger(), "engaged: hold %.2f %.2f %.2f NED, heading %.1f deg", static_cast<double>(in.pos_ned[0]),
			    static_cast<double>(in.pos_ned[1]), static_cast<double>(in.pos_ned[2]),
			    static_cast<double>(in.heading * 57.2958f));
	}

	void onDeactivate() override
	{
		_core.deactivate();
		RCLCPP_INFO(_node.get_logger(), "stock PX4 in charge");
	}

	void updateSetpoint(float /*dt_s*/) override
	{
		if (_params_dirty) {
			applyParameters();
		}

		if (!haveState()) {
			return;
		}

		AppInputs in = inputs();
		in.test_fault = _node.get_parameter("test_fault").as_int() != 0;
		const AppOutput out = _core.update(in);

		if (out.trip != Trip::None) {
			RCLCPP_ERROR(_node.get_logger(), "watchdog trip: %s (%s), handing back", Watchdog::name(out.trip),
				     Watchdog::reason(out.trip));

			if (in.test_fault) {  // one-shot test hook, as SH_TEST_FAULT
				_node.set_parameter(rclcpp::Parameter("test_fault", 0));
			}
		}

		if (out.fallback_changed) {
			RCLCPP_WARN(_node.get_logger(), "%s",
				    out.fallback_active ? "fallback, braking on a stale estimate" : "fresh fix, policy flies again");
		}

		if (!out.policy_ok) {
			RCLCPP_ERROR(_node.get_logger(), "policy produced a non-finite action");
		}

		if (out.request_mode) {
			requestMode(out.target, out.last_target);
		}

		if (out.state == AppState::Standby) {
			return;  // disarmed: commander leaves the mode by itself
		}

		const Eigen::Vector3f rates{out.setpoint.rates_frd[0], out.setpoint.rates_frd[1], out.setpoint.rates_frd[2]};
		const Eigen::Vector3f thrust{0.f, 0.f, -out.setpoint.thrust};
		_rates_sp->update(rates, thrust);

		if (++_status_ticks >= 100) {  // once a second
			_status_ticks = 0;
			RCLCPP_INFO(_node.get_logger(), "%s: hold error %.2f m, estimate age %.2f s%s",
				    out.state == AppState::Engaged ? "holding" : "handing back",
				    static_cast<double>(out.hold_error_m), static_cast<double>(out.stale_s),
				    out.fallback_active ? ", fallback" : "");
		}
	}

private:
	uint64_t nowUs() const { return static_cast<uint64_t>(_node.get_clock()->now().nanoseconds() / 1000); }

	bool haveState() const
	{
		return _local_position->lastValid(std::chrono::seconds(30)) && _attitude->lastValid()
		       && _rates_us > 0 && nowUs() - _rates_us < 500000 && _vehicle_status->lastValid(std::chrono::seconds(5));
	}

	AppInputs inputs() const
	{
		AppInputs in{};
		in.now_us = nowUs();
		in.lpos_us = _lpos_us;
		const Eigen::Vector3f p = _local_position->positionNed();
		const Eigen::Vector3f v = _local_position->velocityNed();

		for (int i = 0; i < 3; i++) {
			in.pos_ned[i] = p(i);
			in.vel_ned[i] = v(i);
		}

		const px4_msgs::msg::VehicleLocalPosition &lp = _local_position->last();
		in.xy_valid = lp.xy_valid;
		in.z_valid = lp.z_valid;
		in.v_xy_valid = lp.v_xy_valid;
		in.heading = lp.heading;
		const Eigen::Quaternionf q = _attitude->attitude();
		in.q[0] = q.w();
		in.q[1] = q.x();
		in.q[2] = q.y();
		in.q[3] = q.z();
		for (int i = 0; i < 3; i++) { in.rates_frd[i] = _rates_frd[i]; }

		in.armed = _vehicle_status->armed();
		// No land detector message yet counts as landed: the conservative side.
		in.landed = !_land_detected->lastValid(std::chrono::seconds(5)) || _land_detected->landed();
		return in;
	}

	void requestMode(HandbackTarget target, bool last)
	{
		px4_msgs::msg::VehicleCommand cmd{};
		cmd.timestamp = 0;  // PX4 stamps it on arrival
		cmd.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE;
		cmd.param1 = 1.f;  // custom mode
		cmd.param2 = MAIN_MODE_POSCTL;
		cmd.param3 = 0.f;

		switch (target) {
		case HandbackTarget::Altitude: cmd.param2 = MAIN_MODE_ALTCTL; break;

		case HandbackTarget::Hold:
			cmd.param2 = MAIN_MODE_AUTO;
			cmd.param3 = SUB_MODE_AUTO_LOITER;
			break;

		case HandbackTarget::StockLand:
		case HandbackTarget::Land:
			cmd.param2 = MAIN_MODE_AUTO;
			cmd.param3 = SUB_MODE_AUTO_LAND;
			break;

		default: break;  // Position (Structure Hover is not in the app yet)
		}

		const px4_msgs::msg::VehicleStatus &st = _vehicle_status->last();
		cmd.target_system = st.system_id;
		cmd.target_component = st.component_id;
		cmd.source_system = st.system_id;
		cmd.source_component = 191;  // MAV_COMP_ID_ONBOARD_COMPUTER
		cmd.from_external = true;
		_command_pub->publish(cmd);

		if (!last || !_logged_last) {
			RCLCPP_WARN(_node.get_logger(), "handing back to %s%s", structure_hold::handbackTargetName(target),
				    last ? " (repeated until it takes)" : "");
		}

		_logged_last = last;
	}

	void logRefusal(Trip why)
	{
		const uint64_t now = nowUs();

		if (why != _last_refusal || now - _last_refusal_log_us > 5000000) {
			RCLCPP_WARN(_node.get_logger(), "engage refused: %s (%s)", Watchdog::name(why), Watchdog::reason(why));
			_last_refusal = why;
			_last_refusal_log_us = now;
		}
	}

	void declareParameters()
	{
		const AppParams d{};
		_node.declare_parameter("wd_est_to", static_cast<double>(d.wd_est_to_s));
		_node.declare_parameter("wd_pos_err", static_cast<double>(d.wd_pos_err_m));
		_node.declare_parameter("wd_tilt", static_cast<double>(d.wd_tilt_deg));
		_node.declare_parameter("wd_frz_t", static_cast<double>(d.wd_frz_t_s));
		_node.declare_parameter("vel_src", static_cast<int64_t>(d.vel_src));
		_node.declare_parameter("fb_en", static_cast<int64_t>(d.fb_en));
		_node.declare_parameter("fb_t", static_cast<double>(d.fb_t_s));
		_node.declare_parameter("fix_src", static_cast<int64_t>(d.fix_src));
		_node.declare_parameter("fix_gap", static_cast<double>(d.fix_gap_s));
		_node.declare_parameter("yaw_guard", static_cast<int64_t>(d.yaw_guard));
		_node.declare_parameter("yaw_p", static_cast<double>(d.yaw_p));
		_node.declare_parameter("rate_max", static_cast<double>(d.rate_max));
		_node.declare_parameter("fb_k", static_cast<double>(d.fb_k));
		_node.declare_parameter("fb_tilt", static_cast<double>(d.fb_tilt));
		_node.declare_parameter("hover_thrust", static_cast<double>(d.hover_thrust));
		_node.declare_parameter("test_fault", static_cast<int64_t>(0));
	}

	void applyParameters()
	{
		_params_dirty = false;
		AppParams p{};
		auto f = [this](const char *name) { return static_cast<float>(_node.get_parameter(name).as_double()); };
		auto i = [this](const char *name) { return static_cast<int32_t>(_node.get_parameter(name).as_int()); };
		p.wd_est_to_s = f("wd_est_to");
		p.wd_pos_err_m = f("wd_pos_err");
		p.wd_tilt_deg = f("wd_tilt");
		p.wd_frz_t_s = f("wd_frz_t");
		p.vel_src = i("vel_src");
		p.fb_en = i("fb_en");
		p.fb_t_s = f("fb_t");
		p.fix_src = i("fix_src");
		p.fix_gap_s = f("fix_gap");
		p.yaw_guard = i("yaw_guard");
		p.yaw_p = f("yaw_p");
		p.rate_max = f("rate_max");
		p.fb_k = f("fb_k");
		p.fb_tilt = f("fb_tilt");
		p.hover_thrust = f("hover_thrust");
		const int changed = _core.configure(p);

		if (changed > 0) {
			RCLCPP_WARN(_node.get_logger(), "%d setting(s) outside their range were clamped or reset", changed);
		}
	}

	rclcpp::Node &_node;
	HoldCore _core{};
	std::shared_ptr<px4_ros2::RatesSetpointType> _rates_sp;
	std::shared_ptr<px4_ros2::OdometryLocalPosition> _local_position;
	std::shared_ptr<px4_ros2::OdometryAttitude> _attitude;
	std::shared_ptr<px4_ros2::VehicleStatus> _vehicle_status;
	std::shared_ptr<px4_ros2::LandDetected> _land_detected;
	rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _odometry_sub;
	rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _vehicle_odometry_sub;
	float _rates_frd[3] {0.f, 0.f, 0.f};
	uint64_t _rates_us{0};
	rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr _command_pub;
	rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr _param_cb;
	uint64_t _lpos_us{0};
	bool _params_dirty{false};
	bool _was_armed{false};
	bool _logged_last{false};
	int _status_ticks{0};
	Trip _last_refusal{Trip::None};
	uint64_t _last_refusal_log_us{0};
};

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	// The constructor waits for PX4 and registers the mode (it throws if commander refuses).
	auto node = std::make_shared<px4_ros2::NodeWithMode<StructureHoldMode>>(kNodeName, false);
	RCLCPP_INFO(node->get_logger(), "%s registered with PX4", kModeName.c_str());
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
