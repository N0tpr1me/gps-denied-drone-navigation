# Structure Hold module status, published every control tick while engaged and at a low
# rate otherwise, so sim, SITL and real flights are analysed by the same script.

uint64 timestamp              # time since system start (microseconds)

uint8 STATE_UNREGISTERED = 0  # mode registration with commander pending or failed
uint8 STATE_STANDBY = 1       # registered, another mode is flying
uint8 STATE_ENGAGED = 2       # Structure Hold is the active mode and commands rates
uint8 STATE_HANDBACK = 3      # watchdog tripped, control handed back to stock PX4
uint8 STATE_LANDING = 4       # Structure Land is the active mode: PX4 descends on the guarded xy setpoint
uint8 STATE_HOVER = 5         # Structure Hover is the active mode: PX4 hovers on the guarded setpoint
uint8 state

uint8 TRIP_NONE = 0
uint8 TRIP_TEST_FAULT = 1
uint8 TRIP_ESTIMATE_INVALID = 2
uint8 TRIP_ESTIMATE_STALE = 3
uint8 TRIP_TILT = 4
uint8 TRIP_POSITION_ERROR = 5
uint8 TRIP_ESTIMATE_RESET = 6 # entry refused: EKF reset its xy position within SH_RST_HOLD (not latched)
uint8 TRIP_LANDED = 7         # entry refused: the land detector says the vehicle is on the ground (not latched)
uint8 trip_reason             # TRIP_*: last watchdog trip or refusal; cleared on disarm
uint8 trip_count              # trips since boot

int8 mode_id                  # nav_state assigned by commander, -1 if unregistered
int8 arming_check_id          # arming check registration id, -1 if unregistered

float32[3] setpoint_ned       # [m] held position, NED
float32 yaw_setpoint          # [rad]
float32[3] position_error_ned # [m] setpoint - estimate
float32 estimate_age_s        # [s] age of the sample the position estimate is based on (last fix)
float32 stale_s               # [s] time since the last fresh position fix (docs/fallback.md)
float32 reset_age_s           # [s] since the EKF last reset its xy position, -1 if never

uint8 CTRL_CASCADE = 0        # placeholder P cascade (SH_CTRL 0)
uint8 CTRL_POLICY = 1         # policy v2 with the brake-and-hover fallback (SH_CTRL 1)
uint8 controller
bool fallback_active          # the brake-and-hover fallback flew this tick
bool yaw_guarded              # the yaw action flown is the module's heading hold (SH_YAW_GUARD)
bool odometry_fix             # fixes counted from vehicle_visual_odometry (else local position)
float32 wall_distance_m       # [m] forward rangefinder, perpendicular to the wall (WallGuard.hpp); -1 without a usable sample
float32 hold_clearance_m      # [m] hold point to the measured wall along the structure x axis; -1 without a sample
float32 standoff_shift_m      # [m] how far the wall guard has moved the hold point back since engage
float32[4] action             # policy-convention action flown (FLU rates / max_rate, thrust in [-1, 1])

float32[3] rates_setpoint     # [rad/s] FRD body rates commanded this tick
float32 thrust_setpoint       # normalised collective thrust commanded this tick
uint32 loop_time_us           # [us] wall time of the control step, policy inference included
