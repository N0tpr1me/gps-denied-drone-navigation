# Structure Hold module status, published every control tick while engaged and at a low
# rate otherwise, so sim, SITL and real flights are analysed by the same script.

uint64 timestamp              # time since system start (microseconds)

uint8 STATE_UNREGISTERED = 0  # mode registration with commander pending or failed
uint8 STATE_STANDBY = 1       # registered, another mode is flying
uint8 STATE_ENGAGED = 2       # Structure Hold is the active mode and commands rates
uint8 STATE_HANDBACK = 3      # watchdog tripped, control handed back to stock PX4
uint8 state

uint8 TRIP_NONE = 0
uint8 TRIP_TEST_FAULT = 1
uint8 TRIP_ESTIMATE_INVALID = 2
uint8 TRIP_ESTIMATE_STALE = 3
uint8 TRIP_TILT = 4
uint8 TRIP_POSITION_ERROR = 5
uint8 trip_reason             # TRIP_*: last watchdog trip, latched until disarm
uint8 trip_count              # trips since boot

int8 mode_id                  # nav_state assigned by commander, -1 if unregistered
int8 arming_check_id          # arming check registration id, -1 if unregistered

float32[3] setpoint_ned       # [m] held position, NED
float32 yaw_setpoint          # [rad]
float32[3] position_error_ned # [m] setpoint - estimate
float32 estimate_age_s        # [s] age of the local position estimate at this tick

float32[3] rates_setpoint     # [rad/s] FRD body rates commanded this tick
float32 thrust_setpoint       # normalised collective thrust commanded this tick
uint32 loop_time_us           # [us] wall time of the control step (policy inference later)
