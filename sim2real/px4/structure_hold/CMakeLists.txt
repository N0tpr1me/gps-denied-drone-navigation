############################################################################
# Structure Hold: one PX4 flight mode that holds a quadrotor near a structure.
# Built into px4_sitl_default (and later the target airframe) through install.sh,
# which links this directory in as src/modules/structure_hold.
############################################################################

px4_add_module(
	MODULE modules__structure_hold
	MAIN structure_hold
	COMPILE_FLAGS
	SRCS
		StructureHold.cpp
		StructureHold.hpp
		Watchdog.hpp
		WallGuard.hpp
		KeepOut.hpp
		Handback.hpp
		RangeHover.hpp
		ParamLimits.hpp
		PassPath.hpp
		HoldController.hpp
		PolicyController.hpp
		MlpPolicy.hpp
		PolicyWeights.hpp
	DEPENDS
		px4_work_queue
		mathlib
	)
