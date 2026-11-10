############################################################################
# Structure Hold: one PX4 flight mode that holds a quadrotor near a structure.
# Built into px4_sitl_default (and later the target airframe) through install.sh,
# which links this directory in as src/modules/structure_hold.
############################################################################

# PX4 v1.18 changed ModuleBase from a template on the module's class to a plain class with
# a per-module Descriptor. No macro or header marks the change, so read module.h and tell
# StructureHold.hpp which one this tree has (v1.16 and v1.17: the template).
file(READ ${PX4_SOURCE_DIR}/platforms/common/include/px4_platform_common/module.h sh_module_h)
set(sh_compile_flags)

if(sh_module_h MATCHES "struct Descriptor")
	list(APPEND sh_compile_flags -DSH_MODULE_DESCRIPTOR=1)
endif()

# PX4 v1.18 also stopped reading parameter definitions from C files in module folders (up
# to v1.17 structure_hold_params.c is found on its own) and takes them from a YAML module
# config only. structure_hold_params.yaml is generated from the C file
# (sim2real/px4/tools/params_yaml.py) and handed over only where the C file is not read,
# so no tree sees the parameters defined twice.
file(READ ${PX4_SOURCE_DIR}/src/lib/parameters/CMakeLists.txt sh_parameters_cmake)
set(sh_module_config)

if(NOT sh_parameters_cmake MATCHES "PX4_MODULE_PATHS")
	set(sh_module_config MODULE_CONFIG structure_hold_params.yaml)
endif()

px4_add_module(
	MODULE modules__structure_hold
	MAIN structure_hold
	COMPILE_FLAGS
		${sh_compile_flags}
	SRCS
		StructureHold.cpp
		StructureHold.hpp
		Watchdog.hpp
		WallGuard.hpp
		KeepOut.hpp
		Handback.hpp
		RangeHover.hpp
		ParamLimits.hpp
		HoldController.hpp
		PolicyController.hpp
		MlpPolicy.hpp
		PolicyWeights.hpp
	${sh_module_config}
	DEPENDS
		px4_work_queue
		mathlib
	)
