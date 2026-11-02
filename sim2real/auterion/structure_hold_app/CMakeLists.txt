cmake_minimum_required(VERSION 3.8)
project(structure_hold_app)

if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)

# The onboard module's PX4-free headers (policy, watchdog, handback, parameter ranges):
# from the repo when built in place, from module/ when staged by stage_app.sh.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/module/PolicyController.hpp)
	set(SH_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/module)
else()
	set(SH_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../px4/structure_hold)
endif()

add_executable(structure_hold_app src/structure_hold_app.cpp)
target_include_directories(structure_hold_app PRIVATE include ${SH_MODULE_DIR})
target_link_libraries(structure_hold_app Eigen3::Eigen px4_ros2_cpp::px4_ros2_cpp rclcpp::rclcpp)
ament_target_dependencies(structure_hold_app px4_msgs)

install(TARGETS structure_hold_app DESTINATION lib/${PROJECT_NAME})

ament_package()
