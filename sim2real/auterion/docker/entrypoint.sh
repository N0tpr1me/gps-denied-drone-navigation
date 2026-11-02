#!/usr/bin/env bash
# app: the Micro XRCE-DDS Agent (UDP 8888, PX4 SITL's default) and the Structure Hold app.
# Extra arguments go to the app as ROS 2 parameters, e.g. -p vel_src:=1.
set -e
# Line-buffered log output, so `docker logs` shows it as it happens.
export RCUTILS_LOGGING_BUFFERED_STREAM=1
. /opt/ros/humble/setup.sh
. /ws/install/setup.sh
case "${1:-app}" in
  app)
    shift || true
    MicroXRCEAgent udp4 -p 8888 >/tmp/agent.log 2>&1 &
    exec ros2 run structure_hold_app structure_hold_app --ros-args "$@"
    ;;
  *) exec "$@" ;;
esac
