# Structure Hold as an AuterionOS app (docs/auterion-app.md). Built by `auterion-cli app
# build` from the folder stage_app.sh writes; `--simulation` builds it for Virtual Skynode.
#
# Package pins are Auterion's compatibility row for AuterionOS 4.6.2 / APX4 1.17.7
# (docs.auterion.com, Auterion SDK installation): px4-ros2-cpp 2.2.1, px4-msgs 3.5.1. A
# mismatch with the vehicle's APX4 stops the mode registering, so moving AuterionOS means
# moving these pins together.
FROM auterion/app-base:v2

ARG PX4_ROS2_CPP=2.2.1-0jammy
ARG PX4_MSGS=3.5.1-0jammy

# The base image's ROS 2 apt key expired (Auterion's own workaround).
RUN rm -f /etc/apt/sources.list.d/ros2.list \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list

# Auterion's public package feed: the PX4 ROS 2 Interface Library and messages matching APX4.
RUN curl -1sLf 'https://dl.cloudsmith.io/public/auterion/public/setup.deb.sh' | bash \
 && apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-px4-ros2-cpp=${PX4_ROS2_CPP} \
      ros-humble-px4-msgs=${PX4_MSGS} \
      ros-humble-eigen3-cmake-module \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY structure_hold_app /app/structure_hold_app
WORKDIR /app/structure_hold_app
RUN . /opt/ros/humble/setup.sh \
 && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)"

CMD . /opt/ros/humble/setup.sh && exec /app/structure_hold_app/build/structure_hold_app
