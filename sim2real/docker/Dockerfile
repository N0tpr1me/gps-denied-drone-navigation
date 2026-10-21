# Headless PX4 SITL + Gazebo Harmonic, gz_x500 airframe.
# Builds natively on arm64 (Apple Silicon via Docker Desktop) and amd64 (CI).
# The PX4 tag is pinned in docs/decisions.md; change both together, and re-tune the
# baseline (eval/baselines/px4_tuned_v1.params) if the tag moves.
FROM ubuntu:24.04

ARG PX4_TAG=v1.17.0
# 4.x wheels fail to load on arm64 Ubuntu 24.04 (libstdc++ symbol mismatch); 3.x works.
ARG MAVSDK_VERSION=3.17.4

ENV DEBIAN_FRONTEND=noninteractive \
    PX4_DIR=/opt/px4 \
    PX4_TAG=${PX4_TAG} \
    HEADLESS=1 \
    PX4_SIM_MODEL=gz_x500

# ubuntu.sh calls sudo and lsb_release, which the base image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git gnupg lsb-release sudo tzdata wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${PX4_TAG}" --recurse-submodules --jobs 8 \
        https://github.com/PX4/PX4-Autopilot.git "${PX4_DIR}"

WORKDIR ${PX4_DIR}

# PX4's own setup script: build tools, Python deps and Gazebo Harmonic. No NuttX toolchain.
RUN bash Tools/setup/ubuntu.sh --no-nuttx \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Structure Hold module (sim2real/px4/): wired into the tree before the build.
COPY px4/ /opt/structure_hold/px4/
RUN /opt/structure_hold/px4/install.sh "${PX4_DIR}"

RUN make px4_sitl_default \
    && rm -rf /root/.ccache

RUN python3 -m pip install --break-system-packages "mavsdk==${MAVSDK_VERSION}" \
    && rm -rf /root/.cache/pip

COPY docker/entrypoint.sh docker/hover_test.py /opt/sitl/
RUN chmod +x /opt/sitl/entrypoint.sh

# MAVLink: 14540 offboard/API (MAVSDK), 14550 ground station.
EXPOSE 14540/udp 14550/udp

ENTRYPOINT ["/opt/sitl/entrypoint.sh"]
CMD ["sitl"]
