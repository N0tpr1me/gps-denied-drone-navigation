# Basalt VIO (TUM, BSD-3) for the second envs/estimate fit (docs/vio-fit.md). Pinned to the
# last commit that builds from Ubuntu packages (the next one moved to vcpkg), plus one patch
# that adds the VIO's world-frame velocity as columns 9-11 of the TUM trajectory.
#   docker build -t dronenav-basalt eval/vio_basalt
#   eval/vio_basalt/run_euroc.sh <SEQ>
FROM ubuntu:22.04

ARG BASALT_COMMIT=47b063e64fa7dd56ac04eb2d579c5cb5e72c743e
ARG JOBS=4

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates build-essential python3-pip git libtbb-dev libeigen3-dev libglew-dev ccache \
        libjpeg-dev libpng-dev liblz4-dev libbz2-dev libboost-regex-dev \
        libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev \
        libgtest-dev libopencv-dev libfmt-dev \
        libepoxy-dev libgl1-mesa-dev libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir cmake==3.28.3  # Sophus needs >= 3.24; jammy ships 3.22

COPY basalt_save_velocity.patch /tmp/
# arm64: -fsigned-char (Basalt assumes x86 signed char, e.g. its ORB pattern table);
# -Wno-error (the bundled Eigen NEON code trips -Wclass-memaccess).
RUN git clone https://gitlab.com/VladyslavUsenko/basalt.git /basalt \
    && cd /basalt && git checkout ${BASALT_COMMIT} \
    && git submodule update --init --recursive --depth 1 \
    && git apply /tmp/basalt_save_velocity.patch \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=OFF \
        -DCMAKE_CXX_FLAGS="-Wno-error -fsigned-char" \
    && cmake --build build -j${JOBS} --target basalt_vio \
    && cp build/basalt_vio /usr/local/bin/ \
    && find build -name "*.so*" -exec cp {} /usr/local/lib/ \; && ldconfig \
    && rm -rf /basalt/build /basalt/thirdparty/*/.git

WORKDIR /work
