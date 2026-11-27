# OpenVINS (RPNG, University of Delaware, GPL-3.0) for the third envs/estimate fit
# (docs/vio-fit.md): a filter-based VIO (MSCKF) beside Basalt's optimiser, run offline on
# the same EuRoC-layout flights. Used for evaluation only: nothing from it ships in the
# module or the evaluation kit. Built without ROS; run_euroc.cpp (ours) reads the ASL
# folders and feeds OpenVINS's VioManager the way its ROS subscriber does.
#   docker build -t dronenav-openvins eval/vio_openvins    (cloud: cloud_build.sh)
FROM ubuntu:22.04

ARG OPENVINS_COMMIT=93adc241390d13e99232652cf05cbe18a93c7bea  # tag v2.7
ARG JOBS=4

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates build-essential cmake git libeigen3-dev libboost-all-dev \
        libceres-dev libopencv-dev libopencv-contrib-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/rpng/open_vins.git /open_vins \
    && cd /open_vins && git checkout ${OPENVINS_COMMIT} \
    && cmake -S ov_msckf -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS=OFF \
        -DENABLE_ARUCO_TAGS=OFF \
    && cmake --build build -j${JOBS} --target ov_msckf_lib \
    && rm -rf /open_vins/.git

COPY run_euroc.cpp /src/run_euroc.cpp
RUN g++ -std=c++17 -O2 -o /usr/local/bin/ov_run_euroc /src/run_euroc.cpp \
        -I/open_vins/ov_core/src -I/open_vins/ov_init/src -I/open_vins/ov_msckf/src \
        -I/usr/include/eigen3 $(pkg-config --cflags opencv4) \
        -DENABLE_ARUCO_TAGS=0 -DROS_AVAILABLE=0 \
        -L/open_vins/build -Wl,-rpath,/open_vins/build -lov_msckf_lib \
        $(pkg-config --libs opencv4) -lboost_system -lboost_filesystem -lboost_thread \
        -lceres -lglog -lpthread

WORKDIR /work
