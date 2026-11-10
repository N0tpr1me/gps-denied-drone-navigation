#!/usr/bin/env bash
# Build the SITL image inside the Claude cloud container, which has Docker but reaches the
# internet only through an HTTPS proxy with its own CA, and blocks packages.osrfoundation.org
# (where PX4's ubuntu.sh gets Gazebo Harmonic). Everywhere else (the GPU server, the Mac,
# CI) use `run.sh build`, which builds the real image from plain Ubuntu.
#
#   sudo dockerd >/tmp/dockerd.log 2>&1 &     # once per container
#   sim2real/docker/cloud_build.sh            # ~30 min on 4 cores
#
# What differs from `run.sh build`: the base is PX4's published Gazebo Harmonic dev image
# (px4io/px4-dev-ros2-gazebo, Ubuntu 24.04, gz-sim 8) instead of ubuntu.sh's install, and the
# proxy CA is trusted in the build. PX4 v1.17.0, the module, MAVSDK and the Gazebo world and
# wind plugin are the same. Good for checking the runner and the x86_64 path; benchmark
# numbers for docs/baseline.md come from the real image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PX4_TAG as in run.sh: v1.17.0 unless another release is being checked (docs/px4-versions.md).
PX4_TAG="${PX4_TAG:-v1.17.0}"
IMAGE="${SITL_IMAGE:-structure-hold/px4-sitl:${PX4_TAG}}"
BASE="${CLOUD_BASE_IMAGE:-px4io/px4-dev-ros2-gazebo:main-jazzy}"
CA="${CLOUD_CA_BUNDLE:-/root/.ccr/ca-bundle.crt}"
[[ -f "${CA}" ]] || { echo "proxy CA not found: ${CA}" >&2; exit 1; }

# The real Dockerfile with a proxy-trust layer right after FROM. The base's own ROS and
# Gazebo apt sources are dropped (their hosts are blocked; nothing new is needed from them).
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
awk '
    { print }
    /^FROM / && !done {
        print "COPY --from=cloudca ca-bundle.crt /usr/local/share/ca-certificates/cloud-proxy.crt"
        print "RUN rm -f /etc/apt/sources.list.d/ros2.sources /etc/apt/sources.list.d/gazebo-stable.list \\"
        print "    && update-ca-certificates"
        print "ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \\"
        print "    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \\"
        print "    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt"
        done = 1
    }' "${HERE}/Dockerfile" > "${tmp}/Dockerfile"
cp "${CA}" "${tmp}/ca-bundle.crt"

docker build --network host -t "${IMAGE}" -f "${tmp}/Dockerfile" \
    --build-context "cloudca=${tmp}" \
    --build-arg "BASE_IMAGE=${BASE}" --build-arg PX4_SETUP=0 --build-arg "PX4_TAG=${PX4_TAG}" \
    --build-arg "HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg "http_proxy=${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg "https_proxy=${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg "NO_PROXY=${NO_PROXY:-localhost,127.0.0.1}" \
    "$@" "${HERE}/.."
