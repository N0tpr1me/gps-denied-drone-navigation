#!/usr/bin/env bash
# Host-side helper for the PX4 SITL image.
#   ./run.sh build         build the image (first build: PX4 + Gazebo, roughly 20-40 min)
#   ./run.sh sitl          run PX4 + Gazebo headless (Ctrl-C to stop)
#   ./run.sh hover-test    takeoff / 30 s hover / land; writes runs/<time>_sitl_hover/
#   ./run.sh shell         bash inside the image
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
IMAGE="${SITL_IMAGE:-structure-hold/px4-sitl:v1.17.0}"

cmd="${1:-sitl}"
shift || true

case "${cmd}" in
    build)
        docker build -t "${IMAGE}" "$@" "${HERE}"
        ;;
    sitl|shell)
        docker run --rm -it --name px4-sitl "${IMAGE}" "${cmd}" "$@"
        ;;
    hover-test)
        out="${REPO_ROOT}/runs/$(date -u +%Y%m%d-%H%M%S)_sitl_hover_$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
        mkdir -p "${out}"
        docker run --rm --name px4-sitl-hover -v "${out}:/out" "${IMAGE}" hover-test "$@"
        echo "run folder: ${out}"
        ;;
    *)
        echo "usage: $0 {build|sitl|hover-test|shell} [args]" >&2
        exit 2
        ;;
esac
