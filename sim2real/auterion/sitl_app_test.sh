#!/usr/bin/env bash
# Fly the Structure Hold app (external-mode variant) against stock PX4 SITL with the SIH
# quadrotor. Needs Docker, a stock PX4 v1.17.0 tree built with `make px4_sitl_default`
# (no install.sh: the module is not compiled in), and python3 with pymavlink.
#
#   sim2real/auterion/sitl_app_test.sh <PX4_DIR> [out_dir] [sitl_app_test.py args]
#
# Builds the app image (docker/Dockerfile.sitl) unless SKIP_BUILD=1, starts PX4, flies
# sitl_app_test.py, and writes <out_dir>/{summary.json,px4.log,app.log}.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PX4_DIR="${1:?usage: sitl_app_test.sh <PX4_DIR> [out_dir] [sitl_app_test.py args]}"
shift
OUT="${REPO_ROOT}/runs/$(date -u +%Y%m%d-%H%M%S)_sitl_auterion_app_$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
if [[ $# -gt 0 && "$1" != -* ]]; then OUT="$1"; shift; fi
IMAGE="${APP_IMAGE:-structure-hold/app-sitl}"
mkdir -p "${OUT}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    secret=()
    # Behind the cloud container's HTTPS proxy, trust its CA in the build.
    [[ -f /root/.ccr/ca-bundle.crt ]] && secret=(--secret id=proxy_ca,src=/root/.ccr/ca-bundle.crt)
    DOCKER_BUILDKIT=1 docker build --network host "${secret[@]}" \
        --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-}" --build-arg "https_proxy=${HTTPS_PROXY:-}" \
        --build-arg "HTTP_PROXY=${HTTP_PROXY:-}" --build-arg "http_proxy=${HTTP_PROXY:-}" \
        -f "${HERE}/docker/Dockerfile.sitl" -t "${IMAGE}" "${REPO_ROOT}"
fi

BUILD_DIR="${PX4_DIR}/build/px4_sitl_default"
[[ -x "${BUILD_DIR}/bin/px4" ]] || { echo "no px4 binary in ${BUILD_DIR}; build first" >&2; exit 1; }
if pgrep -x px4 >/dev/null; then
    echo "a PX4 SITL instance is already running; stop it first (pkill -x px4)" >&2
    exit 1
fi
ROOTFS="${OUT}/rootfs"
mkdir -p "${ROOTFS}"
(cd "${ROOTFS}" && PX4_SIM_MODEL=sihsim_quadx HEADLESS=1 \
    exec "${BUILD_DIR}/bin/px4" -d "${BUILD_DIR}/etc" >"${OUT}/px4.log" 2>&1) &
PX4_PID=$!
trap 'kill ${PX4_PID} 2>/dev/null || true; wait ${PX4_PID} 2>/dev/null || true' EXIT

status=0
python3 "${HERE}/sitl_app_test.py" --out "${OUT}/summary.json" --image "${IMAGE}" "$@" || status=$?
echo "summary: ${OUT}/summary.json"
exit "${status}"
