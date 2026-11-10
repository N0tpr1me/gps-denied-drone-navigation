#!/usr/bin/env bash
# Render the VIO datasets with Isaac Sim on the GPU server (docs/vio-render.md), inside the
# pinned Isaac Lab image, one GPU. Basalt then runs on the host on these renders
# (eval/configs/vio_render_pool.yaml, params.from_run = this run's folder).
#
#   bash eval/vio_render/isaac_render.sh --gpu 1                         # all seeds
#   bash eval/vio_render/isaac_render.sh --gpu 1 --set params.seeds=[2]  # one seed
#   bash eval/vio_render/isaac_render.sh --smoke                         # a few seconds
#
# Everything after the options goes to training/launch.py (so --set, --smoke, --approved).
# Run only after the Isaac Lab parity check passed (training/isaac_parity.sh); the queue
# training/queues/vio_render.yaml checks that first. Output paths are printed as host paths,
# so the queue runner finds the run folder.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
REPO="$(pwd)"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}"
CONFIG="${VIO_RENDER_CONFIG:-eval/configs/vio_render_wall_isaac.yaml}"
GPU=""
ARGS=()
while (($#)); do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --gpu=*) GPU="${1#--gpu=}"; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found: this runs on the GPU server only" >&2
    exit 3
fi
if [[ -z "${GPU}" ]]; then
    GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', *' '$2 < 1024 && $3 < 10 { print $1; exit }')"
    [[ -n "${GPU}" ]] || { echo "no free GPU (nvidia-smi)" >&2; nvidia-smi >&2; exit 3; }
fi
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}" || exit 1

CACHE="${HOME}/docker/isaac-sim"
mkdir -p "${CACHE}"/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}

INSIDE='
set -euo pipefail
cd /workspace/dronenav
ISAAC_DIR="$(ls -d /workspace/isaaclab /workspace/IsaacLab 2>/dev/null | head -1)"
PY="${ISAAC_DIR}/isaaclab.sh -p"
${PY} -m pip freeze | grep -E "^(torch|numpy)==" >/tmp/keep.txt
${PY} -m pip install -q -e ".[dev]" -c /tmp/keep.txt >/tmp/install.log 2>&1 || { tail -20 /tmp/install.log; exit 1; }
exec ${PY} -m training.launch --config "$0" --device cuda "$@"
'

echo "isaac_render: GPU ${GPU}, image ${IMAGE}, config ${CONFIG}"
docker run --rm --name "dronenav-vio-render-$(date -u +%Y%m%d-%H%M%S)" --entrypoint bash \
    --gpus "\"device=${GPU}\"" --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${CACHE}/cache/ov:/root/.cache/ov:rw" \
    -v "${CACHE}/cache/pip:/root/.cache/pip:rw" \
    -v "${CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "${CACHE}/cache/computecache:/root/.nv/ComputeCache:rw" \
    -v "${CACHE}/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "${CACHE}/data:/root/.local/share/ov/data:rw" \
    -v "${CACHE}/documents:/root/Documents:rw" \
    -v "${REPO}:/workspace/dronenav" -w /workspace/dronenav \
    "${IMAGE}" -c "${INSIDE}" "${CONFIG}" "${ARGS[@]}" 2>&1 \
    | sed -u "s#/workspace/dronenav#${REPO}#g"
exit "${PIPESTATUS[0]}"
