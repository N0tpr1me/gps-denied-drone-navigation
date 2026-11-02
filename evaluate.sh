#!/usr/bin/env bash
# The evaluation kit in one command (docs/eval-kit.md): fly the six position-estimate
# failures (VIO reset, restart, link loss, freeze, drift, dust/fog/low light) with Structure Hold and with tuned
# PX4 in Gazebo SITL, then print the comparison table.
# Confidential. Copyright (c) 2026 Zain Ali. All rights reserved.
#
#   ./evaluate.sh                          # everything: checks, image, Python deps, flights
#   ./evaluate.sh --load kit-image.tar     # use a shipped image instead of building it
#   ./evaluate.sh --faults reset drift     # a subset (any eval.eval_kit option passes on)
#   ./evaluate.sh --dry-run                # the plan and the time it should take
#
# Needs Docker and Python 3.10 or newer. No GPU: Gazebo runs headless on CPU cores, 4 per
# flight; with 48 free cores all 12 flights fly at once (a few minutes), on 4 cores about 24.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
IMAGE="${SITL_IMAGE:-structure-hold/px4-sitl:v1.17.0}"
VENV="${KIT_VENV:-${ROOT}/.venv-kit}"

load=""
dry=0
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --load) load="$2"; shift 2 ;;
        --dry-run|--rescore) dry=1; args+=("$1"); shift ;;
        *) args+=("$1"); shift ;;
    esac
done

say() { printf '\n== %s\n' "$*"; }

if [[ "${dry}" == 0 ]]; then
say "1/4 Docker"
command -v docker >/dev/null || { echo "Docker is not installed: https://docs.docker.com/engine/install/" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker is installed but not running (or needs sudo / the docker group)." >&2; exit 2; }
echo "ok: $(docker --version)"

say "2/4 SITL image"
if [[ -n "${load}" ]]; then
    # Fly the image the file holds (the binary-only kit image has its own name).
    loaded="$(docker load -i "${load}" | sed -n 's/^Loaded image: //p' | tail -1)"
    [[ -n "${loaded}" ]] && IMAGE="${loaded}"
    echo "loaded ${IMAGE}"
fi
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ok: ${IMAGE} present"
else
    echo "not found: building it once (PX4 v1.17.0 + Gazebo Harmonic, 20 to 40 min)"
    SITL_IMAGE="${IMAGE}" sim2real/docker/run.sh build
fi
fi  # --dry-run and --rescore fly nothing: no Docker or image needed

say "3/4 Python"
PY="python3"
if ! "${PY}" -c 'import numpy, yaml, pyulog, pymavlink' 2>/dev/null \
        || ! "${PY}" -c 'import eval.eval_kit' 2>/dev/null; then
    if [[ ! -x "${VENV}/bin/python" ]]; then
        "${PY}" -m venv "${VENV}"
    fi
    PY="${VENV}/bin/python"
    "${PY}" -m pip install --quiet --upgrade pip
    "${PY}" -m pip install --quiet -e "${ROOT}[sitl]"
fi
"${PY}" -c 'import sys; assert sys.version_info >= (3, 10), "needs Python 3.10+"'
echo "ok: $("${PY}" --version)"

say "4/4 Flights"
exec "${PY}" -m eval.eval_kit --image "${IMAGE}" ${args[@]+"${args[@]}"}
