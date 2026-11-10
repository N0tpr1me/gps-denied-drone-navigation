#!/usr/bin/env bash
# Check the Structure Hold module against one PX4 release, start to finish:
#
#   1. clone that PX4 tag (skipped when the tree is already there)
#   2. install the module into it (install.sh) and build px4_sitl_default
#   3. compare the parameters that build generated with structure_hold_params.c
#   4. fly the SIH acceptance flight (sitl_test.sh)
#
#   version_check.sh <px4_tag> [work_dir] [-- extra sitl_test.py args]
#
# The tree lands in <work_dir>/<px4_tag> (default work_dir: ~/px4-trees) and the flight in
# runs/<time>_px4_<tag>_<commit>/. Exit 0 when every step passed. This is the check behind
# the table in docs/px4-versions.md; run it again when a release (v1.18.0 final, say)
# comes out. It says the module builds and its modes work on that release, nothing about
# hold quality: the benchmark is flown on the pinned release only (docs/decisions.md).
#
# Needs git, PX4's build tools, and a python3 with PX4's Tools/setup/requirements.txt plus
# pymavlink and pyulog (Python 3.10 to 3.12: empy<4 does not build on 3.13).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
TAG="${1:?usage: version_check.sh <px4_tag> [work_dir] [-- extra sitl_test.py args]}"
shift
WORK="${HOME}/px4-trees"
if [[ $# -gt 0 && "$1" != "--" ]]; then WORK="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
PX4_DIR="${WORK}/${TAG}"

if [[ ! -d "${PX4_DIR}/src/modules" ]]; then
    mkdir -p "${WORK}"
    git clone --depth 1 --branch "${TAG}" --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git "${PX4_DIR}"
    # v1.16's version header reads NuttX's tags, which a shallow submodule clone leaves
    # out (the build then stops in px_update_git_header.py); v1.17 on does without them.
    git -C "${PX4_DIR}/platforms/nuttx/NuttX/nuttx" fetch --quiet --depth 1 --tags origin
fi

echo "== install and build (${TAG})"
"${HERE}/install.sh" "${PX4_DIR}"
make -C "${PX4_DIR}" px4_sitl_default

echo "== parameters as PX4 generated them"
(cd "${REPO_ROOT}" && python3 -m sim2real.px4.tools.params_yaml \
    --against "${PX4_DIR}/build/px4_sitl_default/parameters.json")

echo "== SIH acceptance flight"
OUT="${REPO_ROOT}/runs/$(date -u +%Y%m%d-%H%M%S)_px4_${TAG}_$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
"${HERE}/sitl_test.sh" "${PX4_DIR}" "${OUT}" -- "$@"
echo "PX4 ${TAG}: built, parameters match, acceptance flight passed"
