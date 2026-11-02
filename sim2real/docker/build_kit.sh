#!/usr/bin/env bash
# Package the evaluation kit for a buyer, binary only (docs/eval-kit.md, Decided 2026-11-02):
# the SITL image with the Structure Hold module's source and build objects removed and its
# layers flattened (so the source is not in any lower layer either), plus the Python side
# the kit runs, without the module source, the training code, the policies or our docs.
# Confidential. Copyright (c) 2026 Zain Ali. All rights reserved.
#
#   sim2real/docker/build_kit.sh                 # from the built image -> dist/kit-<commit>/
#
# The buyer unpacks kit-src.tar.gz and runs ./evaluate.sh --load kit-image.tar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${SITL_IMAGE:-structure-hold/px4-sitl:v1.17.0}"
KIT_IMAGE="structure-hold/px4-sitl:v1.17.0-kit"
commit="$(git -C "${ROOT}" rev-parse --short HEAD)"
out="${ROOT}/dist/kit-${commit}"
mkdir -p "${out}"

# 1. Image: strip the module's source and objects in a container, then export its whole
# filesystem as one layer and import it with the base image's settings.
trap 'docker rm -f "kit-strip-${commit}" >/dev/null 2>&1 || true' EXIT
docker start -a "$(docker create --entrypoint bash --name "kit-strip-${commit}" "${BASE}" -c '
    set -e
    rm -rf /opt/structure_hold /opt/px4/src/modules/structure_hold
    find /opt/px4/build -path "*structure_hold*" -type f \
        \( -name "*.hpp" -o -name "*.h" -o -name "*.cpp" -o -name "*.o" -o -name "*.a" \
           -o -name "*.d" -o -name "*.cmake" \) -delete
    test ! -e /opt/px4/src/modules/structure_hold
    ! find / -xdev -name "PolicyWeights.hpp" | grep -q .
')"
env_args=()
while IFS= read -r e; do env_args+=(--change "ENV ${e}"); done < <(
    docker inspect "${BASE}" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed '/^$/d')
docker export "kit-strip-${commit}" | docker import "${env_args[@]}" \
    --change 'WORKDIR /opt/px4' \
    --change 'ENTRYPOINT ["/opt/sitl/entrypoint.sh"]' \
    --change 'CMD ["sitl"]' \
    --change 'EXPOSE 14540/udp 14550/udp' \
    - "${KIT_IMAGE}"
docker rm -f "kit-strip-${commit}" >/dev/null
docker save "${KIT_IMAGE}" -o "${out}/kit-image.tar"

# 2. Python side: what the kit's runners and scorers import, nothing else.
git -C "${ROOT}" archive --format=tar HEAD \
    evaluate.sh pyproject.toml LICENSE \
    eval/__init__.py eval/eval_kit.py eval/field_faults.py eval/metrics.py eval/ulog.py \
    eval/baselines/__init__.py eval/baselines/run_baseline.py eval/baselines/px4_tuned_v2.params \
    envs/__init__.py envs/frames.py envs/disturbance envs/estimate \
    sim2real/estimate_bridge sim2real/gazebo \
    sim2real/px4/run_gazebo.py sim2real/px4/sitl_test.py \
    sim2real/docker/run.sh \
    | gzip > "${out}/kit-src.tar.gz"
ls -lh "${out}"
echo "kit: ${out} (the buyer runs: ./evaluate.sh --load kit-image.tar)"
