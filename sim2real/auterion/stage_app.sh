#!/usr/bin/env bash
# Write a self-contained AuterionOS app folder for `auterion-cli app build`: the manifest
# and Dockerfile (package/), the app's sources (structure_hold_app/) and the onboard
# module's PX4-free headers it shares (copied into structure_hold_app/module/, which its
# CMakeLists.txt prefers when present).
#
#   sim2real/auterion/stage_app.sh <out_dir>
#   cd <out_dir> && auterion-cli app build              # Skynode (linux/arm64)
#   cd <out_dir> && auterion-cli app build --simulation # Virtual Skynode (linux/amd64)
#
# Needs an Auterion developer set-up to go further (docs/auterion-app.md): auterion-cli is
# on PyPI, but Virtual Skynode's image comes from Auterion Suite's developer store.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE="${HERE}/../px4/structure_hold"
OUT="${1:?usage: stage_app.sh <out_dir>}"

# The module headers the app compiles in; HoldCore.hpp includes the first four, and
# PolicyController.hpp the last two.
HEADERS=(PolicyController.hpp Watchdog.hpp Handback.hpp ParamLimits.hpp PolicyWeights.hpp MlpPolicy.hpp)

rm -rf "${OUT}"
mkdir -p "${OUT}/structure_hold_app/module"
cp "${HERE}/package/auterion-app.yml" "${HERE}/package/Dockerfile" "${OUT}/"
cp -r "${HERE}/structure_hold_app/src" "${HERE}/structure_hold_app/include" \
      "${HERE}/structure_hold_app/CMakeLists.txt" "${HERE}/structure_hold_app/package.xml" \
      "${OUT}/structure_hold_app/"
for h in "${HEADERS[@]}"; do cp "${MODULE}/${h}" "${OUT}/structure_hold_app/module/"; done
echo "staged ${OUT}"
