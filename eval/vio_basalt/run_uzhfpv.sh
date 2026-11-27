#!/usr/bin/env bash
# Basalt on every UZH-FPV flight with ground truth (docs/vio-fit.md, "Third fit"):
#   eval/vio_basalt/run_uzhfpv.sh <uzhfpv dir> <out dir>
# <uzhfpv dir> holds one folder per flight (the *_snapdragon_with_gt.zip unzipped into a
# folder named after the flight, e.g. indoor_forward_3/) and calib/<mount>/ with each
# *_calib_snapdragon.zip unzipped. Writes <out dir>/<flight>/{stamped_traj_estimate.txt,
# truth.txt, basalt.log}, which eval/configs/vio_fit_basalt_uzhfpv.yaml reads.
set -euo pipefail
src="$(realpath "$1")"; out="$(realpath -m "$2")"
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
prep="$out/_layout"
for flight in "$src"/indoor_* "$src"/outdoor_*; do
    [ -f "$flight/groundtruth.txt" ] || continue
    name="$(basename "$flight")"
    mount="${name%_*}"                        # indoor_forward_3 -> indoor_forward
    [ -f "$out/$name/stamped_traj_estimate.txt" ] && continue
    (cd "$repo" && python -m eval.vio_basalt.prepare_uzhfpv "$flight" "$src/calib/$mount" \
        "$prep/$name")
    BASALT_DATA_ROOT="$(dirname "$src")" \
        "$here/run_basalt.sh" "$prep/$name" "$out/$name" "$prep/$name/calib.json" \
        || { echo "$name: Basalt failed" >&2; continue; }
    cp "$prep/$name/truth.txt" "$out/$name/truth.txt"
done
