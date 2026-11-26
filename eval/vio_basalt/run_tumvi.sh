#!/usr/bin/env bash
# Basalt on the TUM-VI room sequences (docs/vio-fit.md, "Third fit"):
#   eval/vio_basalt/run_tumvi.sh <tumvi dir> <out dir>
# <tumvi dir> holds the exported EuRoC-format sequences, each dataset-room<N>_512_16.tar
# unpacked (dataset-room<N>_512_16/mav0/...). Only the room sequences have motion-capture
# truth over the whole run; the rest have it at the start and end only. Basalt's own TUM-VI
# calibration and config are used (basalt/data/tumvi_512_*). Writes <out dir>/<seq>/
# {stamped_traj_estimate.txt, truth.csv, basalt.log}; truth.csv is mav0/mocap0/data.csv
# (the IMU's pose, already moved from the markers to the IMU by the dataset).
set -euo pipefail
src="$(realpath "$1")"; out="$(realpath -m "$2")"
here="$(cd "$(dirname "$0")" && pwd)"
for seq in "$src"/dataset-room*_512_16; do
    [ -f "$seq/mav0/mocap0/data.csv" ] || continue
    name="$(basename "$seq")"
    [ -f "$out/$name/stamped_traj_estimate.txt" ] && continue
    "$here/run_basalt.sh" "$seq" "$out/$name" /basalt/data/tumvi_512_ds_calib.json \
        /basalt/data/tumvi_512_config.json || { echo "$name: Basalt failed" >&2; continue; }
    cp "$seq/mav0/mocap0/data.csv" "$out/$name/truth.csv"
done
