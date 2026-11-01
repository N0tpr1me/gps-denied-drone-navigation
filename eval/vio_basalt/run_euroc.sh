#!/usr/bin/env bash
# Run Basalt stereo-inertial VIO on one EuRoC flight (ASL format) and lay out what
# eval/vio_fit.py reads (docs/vio-fit.md, "Second fit"):
#   runs/vio_euroc/out/<SEQ>/stamped_traj_estimate.txt  TUM + velocity, seconds
#   runs/vio_euroc/out/<SEQ>/truth.csv                  EuRoC state_groundtruth_estimate0
#   runs/vio_euroc/out/<SEQ>/stats_sums.json            per-frame backend time (ms)
#   runs/vio_euroc/out/<SEQ>/basalt.log                 stdout, for resets and tracking loss
# Expects runs/vio_euroc/data/<SEQ>/mav0/ and the dronenav-basalt image (see Dockerfile).
set -euo pipefail

seq_name="$1"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
data="$repo/runs/vio_euroc/data/$seq_name"
out="$repo/runs/vio_euroc/out/$seq_name"
[ -d "$data/mav0" ] || { echo "no $data/mav0" >&2; exit 1; }
mkdir -p "$out"

docker run --rm -v "$data:/data:ro" -v "$out:/work" dronenav-basalt \
    basalt_vio --dataset-path /data --dataset-type euroc \
        --cam-calib /basalt/data/euroc_ds_calib.json \
        --config-path /basalt/data/euroc_config.json \
        --show-gui 0 --use-imu 1 --save-trajectory tum \
    > "$out/basalt.log" 2>&1

# Basalt writes trajectory.txt in its working directory (/work = $out)
mv "$out/trajectory.txt" "$out/stamped_traj_estimate.txt"
cp "$data/mav0/state_groundtruth_estimate0/data.csv" "$out/truth.csv"
echo "$seq_name: $(grep -c -v '^#' "$out/stamped_traj_estimate.txt") poses"
