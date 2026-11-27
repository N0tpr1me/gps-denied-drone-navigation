#!/usr/bin/env bash
# Run Basalt stereo-inertial VIO on one flight already in EuRoC's ASL layout (mav0/) and
# keep what eval/vio_fit.py reads (docs/vio-fit.md):
#   <out>/stamped_traj_estimate.txt   TUM + velocity, seconds
#   <out>/basalt.log                  stdout, with the per-frame timing Basalt prints
#   eval/vio_basalt/run_basalt.sh <mav0 parent> <out> <calib.json> [config.json]
# Paths are mounted under the same names inside the container, so relative symlinks in the
# layout (eval/vio_basalt/prepare_uzhfpv.py) keep working. A calib or config path starting
# with /basalt/data/ is one of Basalt's own files (e.g. tumvi_512_ds_calib.json).
set -euo pipefail

data="$(realpath "$1")"
out="$(realpath -m "$2")"
calib="$3"
config="${4:-/basalt/data/euroc_config.json}"
[ -d "$data/mav0" ] || { echo "no $data/mav0" >&2; exit 1; }
mkdir -p "$out"

mounts=(-v "$out:/work")
# Mount the common parent of the data so symlinks that leave the folder still resolve.
root="${BASALT_DATA_ROOT:-$(dirname "$(dirname "$data")")}"
mounts+=(-v "$root:$root:ro")
case "$data" in "$root"/*) ;; *) mounts+=(-v "$data:$data:ro") ;; esac
for f in "$calib" "$config"; do
    case "$f" in /basalt/data/*) ;; *) f="$(realpath "$f")"; mounts+=(-v "$f:$f:ro") ;; esac
done
calib_in="$calib"; case "$calib" in /basalt/data/*) ;; *) calib_in="$(realpath "$calib")" ;; esac
config_in="$config"; case "$config" in /basalt/data/*) ;; *) config_in="$(realpath "$config")" ;; esac

start=$(date +%s)
docker run --rm "${mounts[@]}" dronenav-basalt \
    basalt_vio --dataset-path "$data" --dataset-type euroc \
        --cam-calib "$calib_in" --config-path "$config_in" \
        --show-gui 0 --use-imu 1 --save-trajectory tum \
    > "$out/basalt.log" 2>&1 || { echo "basalt failed, see $out/basalt.log" >&2; exit 1; }
mv "$out/trajectory.txt" "$out/stamped_traj_estimate.txt"
echo "$(basename "$data"): $(grep -c -v '^#' "$out/stamped_traj_estimate.txt") poses in $(( $(date +%s) - start )) s"
