#!/usr/bin/env bash
# Run OpenVINS on one flight in EuRoC's ASL layout (docs/vio-fit.md, "Third fit"):
#   eval/vio_openvins/run_openvins.sh <mav0 parent> <out> <config> [--zero-timeshift]
#       [--set key=value]...
# <config> names one of OpenVINS's own config folders (euroc_mav, tum_vi, uzhfpv_indoor,
# uzhfpv_indoor_45, uzhfpv_outdoor, uzhfpv_outdoor_45). --zero-timeshift sets the
# camera-IMU time shift in its Kalibr file to 0, for layouts whose image stamps already
# carry it (eval/vio_basalt/prepare_uzhfpv.py). --set changes one estimator_config value
# (e.g. init_imu_thresh=0.25, which OpenVINS's own TUM-VI config notes for room6). Writes <out>/stamped_traj_estimate.txt,
# frame_time_ms.txt and openvins.log. The common parent of the data is mounted read-only
# under the same path, so relative symlinks in the layout resolve.
set -euo pipefail
data="$(realpath "$1")"; out="$(realpath -m "$2")"; config="$3"; shift 3
fix=""
while [ $# -gt 0 ]; do
    case "$1" in
        --zero-timeshift) fix="$fix sed -i 's/timeshift_cam_imu:.*/timeshift_cam_imu: 0.0/' \
            /tmp/cfg/kalibr_imucam_chain.yaml &&" ;;
        --set) k="${2%%=*}"; v="${2#*=}"; shift
               fix="$fix sed -i 's/^$k:.*/$k: $v/' /tmp/cfg/estimator_config.yaml &&" ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
[ -d "$data/mav0" ] || { echo "no $data/mav0" >&2; exit 1; }
mkdir -p "$out"
root="${OPENVINS_DATA_ROOT:-$(dirname "$(dirname "$data")")}"
mounts=(-v "$out:/work" -v "$root:$root:ro")
case "$data" in "$root"/*) ;; *) mounts+=(-v "$data:$data:ro") ;; esac
start=$(date +%s)
docker run --rm "${mounts[@]}" dronenav-openvins bash -c \
    "cp -r /open_vins/config/$config /tmp/cfg && $fix \
     sed -i 's/^verbosity:.*/verbosity: \"WARNING\"/' /tmp/cfg/estimator_config.yaml && \
     ov_run_euroc /tmp/cfg/estimator_config.yaml '$data' /work" \
    > "$out/openvins.log" 2>&1 || { echo "OpenVINS failed, see $out/openvins.log" >&2; exit 1; }
echo "$(basename "$data"): $(tail -1 "$out/openvins.log") in $(( $(date +%s) - start )) s"
