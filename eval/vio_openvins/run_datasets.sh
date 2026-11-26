#!/usr/bin/env bash
# OpenVINS on the same flights as Basalt (docs/vio-fit.md, "Third fit"):
#   eval/vio_openvins/run_datasets.sh <uzhfpv layout dir> <tumvi dir> <out dir>
# <uzhfpv layout dir> is the _layout folder eval/vio_basalt/run_uzhfpv.sh writes (EuRoC
# layout per flight, image stamps already shifted onto the IMU clock, truth.txt beside);
# <tumvi dir> holds the unpacked dataset-room<N>_512_16 sequences. Writes
# <out dir>/uzhfpv/<flight>/ and <out dir>/tumvi/<seq>/ with stamped_traj_estimate.txt and
# the truth (truth.txt or truth.csv), for eval/configs/vio_fit_openvins_*.yaml.
set -euo pipefail
uzh="$(realpath "$1")"; tum="$(realpath "$2")"; out="$(realpath -m "$3")"
here="$(cd "$(dirname "$0")" && pwd)"
for flight in "$uzh"/indoor_* "$uzh"/outdoor_*; do
    [ -f "$flight/truth.txt" ] || continue
    name="$(basename "$flight")"; mount="${name%_*}"
    case "$mount" in
        indoor_forward) cfg=uzhfpv_indoor ;; indoor_45) cfg=uzhfpv_indoor_45 ;;
        outdoor_forward) cfg=uzhfpv_outdoor ;; outdoor_45) cfg=uzhfpv_outdoor_45 ;;
        *) echo "$name: unknown mount" >&2; continue ;;
    esac
    dest="$out/uzhfpv/$name"
    [ -s "$dest/stamped_traj_estimate.txt" ] && continue
    OPENVINS_DATA_ROOT="$(dirname "$(dirname "$(dirname "$uzh")")")" \
        "$here/run_openvins.sh" "$flight" "$dest" "$cfg" --zero-timeshift \
        || { echo "$name: OpenVINS failed" >&2; continue; }
    cp "$flight/truth.txt" "$dest/truth.txt"
done
for seq in "$tum"/dataset-room*_512_16; do
    [ -f "$seq/mav0/mocap0/data.csv" ] || continue
    name="$(basename "$seq")"; dest="$out/tumvi/$name"
    [ -s "$dest/stamped_traj_estimate.txt" ] && continue
    "$here/run_openvins.sh" "$seq" "$dest" tum_vi || { echo "$name: OpenVINS failed" >&2; continue; }
    cp "$seq/mav0/mocap0/data.csv" "$dest/truth.csv"
done
