#!/usr/bin/env bash
# Container entrypoint.
#   sitl        run PX4 + Gazebo headless in the foreground (default)
#   hover-test  start SITL, fly takeoff / 30 s hover / land with MAVSDK, write a summary
#   hold-test   start SITL, fly the Structure Hold acceptance sequence (sim2real/px4/sitl_test.py
#               from the repo mounted at /repo), write a summary
#   with-sitl   start SITL (and the bridge), run the given command, copy the ulog to /out
#   shell       bash, for poking around
set -euo pipefail

BUILD_DIR="${PX4_DIR}/build/px4_sitl_default"
OUT_DIR="${OUT_DIR:-/out}"

# WATCH=1 (watch image, Dockerfile.watch): Gazebo's GUI on a virtual display, served to a
# browser on port 6080. Must start before PX4, which reads DISPLAY and HEADLESS.
if [[ "${WATCH:-0}" == "1" ]]; then
    if [[ ! -f /opt/sitl/watch.sh ]]; then
        echo "WATCH=1 needs the watch image (sim2real/docker/run.sh build-watch)" >&2
        exit 2
    fi
    # shellcheck source=watch.sh
    source /opt/sitl/watch.sh
    start_viewer
fi

# A scenario's payload (sim2real/gazebo/scenario.py): fixed under the x500 in its model file
# before PX4 spawns it; this container is the flight's own, so the edit goes with it.
add_payload() {
    if [[ -n "${STRUCTURE_HOLD_PAYLOAD_KG:-}" ]]; then
        python3 /opt/sitl/add_payload.py "${PX4_DIR}/Tools/simulation/gz/models/x500/model.sdf" \
            --mass-kg "${STRUCTURE_HOLD_PAYLOAD_KG}" \
            --offset-down-m "${STRUCTURE_HOLD_PAYLOAD_OFFSET_M:-0.05}"
    fi
}

start_px4_background() {
    mkdir -p "${OUT_DIR}"
    add_payload
    # -d: no interactive shell. PX4 starts gz sim (server only, HEADLESS=1) itself.
    PX4_SYS_AUTOSTART=4001 "${BUILD_DIR}/bin/px4" -d >"${OUT_DIR}/px4.log" 2>&1 &
    PX4_PID=$!
    # Estimate bridge (sim2real/estimate_bridge, repo mounted at /repo): Gazebo truth ->
    # corrupted estimate -> EKF2 external vision. Only when a profile is given.
    BRIDGE_PID=""
    if [[ -n "${ESTIMATE_BRIDGE_PROFILE:-}" ]]; then
        PYTHONPATH=/repo python3 -m sim2real.estimate_bridge.run_bridge \
            --profile "${ESTIMATE_BRIDGE_PROFILE}" --trace "${OUT_DIR}/estimate_trace.csv" \
            >"${OUT_DIR}/bridge.log" 2>&1 &
        BRIDGE_PID=$!
    fi
}

stop_px4() {
    [[ -n "${BRIDGE_PID}" ]] && { kill "${BRIDGE_PID}" 2>/dev/null || true; wait "${BRIDGE_PID}" 2>/dev/null || true; }
    kill "${PX4_PID}" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    wait "${PX4_PID}" 2>/dev/null || true
    if declare -F stop_viewer >/dev/null; then stop_viewer; fi
}

copy_latest_ulog() {
    local ulog
    ulog=$(find "${BUILD_DIR}/rootfs/log" -name '*.ulg' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -n "${ulog}" ]]; then
        cp "${ulog}" "${OUT_DIR}/flight.ulg"
    else
        echo "warning: no ulog found" >&2
    fi
}

case "${1:-sitl}" in
    sitl)
        add_payload
        PX4_SYS_AUTOSTART=4001 exec "${BUILD_DIR}/bin/px4" -d
        ;;
    hover-test)
        shift
        start_px4_background
        trap stop_px4 EXIT
        status=0
        python3 /opt/sitl/hover_test.py --out "${OUT_DIR}/summary.json" "$@" || status=$?
        copy_latest_ulog
        exit "${status}"
        ;;
    hold-test)
        shift
        start_px4_background
        trap stop_px4 EXIT
        status=0
        # Gazebo takes longer than SIH to come up; sitl_test.py's own default is 120 s.
        python3 /repo/sim2real/px4/sitl_test.py --out "${OUT_DIR}/summary.json" \
            --ready-timeout-s 300 "$@" || status=$?
        copy_latest_ulog
        exit "${status}"
        ;;
    with-sitl)
        # Any flight script beside SITL (and the bridge), e.g. eval/baselines/autotune_flight.py.
        shift
        start_px4_background
        trap stop_px4 EXIT
        status=0
        "$@" || status=$?
        copy_latest_ulog
        exit "${status}"
        ;;
    shell)
        exec bash
        ;;
    *)
        exec "$@"
        ;;
esac
