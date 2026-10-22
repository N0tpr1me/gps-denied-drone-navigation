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

start_px4_background() {
    mkdir -p "${OUT_DIR}"
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
