#!/usr/bin/env bash
# Virtual display for watching a flight (watch image only, sourced by entrypoint.sh when
# WATCH=1). Starts Xvfb, x11vnc and noVNC, then points PX4's Gazebo start-up at that display
# so it launches the GUI beside the server and makes the camera follow the drone.
# Open http://localhost:6080/vnc.html?autoconnect=1&resize=scale on the host.

WATCH_DISPLAY="${WATCH_DISPLAY:-:1}"
WATCH_PORT="${WATCH_PORT:-6080}"
WATCH_SIZE="${WATCH_SIZE:-1280x720x24}"

start_viewer() {
    mkdir -p "${OUT_DIR:-/tmp}"
    local log="${OUT_DIR:-/tmp}/watch.log"
    Xvfb "${WATCH_DISPLAY}" -screen 0 "${WATCH_SIZE}" +extension GLX -nolisten tcp >"${log}" 2>&1 &
    VIEWER_PIDS="$!"
    for _ in $(seq 50); do
        [[ -e "/tmp/.X11-unix/X${WATCH_DISPLAY#:}" ]] && break
        sleep 0.1
    done
    x11vnc -display "${WATCH_DISPLAY}" -rfbport 5900 -localhost -nopw -forever -shared -quiet \
        >>"${log}" 2>&1 &
    VIEWER_PIDS+=" $!"
    websockify --web /usr/share/novnc "${WATCH_PORT}" localhost:5900 >>"${log}" 2>&1 &
    VIEWER_PIDS+=" $!"

    export DISPLAY="${WATCH_DISPLAY}"
    export LIBGL_ALWAYS_SOFTWARE=1
    # PX4's px4-rc.gzsim starts `gz sim -g` when HEADLESS is unset and sets up the follow camera.
    unset HEADLESS
    export PX4_GZ_FOLLOW=1
    # Side-on view of the drone and the wall it holds against (wall normal along x).
    export PX4_GZ_FOLLOW_OFFSET_X="${PX4_GZ_FOLLOW_OFFSET_X:--1.5}"
    export PX4_GZ_FOLLOW_OFFSET_Y="${PX4_GZ_FOLLOW_OFFSET_Y:--5.0}"
    export PX4_GZ_FOLLOW_OFFSET_Z="${PX4_GZ_FOLLOW_OFFSET_Z:-1.5}"
    follow_drone >>"${log}" 2>&1 &
    VIEWER_PIDS+=" $!"
    echo "watch: http://localhost:${WATCH_PORT}/vnc.html?autoconnect=1&resize=scale" >&2
}

# PX4 asks the GUI to follow the drone once, as soon as the server is up; the software-rendered
# GUI is usually slower than that, so ask again once its follow service exists.
follow_drone() {
    local model="${WATCH_MODEL:-x500_0}"
    for _ in $(seq 120); do
        if gz service -l 2>/dev/null | grep -q "^/gui/follow$"; then
            sleep 2
            gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean \
                --timeout 5000 --req "data: \"${model}\""
            gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d \
                --reptype gz.msgs.Boolean --timeout 5000 --req \
                "x: ${PX4_GZ_FOLLOW_OFFSET_X}, y: ${PX4_GZ_FOLLOW_OFFSET_Y}, z: ${PX4_GZ_FOLLOW_OFFSET_Z}"
            return 0
        fi
        sleep 1
    done
    echo "watch: Gazebo GUI never offered /gui/follow" >&2
}

stop_viewer() {
    pkill -f "gz sim -g" 2>/dev/null || true
    # shellcheck disable=SC2086
    [[ -n "${VIEWER_PIDS:-}" ]] && kill ${VIEWER_PIDS} 2>/dev/null || true
}
