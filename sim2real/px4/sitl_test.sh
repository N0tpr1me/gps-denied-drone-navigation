#!/usr/bin/env bash
# Run the Structure Hold acceptance flight in PX4 SITL with the built-in SIH quadrotor
# (no Gazebo needed). Starts PX4 from a tree prepared by install.sh and built with
# `make px4_sitl_default`, flies sitl_test.py against it, then copies the ulog.
#
#   sitl_test.sh <PX4_DIR> [out_dir] [-- extra sitl_test.py args]
#
# Writes <out_dir>/{summary.json,px4.log,flight.ulg}; exits 0 when every check passed.
# Needs python3 with pymavlink and pyulog (`pip install pymavlink pyulog`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PX4_DIR="${1:?usage: sitl_test.sh <PX4_DIR> [out_dir] [-- args]}"
shift
OUT="${1:-}"
if [[ -n "${OUT}" && "${OUT}" != "--" ]]; then shift; fi
[[ "${1:-}" == "--" ]] && shift
if [[ -z "${OUT}" || "${OUT}" == "--" ]]; then
    OUT="${REPO_ROOT}/runs/$(date -u +%Y%m%d-%H%M%S)_sitl_structure_hold_$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
fi
mkdir -p "${OUT}"

BUILD_DIR="${PX4_DIR}/build/px4_sitl_default"
[[ -x "${BUILD_DIR}/bin/px4" ]] || { echo "no px4 binary in ${BUILD_DIR}; build first" >&2; exit 1; }

# A fresh rootfs per run: parameters and logs start clean.
ROOTFS="${OUT}/rootfs"
mkdir -p "${ROOTFS}"
export PX4_TAG="${PX4_TAG:-$(git -C "${PX4_DIR}" describe --tags --always 2>/dev/null || echo unknown)}"

if pgrep -x px4 >/dev/null; then
    echo "a PX4 SITL instance is already running; stop it first (pkill -x px4)" >&2
    exit 1
fi
(cd "${ROOTFS}" && PX4_SIM_MODEL=sihsim_quadx HEADLESS=1 \
    exec "${BUILD_DIR}/bin/px4" -d "${BUILD_DIR}/etc" >"${OUT}/px4.log" 2>&1) &
PX4_PID=$!
trap 'kill ${PX4_PID} 2>/dev/null || true; wait ${PX4_PID} 2>/dev/null || true' EXIT

# SIH's EKF velocity is unusable on v1.17.0 (README), so the placeholder controller
# differentiates position here. SITL_EXTRA_ARGS adds or overrides sitl_test.py options.
status=0
python3 "${HERE}/sitl_test.py" --out "${OUT}/summary.json" --param SH_VEL_SRC=1 \
    ${SITL_EXTRA_ARGS:-} "$@" || status=$?

# Stop PX4 cleanly so the logger closes the file, then keep the flight log.
kill "${PX4_PID}" 2>/dev/null || true
wait "${PX4_PID}" 2>/dev/null || true
ulog=$(find "${ROOTFS}/log" -name '*.ulg' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -n "${ulog}" ]]; then
    cp "${ulog}" "${OUT}/flight.ulg"
    # Second pass: add the ulog checks to the summary (re-uses the flight result).
    python3 - "${OUT}/summary.json" "${OUT}/flight.ulg" "${HERE}" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[3])
from pathlib import Path
import sitl_test
summary = json.loads(Path(sys.argv[1]).read_text())
if "error" not in summary:
    sitl_test.check_ulog(Path(sys.argv[2]), summary)
    summary["passed"] = all(summary["checks"].values())
    Path(sys.argv[1]).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("ulog", "checks", "passed") if k in summary}, indent=2))
EOF
    grep -q '"passed": true' "${OUT}/summary.json" || status=1
else
    echo "warning: no ulog found under ${ROOTFS}/log" >&2
    status=1
fi
rm -rf "${ROOTFS}"
echo "run folder: ${OUT}"
exit "${status}"
