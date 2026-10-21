#!/usr/bin/env bash
# Wire the Structure Hold module into a PX4-Autopilot source tree (pinned tag in
# docs/decisions.md), idempotently:
#
#   1. src/modules/structure_hold      -> this folder's structure_hold/ (copy, or symlink with --link)
#   2. msg/StructureHoldStatus.msg      added to the uORB message list
#   3. CONFIG_MODULES_STRUCTURE_HOLD=y  in the board config (default: boards/px4/sitl/default.px4board)
#   4. structure_hold start             in ROMFS init.d/rc.mc_apps, guarded by SH_EN
#   5. structure_hold_status            logged by default (logger/logged_topics.cpp)
#
# Usage: install.sh <PX4_DIR> [--link] [--board boards/<vendor>/<model>/<config>.px4board]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${1:?usage: install.sh <PX4_DIR> [--link] [--board <px4board>]}"
shift
LINK=0
BOARD="boards/px4/sitl/default.px4board"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --link) LINK=1 ;;
        --board) BOARD="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

[[ -f "${PX4_DIR}/CMakeLists.txt" && -d "${PX4_DIR}/src/modules" ]] \
    || { echo "not a PX4-Autopilot tree: ${PX4_DIR}" >&2; exit 1; }

# 1. module sources
dest="${PX4_DIR}/src/modules/structure_hold"
rm -rf "${dest}"
if [[ "${LINK}" -eq 1 ]]; then
    ln -s "${HERE}/structure_hold" "${dest}"
else
    cp -r "${HERE}/structure_hold" "${dest}"
fi

# 2. uORB message, kept in alphabetical order in the unversioned list
cp "${HERE}/msg/StructureHoldStatus.msg" "${PX4_DIR}/msg/StructureHoldStatus.msg"
python3 - "${PX4_DIR}/msg/CMakeLists.txt" <<'EOF'
import re, sys
path = sys.argv[1]
text = open(path).read()
name = "StructureHoldStatus.msg"
if name not in text:
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("set(msg_files"))
    i = start + 1
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(")"):
        entry = lines[i].strip()
        if not entry.startswith("versioned/") and entry > name:
            break
        i += 1
    lines.insert(i, "\t" + name)
    open(path, "w").write("\n".join(lines))
EOF

# 3. board config
board_file="${PX4_DIR}/${BOARD}"
grep -q "^CONFIG_MODULES_STRUCTURE_HOLD=y" "${board_file}" \
    || echo "CONFIG_MODULES_STRUCTURE_HOLD=y" >> "${board_file}"

# 4. start at boot (multicopter airframes), guarded by SH_EN
apps="${PX4_DIR}/ROMFS/px4fmu_common/init.d/rc.mc_apps"
if ! grep -q "structure_hold start" "${apps}"; then
    cat >> "${apps}" <<'EOF'

if param compare -s SH_EN 1
then
	structure_hold start
fi
EOF
fi

# 5. log the status topic by default
topics="${PX4_DIR}/src/modules/logger/logged_topics.cpp"
if ! grep -q "structure_hold_status" "${topics}"; then
    python3 - "${topics}" <<'EOF'
import sys
path = sys.argv[1]
text = open(path).read()
anchor = '\tadd_topic("vehicle_rates_setpoint", 20);\n'
assert anchor in text, "logged_topics.cpp anchor not found"
text = text.replace(anchor, anchor + '\tadd_topic("structure_hold_status", 10);\n', 1)
open(path, "w").write(text)
EOF
fi

echo "structure_hold installed into ${PX4_DIR} (board ${BOARD}, $([[ ${LINK} -eq 1 ]] && echo linked || echo copied))"
