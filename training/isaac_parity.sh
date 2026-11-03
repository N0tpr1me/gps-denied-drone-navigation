#!/usr/bin/env bash
# The Isaac Lab first hour as one command (docs/isaac-lab.md "First hour on the new server",
# steps 3 to 9), run from the repo on the GPU server after training/server_setup.sh has pulled
# the image.
#
#   bash training/isaac_parity.sh               # a free GPU, picked by nvidia-smi
#   bash training/isaac_parity.sh --gpu 2       # that GPU (the queue runner passes it)
#   bash training/isaac_parity.sh --no-train    # stop after the parity report (step 6)
#
# Inside nvcr.io/nvidia/isaac-lab:2.3.2 (one container, this repo mounted, Isaac's caches on
# the host so the second start is fast) it installs the repo into Isaac's Python keeping
# Isaac's torch and numpy, runs the stock headless smoke, the repo's sim and parity tests,
# the parity report (must end "parity: PASS"), then the v22 recipe's smoke run on the mock and
# on Isaac (steps/s of both) and v22 scored on Isaac against its mock numbers. Everything
# lands in runs/isaac_parity_<time>/ with report.md; paste that in the project. Exit 0 only
# when every step passed. Uses only the one GPU it is given.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO="$(pwd)"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}"
GPU=""
TRAIN=1
for ((i = 1; i <= $#; i++)); do
    case "${!i}" in
        --gpu) ((i++)); GPU="${!i}" ;;
        --gpu=*) GPU="${!i#--gpu=}" ;;
        --no-train) TRAIN=0 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: ${!i}" >&2; exit 2 ;;
    esac
done
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT="runs/isaac_parity_${STAMP}"
mkdir -p "${OUT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found: this runs on the GPU server only" >&2
    exit 3
fi
if [[ -z "${GPU}" ]]; then
    # The first GPU with nothing on it, by the same test as training/launch.py.
    GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', *' '$2 < 1024 && $3 < 10 { print $1; exit }')"
    if [[ -z "${GPU}" ]]; then
        echo "no free GPU (nvidia-smi):" >&2
        nvidia-smi >&2
        exit 3
    fi
fi
echo "isaac_parity: GPU ${GPU}, image ${IMAGE}, output ${OUT}"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}" || exit 1
docker inspect --format '{{index .RepoDigests 0}}' "${IMAGE}" >"${OUT}/image_digest.txt" 2>/dev/null

CACHE="${HOME}/docker/isaac-sim"
mkdir -p "${CACHE}"/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}

# The steps, run inside the container. Each writes its own log under OUT and a line in
# steps.txt; the report is built from those on the host afterwards.
cat >"${OUT}/inside.sh" <<'EOF'
set -uo pipefail
cd /workspace/dronenav
OUT="$1"; TRAIN="$2"
ISAAC_DIR="$(ls -d /workspace/isaaclab /workspace/IsaacLab 2>/dev/null | head -1)"
PY="${ISAAC_DIR}/isaaclab.sh -p"
FAILED=0
step() {  # step <name> <log> <command...>
    local name="$1" log="$2"; shift 2
    echo "=== ${name} ==="
    if "$@" >"${OUT}/${log}" 2>&1; then echo "${name} | PASS | ${log}" >>"${OUT}/steps.txt"
    else tail -20 "${OUT}/${log}"; echo "${name} | FAIL | ${log}" >>"${OUT}/steps.txt"; FAILED=1; fi
}
${PY} -m pip freeze | grep -E '^(torch|numpy)==' >/tmp/keep.txt
step "3. repo into Isaac's Python (keeping Isaac's torch and numpy)" install.log \
    ${PY} -m pip install -e '.[torch,dev]' -c /tmp/keep.txt
step "3b. isaac_available" available.log \
    ${PY} -c "import isaaclab, envs.sims.isaac_quad as q; assert q.isaac_available(); print('isaac ok')"
step "4. Isaac starts headless" headless.log \
    ${PY} "${ISAAC_DIR}/scripts/tutorials/00_sim/log_time.py" --headless
step "5. repo tests in Isaac's Python (no Isaac SKIPs allowed)" tests.log \
    ${PY} -m pytest -rs envs/sims eval/tests/test_sim_parity.py training/tests/test_isaac_ready.py
if grep -qi "skipped.*isaac" "${OUT}/tests.log"; then
    echo "5b. tests skipped for an Isaac reason | FAIL | tests.log" >>"${OUT}/steps.txt"; FAILED=1
fi
step "6. parity report torch vs isaac" parity.log \
    ${PY} -m eval.sim_parity --a torch --b isaac --device cuda --out "${OUT}/parity_isaac.json"
grep -q "parity: PASS" "${OUT}/parity.log" || { echo "6b. parity: PASS line | FAIL | parity.log" >>"${OUT}/steps.txt"; FAILED=1; }
if [[ "${TRAIN}" == "1" ]]; then
    step "7. v22 recipe smoke on the mock, GPU" smoke_torch.log \
        ${PY} -m training.launch --config training/configs/structure_hold_ppo_v22.yaml --sim torch --smoke --device cuda --allow-dirty --quiet
    step "8. v22 recipe smoke on Isaac, GPU" smoke_isaac.log \
        ${PY} -m training.launch --config training/configs/structure_hold_ppo_v22.yaml --sim isaac --smoke --device cuda --allow-dirty --quiet
    step "9. v22 scored on Isaac (mock: p95 0.211 / 0.206 / 0.202 m, crash under 1 %)" eval_isaac.log \
        ${PY} -m training.launch --config training/configs/structure_hold_eval_v22.yaml --sim isaac --device cuda --quiet
fi
exit "${FAILED}"
EOF

docker run --rm --name "dronenav-isaac-${STAMP}" --entrypoint bash \
    --gpus "\"device=${GPU}\"" --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${CACHE}/cache/ov:/root/.cache/ov:rw" \
    -v "${CACHE}/cache/pip:/root/.cache/pip:rw" \
    -v "${CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "${CACHE}/cache/computecache:/root/.nv/ComputeCache:rw" \
    -v "${CACHE}/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "${CACHE}/data:/root/.local/share/ov/data:rw" \
    -v "${CACHE}/documents:/root/Documents:rw" \
    -v "${REPO}:/workspace/dronenav" -w /workspace/dronenav \
    "${IMAGE}" "/workspace/dronenav/${OUT}/inside.sh" "${OUT}" "${TRAIN}"
STATUS=$?

{
    echo "# Isaac Lab parity report"
    echo
    echo "$(date -u +%Y-%m-%dT%H:%MZ), commit $(git rev-parse --short HEAD 2>/dev/null), GPU ${GPU}, image $(cat "${OUT}/image_digest.txt" 2>/dev/null)"
    echo
    echo "| Step | Result | Log |"
    echo "| --- | --- | --- |"
    sed 's/^/| /; s/$/ |/' "${OUT}/steps.txt" 2>/dev/null
    echo
    echo "## Parity table"
    echo '```'
    grep -A200 -m1 "^|" "${OUT}/parity.log" 2>/dev/null | head -60
    echo '```'
    echo
    echo "## Steps per second (smoke runs)"
    echo '```'
    grep -h "env_steps_per_s" "${OUT}"/smoke_*.log 2>/dev/null
    echo '```'
    echo
    echo "## v22 on Isaac (step 9 headline)"
    echo '```'
    grep -h "headline\|p95" "${OUT}/eval_isaac.log" 2>/dev/null | tail -5
    echo '```'
} >"${OUT}/report.md"
echo
cat "${OUT}/steps.txt" 2>/dev/null
echo "report: ${OUT}/report.md (paste it in the project)"
exit "${STATUS}"
