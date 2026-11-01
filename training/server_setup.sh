#!/usr/bin/env bash
# One-shot setup and self-test for the GPU server (Linux x86_64, NVIDIA, Docker).
#
#   git clone https://github.com/N0tpr1me/gps-denied-drone-navigation.git
#   cd gps-denied-drone-navigation
#   bash training/server_setup.sh               # everything, about 1 hour the first time
#   bash training/server_setup.sh --no-isaac    # skip the Isaac Lab image pull (~20 GB)
#
# Safe to re-run: every step is skipped or redone cheaply once it has passed. Uses no sudo
# and installs nothing system-wide; the driver, Docker and the NVIDIA Container Toolkit are
# the admin's part (docs/gpu-server.md). Uses only free GPUs. Writes everything to
# runs/server_setup_<time>/ and ends with report.md, which is safe to paste in the project
# (no secrets, no hostnames).
#
# Steps: machine report -> hard checks -> Python env -> lint + tests -> training smoke on
# CPU and on one free GPU -> SITL image (PX4 v1.17.0 + the released policy) -> hover test ->
# parallel flight check (several hover tests at once, real-time factor of each) -> Isaac Lab
# image pull and digest. The first hour on Isaac itself is docs/isaac-lab.md.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO="$(pwd)"
OUT="${REPO}/runs/server_setup_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "${OUT}"
ISAAC=1
SITL=1
ENV_DIR="${DRONENAV_VENV:-${REPO}/.venv}"
ISAAC_IMAGE="nvcr.io/nvidia/isaac-lab:2.3.2"
for arg in "$@"; do
    case "${arg}" in
        --no-isaac) ISAAC=0 ;;
        --no-sitl) SITL=0 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

RESULTS=()
FAILED=0
have() { command -v "$1" >/dev/null 2>&1; }
step() { printf '\n=== %s ===\n' "$1"; }
pass() { RESULTS+=("| $1 | PASS | ${2:-} |"); echo "PASS: $1 ${2:-}"; }
fail() { RESULTS+=("| $1 | FAIL | ${2:-} |"); echo "FAIL: $1 ${2:-}" >&2; FAILED=1; }
skip() { RESULTS+=("| $1 | skipped | ${2:-} |"); echo "skipped: $1 ${2:-}"; }
# run <name> <log> <command...>: run, keep the output in OUT/<log>, record pass/fail.
run() {
    local name="$1" log="$2"
    shift 2
    if "$@" >"${OUT}/${log}" 2>&1; then pass "${name}" "(${log})"; return 0; fi
    tail -20 "${OUT}/${log}" >&2
    fail "${name}" "(see ${log})"
    return 1
}

finish() {
    {
        echo "# Server setup report"
        echo
        echo "$(date -u +%Y-%m-%dT%H:%MZ), commit $(git rev-parse --short HEAD 2>/dev/null)"
        echo
        echo "| Step | Result | Notes |"
        echo "| --- | --- | --- |"
        printf '%s\n' "${RESULTS[@]}"
        echo
        echo '```'
        cat "${OUT}/server_report.txt" 2>/dev/null
        echo '```'
    } >"${OUT}/report.md"
    echo
    printf '%s\n' "${RESULTS[@]}"
    echo
    echo "report: ${OUT}/report.md (paste it in the project)"
    exit "${FAILED}"
}

# --------------------------------------------------------------------------------------
step "1. Machine report"
bash training/server_report.sh >"${OUT}/server_report.txt" 2>&1
pass "machine report" "(server_report.txt)"

step "2. Hard checks"
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    pass "Linux x86_64"
else
    fail "Linux x86_64" "this is $(uname -s) $(uname -m)"
fi
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    pass "NVIDIA driver" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), $(nvidia-smi -L | wc -l) GPUs"
else
    fail "NVIDIA driver" "nvidia-smi missing or failing"
fi
if have docker && docker info >/dev/null 2>&1; then
    pass "Docker without sudo"
else
    fail "Docker without sudo" "install Docker and add this user to the docker group"
fi
free_gb=$(df -BG --output=avail "${REPO}" | tail -1 | tr -dc '0-9')
if [[ "${free_gb:-0}" -ge 80 ]]; then
    pass "disk" "${free_gb} GB free"
else
    fail "disk" "${free_gb} GB free; need 80 GB (SITL image ~10 GB, Isaac Lab ~25 GB, runs)"
fi
if [[ "${FAILED}" -ne 0 ]]; then
    echo "fix the failed checks above (docs/gpu-server.md), then re-run" >&2
    finish
fi
# GPUs inside containers need the NVIDIA Container Toolkit (Isaac Lab; SITL does not).
run "GPU inside Docker" gpu_docker.log \
    docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L

# --------------------------------------------------------------------------------------
step "3. Python env (${ENV_DIR})"
# A plain venv: conda is fine too (environment.yml), but a venv needs nothing installed.
PY="${ENV_DIR}/bin/python"
if [[ ! -x "${PY}" ]]; then
    if have uv; then
        uv venv -p 3.11 "${ENV_DIR}" >"${OUT}/venv.log" 2>&1
    elif have python3.11; then
        python3.11 -m venv "${ENV_DIR}" >"${OUT}/venv.log" 2>&1
    else
        python3 -m venv "${ENV_DIR}" >"${OUT}/venv.log" 2>&1
    fi
fi
if have uv; then
    run "install repo" pip_install.log env VIRTUAL_ENV="${ENV_DIR}" uv pip install -e ".[dev,torch,sitl]"
else
    run "install repo" pip_install.log "${PY}" -m pip install -e ".[dev,torch,sitl]"
fi
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" \
    >"${OUT}/torch.txt" 2>&1
grep -q "cuda True" "${OUT}/torch.txt" && pass "torch sees CUDA" "$(cat "${OUT}/torch.txt")" \
    || fail "torch sees CUDA" "$(tail -1 "${OUT}/torch.txt")"

step "4. Lint and tests"
run "ruff" ruff.log "${ENV_DIR}/bin/ruff" check .
run "pytest" pytest.log "${PY}" -m pytest -q -rs -x

step "5. Training smoke runs"
run "launch smoke (CPU)" smoke_cpu.log \
    "${PY}" -m training.launch --config training/configs/selfcheck.yaml --smoke --device cpu
# The released policy's PPO recipe, a few iterations on one GPU: launch.py reads nvidia-smi
# and picks a free GPU, or refuses.
run "PPO smoke (1 free GPU)" smoke_gpu.log \
    "${PY}" -m training.launch --config training/configs/structure_hold_ppo_v22.yaml --smoke --device cuda

# --------------------------------------------------------------------------------------
if [[ "${SITL}" -eq 1 ]]; then
    step "6. SITL image"
    policy=$(sed -n 's/.*NAME = "\(.*\)";/\1/p' sim2real/px4/structure_hold/PolicyWeights.hpp)
    run "SITL image build" sitl_build.log sim2real/docker/run.sh build \
        && RESULTS[-1]="| SITL image build | PASS | PX4 v1.17.0, policy ${policy} (sitl_build.log) |"

    step "7. Hover test"
    run "hover test" hover_test.log sim2real/docker/run.sh hover-test

    step "8. Parallel flights"
    # As many 60 s hovers at once as the CPUs allow (4 per flight), each pinned to its own
    # CPUs; each must keep Gazebo near real time, as one flight alone on the Mac did (0.97
    # mean, 0.85 min). Stock PX4 hover on the tuned params: nothing is logged from it.
    jobs=$(( $(nproc) / 4 ))
    (( jobs > 8 )) && jobs=8
    seeds=$(seq -s ' ' 1 "${jobs}")
    if run "parallel hovers (${jobs} at once)" parallel.log \
        "${PY}" -m eval.baselines.run_baseline --label server_parallel_check \
            --params eval/baselines/px4_tuned_v1.params --scenarios wall1m_calm_0ms \
            --seeds ${seeds} --jobs "${jobs}" -- --profile vio_at_hover; then
        batch=$(sed -n 's/^batch folder: //p' "${OUT}/parallel.log")
        rtf=$("${PY}" -c "import json,sys; b=json.load(open(sys.argv[1]+'/batch.json')); r=[x['rtf_min'] for x in b['rows']]; print(len(r), 'flights, worst real-time factor', min(v for v in r if v is not None) if any(r) else None)" "${batch}")
        RESULTS[-1]="| parallel hovers (${jobs} at once) | PASS | ${rtf} |"
    fi
else
    skip "SITL image, hover test, parallel flights" "--no-sitl"
fi

# --------------------------------------------------------------------------------------
if [[ "${ISAAC}" -eq 1 ]]; then
    step "9. Isaac Lab image"
    if run "pull ${ISAAC_IMAGE}" isaac_pull.log docker pull "${ISAAC_IMAGE}"; then
        digest=$(docker inspect --format '{{index .RepoDigests 0}}' "${ISAAC_IMAGE}")
        RESULTS[-1]="| pull Isaac Lab 2.3.2 | PASS | ${digest} |"
        echo "next: the first-hour checklist in docs/isaac-lab.md (steps 3 to 9)"
    else
        echo "if the pull says unauthorized: free NGC account, then docker login nvcr.io" >&2
    fi
else
    skip "Isaac Lab image" "--no-isaac"
fi

finish
