#!/usr/bin/env bash
# Run mock training or scoring inside the dronenav-train container (docs/gpu-server.md
# "Runs in Docker"). The repo is mounted at /work, runs/ lands in the checkout as usual and
# files are owned by the calling user. The container sees every GPU so launch.py's
# nvidia-smi check can still pick only free cards (CUDA_VISIBLE_DEVICES uses host indices).
#
#   training/docker/run.sh build                    # once, and after a dependency change
#   training/docker/run.sh smoke                    # tests + the selfcheck smoke on GPU
#   training/docker/run.sh <command ...>            # e.g. python -m training.queue ...
set -euo pipefail

IMAGE="${DRONENAV_TRAIN_IMAGE:-dronenav-train:latest}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

run() {
  docker run --rm --init --gpus all --ipc=host \
    --user "$(id -u):$(id -g)" \
    --cpus "${DRONENAV_CPUS:-16}" \
    -v "$REPO":/work -w /work \
    --name "dronenav-$(date +%H%M%S)-$$" \
    "$IMAGE" "$@"
}

case "${1:-}" in
  build)
    # The build context is this folder only, so runs/ and checkpoints never go to the daemon.
    docker build -t "$IMAGE" "$REPO/training/docker"
    ;;
  smoke)
    run bash -c "python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' \
      && python -m training.launch --config training/configs/selfcheck.yaml --smoke --device cuda"
    ;;
  "")
    echo "usage: $0 build | smoke | <command ...>" >&2
    exit 2
    ;;
  *)
    run "$@"
    ;;
esac
