# The mock training and scoring image for the GPU server (docs/gpu-server.md "Runs in Docker").
# Every run on the shared server goes inside a container (Zain, 2026-11-26). The image holds
# the Python dependencies only; the repo is mounted at /work by run.sh, so a new commit needs
# no rebuild, only a change to pyproject.toml's dependencies does.
#
#   training/docker/run.sh build
#   training/docker/run.sh python -m training.queue training/queues/server_week1.yaml --max-gpus 2
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# The same versions as the server's venv on 2026-11-26 (torch's PyPI wheel brings its own
# CUDA runtime; the host driver 580 / CUDA 13 runs it). rsl-rl-lib is pinned in pyproject.toml.
# The list mirrors pyproject.toml's torch, dev and sitl extras; keep the two in step.
ARG TORCH_VERSION=2.15.0
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir "numpy>=1.24" "pyyaml>=6.0" "rsl-rl-lib==5.5.1" \
        "onnx>=1.16" "onnxruntime>=1.17" "pytest>=8.0" "ruff>=0.6" "pyulog>=1.0" \
        "pymavlink>=2.4.41" tensorboard

ENV PYTHONPATH=/work PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /work
