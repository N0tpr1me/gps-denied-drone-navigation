#!/usr/bin/env bash
# Print the facts we need about a new GPU server before installing anything.
#   bash training/server_report.sh | tee server_report.txt
# Read-only: it installs nothing and changes nothing. Missing tools are reported, not fatal.
# Paste the output into the project (it holds no secrets) so the install path can be chosen.

section() { printf '\n== %s ==\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "OS and CPU"
uname -srm
if [ -r /etc/os-release ]; then . /etc/os-release && echo "distro: ${PRETTY_NAME:-unknown}"; fi
if have nproc; then echo "cpus: $(nproc)"; fi
if have lscpu; then lscpu | grep -E '^Model name' || true; fi

section "Memory and disk"
if have free; then free -g | head -2; fi
df -h "$HOME" / 2>/dev/null | awk 'NR == 1 || !seen[$0]++'

section "NVIDIA driver and GPUs"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
    --format=csv
  echo "processes using GPUs:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    || true
  nvidia-smi | grep -i 'CUDA Version' || true
else
  echo "nvidia-smi: NOT FOUND (no NVIDIA driver, or not on PATH)"
fi

section "Docker"
if have docker; then
  docker --version
  if docker info >/dev/null 2>&1; then
    echo "docker without sudo: yes"
    docker info 2>/dev/null | grep -iE 'runtimes|default runtime' || true
  else
    echo "docker without sudo: NO (daemon not running, or this user lacks permission)"
  fi
else
  echo "docker: NOT FOUND"
fi
if have nvidia-ctk; then nvidia-ctk --version | head -1; else echo "nvidia-ctk: NOT FOUND"; fi

section "Python and conda"
for tool in python3 conda mamba; do
  if have "$tool"; then echo "$tool: $("$tool" --version 2>&1 | head -1)"; else echo "$tool: not found"; fi
done

section "Shared machine?"
echo "logged-in users: $(who 2>/dev/null | awk '{print $1}' | sort -u | tr '\n' ' ')"
for tool in sbatch qsub; do
  if have "$tool"; then echo "job scheduler present: $tool (use it; do not bypass it)"; fi
done

section "Outbound network"
for url in https://github.com https://nvcr.io/v2/ https://pypi.org/simple/; do
  if have curl; then
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$url" || true)
    echo "$url -> ${code:-no answer}"
  fi
done
if ! have curl; then echo "curl not found; network not checked"; fi

section "Done"
echo "Report generated $(date -u +%Y-%m-%dT%H:%MZ)"
