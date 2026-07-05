#!/bin/bash
set -euo pipefail

readonly MARKER_FILE="/opt/nnmcts/.gpu-deps-ready"
readonly CLOUDWATCH_AGENT_CTL="/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"

deps_importable() {
  python -c "import torch, numpy, tqdm" >/dev/null 2>&1
}

if [[ -f "${MARKER_FILE}" ]] && deps_importable; then
  echo "$(date -Is) GPU deps already installed (${MARKER_FILE})."
  exit 0
fi

if [[ ! -x "${CLOUDWATCH_AGENT_CTL}" ]]; then
  echo "$(date -Is) Installing amazon-cloudwatch-agent."
  dnf install -y amazon-cloudwatch-agent
else
  echo "$(date -Is) amazon-cloudwatch-agent already present; skipping dnf install."
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "$(date -Is) Installing unzip."
  dnf install -y unzip
else
  echo "$(date -Is) unzip already present; skipping dnf install."
fi

if [[ -f /opt/pytorch/bin/activate ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/pytorch/bin/activate
  set -u
  echo "$(date -Is) Using DLAMI PyTorch environment."
else
  echo "$(date -Is) DLAMI PyTorch environment not found; installing CUDA torch via pip."
  python3 -m pip install --quiet --upgrade pip
  python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install --quiet --upgrade pip

if ! python -c "import numpy" >/dev/null 2>&1; then
  echo "$(date -Is) Installing numpy."
  python -m pip install --quiet numpy
else
  echo "$(date -Is) numpy already importable; skipping pip install."
fi

if ! python -c "import tqdm" >/dev/null 2>&1; then
  echo "$(date -Is) Installing tqdm."
  python -m pip install --quiet tqdm
else
  echo "$(date -Is) tqdm already importable; skipping pip install."
fi

if ! deps_importable; then
  echo "$(date -Is) GPU dependency verification failed after install."
  exit 1
fi

mkdir -p /opt/nnmcts
date -Is > "${MARKER_FILE}"
echo "$(date -Is) GPU deps ready (marker written to ${MARKER_FILE})."
