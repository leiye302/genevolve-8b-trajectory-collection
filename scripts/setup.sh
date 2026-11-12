#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
python_bin="${GENEVOLVE_PYTHON:-python3.11}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
if [[ "$(uname -s)" != Linux ]]; then
  echo 'CUDA collection setup requires Linux.' >&2
  exit 1
fi
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q 'release 12.8'; then
  echo 'Activate a CUDA 12.8 toolkit for the pinned FlashAttention build.' >&2
  exit 1
fi
"$python_bin" -m venv .venv
py=.venv/bin/python
"$py" -m pip install --upgrade pip wheel setuptools ninja packaging psutil
"$py" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install -r requirements.txt
MAX_JOBS="${GENEVOLVE_BUILD_JOBS:-4}" "$py" -m pip install --no-build-isolation flash-attn==2.8.3
"$py" -m pip check
"$py" collect.py setup
"$py" -c 'import flash_attn, torch; assert torch.version.cuda == "12.8"'
