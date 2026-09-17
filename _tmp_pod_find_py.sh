#!/bin/bash
set -euo pipefail
echo "=== runtime env ==="
source /workspace/otr-config/otr-runtime.env
echo "COMFY_PY=${COMFY_PY-}"
echo "OTR_COMFY_ROOT=${OTR_COMFY_ROOT-}"
echo "OTR_REPO_ROOT=${OTR_REPO_ROOT-}"
echo "=== which python ==="
command -v python3 || true
command -v python || true
ls -ld /workspace/runpod-slim/ComfyUI/.venv* 2>/dev/null || echo "NO_VENV_GLOB"
ls -ld /workspace/runpod-slim/ComfyUI/.venv-py313 2>/dev/null || echo "NO_VENV_PY313"
find /workspace/runpod-slim -maxdepth 4 -type f -name python -path '*venv*' 2>/dev/null | head
find /usr -name python3 -type f 2>/dev/null | head
ls /workspace/runpod-slim/ComfyUI | head
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
