#!/usr/bin/env bash
# BUILD A PYTHON 3.13 VENV FOR OTR ON A POD, the third and working cut.
#
# Why 3.13 at all: the official PyTorch pod template ships 3.11, and the
# operator's instruction on 2026-09-12 was to test the bleeding edge rather
# than the template's default. On 3.13 the Kokoro voices route through
# kokoro-onnx on the CPU instead of the torch package -- same voices.
#
# The two traps this cut exists to avoid, both measured on 2026-09-13:
#   1. The container disk is small and `uv` stages wheels in TMPDIR, so a
#      venv built with the default temp dir dies mid-install. TMPDIR is
#      pointed at the volume and UV_NO_CACHE is set.
#   2. The network volume is SHARED and near its quota. A build that reports
#      "Disk quota exceeded" is not a broken venv; it is a full disk.
#
# Lives in the repo so it survives the pod being stopped.

set -uo pipefail
COMFY="${OTR_COMFY_ROOT:-/workspace/runpod-slim/ComfyUI}"
VENV="$COMFY/.venv-py313"
export PATH="$HOME/.local/bin:$PATH" UV_NO_CACHE=1 TMPDIR=/tmp
echo "=== venv 3.13 build (volume venv, /tmp staging) $(date -u '+%H:%M:%SZ') ==="
df -h / | tail -n 1
[ -x "$VENV/bin/python" ] || uv venv --python 3.13 "$VENV" || { echo "FATAL: uv could not make a 3.13 venv"; exit 1; }
PY="$VENV/bin/python"
"$PY" --version
echo "torch cu128 trio $(date -u '+%H:%M:%SZ')"
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" 2>&1 | tail -n 3 \
  || { echo "FATAL: torch cu128 for 3.13 failed"; exit 1; }
"$PY" -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__, torch.version.cuda, "cuda ok:", torch.cuda.is_available())' || { echo "FATAL: torch import failed"; exit 1; }
du -sh "$VENV"
echo "pip into the venv $(date -u '+%H:%M:%SZ')"
uv pip install --python "$PY" pip setuptools wheel 2>&1 | tail -n 1
echo "ComfyUI requirements $(date -u '+%H:%M:%SZ')"
uv pip install --python "$PY" -r "$COMFY/requirements.txt" 2>&1 | tail -n 2
cd "$COMFY/custom_nodes"
for req in ComfyUI-OldTimeRadio/requirements.txt ComfyUI-GGUF/requirements.txt ComfyUI-LTXVideo/requirements.txt ComfyUI-KJNodes/requirements.txt ComfyUI-AnimateDiff-Evolved/requirements.txt; do
  [ -f "$req" ] || continue
  echo "pack requirements: $req $(date -u '+%H:%M:%SZ')"
  uv pip install --python "$PY" -r "$req" 2>&1 | grep -iE "error|failed|no matching|building" | head -n 5
done
du -sh "$VENV"
echo "=== import smoke ==="
"$PY" - <<'PYEOF'
import importlib
mods = ["torch", "torchaudio", "transformers", "bitsandbytes", "kokoro", "kokoro_onnx", "onnxruntime", "gguf", "safetensors", "av", "soundfile", "librosa"]
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print("ok  ", m, v)
    except Exception as e:
        print("MISS", m, type(e).__name__, str(e)[:80])
PYEOF
echo "=== venv 3.13 build done $(date -u '+%H:%M:%SZ') ==="
