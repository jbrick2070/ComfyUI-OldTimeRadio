#!/usr/bin/env bash
# =============================================================================
# OTR v2.0-alpha cloud setup script
# =============================================================================
# Designed for SimplePod / RunPod / Vast.ai-style rented ComfyUI instances.
#
# Usage (from a bash prompt on the cloud instance):
#   curl -L https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/scripts/setup_cloud.sh | bash
#
# What this does (in 4 phases):
#   1. CODE: clone OTR v2.0-alpha + 6 supporting custom nodes
#   2. PYTHON: install OTR's requirements.txt + custom-node requirements
#   3. HF_HOME: point HF cache at ComfyUI/models/huggingface (one tidy folder)
#   4. PRINT: huggingface-cli download commands for the 50+ GB of weights
#
# Idempotent: re-running is safe; clones skip if directory already exists.
# Stops on first hard failure (set -e).
#
# Important: C7 byte-identity does NOT hold across hardware. A cloud render
# is a different artifact, not a substitute for a local render.
# =============================================================================

set -e
set -o pipefail

# -----------------------------------------------------------------------------
# 0. Detect ComfyUI install location
# -----------------------------------------------------------------------------
echo "[0/4] Detecting ComfyUI install path..."

CANDIDATES=(
  "/workspace/ComfyUI"
  "/app/ComfyUI"
  "/home/comfyui/ComfyUI"
  "/opt/ComfyUI"
  "/ComfyUI"
)

COMFY_ROOT=""
for c in "${CANDIDATES[@]}"; do
  if [ -d "$c/custom_nodes" ] && [ -f "$c/main.py" ]; then
    COMFY_ROOT="$c"
    break
  fi
done

if [ -z "$COMFY_ROOT" ]; then
  echo "  WARN: standard paths not found, scanning filesystem (may take 30s)..."
  COMFY_ROOT=$(find / -maxdepth 5 -name "main.py" -path "*/ComfyUI/main.py" 2>/dev/null | head -1)
  COMFY_ROOT=$(dirname "$COMFY_ROOT" 2>/dev/null || true)
fi

if [ -z "$COMFY_ROOT" ] || [ ! -d "$COMFY_ROOT/custom_nodes" ]; then
  echo "  FATAL: cannot find ComfyUI install. Halt."
  exit 1
fi

echo "  ComfyUI root: $COMFY_ROOT"
CUSTOM_NODES="$COMFY_ROOT/custom_nodes"
MODELS_ROOT="$COMFY_ROOT/models"

# -----------------------------------------------------------------------------
# 1. Detect Python (ComfyUI's venv or system python)
# -----------------------------------------------------------------------------
echo "[1/4] Detecting Python interpreter..."

PYBIN=""
for p in "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/.venv/bin/python" "/usr/bin/python3" "/usr/local/bin/python3" "$(which python3)"; do
  if [ -x "$p" ]; then
    PYBIN="$p"
    break
  fi
done

if [ -z "$PYBIN" ]; then
  echo "  FATAL: no python3 found. Halt."
  exit 1
fi

echo "  Python: $PYBIN"
echo "  Version: $($PYBIN --version)"

PIP="$PYBIN -m pip"

# -----------------------------------------------------------------------------
# 2. Clone OTR v2.0-alpha + supporting custom nodes
# -----------------------------------------------------------------------------
echo "[2/4] Cloning OTR + supporting custom nodes..."
cd "$CUSTOM_NODES"

# OTR (the main repo)
if [ -d "ComfyUI-OldTimeRadio" ]; then
  echo "  ComfyUI-OldTimeRadio already present, pulling latest v2.0-alpha"
  cd ComfyUI-OldTimeRadio
  git fetch origin
  git checkout v2.0-alpha
  git pull origin v2.0-alpha
  cd ..
else
  git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git
fi

# Supporting custom nodes (idempotent: skip if already cloned)
declare -A NODES=(
  [ComfyUI-Manager]="https://github.com/Comfy-Org/ComfyUI-Manager.git"
  [ComfyUI-KJNodes]="https://github.com/kijai/ComfyUI-KJNodes.git"
  [ComfyUI-GGUF]="https://github.com/city96/ComfyUI-GGUF.git"
  [comfyui-videohelpersuite]="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
  [comfyui-kokorotts]="https://github.com/stavsap/comfyui-kokorotts.git"
  [ComfyUI-LTXVideo]="https://github.com/Lightricks/ComfyUI-LTXVideo.git"
)

for name in "${!NODES[@]}"; do
  if [ -d "$name" ]; then
    echo "  $name already present, skipping clone"
  else
    echo "  Cloning $name..."
    git clone --depth 1 "${NODES[$name]}" "$name" || echo "  WARN: $name clone failed (continuing)"
  fi
done

# -----------------------------------------------------------------------------
# 3. Install Python requirements
# -----------------------------------------------------------------------------
echo "[3/4] Installing Python requirements..."

# OTR's own requirements
if [ -f "$CUSTOM_NODES/ComfyUI-OldTimeRadio/requirements.txt" ]; then
  echo "  Installing OTR requirements..."
  $PIP install --no-input -q -r "$CUSTOM_NODES/ComfyUI-OldTimeRadio/requirements.txt" || true
fi

# Each supporting node's requirements
for name in "${!NODES[@]}"; do
  req="$CUSTOM_NODES/$name/requirements.txt"
  if [ -f "$req" ]; then
    echo "  Installing $name requirements..."
    $PIP install --no-input -q -r "$req" || echo "  WARN: $name requirements install had errors (continuing)"
  fi
done

# Extra packages OTR commonly needs that aren't in requirements.txt
echo "  Installing extra deps (huggingface_hub, sounddevice, etc.)..."
$PIP install --no-input -q huggingface_hub feedparser bitsandbytes sentencepiece || true

# -----------------------------------------------------------------------------
# 4. Set up HF_HOME pointing into ComfyUI/models/huggingface
# -----------------------------------------------------------------------------
echo "[4/4] Setting up HF_HOME..."

HF_HOME_DIR="$MODELS_ROOT/huggingface"
mkdir -p "$HF_HOME_DIR"

# Persist for future shells
if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
  echo "export HF_HOME=$HF_HOME_DIR" >> ~/.bashrc
fi
export HF_HOME="$HF_HOME_DIR"
echo "  HF_HOME = $HF_HOME"

# -----------------------------------------------------------------------------
# Summary + next steps
# -----------------------------------------------------------------------------
cat <<EOF

=============================================================================
SETUP COMPLETE -- code is in place. Now download the model weights.
=============================================================================

ComfyUI root:     $COMFY_ROOT
Custom nodes:     $CUSTOM_NODES
Models root:      $MODELS_ROOT
HF_HOME:          $HF_HOME_DIR
Python:           $PYBIN

=============================================================================
NEXT: download model weights (50+ GB total, run in parallel terminals)
=============================================================================

# Set HF token if you have one (gated models like FLUX.1-dev / Mistral need it):
# export HF_TOKEN=hf_xxxxx

# 1. HuMo 14B fp8 (the lip-sync video model, ~16 GB)
huggingface-cli download Kijai/WanVideo_comfy \\
    Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors \\
    --local-dir $MODELS_ROOT/diffusion_models/

# 2. FLUX.1-dev (image generation, ~24 GB) -- gated, needs HF_TOKEN
huggingface-cli download black-forest-labs/FLUX.1-dev \\
    --local-dir $MODELS_ROOT/diffusion_models/FLUX.1-dev

# 3. WanTEModel + WanVAE (~7 GB combined)
huggingface-cli download Kijai/WanVideo_comfy \\
    umt5-xxl-enc-bf16.safetensors \\
    --local-dir $MODELS_ROOT/text_encoders/
huggingface-cli download Kijai/WanVideo_comfy \\
    Wan2_1_VAE_fp32.safetensors \\
    --local-dir $MODELS_ROOT/vae/

# 4. Whisper Large v3 (audio encoder for HuMo, ~3 GB)
huggingface-cli download openai/whisper-large-v3 \\
    --local-dir $MODELS_ROOT/audio_encoders/whisper-large-v3

# 5. Story-writing LLM (smaller is faster on cloud; pick ONE):
huggingface-cli download google/gemma-4-E2B-it       # ~5 GB, fastest
# OR
# huggingface-cli download mistralai/Mistral-Nemo-Instruct-2407   # ~24 GB, gated

# 6. Bark TTS (~3.5 GB, character voices)
huggingface-cli download suno/bark

# 7. Kokoro TTS (~300 MB, fallback / utility voices)
huggingface-cli download hexgrad/Kokoro-82M

# 8. MusicGen + AudioGen (~2 GB combined, optional but recommended)
huggingface-cli download facebook/musicgen-small
huggingface-cli download facebook/audiogen-medium

=============================================================================
After models are downloaded:
  1. Restart ComfyUI (stop + start the cloud instance, or restart inside)
  2. Upload your workflow JSON via File Browser
  3. Open the Web UI (PORT 8188), load workflow, queue prompt
=============================================================================

EOF
