#!/usr/bin/env bash
# =============================================================================
# OTR cloud model download script
# =============================================================================
# Usage (after running setup_cloud.sh first):
#   export HF_TOKEN=hf_yourtoken    # required for gated models
#   curl -L https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/scripts/download_models.sh | bash
#
# Downloads ~60 GB total. Takes 10-15 min on a healthy cloud connection.
#
# Required (downloaded by default):
#   - HuMo 14B fp8                        ~16 GB
#   - WanTEModel + WanVAE                 ~8 GB
#   - Whisper Large v3                    ~3 GB
#   - Gemma-4-E2B-it (story LLM, GATED)   ~5 GB
#   - Bark TTS                            ~3.5 GB
#   - FLUX.1-dev (stills, GATED)          ~24 GB
#
# Skipped by default (run later if needed):
#   - Mistral-Nemo-Instruct-2407          ~24 GB (alt LLM)
#   - MusicGen + AudioGen                 ~2 GB (music/SFX)
#
# Idempotent: huggingface-cli skips files already cached.
# =============================================================================

set -e
set -o pipefail

# -----------------------------------------------------------------------------
# 0. Detect ComfyUI install + check HF_TOKEN
# -----------------------------------------------------------------------------
echo "[0/7] Detecting ComfyUI install..."

CANDIDATES=(
  "/app/ComfyUI"
  "/workspace/ComfyUI"
  "/home/comfyui/ComfyUI"
  "/opt/ComfyUI"
)

COMFY_ROOT=""
for c in "${CANDIDATES[@]}"; do
  if [ -d "$c/custom_nodes" ] && [ -f "$c/main.py" ]; then
    COMFY_ROOT="$c"
    break
  fi
done

if [ -z "$COMFY_ROOT" ]; then
  COMFY_ROOT=$(find / -maxdepth 5 -name "main.py" -path "*/ComfyUI/main.py" 2>/dev/null | head -1)
  COMFY_ROOT=$(dirname "$COMFY_ROOT" 2>/dev/null || true)
fi

if [ -z "$COMFY_ROOT" ] || [ ! -d "$COMFY_ROOT/custom_nodes" ]; then
  echo "  FATAL: cannot find ComfyUI install."
  exit 1
fi

echo "  ComfyUI root: $COMFY_ROOT"
MODELS_ROOT="$COMFY_ROOT/models"
cd "$MODELS_ROOT"
mkdir -p diffusion_models text_encoders vae audio_encoders huggingface

export HF_HOME="$MODELS_ROOT/huggingface"

if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
  echo "  WARN: HF_TOKEN is not set. Gated models (FLUX.1-dev, Gemma-4) will 401."
  echo "  To set: export HF_TOKEN=hf_yourtoken"
  echo "  Continuing with non-gated models only..."
  echo ""
  SKIP_GATED=1
else
  echo "  HF_TOKEN is set (length=${#HF_TOKEN})"
  SKIP_GATED=0
fi

# Make sure huggingface-cli is present
pip install -q -U huggingface_hub 2>/dev/null || true

# -----------------------------------------------------------------------------
# 1. HuMo 14B fp8 (the lip-sync video model)
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] HuMo 14B fp8 (~16 GB)..."
huggingface-cli download Kijai/WanVideo_comfy \
    Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors \
    --local-dir "$MODELS_ROOT/diffusion_models/"

# -----------------------------------------------------------------------------
# 2. WanTEModel (text encoder for HuMo)
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] WanTEModel umt5-xxl-enc-bf16 (~7 GB)..."
huggingface-cli download Kijai/WanVideo_comfy \
    umt5-xxl-enc-bf16.safetensors \
    --local-dir "$MODELS_ROOT/text_encoders/"

# -----------------------------------------------------------------------------
# 3. WanVAE
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] WanVAE Wan2_1_VAE_fp32 (~1 GB)..."
huggingface-cli download Kijai/WanVideo_comfy \
    Wan2_1_VAE_fp32.safetensors \
    --local-dir "$MODELS_ROOT/vae/"

# -----------------------------------------------------------------------------
# 4. Whisper Large v3 (audio encoder for HuMo)
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Whisper Large v3 (~3 GB)..."
huggingface-cli download openai/whisper-large-v3 \
    --local-dir "$MODELS_ROOT/audio_encoders/whisper-large-v3"

# -----------------------------------------------------------------------------
# 5. Bark TTS (always required)
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Bark TTS (~3.5 GB)..."
huggingface-cli download suno/bark

# -----------------------------------------------------------------------------
# 6. Gemma-4-E2B (story writer, GATED)
# -----------------------------------------------------------------------------
echo ""
if [ "$SKIP_GATED" -eq 1 ]; then
  echo "[6/7] Gemma-4-E2B SKIPPED (HF_TOKEN not set)"
else
  echo "[6/7] Gemma-4-E2B-it (~5 GB, gated)..."
  huggingface-cli download google/gemma-4-E2B-it
fi

# -----------------------------------------------------------------------------
# 7. FLUX.1-dev (cast/env stills + radio bookend, GATED, BIG)
# -----------------------------------------------------------------------------
echo ""
if [ "$SKIP_GATED" -eq 1 ]; then
  echo "[7/7] FLUX.1-dev SKIPPED (HF_TOKEN not set)"
else
  echo "[7/7] FLUX.1-dev (~24 GB, gated, the big one)..."
  huggingface-cli download black-forest-labs/FLUX.1-dev \
      --local-dir "$MODELS_ROOT/diffusion_models/FLUX.1-dev"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "============================================================="
echo "Model downloads complete."
echo "============================================================="
df -h "$MODELS_ROOT"
echo ""
echo "Next steps:"
echo "  1. Restart ComfyUI from SimplePod dashboard"
echo "  2. Upload workflow JSON via File Browser"
echo "  3. Open PORT 8188 web UI, load workflow, queue prompt"
echo ""

if [ "$SKIP_GATED" -eq 1 ]; then
  echo "WARNING: gated models were skipped (HF_TOKEN was not set)."
  echo "Without FLUX.1-dev and Gemma-4-E2B, OTR cannot run."
  echo "Set HF_TOKEN and re-run this script:"
  echo "  export HF_TOKEN=hf_yourtoken"
  echo "  curl -L https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/scripts/download_models.sh | bash"
fi
