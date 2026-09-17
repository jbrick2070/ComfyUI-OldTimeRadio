#!/bin/bash
set +e
source /workspace/otr-config/otr-runtime.env 2>/dev/null || true
echo "=== qwen / vae / unet dirs ==="
ls /workspace/runpod-slim/ComfyUI/models/unet /workspace/runpod-slim/ComfyUI/models/diffusion_models /workspace/runpod-slim/ComfyUI/models/vae /workspace/runpod-slim/ComfyUI/models/text_encoders /workspace/runpod-slim/ComfyUI/models/clip 2>/dev/null | head -n 80
echo "=== named files ==="
for n in flux-2-klein-4b-Q4_K_M.gguf flux2-vae.safetensors qwen_3_4b.safetensors flux1-dev-fp8.safetensors; do
  echo -n "$n "
  find /workspace/runpod-slim/ComfyUI/models -name "$n" 2>/dev/null | head -n 3
done
