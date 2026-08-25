<#
.SYNOPSIS
Downloads the models required by the 4060 Nano profile (8 GB VRAM tier).

NOTE: if you only want a FIRST RENDER on an 8 GB card, you do not need this
script at all. Use the `otr_4060_floor` profile / variant instead -- it runs on
viz_camera + bark + musicgen, none of which need a hand-placed file. This script
is for the heavier `otr_4060_nano` lane (LTX video + Kokoro voices).
#>
$ErrorActionPreference = "Stop"

Write-Host "============================================================="
Write-Host "Downloading models for 4060 Nano Profile (8GB VRAM Tier)"
Write-Host "============================================================="

# 2026-08-25: the old warning here said a token was required "for gated models
# like Gemma". That is not true -- the Hugging Face API reports "gated": false
# for google/gemma-4-E2B-it, google/gemma-4-E4B-it, google/gemma-4-12b-it and
# mistralai/Mistral-Nemo-Instruct-2407. Gemma 4 is Apache-2.0 and ungated
# (unlike Gemma 2). Nothing this script fetches needs a token today.
if (-not $env:HF_TOKEN) {
    Write-Host "HF_TOKEN is not set -- that is fine, nothing here is gated." -ForegroundColor DarkGray
}

$modelsDir = "C:\ComfyUI-Models"
if (-not (Test-Path $modelsDir)) {
    Write-Host "Creating $modelsDir..."
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

$env:HF_HOME = "$modelsDir\huggingface"

# 1. Gemma-4-E2B-it (LLM)
Write-Host "`n[1/4] Downloading google/gemma-4-E2B-it (Small LLM, ~5GB)..." -ForegroundColor Cyan
& huggingface-cli download google/gemma-4-E2B-it

# 2. LTX Video 0.9.8 (video lane)
Write-Host "`n[2/4] Downloading LTX Video 0.9.8..." -ForegroundColor Cyan
& ".\scripts\download_ltx_0_9_8.ps1"

# 3. T5 XXL (text encoder for the video/image lanes)
Write-Host "`n[3/4] Downloading T5 XXL fp16..." -ForegroundColor Cyan
& huggingface-cli download comfyanonymous/edit_t5 t5xxl_fp16.safetensors --local-dir "$modelsDir\text_encoders"

# 4. Kokoro voices.
#
# 2026-08-25 CORRECTION: this step used to fetch `hexgrad/Kokoro-82M` into the
# HF hub cache. That satisfies KPipeline's MODEL load but does nothing for the
# VOICES -- nodes/_otr_audio_engines/eng_kokoro.py never consults the HF cache
# for them. It does a bare os.path.exists on
#     <models_dir>\TTS\KokoroTTS\voices\<voice_id>.pt
# and raises EngineUnusable(MISSING_MODEL) before the pipeline is built. So the
# old command printed a green success and left the render failing exactly as
# before. The correct repo is 1038lab/KokoroTTS and the correct destination is
# the TTS tree, which is what the engine's own error message tells you.
#
# Also note: since 2026-08-24 the pack auto-prefetches these at boot via
# nodes/_otr_kokoro_voice_prefetch.py (wired into prestartup_script.py), so this
# step is usually redundant. It stays for offline/air-gapped setups.
Write-Host "`n[4/4] Downloading Kokoro TTS voices..." -ForegroundColor Cyan
& huggingface-cli download 1038lab/KokoroTTS --local-dir "$modelsDir\TTS\KokoroTTS"

Write-Host "`n============================================================="
Write-Host "NOTE: the nano profile's image role uses 'z_image_turbo', which has"
Write-Host "NO auto-download -- no image engine in the pack does. Copy"
Write-Host "z_image_turbo_bf16.safetensors into the 4060's diffusion_models"
Write-Host "folder, or use the otr_4060_floor variant, whose viz_camera engine"
Write-Host "declares accepts_still=False and skips the image phase entirely."
Write-Host "============================================================="
Write-Host "All automated downloads completed successfully." -ForegroundColor Green
