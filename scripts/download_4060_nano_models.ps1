<#
.SYNOPSIS
Downloads all models required for the 4060 Nano profile.
Requires HF_TOKEN environment variable for gated models.
#>
$ErrorActionPreference = "Stop"

Write-Host "============================================================="
Write-Host "Downloading models for 4060 Nano Profile (8GB VRAM Tier)"
Write-Host "============================================================="

if (-not $env:HF_TOKEN) {
    Write-Host "WARNING: HF_TOKEN is not set. You must set it for gated models like Gemma!" -ForegroundColor Yellow
}

$modelsDir = "C:\ComfyUI-Models"
if (-not (Test-Path $modelsDir)) {
    Write-Host "Creating $modelsDir..."
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

$env:HF_HOME = "$modelsDir\huggingface"

# 1. Download Gemma-4-E2B-it (LLM)
Write-Host "`n[1/4] Downloading google/gemma-4-E2B-it (Small LLM, ~5GB)..." -ForegroundColor Cyan
& huggingface-cli download google/gemma-4-E2B-it

# 2. Download LTX Video 0.9.8 (Video)
Write-Host "`n[2/4] Downloading LTX Video 0.9.8..." -ForegroundColor Cyan
& "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe" ".\scripts\hf_download_driver.py" "dummy" -ErrorAction SilentlyContinue # Just testing path
& ".\scripts\download_ltx_0_9_8.ps1"

# 3. Download T5 XXL (Text Encoder for Video/Image)
Write-Host "`n[3/4] Downloading T5 XXL fp16..." -ForegroundColor Cyan
& huggingface-cli download comfyanonymous/edit_t5 t5xxl_fp16.safetensors --local-dir "$modelsDir\text_encoders"

# 4. Download Kokoro (TTS)
Write-Host "`n[4/4] Downloading Kokoro-82M TTS..." -ForegroundColor Cyan
& huggingface-cli download hexgrad/Kokoro-82M --local-dir "$modelsDir\huggingface\hub\models--hexgrad--Kokoro-82M"

Write-Host "`n============================================================="
Write-Host "NOTE: Image generation uses 'z_image_turbo'. Make sure"
Write-Host "z_image_turbo_bf16.safetensors is copied to the 4060's"
Write-Host "checkpoints folder if not already present from the 5080."
Write-Host "============================================================="
Write-Host "All automated downloads completed successfully." -ForegroundColor Green
