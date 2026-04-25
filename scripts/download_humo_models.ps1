# Download all 5 HuMo POC model files into Jeffrey's tidy ComfyUI models
# folder. Run via:
#   powershell -ExecutionPolicy Bypass -File scripts\download_humo_models.ps1
#
# Or kicked off in background by another script via Start-Process.
#
# All files are placed in the canonical ComfyUI subdirectories so the
# WanHuMoImageToVideo workflows resolve them by filename without any
# path edits required in the JSON.

$ErrorActionPreference = 'Stop'

$ModelBase = 'C:\Users\jeffr\Documents\ComfyUI\models'
$LogFile = "$env:TEMP\humo_download.log"

function Log {
    param([string]$msg)
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Ensure-Dir {
    param([string]$path)
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Log "created $path"
    }
}

# Each entry: { url; subdir under models/; filename }
$files = @(
    @{
        url = 'https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/main/split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors'
        subdir = 'diffusion_models'
        filename = 'humo_17B_fp8_e4m3fn.safetensors'
        approx_gb = 10
    },
    @{
        url = 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors'
        subdir = 'text_encoders'
        filename = 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
        approx_gb = 5
    },
    @{
        url = 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors'
        subdir = 'vae'
        filename = 'wan_2.1_vae.safetensors'
        approx_gb = 0.5
    },
    @{
        url = 'https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/main/split_files/audio_encoders/whisper_large_v3_fp16.safetensors'
        subdir = 'audio_encoders'
        filename = 'whisper_large_v3_fp16.safetensors'
        approx_gb = 3
    },
    @{
        url = 'https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors'
        subdir = 'loras'
        filename = 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors'
        approx_gb = 0.15
    }
)

Set-Content -Path $LogFile -Value "" -Encoding UTF8
Log "==== HuMo POC model download starting ===="
Log "model base: $ModelBase"
Log "log file:   $LogFile"

# Ensure all subdirs exist before any downloads
foreach ($f in $files) {
    Ensure-Dir (Join-Path $ModelBase $f.subdir)
}

$total = $files.Count
$idx = 0
$totalBytesDownloaded = 0
$startedAt = Get-Date

foreach ($f in $files) {
    $idx++
    $target = Join-Path (Join-Path $ModelBase $f.subdir) $f.filename
    $sizeStr = "$($f.approx_gb) GB"

    if (Test-Path $target) {
        $existing = (Get-Item $target).Length
        $existingMB = [math]::Round($existing / 1MB, 1)
        Log "[$idx/$total] SKIP $($f.filename) (already on disk, $existingMB MB)"
        continue
    }

    Log "[$idx/$total] DOWNLOAD $($f.filename) (~$sizeStr)"
    Log "    from: $($f.url)"
    Log "    to:   $target"

    $t0 = Get-Date
    try {
        # Use BITS for robust resumable transfers on big files; falls back
        # to Invoke-WebRequest for small ones / non-Windows-features sites.
        if ($f.approx_gb -ge 1) {
            Start-BitsTransfer -Source $f.url -Destination $target `
                -DisplayName "HuMo download $($f.filename)" `
                -Description "OTR HuMo POC" `
                -Priority Foreground `
                -ErrorAction Stop
        } else {
            Invoke-WebRequest -Uri $f.url -OutFile $target -UseBasicParsing -ErrorAction Stop
        }
        $elapsed = (Get-Date) - $t0
        $sizeBytes = (Get-Item $target).Length
        $sizeMB = [math]::Round($sizeBytes / 1MB, 1)
        $rate = if ($elapsed.TotalSeconds -gt 0) {
            [math]::Round($sizeBytes / 1MB / $elapsed.TotalSeconds, 1)
        } else { 0 }
        $totalBytesDownloaded += $sizeBytes
        Log "[$idx/$total] DONE  $($f.filename)  $sizeMB MB in $([math]::Round($elapsed.TotalSeconds,1))s  ($rate MB/s)"
    } catch {
        Log "[$idx/$total] FAILED $($f.filename): $($_.Exception.Message)"
        Log "    target file may be partial; delete it to retry"
    }
}

$totalElapsed = (Get-Date) - $startedAt
$totalMB = [math]::Round($totalBytesDownloaded / 1MB, 1)
$totalGB = [math]::Round($totalBytesDownloaded / 1GB, 2)
Log "==== ALL DONE ===="
Log "total downloaded: $totalGB GB ($totalMB MB) in $([math]::Round($totalElapsed.TotalMinutes,1)) min"
Log ""

# Final inventory
Log "=== final inventory ==="
foreach ($f in $files) {
    $target = Join-Path (Join-Path $ModelBase $f.subdir) $f.filename
    if (Test-Path $target) {
        $sz = [math]::Round((Get-Item $target).Length / 1GB, 2)
        Log "  OK   $($f.filename) ($sz GB)  $($f.subdir)/"
    } else {
        Log "  MISSING $($f.filename)"
    }
}

Log ""
Log "Next: restart ComfyUI Desktop, then load:"
Log "  workflows/external_examples/video_humo_official_native.json"
Log "  -- or --"
Log "  workflows/external_examples/video_humo_native_unlimited_workflow.json"
