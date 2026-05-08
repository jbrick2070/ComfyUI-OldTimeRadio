# scripts/prep_full_run.ps1
#
# Pre-flight checklist + cleanup for a healthy OTR full acceptance run.
# Reports state and (optionally) wipes stale per-line video clips so the
# BUG-LOCAL-117f duration-aware anti-clobber doesn't keep stale-content
# clips from a prior incomplete run.
#
# Read-only by default. Pass -Wipe to actually delete stale clips, or
# pass -Episode <ep_id> to target a specific folder.
#
# Usage (PowerShell, from repo root):
#     .\scripts\prep_full_run.ps1                 # report only
#     .\scripts\prep_full_run.ps1 -Episode signal_lost_silent_countdown_20260507_193951
#     .\scripts\prep_full_run.ps1 -Wipe           # report + wipe newest pending episode's videos/
#
# What it checks:
#   1. OTR_LTX_ENGINE / OTR_LTX_LOOP_VIA_REVERSE env vars (HKCU\Environment).
#   2. Active comfyui_<port>.log (most-recently-modified non-rotated file).
#   3. Newest pending episode folder + its videos/ contents.
#   4. Disk free space on the output drive.
#
# What it does NOT do:
#   - Restart ComfyUI Desktop (you must do this manually after env-var changes).
#   - Touch the in-flight run.
#   - Modify any node code.

param(
    [string]$Episode = "",
    [switch]$Wipe = $false
)

$ErrorActionPreference = "Stop"

$EpisodesDir = "C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
$LogDir = "C:\Users\jeffr\Documents\ComfyUI\user"

Write-Host ""
Write-Host "=== OTR pre-flight check ===" -ForegroundColor Yellow
Write-Host ""

# 1. Env vars
$ltxEngine = [Environment]::GetEnvironmentVariable("OTR_LTX_ENGINE", "User")
$ltxLoop = [Environment]::GetEnvironmentVariable("OTR_LTX_LOOP_VIA_REVERSE", "User")
if ([string]::IsNullOrEmpty($ltxEngine)) {
    Write-Host "OTR_LTX_ENGINE        = (unset -- defaults to v0_9 sampler path)" -ForegroundColor Gray
} else {
    Write-Host "OTR_LTX_ENGINE        = $ltxEngine" -ForegroundColor Green
}
if ([string]::IsNullOrEmpty($ltxLoop)) {
    Write-Host "OTR_LTX_LOOP_VIA_REVERSE = (unset -- defaults to ON)" -ForegroundColor Gray
} else {
    Write-Host "OTR_LTX_LOOP_VIA_REVERSE = $ltxLoop" -ForegroundColor Green
}

# 2. Active log
$activeLog = Get-ChildItem -Path $LogDir -Filter "comfyui_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch '\.prev2?\.log$' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($activeLog) {
    $age = (Get-Date) - $activeLog.LastWriteTime
    Write-Host ("Active log:           {0} (last write {1:N1}m ago)" -f $activeLog.Name, $age.TotalMinutes) -ForegroundColor Green
} else {
    Write-Host "Active log:           (none found)" -ForegroundColor Red
}

# 3. Newest pending episode
if ([string]::IsNullOrEmpty($Episode)) {
    $candidate = Get-ChildItem -Path $EpisodesDir -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($candidate) {
        $Episode = $candidate.Name
        Write-Host "Newest episode:       $Episode" -ForegroundColor Green
    } else {
        Write-Host "Newest episode:       (none found in $EpisodesDir)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Target episode:       $Episode (user-specified)" -ForegroundColor Green
}

# 4. videos/ folder state
if (-not [string]::IsNullOrEmpty($Episode)) {
    $videosDir = Join-Path $EpisodesDir "$Episode\videos"
    if (Test-Path $videosDir) {
        $clips = Get-ChildItem -Path $videosDir -Filter "*.mp4" -ErrorAction SilentlyContinue
        $count = ($clips | Measure-Object).Count
        $totalMB = [math]::Round((($clips | Measure-Object Length -Sum).Sum / 1MB), 1)
        Write-Host "videos/ contents:     $count clip(s), $totalMB MB total" -ForegroundColor Gray

        $compositedDir = Join-Path $EpisodesDir "$Episode\composited"
        if (Test-Path $compositedDir) {
            $finals = Get-ChildItem -Path $compositedDir -Filter "*.mp4" -ErrorAction SilentlyContinue
            if ($finals) {
                Write-Host "composited/ final:    $($finals[0].Name) ($([math]::Round($finals[0].Length/1MB,1)) MB)" -ForegroundColor Green
            } else {
                Write-Host "composited/ final:    (folder exists, no mp4 -- composite never ran)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "composited/ final:    (no composited/ folder -- composite never ran)" -ForegroundColor Yellow
        }

        if ($Wipe -and $count -gt 0) {
            Write-Host ""
            Write-Host "WIPE requested. Removing $count clip(s) from $videosDir" -ForegroundColor Red
            Remove-Item -Path "$videosDir\*.mp4" -Force -ErrorAction Continue
            $remaining = (Get-ChildItem -Path $videosDir -Filter "*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count
            if ($remaining -eq 0) {
                Write-Host "videos/ wiped clean" -ForegroundColor Green
            } else {
                Write-Host "WARN: $remaining clip(s) could not be deleted (file lock?)" -ForegroundColor Red
            }
        } elseif ($Wipe -and $count -eq 0) {
            Write-Host "WIPE requested but videos/ is already empty" -ForegroundColor Gray
        }
    } else {
        Write-Host "videos/ contents:     (folder does not exist)" -ForegroundColor Gray
    }
}

# 5. Disk space
$drive = Get-PSDrive -Name C
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -lt 50) {
    Write-Host "C: free space:        $freeGB GB" -ForegroundColor Red
    Write-Host "  WARN: full run produces ~5-8 GB; consider cleanup" -ForegroundColor Yellow
} else {
    Write-Host "C: free space:        $freeGB GB" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Next steps ===" -ForegroundColor Yellow
Write-Host "1. If env vars changed: exit ComfyUI Desktop fully + relaunch" -ForegroundColor Gray
Write-Host "2. In ComfyUI Desktop: File -> Load -> otr_scifi_16gb_full.json" -ForegroundColor Gray
Write-Host "3. Optionally start tail in another window: .\scripts\tail_otr_run.ps1" -ForegroundColor Gray
Write-Host "4. Queue the prompt" -ForegroundColor Gray
Write-Host "5. After completion: python scripts\audit_otr_full_run.py --episode <path>" -ForegroundColor Gray
Write-Host ""
