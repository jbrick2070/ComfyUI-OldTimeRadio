# scripts/tail_otr_run.ps1
#
# Live tail of the active ComfyUI Desktop log with color-coded matching for
# the pre-FULL acceptance soak watchlist (load-bearing banner lines, STOP
# signals, and pipeline node markers).
#
# Auto-discovers the most recently modified comfyui_<port>.log under
# C:\Users\jeffr\Documents\ComfyUI\user (rotated *.prev*.log files are
# ignored). Pass -Port NNNN to pin a specific port instead.
#
# Usage (PowerShell):
#     cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
#     .\scripts\tail_otr_run.ps1
#     .\scripts\tail_otr_run.ps1 -Port 8001
#     .\scripts\tail_otr_run.ps1 -Tail 50    # show last 50 lines first

param(
    [int]$Port = 0,
    [int]$Tail = 0
)

$LogDir = "C:\Users\jeffr\Documents\ComfyUI\user"

if ($Port -gt 0) {
    $LogPath = Join-Path $LogDir "comfyui_$Port.log"
    if (-not (Test-Path -Path $LogPath -PathType Leaf)) {
        Write-Host "ERROR: $LogPath not found" -ForegroundColor Red
        exit 1
    }
} else {
    $candidates = Get-ChildItem -Path $LogDir -Filter "comfyui_*.log" |
        Where-Object { $_.Name -notmatch '\.prev2?\.log$' } |
        Sort-Object LastWriteTime -Descending
    if ($candidates.Count -eq 0) {
        Write-Host "ERROR: no comfyui_*.log found in $LogDir" -ForegroundColor Red
        exit 1
    }
    $LogPath = $candidates[0].FullName
}

Write-Host ("Tailing: " + $LogPath) -ForegroundColor Yellow
Write-Host "Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

$failPattern = 'Non-monotonous DTS|boomerang FAILED|duration contract VIOLATED|STALE-LOCKED|derived ledger from \.mp4 not found|audio may be truncated|\[BatchLTXRender\] .* failed:|strict_c7=True'
$loadBearingPattern = 'BUG-LOCAL-117 engine=|boomerang: ON \(BUG-LOCAL-117d\)|post-BUG-117e: music chunks|cast-still binding:|CRITIQUE: Character line counts|silent_combined\.mp4|AUDIT PASSED|AUDIT FAILED'
$nodePattern = '\[BatchLTXRender\]|\[BatchHumoRender\]|\[VideoComposite\]|\[EpisodeAssembler\]|\[FluxAnchor\]|\[BatchBark\]'

Get-Content -Path $LogPath -Wait -Tail $Tail | ForEach-Object {
    $line = $_
    if ($line -match $failPattern) {
        Write-Host $line -ForegroundColor Red
    }
    elseif ($line -match $loadBearingPattern) {
        Write-Host $line -ForegroundColor Green
    }
    elseif ($line -match $nodePattern) {
        Write-Host $line -ForegroundColor Cyan
    }
    else {
        Write-Host $line
    }
}
