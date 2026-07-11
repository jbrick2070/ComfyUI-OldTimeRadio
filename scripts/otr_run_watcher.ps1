# Watches a render to completion and reports the result WITHOUT being asked.
#
# The failure this exists to kill: a detached render finishes, nobody is told, and the
# operator has to remember to come and ask. A run that reports only when polled spends
# the operator's attention instead of saving it.
#
# It watches; it never drives. It does not kill the render and does not touch assets.
#
# Two hard lessons are baked in:
#   1. The STATUS FILE IS WRITTEN FIRST and updated continuously. The first version
#      wrote nothing until the end, so when it died at startup there was no file at all
#      -- a watcher that can fail silently is worse than no watcher.
#   2. The completion popup NEVER BLOCKS THE STATUS. A MessageBox waits for a click
#      forever; if it is the last statement, an unclicked dialog leaves the watcher
#      hanging and looking alive. Status is committed to disk before any UI is shown.
param(
    [Parameter(Mandatory = $true)][int]$WatchPid,
    [string]$StatusFile = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\RUN_STATUS.txt',
    [string]$Label = 'OTR render',
    # The REAL obs base is ComfyUI's output\ root -- `otr\` resolves there on disk, NOT
    # under the custom_nodes package. Pointing at the package path made the watcher call
    # a PUBLISHED run FAILED (2026-07-11): the episode was sitting in output\otr\obs the
    # whole time. A watcher that reports a false negative is worse than no watcher, so
    # this path is the one thing in here that must never be guessed.
    [string]$ObsDir = 'C:\Users\jeffr\Documents\ComfyUI\output\otr\obs',
    [string]$ServerLog = 'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log',
    [switch]$NoPopup
)

$ErrorActionPreference = 'Continue'

function Write-Status([string]$state, [string]$detail) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Label  $state  $detail"
    Write-Host $line
    # A swallowed write is how a status file goes stale while looking healthy -- which
    # is the exact failure this watcher exists to prevent. If the file is locked (a
    # previous watcher still holding a handle), say so LOUDLY in the window rather than
    # letting the operator read a status from an hour ago and believe it.
    try {
        [System.IO.File]::WriteAllText($StatusFile, $line + [Environment]::NewLine)
    } catch {
        Write-Host "!! CANNOT WRITE $StatusFile -- $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Status 'STARTING' "watching pid=$WatchPid"

# Snapshot what was already published, so "new episode" means NEW.
$before = @{}
if (Test-Path $ObsDir) {
    Get-ChildItem $ObsDir -File -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { $before[$_.FullName] = $true }
}

$started = Get-Date
while (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue) {
    $mins = [int]((Get-Date) - $started).TotalMinutes
    Write-Status 'RUNNING' "pid=$WatchPid  ${mins}m elapsed"
    Start-Sleep -Seconds 20
}
$elapsed = [int]((Get-Date) - $started).TotalMinutes

# Truth comes from the ASSET ON DISK -- never from VRAM, never from a quiet log. A
# finished-but-resident server is indistinguishable from a working one, and that has
# been misread on this box before.
$new = @()
if (Test-Path $ObsDir) {
    $new = Get-ChildItem $ObsDir -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { -not $before.ContainsKey($_.FullName) }
}

$failLine = ''
if (Test-Path $ServerLog) {
    $hit = Select-String -LiteralPath $ServerLog -Pattern 'Exception during processing|failed after' |
        Select-Object -Last 1
    if ($hit) { $failLine = $hit.Line }
}

if ($new.Count -gt 0) {
    $names = ($new | ForEach-Object { $_.Name }) -join ', '
    $state = 'PUBLISHED'
    Write-Status $state "${elapsed}m  ->  $names"
    $banner = "$Label PUBLISHED after ${elapsed} min`n`n$names"
    $icon = 'Information'
} else {
    $state = 'FAILED'
    $short = if ($failLine) {
        $failLine.Substring(0, [Math]::Min(300, $failLine.Length))
    } else {
        'no obvious error in the server log'
    }
    Write-Status $state "${elapsed}m  no new asset in otr\obs  |  $short"
    $banner = "$Label FAILED after ${elapsed} min`n`nNo new episode in otr\obs.`n`n$short"
    $icon = 'Warning'
}

# Status is already on disk. The popup is a courtesy, and it must never be load-bearing.
if (-not $NoPopup) {
    try {
        Add-Type -AssemblyName System.Windows.Forms | Out-Null
        [System.Windows.Forms.MessageBox]::Show(
            $banner, "OTR - $state",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::$icon
        ) | Out-Null
    } catch { }
}
