# scripts/soak_watch.ps1
#
# Polls the OTR FULL acceptance soak's episode dir; when it goes quiet
# (no new files for QUIET_MINUTES) runs audit_otr_full_run.py against
# the most recent ComfyUI log, then writes a summary to
# `outputs/soak_status.txt` so Jeffrey can read the verdict first thing
# in the morning.
#
# 2026-05-08 round-robin Element 5 fix: the log path is now resolved
# dynamically. ComfyUI Desktop App writes to %APPDATA%\ComfyUI\logs\
# not the legacy C:\Users\jeffr\Documents\ComfyUI\comfyui_<port>.log
# location. The watcher checks the Desktop path FIRST, then falls
# back to the legacy path so manual-launch users still work. Without
# this fix the watcher's audit ran without a log argument, so the
# `Fatal Python error: Aborted` FAIL_PATTERN never matched.
#
# Idempotent: safe to launch as a Scheduled Task or just run from a
# PowerShell window. Loops until the audit fires once OR the user
# Ctrl-C's the script. Uses ComfyUI's outputs directory directly --
# does not require any sandbox-specific paths.
#
# Usage:
#   pwsh.exe -File C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\soak_watch.ps1
#
# Optional parameters:
#   -EpisodesRoot <path> : override the episodes folder
#   -StatusOut    <path> : override the status file path
#   -PollSeconds  <int>  : how often to wake up (default 300 = 5 min)
#   -QuietMinutes <int>  : trigger threshold (default 10 min idle)
#   -MaxWaitMinutes <int>: hard cap, exit even if soak still active
#                          (default 720 = 12 h)

[CmdletBinding()]
param(
    [string]$EpisodesRoot = 'C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes',
    [string]$StatusOut    = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\outputs\soak_status.txt',
    [int]$PollSeconds     = 300,
    [int]$QuietMinutes    = 10,
    [int]$MaxWaitMinutes  = 720
)

$ErrorActionPreference = 'Continue'
$RepoRoot = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$Python   = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$AuditScript = Join-Path $RepoRoot 'scripts\audit_otr_full_run.py'

# Resolve the active ComfyUI log dynamically. Desktop App (the actual
# 2026-05-08 install) writes to %APPDATA%\ComfyUI\logs\comfyui.log;
# manual-launch users still drop comfyui_<port>.log under the OTR
# Documents tree. Pick whichever exists AND was most recently
# written to.
$LogCandidates = @(
    (Join-Path $env:APPDATA 'ComfyUI\logs\comfyui.log'),
    'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log',
    'C:\Users\jeffr\Documents\ComfyUI\comfyui_8188.log'
)
$ComfyLog = $null
$LogMtime = [DateTime]::MinValue
foreach ($cand in $LogCandidates) {
    if (Test-Path $cand) {
        $info = Get-Item $cand
        if ($info.LastWriteTime -gt $LogMtime) {
            $ComfyLog = $cand
            $LogMtime = $info.LastWriteTime
        }
    }
}

# Ensure outputs dir exists for the status file.
$statusDir = Split-Path -Parent $StatusOut
if (-not (Test-Path $statusDir)) { New-Item -ItemType Directory -Force -Path $statusDir | Out-Null }

function Write-Status([string]$msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "[$ts] $msg" | Out-File -FilePath $StatusOut -Append -Encoding utf8
    Write-Host "[$ts] $msg"
}

# ---- discover the in-flight episode dir --------------------------------
function Get-LatestEpisodeDir() {
    if (-not (Test-Path $EpisodesRoot)) { return $null }
    Get-ChildItem -Path $EpisodesRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Get-LatestFileMtime([string]$dir) {
    if (-not (Test-Path $dir)) { return $null }
    $latest = Get-ChildItem -Path $dir -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) { return $null }
    return $latest.LastWriteTime
}

# ---- main loop ---------------------------------------------------------

# Reset status file at start (don't accumulate across runs).
"OTR soak watcher started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
    Out-File -FilePath $StatusOut -Encoding utf8

Write-Status "polling $EpisodesRoot every $PollSeconds s; trigger threshold $QuietMinutes min idle"

$startTime = Get-Date
$episode = Get-LatestEpisodeDir
if ($null -eq $episode) {
    Write-Status "ERROR: no episode dir found under $EpisodesRoot"
    exit 1
}
Write-Status "watching episode: $($episode.Name)"

while ($true) {
    $now = Get-Date
    $elapsedMin = ($now - $startTime).TotalMinutes
    if ($elapsedMin -gt $MaxWaitMinutes) {
        Write-Status "hit MaxWaitMinutes=$MaxWaitMinutes; giving up without firing audit"
        exit 2
    }

    $latestMtime = Get-LatestFileMtime $episode.FullName
    if ($null -eq $latestMtime) {
        Write-Status "WARN: episode dir empty or unreadable; will retry"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $idleMin = ($now - $latestMtime).TotalMinutes
    Write-Status ("episode idle for {0:N1} min (newest file at {1:HH:mm:ss})" -f $idleMin, $latestMtime)

    if ($idleMin -ge $QuietMinutes) {
        Write-Status "idle threshold reached -- running audit"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

# ---- run the audit -----------------------------------------------------

if (-not (Test-Path $AuditScript)) {
    Write-Status "ERROR: audit script not found at $AuditScript"
    exit 3
}
if ($null -eq $ComfyLog) {
    Write-Status ("WARN: no ComfyUI log found at any candidate path: " + ($LogCandidates -join ', ') + " -- audit will inspect episode dir only")
} else {
    Write-Status "comfy log resolved to $ComfyLog (mtime $LogMtime)"
}

$auditOut = Join-Path $statusDir 'soak_audit.txt'
"OTR soak audit run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $auditOut -Encoding utf8
"episode: $($episode.Name)" | Out-File -FilePath $auditOut -Append -Encoding utf8
"" | Out-File -FilePath $auditOut -Append -Encoding utf8

# Push-Location into the repo so the audit script's relative imports work.
Push-Location $RepoRoot
try {
    if ($null -ne $ComfyLog -and (Test-Path $ComfyLog)) {
        & $Python $AuditScript $ComfyLog *>> $auditOut
    } else {
        & $Python $AuditScript *>> $auditOut
    }
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Status "audit completed exit=$exitCode -- full output at $auditOut"

# ---- post-audit summary line -------------------------------------------

# audit_otr_full_run.py exits non-zero on FAIL; zero on PASS. Surface
# the verdict at the top of the status file with a clear marker so a
# scan of the file's first few lines tells the story.
$verdict = if ($exitCode -eq 0) { 'PASS' } else { 'FAIL' }
$header = @(
    "==== SOAK VERDICT: $verdict ===="
    "episode: $($episode.Name)"
    "audit detail: $auditOut"
    "completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    ""
)
$existing = Get-Content -Path $StatusOut -Raw -ErrorAction SilentlyContinue
if ($null -eq $existing) { $existing = "" }
($header -join "`r`n") + $existing | Out-File -FilePath $StatusOut -Encoding utf8

Write-Status "verdict surfaced at top of $StatusOut"
exit $exitCode
