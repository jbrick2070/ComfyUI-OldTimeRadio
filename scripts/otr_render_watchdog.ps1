<#
  otr_render_watchdog.ps1 -- declare a render DEAD fast instead of waiting the
  full ~24 min (operator directive 2026-06-12). Monitors a soak/coverage leg:

    * HEARTBEAT STALL: the leg log's "[soak] t= <N>s" heartbeat must advance.
      If it does not change for -StallSeconds (default 300 = 5 min) -> DEAD.
    * QUEUE DOWN: http://127.0.0.1:8000/queue (or /system_stats) must answer.
      -QueueFails consecutive failures (default 3) -> DEAD.
    * DONE: the leg log shows PASS / SOAK_FAIL / COMPLETE -> exit 0 with verdict.

  Usage:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_render_watchdog.ps1 `
        -LegLog scripts\_otr_render_b_a.log

  Exit codes: 0 = leg finished (verdict printed); 2 = DEAD (stall or queue down).
  It does NOT kill anything -- it REPORTS. The caller resets per the CLAUDE.md
  directive (kill all python, confirm idle GPU, reboot canonical launcher).
#>
param(
    [Parameter(Mandatory = $true)][string]$LegLog,
    [int]$StallSeconds = 300,
    [int]$PollSeconds = 30,
    [int]$QueueFails = 3,
    [string]$QueueUrl = 'http://127.0.0.1:8000/queue',
    [string]$VerdictFile = ''
)
$ErrorActionPreference = 'SilentlyContinue'
if (-not $VerdictFile) { $VerdictFile = "$LegLog.watchdog" }
function Write-Verdict($state, $msg) {
    "{0} | {1} | {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $state, $msg |
        Set-Content -Path $VerdictFile -Encoding ascii
}
Write-Verdict 'RUNNING' 'watchdog armed'

function Get-Heartbeat($path) {
    $m = Select-String -Path $path -Pattern '\[soak\] t=\s*(\d+)s' |
        Select-Object -Last 1
    if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    return -1
}
function Get-Verdict($path) {
    $m = Select-String -Path $path -Pattern 'sweep_.* -> (PASS|SOAK_FAIL|RC_\d+|ERROR)|COMPLETE:' |
        Select-Object -Last 1
    if ($m) { return $m.Line.Trim() }
    return $null
}

$lastBeat = -1
$lastProgress = Get-Date
$qFail = 0
Write-Host ("[watchdog] start {0} :: leg={1} stall={2}s poll={3}s" -f `
    (Get-Date -Format HH:mm:ss), $LegLog, $StallSeconds, $PollSeconds)

while ($true) {
    Start-Sleep -Seconds $PollSeconds
    if (-not (Test-Path $LegLog)) { continue }

    $verdict = Get-Verdict $LegLog
    if ($verdict) {
        Write-Host ("[watchdog] DONE {0} :: {1}" -f (Get-Date -Format HH:mm:ss), $verdict)
        Write-Verdict 'DONE' $verdict
        exit 0
    }

    $beat = Get-Heartbeat $LegLog
    if ($beat -ge 0 -and $beat -ne $lastBeat) {
        $lastBeat = $beat
        $lastProgress = Get-Date
    }
    $stalledFor = [int]((Get-Date) - $lastProgress).TotalSeconds

    # queue endpoint health
    try {
        $r = Invoke-WebRequest -Uri $QueueUrl -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { $qFail = 0 } else { $qFail++ }
    } catch { $qFail++ }

    Write-Host ("[watchdog] {0} t={1}s stalled={2}s qfail={3}" -f `
        (Get-Date -Format HH:mm:ss), $beat, $stalledFor, $qFail)

    if ($qFail -ge $QueueFails) {
        $m = "queue endpoint down ${qFail}x ($QueueUrl)"
        Write-Host ("[watchdog] DEAD {0} :: {1}" -f (Get-Date -Format HH:mm:ss), $m)
        Write-Verdict 'DEAD' $m
        exit 2
    }
    if ($stalledFor -ge $StallSeconds) {
        $m = "heartbeat stalled ${stalledFor}s (last t=${lastBeat}s) >= ${StallSeconds}s"
        Write-Host ("[watchdog] DEAD {0} :: {1}" -f (Get-Date -Format HH:mm:ss), $m)
        Write-Verdict 'DEAD' $m
        exit 2
    }
}
