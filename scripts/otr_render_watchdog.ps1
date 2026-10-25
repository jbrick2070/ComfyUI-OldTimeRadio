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
  It does NOT kill anything -- it REPORTS. The caller uses the canonical
  harness's selective OTR-headless reset, confirms idle GPU, then reboots.
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

# Phone ping on DEAD (PLAN row 1.3; the same ntfy topic scripts/otr_notify.py
# uses). Like everything in this script it REPORTS: a failed POST is printed and
# changes nothing. Only the topic HOST is ever printed, never its path or a
# password. The URL is trimmed once, exactly as otr_notify.py reads it, so a
# blank value is "off" in both places.
$NotifyUrl = ("$env:OTR_NOTIFY_URL").Trim()
function Get-NotifyState {
    if (-not $NotifyUrl) { return 'notify: off (OTR_NOTIFY_URL unset)' }
    try { return ('notify: on -> {0}' -f ([System.Uri]$NotifyUrl).Authority) }
    catch { return 'notify: on -> unreadable url' }
}
function Send-Ping($title, $body) {
    if (-not $NotifyUrl) { return }
    $secrets = @($NotifyUrl)
    try {
        $headers = @{ Title = $title; Priority = 'high' }
        $target = $NotifyUrl
        $u = [System.Uri]$NotifyUrl
        if ($u.AbsolutePath.Length -gt 1) { $secrets += $u.AbsolutePath }
        if ($u.UserInfo) {
            # user:password@host goes in an HTTP Basic header, as
            # scripts/otr_notify.py does, and never rides in the URL.
            $secrets += $u.UserInfo
            $pair = [System.Uri]::UnescapeDataString($u.UserInfo)
            $headers['Authorization'] = 'Basic ' + [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($pair))
            $target = $u.GetComponents([System.UriComponents]'SchemeAndServer, PathAndQuery', [System.UriFormat]::UriEscaped)
        }
        Invoke-RestMethod -Uri $target -Method Post -TimeoutSec 10 `
            -ContentType 'text/plain; charset=utf-8' `
            -Headers $headers `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) | Out-Null
        Write-Host '[watchdog] notify sent'
    } catch {
        # The message can quote the URL or its path (the topic is a secret).
        $why = $_.Exception.Message
        foreach ($secret in ($secrets | Sort-Object Length -Descending)) {
            $why = $why.Replace($secret, '<hidden>')
        }
        Write-Host ('[watchdog] notify FAILED: {0}' -f $why)
    }
}
Write-Host ('[watchdog] ' + (Get-NotifyState))

function Get-Heartbeat($path) {
    $m = Select-String -Path $path -Pattern '\[(?:soak|canonical-api)\]\s+t=\s*(\d+)s' |
        Select-Object -Last 1
    if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    return -1
}
function Get-Verdict($path) {
    $m = Select-String -Path $path -Pattern 'sweep_.* -> (PASS|SOAK_FAIL|RC_\d+|ERROR)|COMPLETE:|\[canonical-api\].*RESULT\s+(SUCCESS|FAIL|TIMEOUT)|\[canonical-headless\].*RESULT\s+FAIL' |
        Select-Object -Last 1
    if ($m) { return $m.Line.Trim() }
    return $null
}

$lastBeat = -1
$lastProgress = Get-Date
$qFail = 0
$armed = $false
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
        $armed = $true
    }
    if (-not $armed) {
        Write-Host ("[watchdog] {0} waiting for first heartbeat" -f (Get-Date -Format HH:mm:ss))
        continue
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
        Send-Ping 'OTR leg dead: queue down' ("leg dead: $m`n" + (Split-Path $LegLog -Leaf))
        exit 2
    }
    if ($stalledFor -ge $StallSeconds) {
        $m = "heartbeat stalled ${stalledFor}s (last t=${lastBeat}s) >= ${StallSeconds}s"
        Write-Host ("[watchdog] DEAD {0} :: {1}" -f (Get-Date -Format HH:mm:ss), $m)
        Write-Verdict 'DEAD' $m
        Send-Ping 'OTR leg dead: stalled' ("leg dead: $m`n" + (Split-Path $LegLog -Leaf))
        exit 2
    }
}
