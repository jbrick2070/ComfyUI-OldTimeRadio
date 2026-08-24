<#
.SYNOPSIS
Proves the canonical workflow runs on 4060 (8GB) with H3 and Gemma E2B.
#>
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO = Split-Path -Parent $HERE
$LAUNCHCMD = Join-Path $HERE "_otr_soak_server_launch.cmd"
$APIRUN = Join-Path $HERE "otr_canonical_api_run.py"
$AUDIT = Join-Path $HERE "audit_otr_full_run.py"

$VenvPython = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"

Write-Host "Killing old servers..." -ForegroundColor Cyan
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "main\.py.*--port 8000" } | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

# Boot in HUMO/H3 lane? Wait, _otr_soak_server_launch.cmd doesn't have an "H3" lane.
# Let's check what lane H3 uses. Usually it just needs to be booted without LTX or WAN.
Write-Host "Booting ComfyUI server..." -ForegroundColor Cyan
$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = $LAUNCHCMD
$pinfo.Arguments = "`"otr_4060_h3_server.log`" FLOOR"
$pinfo.UseShellExecute = $false
$pinfo.WorkingDirectory = $REPO
$proc = [System.Diagnostics.Process]::Start($pinfo)

Write-Host "Waiting up to 45s for port 8000..." -ForegroundColor Yellow
for ($i=0; $i -lt 45; $i++) {
    $tcp = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($tcp) { break }
    Start-Sleep -Seconds 1
}

Write-Host "Triggering 1-Act run with otr_4060_h3_nano profile..." -ForegroundColor Cyan
$runProc = Start-Process -FilePath $VenvPython -ArgumentList "$APIRUN --profile otr_4060_h3_nano --act-count 1" -NoNewWindow -PassThru -RedirectStandardOutput "otr_4060_h3_client.log"
$runProc.WaitForExit()

Write-Host "Run finished. Looking for Episode ID to audit..." -ForegroundColor Cyan
$matches = (Select-String -Path "otr_4060_h3_client.log" -Pattern "\[ledger\] writing manifest to (.*\\episodes\\(.*?)\\).*")
if ($matches) {
    $epId = $matches.Matches.Groups[2].Value
    Write-Host "Auditing Episode: $epId"
    $auditProc = Start-Process -FilePath $VenvPython -ArgumentList "$AUDIT --episode `"otr\episodes\$epId`"" -NoNewWindow -PassThru
    $auditProc.WaitForExit()
} else {
    Write-Host "Could not find episode ID. Check otr_4060_h3_client.log for errors." -ForegroundColor Red
}
