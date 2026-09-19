# Stop every overnight driver and every leg client, leave the SERVER alone.
# Quoting lives in the file so the shell never sees a backslash it wants to
# read as an escape.
$ErrorActionPreference = "Continue"

Write-Output "--- drivers ---"
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
  Where-Object { $_.CommandLine -like "*shakespeare_overnight*" } |
  ForEach-Object {
      Write-Output ("  stopping driver pid " + $_.ProcessId)
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }

Write-Output "--- leg clients ---"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*otr_canonical_api_run*" } |
  ForEach-Object {
      Write-Output ("  stopping leg pid " + $_.ProcessId)
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }

Start-Sleep -Seconds 3
$drv = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
         Where-Object { $_.CommandLine -like "*shakespeare_overnight*" }).Count
$leg = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like "*otr_canonical_api_run*" }).Count
Write-Output ("  remaining drivers=" + $drv + "  legs=" + $leg)

$srv = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($srv) { Write-Output "  server: STILL UP (intentional)" } else { Write-Output "  server: DOWN" }
