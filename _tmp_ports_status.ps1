$ErrorActionPreference = "Stop"
Write-Output "=== listen 8000/8188 ==="
Get-NetTCPConnection -LocalPort 8000,8188 -State Listen -ErrorAction SilentlyContinue |
  Select-Object LocalPort, OwningProcess |
  Format-Table -AutoSize |
  Out-String

Write-Output "=== comfy python (CommandLine match) ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object {
    $_.CommandLine -and (
      $_.CommandLine -match '--port 8000' -or
      $_.CommandLine -match '--port 8188' -or
      $_.CommandLine -match 'ComfyUI\\main.py'
    )
  } |
  ForEach-Object {
    $cl = [string]$_.CommandLine
    if ($cl.Length -gt 240) { $cl = $cl.Substring(0, 240) }
    Write-Output ("PID=" + $_.ProcessId + " " + $cl)
  }
