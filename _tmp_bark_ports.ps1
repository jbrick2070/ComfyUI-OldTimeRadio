$ErrorActionPreference = "Stop"
Write-Host "=== ports ==="
foreach ($p in 8000, 8188) {
  $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
  if ($c) { Write-Host "LISTEN $p pid=$($c.OwningProcess | Select-Object -First 1)" }
  else { Write-Host "FREE $p" }
}
Write-Host "=== nvidia ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
Write-Host "=== python comfy ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'ComfyUI|main.py|otr_' } |
  Select-Object ProcessId, @{n='cmd';e={ $_.CommandLine.Substring(0, [Math]::Min(180, $_.CommandLine.Length)) }} |
  Format-List
