$ErrorActionPreference = "Continue"
Write-Host "=== 8000 listen ==="
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("LISTEN pid {0}" -f $_.OwningProcess) }
Write-Host "=== 8000 any ==="
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object State, OwningProcess | Format-Table -AutoSize
Write-Host "=== rainbow obs/episodes ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter "*rainbow*" -ErrorAction SilentlyContinue |
    Select-Object FullName, Length, LastWriteTime
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes" -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "rainbow" } |
    Select-Object Name, LastWriteTime
Write-Host "=== cpu log tail ==="
$log = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000.log"
if (Test-Path $log) {
    Write-Host ((Get-Item $log).LastWriteTime)
    Get-Content $log -Tail 25
}
Write-Host "=== retry 8000 queue 12s ==="
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/queue" -UseBasicParsing -TimeoutSec 12
    Write-Host $r.Content.Substring(0, [Math]::Min(200, $r.Content.Length))
} catch { Write-Host $_.Exception.Message }
