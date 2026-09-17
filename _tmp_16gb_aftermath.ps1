$ErrorActionPreference = "Continue"
Write-Host "=== 8188 listen ==="
Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("LISTEN pid {0}" -f $_.OwningProcess) }
if (-not (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "NO LISTEN"
}
Write-Host "=== 8188 queue ==="
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8188/queue" -TimeoutSec 5
    Write-Host ("running={0} pending={1}" -f @($q.queue_running).Count, @($q.queue_pending).Count)
} catch { Write-Host $_.Exception.Message }
Write-Host "=== 8000 queue ==="
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 8
    $id = if (@($q.queue_running).Count) { $q.queue_running[0][1] } else { "-" }
    Write-Host ("running={0} pending={1} id={2}" -f @($q.queue_running).Count, @($q.queue_pending).Count, $id)
} catch { Write-Host $_.Exception.Message }
Write-Host "=== obs after 23:40 ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -gt (Get-Date "2026-09-14 23:40") } |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "=== nvidia ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
