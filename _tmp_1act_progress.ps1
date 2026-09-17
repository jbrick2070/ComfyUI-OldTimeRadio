$ErrorActionPreference = "Continue"
Write-Host "=== queue ==="
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 5
    Write-Host ("running={0} pending={1}" -f @($q.queue_running).Count, @($q.queue_pending).Count)
    if (@($q.queue_running).Count) { Write-Host ("id={0}" -f $q.queue_running[0][1]) }
} catch { Write-Host $_.Exception.Message }
Write-Host "=== history 4fc14cfb ==="
try {
    $raw = (Invoke-WebRequest -Uri "http://127.0.0.1:8000/history/4fc14cfb-2260-43cf-92dd-6de932641740" -UseBasicParsing -TimeoutSec 8).Content
    Write-Host ("len={0}" -f $raw.Length)
    Write-Host $raw.Substring(0, [Math]::Min(400, $raw.Length))
} catch { Write-Host $_.Exception.Message }
Write-Host "=== runners ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "4fc14cfb|_run_vidu|otr_canonical_api_run"
} | ForEach-Object {
    Write-Host ("pid={0}" -f $_.ProcessId)
    Write-Host $_.CommandLine.Substring(0, [Math]::Min(300, $_.CommandLine.Length))
}
Write-Host "=== log hits ==="
$logs = @(
    "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000_live.log",
    "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000.log",
    "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_server.log"
)
foreach ($log in $logs) {
    if (Test-Path $log) {
        $item = Get-Item $log
        Write-Host ("LOG {0} bytes={1} mtime={2}" -f $item.Name, $item.Length, $item.LastWriteTime)
    }
}
$live = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000_live.log"
# Agy may have a different log next to pid 22752 - search recent tmp logs
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp" -File |
    Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-30) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "=== obs newest ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 Name, Length, LastWriteTime |
    Format-Table -AutoSize
