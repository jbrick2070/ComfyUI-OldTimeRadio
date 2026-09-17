$ErrorActionPreference = "Continue"
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 5
    Write-Host ("running={0} pending={1}" -f @($q.queue_running).Count, @($q.queue_pending).Count)
    if (@($q.queue_running).Count) { Write-Host ("id={0}" -f $q.queue_running[0][1]) }
} catch { Write-Host ("queue FAIL {0}" -f $_.Exception.Message) }
foreach ($id in @("4fc14cfb-2260-43cf-92dd-6de932641740", "7b0a5552-f6b5-4df8-b109-da510d26643f")) {
    try {
        $h = Invoke-RestMethod -Uri ("http://127.0.0.1:8000/history/{0}" -f $id) -TimeoutSec 8
        $keys = @($h.PSObject.Properties.Name)
        Write-Host ("history {0} keys={1}" -f $id, ($keys -join ","))
        if ($keys.Count -and $h.($keys[0]).status) {
            $h.($keys[0]).status | ConvertTo-Json -Compress
        }
    } catch { Write-Host ("history {0} FAIL" -f $id) }
}
Write-Host "=== obs since 23:40 ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 |
    Where-Object { $_.LastWriteTime -gt (Get-Date "2026-09-14 23:40") } |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
