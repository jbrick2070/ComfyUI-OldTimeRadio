$ErrorActionPreference = "Continue"
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 5
    $run = @($q.queue_running)
    $pend = @($q.queue_pending)
    Write-Host ("running={0} pending={1}" -f $run.Count, $pend.Count)
    if ($run.Count) {
        Write-Host ("running_id={0}" -f $run[0][1])
    }
} catch { Write-Host ("queue FAIL {0}" -f $_.Exception.Message) }
try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/history/4fc14cfb-2260-43cf-92dd-6de932641740" -TimeoutSec 8
    $h | ConvertTo-Json -Depth 4 -Compress | ForEach-Object { $_.Substring(0, [Math]::Min(800, $_.Length)) }
} catch { Write-Host ("history FAIL {0}" -f $_.Exception.Message) }
Write-Host "=== obs smoke still there ==="
Get-Item "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\comfy_mcp_vidu_q2_t2v_radio_studio_smoke.mp4" |
    Select-Object Name, Length, LastWriteTime | Format-List
