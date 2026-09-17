$ErrorActionPreference = "Continue"
Write-Host "=== queues ==="
foreach ($pair in @(
    @{n="8000-cloud"; u="http://127.0.0.1:8000/queue"},
    @{n="8188-5080"; u="http://127.0.0.1:8188/queue"}
)) {
    try {
        $q = Invoke-RestMethod -Uri $pair.u -TimeoutSec 8
        $id = if (@($q.queue_running).Count) { $q.queue_running[0][1] } else { "-" }
        Write-Host ("{0} running={1} pending={2} id={3}" -f $pair.n, @($q.queue_running).Count, @($q.queue_pending).Count, $id)
    } catch { Write-Host ("{0} FAIL" -f $pair.n) }
}
Write-Host "=== 16gb summary ==="
$sum = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs\shipping_set_20260915_000148\SUMMARY.txt"
if (Test-Path $sum) { Get-Content $sum }
Write-Host "=== night cpu log last 8 ==="
Get-Content "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000_night.log" -Tail 8
