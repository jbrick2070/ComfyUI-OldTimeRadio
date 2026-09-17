$ErrorActionPreference = "Continue"
foreach ($pair in @(
    @{n="8000-cloud"; u="http://127.0.0.1:8000/queue"},
    @{n="8188-5080"; u="http://127.0.0.1:8188/queue"}
)) {
    try {
        $q = Invoke-RestMethod -Uri $pair.u -TimeoutSec 5
        $id = if (@($q.queue_running).Count) { $q.queue_running[0][1] } else { "-" }
        Write-Host ("{0} running={1} pending={2} id={3}" -f $pair.n, @($q.queue_running).Count, @($q.queue_pending).Count, $id)
    } catch { Write-Host ("{0} FAIL" -f $pair.n) }
}
Write-Host "=== obs after 23:40 ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 |
    Where-Object { $_.LastWriteTime -gt (Get-Date "2026-09-14 23:40") } |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "=== 16gb summary ==="
$sum = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs\shipping_set_20260915_000148\SUMMARY.txt"
if (Test-Path $sum) { Get-Content $sum }
