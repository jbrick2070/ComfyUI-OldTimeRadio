$ErrorActionPreference = "Continue"
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 5
    Write-Host ("running={0} pending={1}" -f @($q.queue_running).Count, @($q.queue_pending).Count)
    if (@($q.queue_running).Count) { Write-Host ("id={0}" -f $q.queue_running[0][1]) }
} catch { Write-Host $_.Exception.Message }
$log = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000.log"
if (Test-Path $log) {
    $item = Get-Item $log
    Write-Host ("log bytes={0} mtime={1}" -f $item.Length, $item.LastWriteTime)
    Select-String -Path $log -Pattern "obs_publish|Prompt executed|ERROR|Exception|ImageGen|cloud_luma|cloud_vidu|rainbow_bowl" |
        Select-Object -Last 15 |
        ForEach-Object { Write-Host $_.Line }
}
Write-Host "=== obs newest ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 Name, Length, LastWriteTime |
    Format-Table -AutoSize
