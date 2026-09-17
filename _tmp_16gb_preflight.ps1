$ErrorActionPreference = "Continue"
Write-Host "=== queues ==="
foreach ($url in @("http://127.0.0.1:8000/queue", "http://127.0.0.1:8188/queue")) {
    try {
        $q = Invoke-RestMethod -Uri $url -TimeoutSec 5
        Write-Host ("{0} running={1} pending={2}" -f $url, @($q.queue_running).Count, @($q.queue_pending).Count)
        if (@($q.queue_running).Count) { Write-Host ("  id={0}" -f $q.queue_running[0][1]) }
    } catch { Write-Host ("FAIL {0}" -f $url) }
}
Write-Host "=== nvidia ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
Write-Host "=== 8188 writer act_count ==="
try {
    $w = Invoke-RestMethod -Uri "http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter" -TimeoutSec 15
    $acts = $w.OTR_LedgerScriptWriter.input.required.act_count
    if (-not $acts) { $acts = $w.OTR_LedgerScriptWriter.input.optional.act_count }
    Write-Host (($acts | ConvertTo-Json -Compress).Substring(0, [Math]::Min(180, ($acts | ConvertTo-Json -Compress).Length)))
} catch { Write-Host $_.Exception.Message }
Write-Host "=== 16gb variants ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\variants\otr_16gb_*.json" |
    Select-Object Name, Length | Format-Table -AutoSize
