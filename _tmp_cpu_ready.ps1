$ErrorActionPreference = "Continue"
Write-Host "=== 8000 queue ==="
try {
    (Invoke-WebRequest -Uri "http://127.0.0.1:8000/queue" -UseBasicParsing -TimeoutSec 5).Content
} catch { Write-Host $_.Exception.Message }
Write-Host "=== act_count ==="
try {
    $w = Invoke-RestMethod -Uri "http://127.0.0.1:8000/object_info/OTR_LedgerScriptWriter" -TimeoutSec 20
    $acts = $w.OTR_LedgerScriptWriter.input.required.act_count
    if (-not $acts) { $acts = $w.OTR_LedgerScriptWriter.input.optional.act_count }
    $acts | ConvertTo-Json -Compress
} catch { Write-Host $_.Exception.Message }
Write-Host "=== vidu/luma nodes ==="
try {
    $keys = (Invoke-RestMethod -Uri "http://127.0.0.1:8000/object_info" -TimeoutSec 30).PSObject.Properties.Name
    $keys | Where-Object { $_ -match "Vidu|Luma|Eleven|Sonilo" } | ForEach-Object { Write-Host $_ }
} catch { Write-Host $_.Exception.Message }
Write-Host "=== cpu contract ==="
Select-String -Path "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000_live.log" -Pattern "Device:|vram state|Starting server|ERROR|All 25" |
    Select-Object -Last 20 | ForEach-Object { Write-Host $_.Line }
