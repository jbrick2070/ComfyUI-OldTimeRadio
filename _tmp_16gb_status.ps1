$ErrorActionPreference = "Continue"
$legRoot = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs"
Get-ChildItem $legRoot | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object {
    Write-Host ("LEGDIR {0}" -f $_.FullName)
    Get-ChildItem $_.FullName | Format-Table Name, Length, LastWriteTime -AutoSize
    $sum = Join-Path $_.FullName "SUMMARY.txt"
    if (Test-Path $sum) { Get-Content $sum }
    Get-ChildItem $_.FullName -Filter *.log | ForEach-Object {
        Write-Host ("--- {0} ---" -f $_.Name)
        Get-Content $_.FullName -Tail 20
    }
}
Write-Host "=== 8188 queue ==="
try { (Invoke-WebRequest http://127.0.0.1:8188/queue -UseBasicParsing -TimeoutSec 5).Content.Substring(0,400) } catch { $_.Exception.Message }
Write-Host "=== 8000 still cloud ==="
try {
    $q = Invoke-RestMethod http://127.0.0.1:8000/queue -TimeoutSec 5
    Write-Host ("running={0}" -f @($q.queue_running).Count)
    if (@($q.queue_running).Count) { Write-Host $q.queue_running[0][1] }
} catch { Write-Host $_.Exception.Message }
