$ErrorActionPreference = "Continue"
Write-Host "=== listeners ==="
@(8000, 8001, 8188, 8189) | ForEach-Object {
    $p = $_
    Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host ("LISTEN {0} pid {1}" -f $p, $_.OwningProcess) }
}
Write-Host "=== Comfy processes ==="
Get-Process | Where-Object { $_.ProcessName -match "Comfy|electron" } |
    Select-Object Id, ProcessName | Format-Table -AutoSize
Write-Host "=== /queue ==="
foreach ($url in @("http://127.0.0.1:8000/queue", "http://127.0.0.1:8001/queue", "http://127.0.0.1:8188/queue")) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        Write-Host ("OK {0} {1}" -f $url, $r.StatusCode)
        Write-Host $r.Content.Substring(0, [Math]::Min(180, $r.Content.Length))
    } catch {
        Write-Host ("FAIL {0}" -f $url)
    }
}
Write-Host "=== object_info OTR sample ==="
try {
    $oi = Invoke-WebRequest -Uri "http://127.0.0.1:8000/object_info/OTR_LedgerScriptWriter" -UseBasicParsing -TimeoutSec 5
    Write-Host ("writer 8000 {0}" -f $oi.StatusCode)
} catch {
    Write-Host "no OTR writer on 8000"
}
