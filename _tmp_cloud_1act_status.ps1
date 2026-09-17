$ports = 8000, 8188
foreach ($p in $ports) {
    $listens = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    $est = Get-NetTCPConnection -LocalPort $p -State Established -ErrorAction SilentlyContinue
    Write-Host "PORT $p listen=$($listens.Count) established=$($est.Count)"
}
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 5
    Write-Host "8000 running=$($q.queue_running.Count) pending=$($q.queue_pending.Count)"
} catch { Write-Host "8000 queue_err $($_.Exception.Message)" }
try {
    $q2 = Invoke-RestMethod -Uri "http://127.0.0.1:8188/queue" -TimeoutSec 5
    Write-Host "8188 running=$($q2.queue_running.Count) pending=$($q2.queue_pending.Count)"
} catch { Write-Host "8188 queue_err $($_.Exception.Message)" }
