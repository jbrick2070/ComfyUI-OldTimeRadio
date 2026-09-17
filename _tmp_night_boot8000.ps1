$ErrorActionPreference = "Continue"
Write-Host "=== 8188 must stay ==="
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8188/queue" -TimeoutSec 5
    $id = if (@($q.queue_running).Count) { $q.queue_running[0][1] } else { "-" }
    Write-Host ("8188 running={0} pending={1} id={2}" -f @($q.queue_running).Count, @($q.queue_pending).Count, $id)
} catch {
    Write-Host "8188 FAIL -- abort boot"
    exit 3
}
Write-Host "=== 8000 must be down ==="
$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listen) {
    Write-Host ("8000 already LISTEN pid {0}" -f $listen[0].OwningProcess)
    exit 4
}
Write-Host "8000 empty -- booting cpu server"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py _tmp_boot_cpu_8000.py
exit $LASTEXITCODE
