$ErrorActionPreference = "Continue"
Write-Host "=== 8000 any state ==="
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess |
    Format-Table -AutoSize
Write-Host "=== pids 17996 2956 47828 ==="
foreach ($id in 17996, 2956, 47828) {
    $p = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $id) -ErrorAction SilentlyContinue
    if ($p) {
        Write-Host ("alive pid={0} name={1}" -f $id, $p.Name)
    } else {
        Write-Host ("dead pid={0}" -f $id)
    }
}
Write-Host "=== try 8000 system_stats ==="
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/system_stats" -UseBasicParsing -TimeoutSec 8
    Write-Host ("stats {0} {1}" -f $r.StatusCode, $r.Content.Substring(0, [Math]::Min(300, $r.Content.Length)))
} catch {
    Write-Host $_.Exception.Message
}
Write-Host "=== latest shipping log names ==="
$leg = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs\shipping_set_20260914_233405"
Get-ChildItem $leg -Recurse -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 15 FullName, Length, LastWriteTime |
    Format-Table -AutoSize
$leg2 = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs\shipping_set_20260914_233155"
Get-ChildItem $leg2 -Recurse -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 Name, Length |
    Format-Table -AutoSize
