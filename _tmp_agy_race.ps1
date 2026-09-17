$ErrorActionPreference = "Continue"
Write-Host "=== 8000 listen ==="
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("LISTEN 8000 pid {0}" -f $_.OwningProcess) }
Write-Host "=== 8188 listen ==="
Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("LISTEN 8188 pid {0}" -f $_.OwningProcess) }
Write-Host "=== queues ==="
foreach ($url in @("http://127.0.0.1:8000/queue", "http://127.0.0.1:8188/queue")) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 4
        Write-Host $url
        Write-Host $r.Content.Substring(0, [Math]::Min(400, $r.Content.Length))
    } catch { Write-Host ("FAIL {0}" -f $url) }
}
Write-Host "=== python Comfy cmdlines ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "ComfyUI|otr_canonical|shipping_set"
} | ForEach-Object {
    Write-Host ("pid={0}" -f $_.ProcessId)
    Write-Host $_.CommandLine.Substring(0, [Math]::Min(450, $_.CommandLine.Length))
    Write-Host "---"
}
Write-Host "=== latest legs ==="
if (Test-Path "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs") {
    Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 8 Name, Length, LastWriteTime |
        Format-Table -AutoSize
}
Write-Host "=== obs newest ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "=== tmp cpu logs ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 Name, Length, LastWriteTime |
    Format-Table -AutoSize
