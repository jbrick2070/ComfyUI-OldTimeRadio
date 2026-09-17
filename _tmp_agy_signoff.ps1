$ErrorActionPreference = "Continue"
Write-Host "=== 8000/8188 ==="
@(8000, 8188) | ForEach-Object {
    $p = $_
    Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host ("LISTEN {0} pid {1}" -f $p, $_.OwningProcess) }
}
Write-Host "=== queues ==="
foreach ($url in @("http://127.0.0.1:8000/queue", "http://127.0.0.1:8188/queue")) {
    try {
        Write-Host $url
        Write-Host (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 4).Content.Substring(0, [Math]::Min(500, (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 4).Content.Length))
    } catch { Write-Host ("FAIL {0}" -f $url) }
}
Write-Host "=== comfy python ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "main.py|otr_canonical|_run_vidu|_tmp_submit|_tmp_queue|_tmp_boot"
} | ForEach-Object {
    Write-Host ("pid={0}" -f $_.ProcessId)
    Write-Host $_.CommandLine.Substring(0, [Math]::Min(420, $_.CommandLine.Length))
    Write-Host "---"
}
Write-Host "=== newest legs ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\legs" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name, LastWriteTime |
    Format-Table -AutoSize
Write-Host "=== obs newest mp4 ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 6 Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "=== cpu log tail ==="
$log = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000_live.log"
if (Test-Path $log) {
    Get-Content $log -Tail 25
}
