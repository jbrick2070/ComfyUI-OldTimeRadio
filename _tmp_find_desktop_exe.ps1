$ErrorActionPreference = "Continue"
Write-Host "=== find desktop exe ==="
$places = @(
    "$env:LOCALAPPDATA\Programs\ComfyUI",
    "$env:LOCALAPPDATA\Programs\Comfy Desktop",
    "$env:LOCALAPPDATA\comfyorg",
    "C:\Program Files\ComfyUI",
    "C:\Program Files\Comfy Desktop"
)
Get-ChildItem $env:LOCALAPPDATA\Programs -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "Comfy" } |
    ForEach-Object { Write-Host $_.FullName }
Get-ChildItem "$env:LOCALAPPDATA\Programs" -Recurse -Filter "*.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "Comfy" } |
    Select-Object -First 20 FullName
Write-Host "=== log tail time ==="
Get-Item "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\user\comfyui.log" |
    Select-Object Length, LastWriteTime
Write-Host "=== 8000 still ==="
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 8
    $id = if (@($q.queue_running).Count) { $q.queue_running[0][1] } else { "-" }
    Write-Host ("8000 running={0} id={1}" -f @($q.queue_running).Count, $id)
} catch { Write-Host "8000 FAIL" }
