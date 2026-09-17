$ErrorActionPreference = "Continue"
$names = @("*count_of_three*","*signal_lost*")
$roots = @(
    "C:\Users\jeffr\Documents\ComfyUI\output\otr",
    "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\output\otr",
    "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr"
)
foreach ($root in $roots) {
    Write-Host "=== $root ==="
    if (-not (Test-Path $root)) { Write-Host "missing"; continue }
    Get-ChildItem $root -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "count_of_three|20260915_001532" } |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize
}
Write-Host "=== Comfy Desktop process ==="
Get-Process | Where-Object { $_.Name -match "Comfy|Electron" } |
    Select-Object Name, Id, StartTime |
    Format-Table -AutoSize
Write-Host "=== recent mp4 Documents obs ==="
Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 Name, Length, LastWriteTime
Write-Host "=== recent mp4 Installs obs ==="
$alt = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\output\otr\obs"
if (Test-Path $alt) {
    Get-ChildItem $alt | Sort-Object LastWriteTime -Descending |
        Select-Object -First 8 Name, Length, LastWriteTime
}
