$ErrorActionPreference = "Continue"
$src = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\output\otr\obs\the_count_of_three_20260915_001532_silent__pori__vcam__none__koko__myst__g412__sa3_final.mp4"
$dstDir = "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
$dst = Join-Path $dstDir (Split-Path $src -Leaf)
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
if (-not (Test-Path $src)) { Write-Host "MISSING src"; exit 2 }
if (-not (Test-Path $dst)) {
    Copy-Item -LiteralPath $src -Destination $dst
    Write-Host "COPIED to Documents obs"
} else {
    Write-Host "ALREADY in Documents obs"
}
Get-Item -LiteralPath $dst | Select-Object Name, Length, LastWriteTime
