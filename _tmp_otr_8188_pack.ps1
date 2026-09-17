$ErrorActionPreference = "Continue"
Write-Host "=== custom_nodes OTR path ==="
$root = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
Get-ChildItem $root -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "OldTime|OTR" } | ForEach-Object {
    Write-Host ("{0} Link={1} Target={2}" -f $_.FullName, $_.Attributes, $_.Target)
}
$doc = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Write-Host ("doc pack exists: {0}" -f (Test-Path $doc))
Write-Host "=== 8188 OTR node keys ==="
try {
    $oi = Invoke-RestMethod -Uri "http://127.0.0.1:8188/object_info" -TimeoutSec 30
    $keys = $oi.PSObject.Properties.Name | Where-Object { $_ -like "OTR_*" }
    Write-Host ("OTR node count {0}" -f $keys.Count)
    $keys | Sort-Object | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host $_.Exception.Message
}
Write-Host "=== obs dir ==="
Write-Host ("desktop obs: {0}" -f (Test-Path "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"))
Write-Host ("repo obs: {0}" -f (Test-Path "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr\obs"))
