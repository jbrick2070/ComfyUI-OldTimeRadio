$ErrorActionPreference = "Stop"
$custom = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
$bak = Join-Path $custom "comfyui-old-time-radio._registry_2.1.2.bak"
$destRoot = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\_disabled_packs"
$dest = Join-Path $destRoot "comfyui-old-time-radio._registry_2.1.2.bak"
$link = Join-Path $custom "comfyui-old-time-radio"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

if (-not (Test-Path -LiteralPath $destRoot)) {
    New-Item -ItemType Directory -Path $destRoot | Out-Null
}

if (Test-Path -LiteralPath $bak) {
    if (Test-Path -LiteralPath $dest) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $dest = Join-Path $destRoot ("comfyui-old-time-radio._registry_2.1.2.{0}.bak" -f $stamp)
    }
    Write-Host ("MOVE {0}" -f $bak)
    Write-Host ("  -> {0}" -f $dest)
    Move-Item -LiteralPath $bak -Destination $dest
} else {
    Write-Host "no bak folder in custom_nodes"
}

$item = Get-Item -LiteralPath $link
Write-Host ("live pack attrs={0} target={1}" -f $item.Attributes, (@($item.Target) -join ";"))
if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw "live pack is not a junction"
}
if ((@($item.Target) -join ";") -ne $repo) {
    throw "junction does not point at git repo"
}

Write-Host "--- custom_nodes dirs ---"
Get-ChildItem -LiteralPath $custom -Directory | ForEach-Object { $_.Name }
Write-Host "OK bak is outside custom_nodes"
exit 0
