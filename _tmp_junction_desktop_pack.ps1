$ErrorActionPreference = "Stop"
$deskNodes = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
$stale = Join-Path $deskNodes "comfyui-old-time-radio"
$bakName = "comfyui-old-time-radio._registry_2.1.2.bak"
$bak = Join-Path $deskNodes $bakName
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

function Stop-DesktopPackHolders {
    Get-Process -Name "Comfy Desktop" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host ("STOP Comfy Desktop pid={0}" -f $_.Id)
        Stop-Process -Id $_.Id -Force
    }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -match "ComfyUI-Installs\\ComfyUI \(1\)"
    } | ForEach-Object {
        Write-Host ("KILL desktop python pid={0}" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force
    }
}

Stop-DesktopPackHolders
Start-Sleep -Seconds 2

if (Test-Path -LiteralPath $stale) {
    $item = Get-Item -LiteralPath $stale
    $isLink = [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
    if ($isLink) {
        $target = @($item.Target) -join ";"
        Write-Host ("already a reparse point -> {0}" -f $target)
        if ($target -eq $repo) {
            Write-Host "junction already points at git repo"
            exit 0
        }
        cmd /c "rmdir `"$stale`""
        if ($LASTEXITCODE -ne 0) { throw "could not remove old junction" }
    } else {
        if (Test-Path -LiteralPath $bak) {
            $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
            $bak = Join-Path $deskNodes ("comfyui-old-time-radio._registry_2.1.2.{0}.bak" -f $stamp)
        }
        Write-Host ("renaming 2.1.2 pack to {0}" -f $bak)
        Rename-Item -LiteralPath $stale -NewName (Split-Path $bak -Leaf)
    }
}

cmd /c "mklink /J `"$stale`" `"$repo`""
if ($LASTEXITCODE -ne 0) { throw "mklink failed" }

$link = Get-Item -LiteralPath $stale
Write-Host ("link={0}" -f $link.FullName)
Write-Host ("attrs={0}" -f $link.Attributes)
Write-Host ("target={0}" -f (@($link.Target) -join ";"))
$ver = Select-String -LiteralPath (Join-Path $stale "pyproject.toml") -Pattern "^version"
Write-Host ("pyproject {0}" -f $ver.Line)
$spacesaver = Select-String -LiteralPath (Join-Path $stale "nodes\OTR_LedgerScriptWriter.py") -Pattern "perfect_run_spacesaver" -SimpleMatch
if ($spacesaver) {
    throw "junctioned tree still contains perfect_run_spacesaver"
}
Write-Host "writer has no perfect_run_spacesaver -- git pack is live"
exit 0
