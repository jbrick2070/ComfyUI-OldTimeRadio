$ErrorActionPreference = "Continue"
$desk = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-old-time-radio"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Write-Host "=== desktop pack ==="
if (Test-Path "$desk\.git") {
    Set-Location $desk
    git rev-parse --short HEAD
    git status -sb
    git remote -v
} else {
    Write-Host "NO GIT"
}
Write-Host "pyproject:"
Select-String -Path "$desk\pyproject.toml" -Pattern "^version" -ErrorAction SilentlyContinue
Write-Host "spacesaver in desktop writer?"
Select-String -Path "$desk\nodes\OTR_LedgerScriptWriter.py" -Pattern "perfect_run_spacesaver" |
    Select-Object -First 3 LineNumber, Line
Write-Host "=== repo ==="
Set-Location $repo
git rev-parse --short HEAD
Select-String -Path "$repo\pyproject.toml" -Pattern "^version"
Write-Host "spacesaver in repo INPUT_TYPES?"
Select-String -Path "$repo\nodes\OTR_LedgerScriptWriter.py" -Pattern '"perfect_run_spacesaver"'
Write-Host "=== sizes ==="
(Get-Item $desk).FullName
(Get-ChildItem $desk -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
