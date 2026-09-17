$ErrorActionPreference = "Continue"
Write-Host "=== process start ==="
Get-CimInstance Win32_Process -Filter "ProcessId=34828" | Select-Object ProcessId, CreationDate, CommandLine | Format-List
Get-CimInstance Win32_Process -Filter "ProcessId=34324" | Select-Object ProcessId, CreationDate | Format-List

Write-Host "=== Installs custom_nodes listing ==="
$cn = "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
if (Test-Path $cn) {
    Get-ChildItem $cn | Select-Object Mode, Name, LinkType, Target | Format-Table -AutoSize
}

Write-Host "=== find OTR_LedgerScriptWriter.py ==="
$hits = @(
    "C:\Users\jeffr\ComfyUI-Installs",
    "C:\Users\jeffr\Documents\ComfyUI\custom_nodes",
    "C:\Users\jeffr\AppData\Roaming\Comfy Desktop"
)
foreach ($root in $hits) {
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem $root -Recurse -Filter "OTR_LedgerScriptWriter.py" -ErrorAction SilentlyContinue |
        Select-Object -First 20 FullName, LastWriteTime, Length
}

Write-Host "=== live file has spacesaver? ==="
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\OTR_LedgerScriptWriter.py"
Select-String -Path $repo -Pattern "perfect_run_spacesaver" | Select-Object -First 5 LineNumber, Line
