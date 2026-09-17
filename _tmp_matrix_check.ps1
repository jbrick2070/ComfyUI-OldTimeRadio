$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py scripts/otr_dropdown_matrix.py --check
Write-Output "dropdown_exit=$LASTEXITCODE"
& $py scripts/otr_tier_matrix.py --check
Write-Output "tier_exit=$LASTEXITCODE"
& $py scripts/otr_machine_matrix.py --check
Write-Output "machine_exit=$LASTEXITCODE"
