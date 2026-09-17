$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py scripts/build_variants.py --check
exit $LASTEXITCODE
