$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py -m pytest -q -p no:cacheprovider tests/test_comfy_slot_widgets.py tests/test_cloud_video_adapters.py
Write-Host "PYTEST_RC=$LASTEXITCODE"
