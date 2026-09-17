$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py -m pytest -q -p no:cacheprovider tests/test_pod_obs_bridge_ssh_watch.py tests/test_obs_published_filename.py
exit $LASTEXITCODE
