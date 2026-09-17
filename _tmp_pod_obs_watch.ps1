$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$dest = "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py scripts/otr_pod_obs_bridge.py 3o8mxio6jm3t3n --watch --host 213.173.105.175 --port 13317 --key "$env:USERPROFILE\.ssh\runpod_otr" --dest $dest --poll-s 60 --max-wait-s 43200
exit $LASTEXITCODE
