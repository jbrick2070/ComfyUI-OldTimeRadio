$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:OTR_ENABLE_COMFY_CREDITS = "1"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$log = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\cloud_3act_submit.log"
& $py _tmp_night_submit_3act.py *>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
