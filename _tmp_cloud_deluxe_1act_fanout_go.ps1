$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:OTR_ENABLE_COMFY_CREDITS = "1"
$env:OTR_CLOUD_VIDEO_FANOUT = "8"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$bootLog = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\recycle_8000_deluxe_1act.log"
$subLog = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\cloud_deluxe_1act_fanout_submit.log"
& $py _tmp_recycle_8000_fanout.py *>&1 | Tee-Object -FilePath $bootLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py _tmp_night_submit_deluxe_1act.py *>&1 | Tee-Object -FilePath $subLog
exit $LASTEXITCODE
