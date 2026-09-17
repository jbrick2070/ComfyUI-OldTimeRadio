$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$root = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$log = Join-Path $root "tmp\cloud_1act_night.log"
Set-Location $root
if (Test-Path $log) { Remove-Item $log -Force }
$p = Start-Process -FilePath $py -ArgumentList "_tmp_night_submit_1act.py" -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $log -PassThru -WindowStyle Hidden
Write-Host ("SUBMIT_PID={0} log={1}" -f $p.Id, $log)
