@echo off
REM ============================================================
REM Sprint H bug-hunt: pre-launch python sweep + supervisor spawn.
REM
REM Step 1 (blanket sweep, safe because no python the loop cares
REM         about is alive yet): enumerate every python.exe and
REM         pythonw.exe on the box. For each: log timestamp + PID +
REM         truncated CommandLine to logs\sweep_prelaunch.log, then
REM         Stop-Process -Force.
REM Step 2 (supervisor launch): start
REM         scripts\overnight_bug_hunt.py as a normal python
REM         subprocess. The supervisor's argv is forwarded verbatim
REM         via %*. This bat exits after the python process is
REM         spawned -- the supervisor inherits stdin/stdout from
REM         the launching cmd window, so the operator can attach
REM         a console or close the window and the supervisor keeps
REM         running. (For fully detached overnight use, wrap this
REM         bat in PowerShell Start-Process -WindowStyle Hidden.)
REM
REM Per synthesis 2026-05-17 section 1.2 + 3.5. Do NOT call this
REM mid-loop; the supervisor relies on its own PID staying alive
REM and a second invocation of this bat would kill it.
REM
REM Logs (UTF-8, append-only):
REM   logs\sweep_prelaunch.log     -- one line per killed PID
REM                                   with timestamp + cmdline
REM ============================================================
setlocal EnableDelayedExpansion

set LOGSDIR=C:\Users\jeffr\Documents\ComfyUI\logs
set SWEEP_LOG=%LOGSDIR%\sweep_prelaunch.log
set REPO=C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
set PY=C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
set SUPERVISOR=%REPO%\scripts\overnight_bug_hunt.py

if not exist "%LOGSDIR%" mkdir "%LOGSDIR%"

echo === sweep_and_launch.bat START ===
echo log: %SWEEP_LOG%

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$log = '%SWEEP_LOG%';" ^
  "$now = (Get-Date).ToString('o');" ^
  "Add-Content -Path $log -Value ('[' + $now + '] === PRELAUNCH SWEEP START ===') -Encoding UTF8;" ^
  "try { $before = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) -as [int] } catch { $before = $null };" ^
  "Add-Content -Path $log -Value ('[' + $now + '] VRAM_BEFORE_MB=' + $before) -Encoding UTF8;" ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\";" ^
  "if ($procs.Count -eq 0) { Add-Content -Path $log -Value ('[' + $now + '] NO_PYTHON_RUNNING') -Encoding UTF8 }" ^
  "else { foreach ($p in $procs) {" ^
  "    $cmd = if ($p.CommandLine) { $p.CommandLine.Substring(0,[Math]::Min(180,$p.CommandLine.Length)) } else { '(no cmd)' };" ^
  "    Add-Content -Path $log -Value ('[' + $now + '] KILL PID=' + $p.ProcessId + ' CMD=' + $cmd) -Encoding UTF8;" ^
  "    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }" ^
  "    catch { Add-Content -Path $log -Value ('[' + $now + ']   stop-process FAILED PID=' + $p.ProcessId + ' err=' + $_.Exception.Message) -Encoding UTF8 }" ^
  "} ; Start-Sleep -Seconds 4 };" ^
  "$now2 = (Get-Date).ToString('o');" ^
  "try { $after = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) -as [int] } catch { $after = $null };" ^
  "Add-Content -Path $log -Value ('[' + $now2 + '] VRAM_AFTER_MB=' + $after) -Encoding UTF8;" ^
  "if ($before -ne $null -and $after -ne $null) { Add-Content -Path $log -Value ('[' + $now2 + '] FREED_MB=' + ($before - $after)) -Encoding UTF8 };" ^
  "Add-Content -Path $log -Value ('[' + $now2 + '] === PRELAUNCH SWEEP DONE ===') -Encoding UTF8;" ^
  "Write-Host ('VRAM_BEFORE_MB=' + $before + '  VRAM_AFTER_MB=' + $after)"

echo.
echo === launching supervisor ===
echo cmd: "%PY%" "%SUPERVISOR%" %*
"%PY%" "%SUPERVISOR%" %*
set RC=%ERRORLEVEL%

echo === sweep_and_launch.bat DONE rc=%RC% ===
endlocal & exit /b %RC%
