@echo off
REM ============================================================
REM Sprint H bug-hunt: between-iter filtered python sweep.
REM
REM Usage:
REM     sweep_python_excluding.bat <SUPERVISOR_PID>
REM
REM Kills every python.exe / pythonw.exe on the box EXCEPT the
REM PID passed as argv[1]. The supervisor invokes this between
REM every iter to drop the previous iter's worker tree (ComfyUI
REM child + any leftover ffmpeg / cuda CPU-spinner processes) while
REM keeping its own loop alive.
REM
REM Per synthesis 2026-05-17 section 1.2:
REM   - PID passed via argv -- NEVER hardcoded.
REM   - Every killed PID + cmdline + timestamp logged to
REM     logs\sweep_betweeniter.log. No silent kills.
REM   - Operator note: this WILL kill unrelated python tools
REM     (editors, jupyter, pytest in another window). Acceptable
REM     for a closed overnight box; do not run while doing other
REM     python work in parallel.
REM
REM Exit code: 0 always. The sweep is best-effort; a failed
REM Stop-Process logs to the sweep log but does not abort the
REM supervisor.
REM ============================================================
setlocal EnableDelayedExpansion

if "%~1"=="" (
  echo ERROR: missing SUPERVISOR_PID argv[1]
  echo Usage: sweep_python_excluding.bat ^<SUPERVISOR_PID^>
  exit /b 2
)
set SUPERVISOR_PID=%~1

set LOGSDIR=C:\Users\jeffr\Documents\ComfyUI\logs
set SWEEP_LOG=%LOGSDIR%\sweep_betweeniter.log

if not exist "%LOGSDIR%" mkdir "%LOGSDIR%"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$log = '%SWEEP_LOG%';" ^
  "$keep = [int]%SUPERVISOR_PID%;" ^
  "$now = (Get-Date).ToString('o');" ^
  "Add-Content -Path $log -Value ('[' + $now + '] === BETWEEN-ITER SWEEP START keep=' + $keep + ' ===') -Encoding UTF8;" ^
  "try { $before = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) -as [int] } catch { $before = $null };" ^
  "Add-Content -Path $log -Value ('[' + $now + '] VRAM_BEFORE_MB=' + $before) -Encoding UTF8;" ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Where-Object { $_.ProcessId -ne $keep };" ^
  "if ($procs.Count -eq 0) { Add-Content -Path $log -Value ('[' + $now + '] NO_KILLABLE_PYTHON') -Encoding UTF8 }" ^
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
  "Add-Content -Path $log -Value ('[' + $now2 + '] === BETWEEN-ITER SWEEP DONE ===') -Encoding UTF8;" ^
  "Write-Host ('between-iter sweep done: VRAM_BEFORE_MB=' + $before + '  VRAM_AFTER_MB=' + $after)"

endlocal & exit /b 0
