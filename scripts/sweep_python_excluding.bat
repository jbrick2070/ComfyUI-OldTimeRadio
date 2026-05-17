@echo off
REM ============================================================
REM Sprint H bug-hunt: between-iter filtered python sweep.
REM
REM Usage:
REM     sweep_python_excluding.bat <KEEP_PID> [KEEP_PID ...]
REM
REM Kills every python.exe / pythonw.exe on the box EXCEPT the
REM PIDs passed via argv. Variadic so callers can preserve
REM multiple PIDs simultaneously -- required under the uv
REM launcher-stub model where the supervisor "process" is
REM actually two python.exe processes (stub + real cpython, both
REM with Name='python.exe' to WMI).
REM
REM Per synthesis 2026-05-17 section 1.2 + Jeffrey 2026-05-17
REM round-robin section B2:
REM   - PIDs passed via argv -- NEVER hardcoded.
REM   - Every killed PID + cmdline + timestamp logged to
REM     logs\sweep_betweeniter.log. Every kept PID also logged
REM     so a post-run audit can confirm the keep-list reached
REM     PowerShell intact.
REM   - Operator note: this WILL kill unrelated python tools
REM     (editors, jupyter, pytest in another window) -- they're
REM     not in the keep-list. Acceptable for a closed overnight
REM     box; do not run while doing other python work in parallel.
REM
REM Implementation note (round-robin B2, caught on review +
REM iter-1 standalone test 2026-05-17):
REM   - First-draft used `-- %*` to push argv into PowerShell's
REM     $args. cmd.exe parsed this fine, but PowerShell's
REM     -Command parser consumed everything after the script
REM     close-quote AS the script -- $args came out empty and
REM     the keep-list was silently lost. Bug confirmed by
REM     standalone test 2026-05-17.
REM   - Fix: build a CSV in cmd via shift+goto loop, hand it to
REM     PowerShell as a single env var, and have PowerShell
REM     -split it back to an int[]. The downside Jeffrey flagged
REM     in the round-robin -- "scalar string compared by
REM     -notcontains" -- is closed by the explicit -split and
REM     [int] cast on each element.
REM   - `[int]` casts on BOTH sides of `-notcontains` remain
REM     load-bearing: `$p.ProcessId` is UInt32 from CIM,
REM     parsed-from-string `$_` is Int32 -- without aligned
REM     casts the containment check can type-mismatch and
REM     false-negative.
REM
REM Exit code: 0 always. The sweep is best-effort; a failed
REM Stop-Process logs to the sweep log but does not abort the
REM supervisor.
REM ============================================================
setlocal EnableDelayedExpansion

if "%~1"=="" (
  echo ERROR: missing KEEP_PID argv list
  echo Usage: sweep_python_excluding.bat ^<KEEP_PID^> [KEEP_PID ...]
  exit /b 2
)

set LOGSDIR=C:\Users\jeffr\Documents\ComfyUI\logs
set SWEEP_LOG=%LOGSDIR%\sweep_betweeniter.log

if not exist "%LOGSDIR%" mkdir "%LOGSDIR%"

REM Build a CSV of keep PIDs via shift+goto. Variadic-safe.
REM Result: SWEEP_KEEP_CSV=pid1,pid2,...,pidN
set "SWEEP_KEEP_CSV="
:_keep_loop
if "%~1"=="" goto :_keep_done
if defined SWEEP_KEEP_CSV (
  set "SWEEP_KEEP_CSV=!SWEEP_KEEP_CSV!,%~1"
) else (
  set "SWEEP_KEEP_CSV=%~1"
)
shift
goto :_keep_loop
:_keep_done

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$log = '%SWEEP_LOG%';" ^
  "$csv = $env:SWEEP_KEEP_CSV;" ^
  "$keep = @($csv -split ',') | Where-Object { $_ -and $_.Trim().Length -gt 0 } | ForEach-Object { [int]$_.Trim() };" ^
  "$now = (Get-Date).ToString('o');" ^
  "Add-Content -Path $log -Value ('[' + $now + '] === BETWEEN-ITER SWEEP START keep_pids=' + ($keep -join ',') + ' ===') -Encoding UTF8;" ^
  "Write-Host ('keep_pids=' + ($keep -join ','));" ^
  "try { $before = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) -as [int] } catch { $before = $null };" ^
  "Add-Content -Path $log -Value ('[' + $now + '] VRAM_BEFORE_MB=' + $before) -Encoding UTF8;" ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\";" ^
  "$killed = 0; $kept = 0;" ^
  "foreach ($p in $procs) {" ^
  "    $pid_i = [int]$p.ProcessId;" ^
  "    $cmd = if ($p.CommandLine) { $p.CommandLine.Substring(0,[Math]::Min(180,$p.CommandLine.Length)) } else { '(no cmd)' };" ^
  "    if ($keep -notcontains $pid_i) {" ^
  "        Add-Content -Path $log -Value ('[' + $now + '] KILL PID=' + $pid_i + ' CMD=' + $cmd) -Encoding UTF8;" ^
  "        try { Stop-Process -Id $pid_i -Force -ErrorAction Stop; $killed++ }" ^
  "        catch { Add-Content -Path $log -Value ('[' + $now + ']   stop-process FAILED PID=' + $pid_i + ' err=' + $_.Exception.Message) -Encoding UTF8 }" ^
  "    } else {" ^
  "        Add-Content -Path $log -Value ('[' + $now + '] KEEP PID=' + $pid_i + ' CMD=' + $cmd) -Encoding UTF8;" ^
  "        $kept++" ^
  "    }" ^
  "}" ^
  "if ($procs.Count -eq 0) { Add-Content -Path $log -Value ('[' + $now + '] NO_PYTHON_RUNNING') -Encoding UTF8 } else { Start-Sleep -Seconds 4 };" ^
  "$now2 = (Get-Date).ToString('o');" ^
  "try { $after = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) -as [int] } catch { $after = $null };" ^
  "Add-Content -Path $log -Value ('[' + $now2 + '] VRAM_AFTER_MB=' + $after) -Encoding UTF8;" ^
  "if ($before -ne $null -and $after -ne $null) { Add-Content -Path $log -Value ('[' + $now2 + '] FREED_MB=' + ($before - $after)) -Encoding UTF8 };" ^
  "Add-Content -Path $log -Value ('[' + $now2 + '] === BETWEEN-ITER SWEEP DONE killed=' + $killed + ' kept=' + $kept + ' ===') -Encoding UTF8;" ^
  "Write-Host ('between-iter sweep done: killed=' + $killed + ' kept=' + $kept + ' VRAM_BEFORE_MB=' + $before + '  VRAM_AFTER_MB=' + $after)"

endlocal & exit /b 0
