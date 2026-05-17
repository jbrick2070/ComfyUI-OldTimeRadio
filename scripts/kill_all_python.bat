@echo off
REM Sprint H bug-hunt: NUCLEAR. Kill every python.exe / pythonw.exe on
REM the box. VRAM + port 8000 are returned to the pool.
REM
REM ============================================================
REM MANUAL OPERATOR TOOL ONLY -- NOT for the loop codepath.
REM Per synthesis 2026-05-17 section 1.2, the autonomous bug-hunt
REM supervisor does NOT call this. The supervisor uses:
REM   scripts\sweep_and_launch.bat       (outer, pre-launch blanket)
REM   scripts\sweep_python_excluding.bat (between-iter, filtered by
REM                                       SUPERVISOR_PID)
REM Both of those log every killed PID + cmdline + timestamp to
REM logs\sweep_*.log so the loop is auditable. This bat logs only
REM to stdout and kills the supervisor along with everything else,
REM which is fine for a manual reset but fatal mid-loop.
REM
REM When to run THIS bat: before kicking off a fresh attended
REM session, when something has clearly wedged the GPU, or when
REM the operator wants a clean slate before anything autonomous
REM starts. Never invoke it from a loop driver.
REM ============================================================

powershell.exe -NoProfile -Command "$before = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -as [int]; Write-Host ('VRAM_BEFORE_MB ' + $before); $procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\"; if ($procs.Count -eq 0) { Write-Host 'NO_PYTHON_RUNNING' } else { Write-Host ('FOUND ' + $procs.Count + ' python process(es)'); foreach ($p in $procs) { $cmd = if ($p.CommandLine) { $p.CommandLine.Substring(0,[Math]::Min(140,$p.CommandLine.Length)) } else { '(no cmd)' }; Write-Host ('KILLING PID ' + $p.ProcessId + ' :: ' + $cmd); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } ; Start-Sleep -Seconds 5 } ; $after = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -as [int]; Write-Host ('VRAM_AFTER_MB ' + $after); $delta = $before - $after; if ($delta -gt 0) { Write-Host ('FREED_MB ' + $delta) }"
