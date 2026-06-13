<#
  otr_run_leg.ps1 -- the ONE render-launch path (operator directive 2026-06-12:
  wire the watchdog into EVERY render, auto not manual). It:

    1. RESETS aggressively: kills all python, confirms no :8000 listener +
       nvidia-smi back to the desktop baseline (per the CLAUDE.md directive).
    2. Stages the 0-E enable flags + OTR_LTX_I2V_STRENGTH (+ any -ExtraEnv) on
       the launcher env seam.
    3. Boots the CANONICAL launcher (_otr_soak_server_launch.cmd) and waits for
       health (system_stats + OTR_VideoDirector).
    4. Launches the coverage-sweep leg detached.
    5. AUTO-arms otr_render_watchdog.ps1 detached -> writes <leglog>.watchdog
       (RUNNING/DONE/DEAD) so a 5-min stall or a down queue is declared fast.

  Non-blocking: prints the leg log + verdict file + pids and returns. The caller
  polls <leglog>.watchdog for DONE/DEAD instead of waiting the full ~24 min.

  Usage:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_run_leg.ps1 `
        -Leg other_beats_visual_ltx_orbit
#>
param(
    [string]$Leg = 'other_beats_visual_ltx_orbit',
    [string]$StrengthI2V = '0.6',
    [string[]]$ExtraEnv = @(),
    [int]$StallSeconds = 300,
    [string]$Tag = ''
)
$ErrorActionPreference = 'Stop'
$REPO = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$VENV = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$COMFY = 'C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI'
$LAUNCH = Join-Path $REPO 'scripts\_otr_soak_server_launch.cmd'
$SRVLOG = 'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log'
$EXTRA = Join-Path $REPO 'scripts\_otr_soak_capstone_results\_marathon_extra_env.cmd'
$WD = Join-Path $REPO 'scripts\otr_render_watchdog.ps1'
if (-not $Tag) { $Tag = $Leg }
$LEGLOG = Join-Path $REPO ("scripts\_otr_leg_{0}.log" -f $Tag)
$VERDICT = "$LEGLOG.watchdog"

function Say($m) { Write-Host ("[run_leg] {0} {1}" -f (Get-Date -Format HH:mm:ss), $m) }

# 1. RESET ------------------------------------------------------------------
Say "reset: killing all python"
Stop-Process -Name python, pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listen) { foreach ($p in ($listen.OwningProcess | Select-Object -Unique)) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }; Start-Sleep 2 }
$vram = (& 'nvidia-smi' --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1
Say "after reset: :8000 listeners=$((Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count) vram=${vram}MiB"

# 2. STAGE ENV --------------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Split-Path $EXTRA) | Out-Null
Remove-Item Env:OTR_C7 -ErrorAction SilentlyContinue
$envLines = @('set OTR_ENABLE_LTX_ORBIT=1', 'set OTR_ENABLE_STILL_PARALLAX=1',
    'set OTR_ENABLE_MESH_STAGE=1', "set OTR_LTX_I2V_STRENGTH=$StrengthI2V") + $ExtraEnv
$envLines | Set-Content -Path $EXTRA -Encoding ascii
Say "staged env: $($envLines -join ' | ')"

# 3. BOOT + HEALTH ----------------------------------------------------------
$srv = Start-Process -FilePath $LAUNCH -ArgumentList "`"$SRVLOG`"" -WindowStyle Hidden -PassThru
Say "server boot pid=$($srv.Id); waiting for health..."
$up = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 10
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/system_stats' -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) {
            $oi = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/object_info/OTR_VideoDirector' -UseBasicParsing -TimeoutSec 5
            if ($oi.StatusCode -eq 200) { $up = $true; Say "server healthy after ~$([int](($i+1)*10))s"; break }
        }
    } catch {}
}
if (-not $up) {
    Say "SERVER DID NOT COME UP -- aborting"
    "{0} | DEAD | server did not come up" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Set-Content -Path $VERDICT -Encoding ascii
    exit 2
}

# 4. LAUNCH LEG -------------------------------------------------------------
$env:PYTHONPATH = $COMFY
$leg = Start-Process -FilePath $VENV -ArgumentList @('-u', 'scripts\otr_coverage_sweep.py', '--only', $Leg) `
    -WorkingDirectory $REPO -RedirectStandardOutput $LEGLOG -RedirectStandardError ($LEGLOG + '.err') `
    -WindowStyle Hidden -PassThru
Say "leg launched pid=$($leg.Id) -> $LEGLOG"

# 5. AUTO-ARM WATCHDOG (detached) -------------------------------------------
$wd = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$WD`"",
        '-LegLog', "`"$LEGLOG`"", '-StallSeconds', "$StallSeconds", '-VerdictFile', "`"$VERDICT`"") `
    -WindowStyle Hidden -PassThru
Say "watchdog armed pid=$($wd.Id) -> $VERDICT (stall=${StallSeconds}s)"
Say "POLL: $VERDICT  (RUNNING|DONE|DEAD)"
