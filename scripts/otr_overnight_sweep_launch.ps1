<#
  otr_overnight_sweep_launch.ps1 -- ONE-SHOT overnight coverage-sweep launcher.

  Boots a FRESH headless ComfyUI on :8000 (loads whatever code is current in the
  live repo via the custom_nodes junction = "updated code end to end"), waits for
  health, then launches the full 27-leg dropdown coverage sweep DETACHED on the
  humo_1.7B default. The otr-sweep-monitor scheduled task tracks per-leg verdicts,
  writes the morning digest, and creates scripts\_otr_0e_gpu_go.txt on a clean
  27/27 PASS.

  Clean-env boot: NO OTR_C7 (so the C7 seed pins stay off), NO OpenRouter, NO
  cast/style seed override -- the sweep applies its own profile (per-leg seed 42).
  Normal VRAM (no --highvram, per BUG-291). 100% local.

  Run it via the otr-overnight-sweep scheduled task ("Run now" at bedtime) or
  directly:  powershell -ExecutionPolicy Bypass -File <this file>
#>
$ErrorActionPreference = 'Stop'

$VENV   = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$MAIN   = 'C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py'
$COMFYBASE = 'C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI'
$REPO   = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$USERDIR = 'C:\Users\jeffr\Documents\ComfyUI'
$SRVLOG = 'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log'
$SRVERR = 'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.err.log'
$SWEEPLOG = Join-Path $REPO 'scripts\coverage_sweep_latest.log'
$SWEEPERR = Join-Path $REPO 'scripts\coverage_sweep_latest.err.log'
$MARKER  = Join-Path $REPO 'scripts\.sweep_run_active'
$SUMMARY = Join-Path $REPO 'scripts\coverage_sweep_summary.json'

function Log($m){ Write-Host ("[overnight {0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

# 0. Sanity
if(-not (Test-Path $VENV)){ throw "venv python not found: $VENV" }
if(-not (Test-Path $MAIN)){ throw "ComfyUI main.py not found: $MAIN (Desktop install path may have changed -- re-locate comfyui_version.py)" }

# 1. Kill any stale ComfyUI on :8000 so we boot fresh on current code.
$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if($listen){ foreach($pid in ($listen.OwningProcess | Select-Object -Unique)){ Log "killing stale :8000 pid=$pid"; Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } ; Start-Sleep -Seconds 3 }
# also clear any leftover repo-venv python (sweep / server) from a prior run
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Documents\ComfyUI\.venv*' } | ForEach-Object { Log "killing leftover venv python pid=$($_.Id)"; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

# 2. Boot fresh headless server (clean env, normal VRAM, cuda-malloc).
$env:HF_HOME = 'C:\ComfyUI-Models\huggingface'
# R3 A2a: soak escape hatch -- a fabricated news key_term degrades (raw
# news_seed) instead of HALTING the leg. Production leaves this unset so the
# graph widget default (news_briefs_required=True, fail-closed) governs.
$env:OTR_NEWS_BRIEFS_REQUIRED = '0'
# R2 + 0-E coverage: make the engine enable-set EXPLICIT in this launcher so the
# nightly never depends on ambient env. The 2026-06-13 soak floored humo_1.7B
# (58x) / humo (10x) to still_kenburns with reason=gated_by_flag because
# OTR_ENABLE_HUMO was unset at this direct main.py boot (the canonical
# _otr_soak_server_launch.cmd sets it; this ps1 bypassed it). That is NOT an OOM
# or a forward/wrapper bug -- the HuMo in-process VRAM path (CS-4/BUG-291) is left
# untouched. Production ship-defaults still gate HuMo OFF; this only makes the
# engine selectable so the soak permutation legs actually exercise it.
$env:OTR_ENABLE_HUMO = '1'
$env:OTR_ENABLE_LTX_ORBIT = '1'
$env:OTR_ENABLE_STILL_PARALLAX = '1'
$env:OTR_ENABLE_MESH_STAGE = '1'
Remove-Item Env:OTR_C7 -ErrorAction SilentlyContinue
Remove-Item Env:OTR_ENABLE_OPENROUTER -ErrorAction SilentlyContinue
Remove-Item Env:OTR_CAST_SEED -ErrorAction SilentlyContinue
Remove-Item Env:OTR_STYLE_SEED -ErrorAction SilentlyContinue
$srvArgs = @("`"$MAIN`"", '--port','8000','--cuda-malloc','--user-directory',"`"$USERDIR`"")
$srv = Start-Process -FilePath $VENV -ArgumentList $srvArgs -WorkingDirectory $COMFYBASE -RedirectStandardOutput $SRVLOG -RedirectStandardError $SRVERR -WindowStyle Hidden -PassThru
Log "server launched pid=$($srv.Id); waiting for health..."

# 3. Wait for /system_stats + the OTR node to confirm the live code loaded.
$up = $false
for($i=0; $i -lt 30; $i++){
  Start-Sleep -Seconds 10
  try {
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/system_stats' -UseBasicParsing -TimeoutSec 5
    if($r.StatusCode -eq 200){
      $oi = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/object_info/OTR_VideoDirector' -UseBasicParsing -TimeoutSec 5
      if($oi.StatusCode -eq 200){ $up = $true; Log "server healthy after ~$([int](($i+1)*10))s; OTR nodes loaded"; break }
    }
  } catch { }
}
if(-not $up){ Log "SERVER DID NOT COME UP in 300s -- aborting. See $SRVERR"; if(Test-Path $SRVERR){ Get-Content $SRVERR -Tail 15 }; exit 2 }

# 4. Launch the full 27-leg sweep DETACHED.
$env:PYTHONPATH = $COMFYBASE
$swArgs = @('-u','scripts\otr_coverage_sweep.py')
$sw = Start-Process -FilePath $VENV -ArgumentList $swArgs -WorkingDirectory $REPO -RedirectStandardOutput $SWEEPLOG -RedirectStandardError $SWEEPERR -WindowStyle Hidden -PassThru
$started = (Get-Date).ToString('o')
"server_pid=$($srv.Id)`nsweep_pid=$($sw.Id)`nstarted=$started" | Set-Content -Path $MARKER -Encoding ascii
Log "SWEEP LAUNCHED pid=$($sw.Id); log=$SWEEPLOG"
Log "marker written: $MARKER"
Log "DONE -- the otr-sweep-monitor task will track verdicts + release 0-E Phase B on a clean 27/27."
