<#
  otr_3d_quick_tests.ps1 -- DECOUPLED 3D smoke (operator-directed 2026-06-12):
  one 30-word FULL-episode test per 0-E 3D engine, EASIEST -> HARDEST, on the
  other_beats (character) slot. Does NOT wait for the 27-leg coverage sweep.

  Boots a FRESH headless ComfyUI on :8000 (current code), then runs the three
  0-E engines in order via the proven coverage-sweep harness (--only), each with
  the capstone gates (playable obs final, byte-identical master audio, hygiene,
  VRAM ceiling). Writes a running digest + per-engine verdict.

  Order: ltx_orbit (no new deps) -> still_parallax (DepthAnythingV2-S) ->
  mesh_stage (hy3d-2mv mesh + Blender headless). humo_1.7B stays the default for
  the non-3D character beats; the 3D engine under test owns other_beats_visual.
#>
$ErrorActionPreference = 'Stop'

$VENV   = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$MAIN   = 'C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py'
$COMFYBASE = 'C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI'
$REPO   = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$USERDIR = 'C:\Users\jeffr\Documents\ComfyUI'
$SRVLOG = 'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log'
$SRVERR = 'C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.err.log'
$DIGEST = Join-Path $REPO 'scripts\otr_3d_quick_digest.md'
$MARKER = Join-Path $REPO 'scripts\.otr_3d_quick_active'

$ENGINES = @('ltx_orbit','still_parallax','mesh_stage')   # easiest -> hardest

function Dig($m){ $line = "{0} | {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Add-Content -Path $DIGEST -Value $line -Encoding utf8; Write-Host $line }

"# OTR 3D quick-test digest (decoupled from the 27-leg sweep)" | Set-Content -Path $DIGEST -Encoding utf8
Dig "START -- engines (easiest first): $($ENGINES -join ', ')"
"started=$((Get-Date).ToString('o'))" | Set-Content -Path $MARKER -Encoding ascii

# 1. Fresh server on current code.
$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if($listen){ foreach($pid in ($listen.OwningProcess | Select-Object -Unique)){ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 3 }
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Documents\ComfyUI\.venv*' } | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$env:HF_HOME = 'C:\ComfyUI-Models\huggingface'
Remove-Item Env:OTR_C7 -ErrorAction SilentlyContinue
$srvArgs = @("`"$MAIN`"", '--port','8000','--cuda-malloc','--user-directory',"`"$USERDIR`"")
$srv = Start-Process -FilePath $VENV -ArgumentList $srvArgs -WorkingDirectory $COMFYBASE -RedirectStandardOutput $SRVLOG -RedirectStandardError $SRVERR -WindowStyle Hidden -PassThru
Dig "server launched pid=$($srv.Id); waiting for health..."
$up=$false
for($i=0;$i -lt 30;$i++){ Start-Sleep -Seconds 10; try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/system_stats' -UseBasicParsing -TimeoutSec 5; if($r.StatusCode -eq 200){ $oi=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/object_info/OTR_VideoDirector' -UseBasicParsing -TimeoutSec 5; if($oi.StatusCode -eq 200){ $up=$true; Dig "server healthy after ~$([int](($i+1)*10))s"; break } } } catch {} }
if(-not $up){ Dig "SERVER DID NOT COME UP -- aborting"; Remove-Item $MARKER -ErrorAction SilentlyContinue; exit 2 }

# 2. Run each 3D engine, easiest first, on the character slot.
$env:PYTHONPATH = $COMFYBASE
foreach($eng in $ENGINES){
  $leg = "other_beats_visual_$eng"
  $log  = Join-Path $REPO ("scripts\otr_3d_quick_{0}.log" -f $eng)
  $elog = Join-Path $REPO ("scripts\otr_3d_quick_{0}.err.log" -f $eng)
  Dig "=== TEST $eng (leg $leg) START ==="
  $t0 = Get-Date
  # Start-Process redirect streams stdout LIVE to the file (unlike '*>' which buffers).
  $proc = Start-Process -FilePath $VENV -ArgumentList @('-u','scripts\otr_coverage_sweep.py','--only',$leg) -WorkingDirectory $REPO -RedirectStandardOutput $log -RedirectStandardError $elog -WindowStyle Hidden -PassThru
  $beat = 0
  while(-not $proc.HasExited){
    Start-Sleep -Seconds 60
    $beat++
    $last = (Select-String -Path $log -Pattern 't=\s*\d+s.*vram_peak=\d+MB' -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
    $now = (Select-String -Path $log -Pattern '=== LEG|QUEUED|vram_peak=\d+MB' -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
    Dig ("   {0} heartbeat #{1}: {2}" -f $eng, $beat, ($(if($last){$last.Trim()}elseif($now){$now.Trim()}else{'(no progress line yet)'})))
  }
  $rc = $proc.ExitCode
  $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes,1)
  $verdict = (Select-String -Path $log -Pattern 'sweep_.* -> (PASS|RC_\d+|SOAK_FAIL|ERROR|PROFILE_INVALID)' | Select-Object -Last 1).Line
  if(-not $verdict){ $verdict = (Select-String -Path $log -Pattern '(COMPLETE:|SOAK_FAIL|ERROR|PROFILE_INVALID)' | Select-Object -Last 1).Line }
  Dig "=== TEST $eng DONE rc=$rc ${mins}min :: $($verdict) ==="
}
Dig "ALL DONE"
Remove-Item $MARKER -ErrorAction SilentlyContinue
