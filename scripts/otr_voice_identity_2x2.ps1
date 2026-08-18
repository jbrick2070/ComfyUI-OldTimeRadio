# The 2x2 voice-identity proof (PBUG-20260817-09, QA-8) -- a permanent instrument.
#
#   seed axis:     char_v1 (fixed)      vs  line_v1 (the shipped pre-fix seed)
#   emotion axis:  alpha 0.4 + cap 0.4  vs  alpha 1.0 + no cap (the pre-fix blend)
#
# THE SPEC SAID "alpha 1.0 vs 0.4" AND THAT AXIS WOULD HAVE PROVED NOTHING.
# On a neutral line -- calm=1.0, the shape of the very beat the operator
# reported -- alpha 1.0 is rescaled to 0.4 by the emotion ceiling and alpha 0.4
# lands on 0.4 by the vendor's own scaling. Identical effective mass, so both
# "alpha arms" were one arm and no arm reproduced pre-fix behaviour. The emotion
# axis therefore varies the CEILING too (OTR_INDEXTTS2_EMO_MASS_CAP=8 means no
# ceiling), which is the only way the fix can be attributed between its two
# causes. A structural review gate caught this; the first cut had it wrong.
#
# BOTH AXES ARE ENVIRONMENT, so every arm gets its own reset and its own fresh
# server boot -- QA-8 requires it, and a knob read at boot cannot be changed
# under a running server anyway. Every arm renders through the CANONICAL
# workflow via the canonical runner and PUBLISHES TO otr/obs/: publication is
# never reduced, gated or relocated, because it is how the operator reads
# success.
#
# Read the arms afterwards with:
#   python scripts/otr_voice_identity_acceptance.py --log <arm>.pobs.log `
#          --expect-policy char_v1 --expect-alpha 0.4 --expect-mass-cap 0.4
# passing each arm its OWN contract -- a control arm booted without a ceiling
# is supposed to exceed 0.4, and reporting that as a failure teaches the reader
# to ignore the arms that matter.
#
# All quoting and variables live inside this script on purpose: anything
# needing nested quotes or $-interpolation becomes a launcher script rather
# than an inline command.

param(
    [string]$OutDir = "",
    [int]$ActCount = 2,
    [int]$LegTimeoutSeconds = 3600
)

$ErrorActionPreference = 'Continue'
$REPO   = Split-Path -Parent $PSScriptRoot
$PY     = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$LAUNCH = Join-Path $REPO 'scripts\_otr_soak_server_launch.cmd'
$RESET  = Join-Path $REPO 'scripts\otr_reset_gpu.ps1'
$RUNNER = Join-Path $REPO 'scripts\otr_canonical_api_run.py'
$OBS    = 'C:\Users\jeffr\Documents\ComfyUI\output\otr\obs'

if (-not $OutDir) { $OutDir = Join-Path $REPO 'tmp\voice_2x2' }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

#   A = the whole fix        B = seed fix only      C = emotion fix only
#   D = the shipped defect, reproduced end to end
$ARMS = @(
  @{ Name = 'a_fix_both';       Seed = '1'; Alpha = '0.4'; Cap = '0.4' },
  @{ Name = 'b_fix_seed_only';  Seed = '1'; Alpha = '1.0'; Cap = '8'   },
  @{ Name = 'c_fix_blend_only'; Seed = '0'; Alpha = '0.4'; Cap = '0.4' },
  @{ Name = 'd_prefix_control'; Seed = '0'; Alpha = '1.0'; Cap = '8'   }
)

function Say([string]$m) {
  Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $m)
}

function Wait-ForServer([int]$timeoutSeconds) {
  $deadline = (Get-Date).AddSeconds($timeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction Stop) { return $true }
    } catch { }
    Start-Sleep -Seconds 2
  }
  return $false
}

$obsBefore = (Get-ChildItem $OBS -File -ErrorAction SilentlyContinue).Count
Say ("obs artifacts BEFORE the sweep: {0}" -f $obsBefore)

foreach ($arm in $ARMS) {
  $name   = $arm.Name
  $legLog = Join-Path $OutDir ("{0}.leg.log" -f $name)
  $srvLog = Join-Path $OutDir ("{0}.server.log" -f $name)

  Say ("==== ARM {0}: seed={1} alpha={2} mass_cap={3}" -f $name, $arm.Seed, $arm.Alpha, $arm.Cap)

  # --- reset: never a blanket python kill (it would sever the MCP pythons) ---
  Say 'reset: selective kill + port + VRAM baseline'
  $resetOut = & powershell -NoProfile -ExecutionPolicy Bypass -File $RESET *>&1
  $resetCode = $LASTEXITCODE
  $resetOut | Select-Object -Last 6 | ForEach-Object { Say ("  reset| {0}" -f $_) }

  # THE PORT IS THE HARD GATE. If something still owns 8000 the leg would submit
  # into a server booted under a DIFFERENT arm's environment -- an arm silently
  # measuring the wrong thing is worse than a missing arm.
  if (-not ($resetOut | Select-String -Quiet 'port 8000 : free')) {
    Say 'RESET FAILED: port 8000 is still held -- skipping this arm rather than'
    Say 'submitting into a server somebody else booted.'
    continue
  }
  if ($resetCode -ne 0) {
    # Port free and the OTR pythons gone, so whatever remains on the card is not
    # ours -- on a logged-in desktop it is Chrome, Edge WebView, Snagit and the
    # shell. Name the number and carry on; the ceiling is tuned for a quiet box.
    $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader)
    Say ("reset reported incomplete (exit {0}) but port 8000 is FREE -- residual {1} is the logged-in desktop, not OTR. Proceeding." -f $resetCode, $vram)
  }

  # The arm's environment, exported BEFORE the boot so the server log and the
  # command line stay auditable (the launcher consumes no hidden hook files).
  $env:OTR_VOICE_CHARACTER_SEED   = $arm.Seed
  $env:OTR_INDEXTTS2_EMO_ALPHA    = $arm.Alpha
  $env:OTR_INDEXTTS2_EMO_MASS_CAP = $arm.Cap

  Say 'booting headless server'
  Start-Process -FilePath $LAUNCH -ArgumentList ('"{0}"' -f $srvLog) -WindowStyle Hidden
  if (-not (Wait-ForServer 180)) {
    Say 'SERVER DID NOT COME UP -- reading the boot log and skipping this arm'
    if (Test-Path $srvLog) { Get-Content $srvLog -Tail 25 | ForEach-Object { Say ("  boot| {0}" -f $_) } }
    continue
  }
  Say 'server is listening on :8000'

  # --- the leg: the CANONICAL workflow, never a saved or generated graph ------
  Say 'submitting canonical episode'
  $started = Get-Date
  & $PY $RUNNER --profile none --act-count $ActCount --timeout $LegTimeoutSeconds *> $legLog
  $code = $LASTEXITCODE
  $mins = [Math]::Round(((Get-Date) - $started).TotalMinutes, 1)
  Say ("leg finished exit={0} in {1} min" -f $code, $mins)

  # --- the receipt that matters: did it reach obs? ---------------------------
  $obsNow = (Get-ChildItem $OBS -File -ErrorAction SilentlyContinue).Count
  Say ("obs artifacts now: {0}" -f $obsNow)
  if ($obsNow -le $obsBefore) {
    Say 'WARNING: nothing new in otr/obs/ for this arm -- treat the arm as FAILED and read the leg log'
  }
  $obsBefore = $obsNow

  # The voice receipts are written by the SERVER process, not the runner, so
  # lift them out next to the leg log for the acceptance reader.
  if (Test-Path $srvLog) {
    Select-String -Path $srvLog -Pattern 'voice P-OBS' |
      ForEach-Object { $_.Line } |
      Set-Content -Path (Join-Path $OutDir ("{0}.pobs.log" -f $name)) -Encoding utf8
  }
}

Say '==== all arms done; final teardown reset'
& powershell -NoProfile -ExecutionPolicy Bypass -File $RESET *>&1 |
  Select-Object -Last 4 | ForEach-Object { Say ("  reset| {0}" -f $_) }
Say ("obs artifacts AFTER the sweep: {0}" -f (Get-ChildItem $OBS -File -ErrorAction SilentlyContinue).Count)
Say ("arm logs: {0}" -f $OutDir)
