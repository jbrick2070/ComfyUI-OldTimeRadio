<#
.SYNOPSIS
  Walk every LOCAL video lane with gemma-4-E2B-it in BOTH writer slots,
  rebooting the headless server between BOOT GROUPS.

.DESCRIPTION
  WHY THIS EXISTS (operator, 2026-08-26): "test gemma-4-E2B-it on all valid
  local image and video lanes", then "kill and reboot server all you want".
  That second sentence is what makes this script possible -- four of the
  fifteen local video lanes cannot share a server with the other eleven.

  A profile's `launch.env` is a BOOT contract, not a per-leg switch. The
  headless launcher reads OTR_HEADLESS_RESERVE_VRAM_GB and
  OTR_HEADLESS_DISABLE_PINNED from its own process environment and turns them
  into `--reserve-vram` / `--disable-pinned-memory` on the ComfyUI command
  line. Those are start-up flags: nothing can change them on a running server.
  So the lanes group by the boot they need, and each group costs one reboot.

    A  default        11 lanes  no clamp
    B  humo_diet       4 lanes  reserve 2.921 GB + no pinned host memory
    C  ltx_av_diet     1 lane   no pinned host memory, NO reserve
    D  h3              2 lanes  reserve 12 GB + no pinned host memory

  Group D is the only one the ENGINE itself forces apart -- minimax_h3
  declares `compatible_boot_contracts = ("h3",)` and enforces it. B and C are
  separated because their shipping profiles ask for a different boot, not
  because the engine refuses one.

  KILLING IS SELECTIVE, ALWAYS. The server is matched by CommandLine against
  `_otr_headless_model_paths` and killed by PID. A blanket
  `Stop-Process -Name python` would also kill the Claude MCP extension
  pythons and sever the tools driving this run -- that is a documented,
  already-paid-for mistake, not a hypothetical.

  A group whose server fails to come up is SKIPPED, loudly, and the run
  continues to the next group. One bad boot at 3am must not cost the whole
  night. Each leg appends to a receipt so killing this mid-flight still
  leaves a complete record.

.PARAMETER Groups
  Comma-separated group ids to run. Default: A,B,C,D.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\otr_video_lane_soak.ps1
  powershell -ExecutionPolicy Bypass -File scripts\otr_video_lane_soak.ps1 -Groups B,C,D
#>
param(
    [string]$Groups = "A,B,C,D",
    [int]$TimeoutSeconds = 7200
)

$ErrorActionPreference = "Continue"
$REPO = Split-Path -Parent $PSScriptRoot
$VENV = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$RUNNER = Join-Path $REPO "scripts\otr_canonical_api_run.py"
$LAUNCH = Join-Path $REPO "scripts\_otr_soak_server_launch.cmd"
$RECEIPT = Join-Path $REPO "docs\2026-08-26-video-lane-soak-receipt.json"
$E2B = "google/gemma-4-E2B-it (3.0 GB)"

# Bank rotation, so the video sweep also broadens BANK coverage instead of
# re-proving one lane fifteen times. Style is the one authored for that bank.
$BANKS = @(
    @{ bank = "media_archive"; style = "archival_documentary" },
    @{ bank = "original";      style = "sci_fi_radio" },
    @{ bank = "public_domain"; style = "storybook_engraving" },
    @{ bank = "shakespeare";   style = "shakespeare_stage_realism" }
)

# Smallest model first inside each group, so a night that runs short still
# covers the most distinct engines.
$BOOT_GROUPS = [ordered]@{
    A = @{
        env    = @{}
        lanes  = @(
            "otr_w45_ltx_8gb", "otr_w45_wan_ti2v", "otr_w45_fastwan",
            "otr_w45_mesh_stage", "otr_ghost_signal_v3_haunted",
            "otr_w45_ltx_video", "otr_ltx25_high_video", "otr_w45_wan_i2v"
        )
    }
    B = @{
        env    = @{ OTR_HEADLESS_RESERVE_VRAM_GB = "2.921"; OTR_HEADLESS_DISABLE_PINNED = "1" }
        lanes  = @(
            "otr_w45_humo_1_7b", "otr_w45_humo_1_7b_169",
            "otr_w45_humo", "otr_w45_humo_14b_169"
        )
    }
    C = @{
        env    = @{ OTR_HEADLESS_DISABLE_PINNED = "1" }
        lanes  = @("otr_w45_ltx_audio_in")
    }
    D = @{
        env    = @{ OTR_HEADLESS_RESERVE_VRAM_GB = "12"; OTR_HEADLESS_DISABLE_PINNED = "1" }
        lanes  = @("otr_w45_minimax_h3_video", "otr_w45_minimax_h3_audio_in")
    }
}

$script:Results = @()

function Write-Receipt {
    $payload = [ordered]@{
        generated_at     = (Get-Date).ToString("s")
        model_under_test = $E2B
        slots            = "creative AND technical (both)"
        legs_run         = $script:Results.Count
        pass             = @($script:Results | Where-Object { $_.ok }).Count
        fail             = @($script:Results | Where-Object { -not $_.ok }).Count
        results          = $script:Results
    }
    # UTF-8 WITHOUT BOM, and it has to be written this way deliberately.
    # `Out-File -Encoding utf8` on Windows PowerShell 5.1 emits a BOM, which
    # makes the receipt unreadable by the very idiom its sibling receipts are
    # read with: `json.load(open(p, encoding="utf-8"))` raises
    # "Unexpected UTF-8 BOM (decode using utf-8-sig)". Verified against the
    # first receipt this script wrote. Project rule is UTF-8, no BOM, always.
    $json = $payload | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText($RECEIPT, $json, (New-Object System.Text.UTF8Encoding($false)))
}

function Stop-OtrServer {
    <#
      SWEEP THE WHOLE OTR SIDE, NOT JUST THE SERVER PROCESS.

      Operator, 2026-08-26: "killing all before rebooting can help a clean
      boot." The mechanism is real -- a finished render leaves the server
      RESIDENT holding ~9-10 GB, and a timed-out stage leaves an ORPHAN WORKER
      draining in the background (Gemma generation is not cancellable
      mid-token) still holding a CUDA context. Boot on top of that and the new
      server competes for VRAM with a ghost nobody is watching. So this kills
      the server AND any leg runner / sweep harness / orphaned OTR python,
      then waits for the GPU to fall back to the desktop baseline -- the only
      real proof the contexts went away.

      WHAT IT STILL WILL NOT DO IS A BLANKET `Stop-Process -Name python`.
      That also kills the Claude MCP extension pythons (Desktop Commander,
      windows-mcp, comfy-mcp) -- the tools driving this very run -- and
      severing them mid-sweep ends the night. Documented, already paid for
      once. Matching on CommandLine buys the clean boot without cutting our
      own wire, so the Claude/Codex tool paths are excluded explicitly.
    #>
    $mine = '_otr_headless_model_paths|ComfyUI\\main\.py|otr_canonical_api_run|otr_bank_engine_sweep|otr_llm_image_upscale_sweep|otr_gpu_soak'
    $keep = 'Claude Extensions|\.comfy-mcp|\.codex|windows-mcp|comfy-mcp'
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
             Where-Object { $_.CommandLine -and $_.CommandLine -match $mine -and $_.CommandLine -notmatch $keep }
    foreach ($p in $procs) {
        Write-Host "[videosoak]   killing OTR PID $($p.ProcessId)"
        try { Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction Stop } catch {}
    }
    # Anything still holding :8000 the CommandLine match missed -- a
    # half-booted server reports a BLANK .Path, so the port is the backstop.
    foreach ($c in (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) {
        Write-Host "[videosoak]   killing :8000 holder PID $($c.OwningProcess)"
        try { Stop-Process -Id $c.OwningProcess -Force -Confirm:$false -ErrorAction Stop } catch {}
    }
    # Wait for the port to actually clear; a listening socket outlives the
    # process briefly and a fresh boot would then fail to bind.
    $portClear = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (-not (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) {
            $portClear = $true; break
        }
    }
    if (-not $portClear) { Write-Host "[videosoak]   WARNING: port 8000 still listening after kill" }
    # VRAM is the real proof. Desktop baseline is ~1.5-2 GB; much above that
    # after the kill means a context outlived its process.
    for ($i = 0; $i -lt 20; $i++) {
        $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ([int]$used -lt 3000) {
            Write-Host "[videosoak]   GPU back to baseline ($used MiB)"
            return $portClear
        }
        Start-Sleep -Seconds 3
    }
    Write-Host "[videosoak]   WARNING: GPU still holding memory after kill; booting anyway"
    return $portClear
}

function Start-OtrServer {
    param([hashtable]$BootEnv, [string]$GroupId)

    # These are BOOT flags. Clear both first so a previous group's clamp can
    # never leak into this boot and silently change what is being measured.
    Remove-Item Env:OTR_HEADLESS_RESERVE_VRAM_GB -ErrorAction SilentlyContinue
    Remove-Item Env:OTR_HEADLESS_DISABLE_PINNED  -ErrorAction SilentlyContinue
    foreach ($k in $BootEnv.Keys) {
        Set-Item -Path "Env:$k" -Value $BootEnv[$k]
        Write-Host "[videosoak]   boot env $k=$($BootEnv[$k])"
    }
    if ($BootEnv.Count -eq 0) { Write-Host "[videosoak]   boot env: (default, no clamp)" }

    $log = Join-Path $REPO ("tmp\otr_videosoak_server_{0}.log" -f $GroupId)
    Start-Process -FilePath $LAUNCH -ArgumentList "`"$log`"" -WindowStyle Hidden
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 3
        $listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if ($listen) {
            Start-Sleep -Seconds 5   # let node registration finish
            Write-Host "[videosoak]   server UP (group $GroupId), log $log"
            return $true
        }
    }
    Write-Host "[videosoak]   SERVER DID NOT COME UP for group $GroupId -- see $log"
    return $false
}

function Wait-QueueIdle {
    for ($i = 0; $i -lt 900; $i++) {
        try {
            $q = Invoke-RestMethod -Uri "http://127.0.0.1:8000/queue" -TimeoutSec 10
            $busy = @($q.queue_running).Count + @($q.queue_pending).Count
            if ($busy -eq 0) { return $true }
        } catch { return $false }
        Start-Sleep -Seconds 15
    }
    return $false
}

$wanted = @($Groups -split "," | ForEach-Object { $_.Trim().ToUpper() } |
            Where-Object { $_ })
$legIndex = 0

# A SWEEP THAT SELECTS NOTHING MUST NOT REPORT SUCCESS.
#
# This guard exists because the first run of this script did exactly that:
# "[videosoak] DONE 0/0 passed", exit 0, GPU idle all night, and the log said
# nothing was wrong. The cause was a PowerShell trap worth naming -- VARIABLE
# NAMES ARE CASE-INSENSITIVE, so the boot-group table `$GROUPS` and the
# `-Groups` parameter were ONE variable. The table overwrote the parameter,
# `$Groups.Split(",")` then failed on an OrderedDictionary, `$wanted` came out
# null, and `-notcontains` was therefore true for every group, skipping all of
# them. The table is `$BOOT_GROUPS` now, but a rename alone would leave the
# silent-no-op shape intact for the next mistake to fall into, so the run
# refuses instead.
$selected = @($BOOT_GROUPS.Keys | Where-Object { $wanted -contains $_ })
if ($selected.Count -eq 0) {
    Write-Host "[videosoak] REFUSING: -Groups '$Groups' selected no boot group."
    Write-Host "[videosoak] known groups: $(($BOOT_GROUPS.Keys) -join ', ')"
    Write-Host "[videosoak] parsed filter: '$($wanted -join "', '")'"
    exit 2
}
Write-Host "[videosoak] groups selected: $($selected -join ', ') ($(($selected | ForEach-Object { $BOOT_GROUPS[$_].lanes.Count } | Measure-Object -Sum).Sum) lane(s) total)"

foreach ($gid in $BOOT_GROUPS.Keys) {
    if ($wanted -notcontains $gid) { continue }
    $g = $BOOT_GROUPS[$gid]
    Write-Host "[videosoak] === BOOT GROUP $gid ($($g.lanes.Count) lane(s)) ==="
    Stop-OtrServer | Out-Null
    if (-not (Start-OtrServer -BootEnv $g.env -GroupId $gid)) {
        Write-Host "[videosoak] group $gid SKIPPED (no server). Continuing."
        foreach ($lane in $g.lanes) {
            $legIndex++
            $script:Results += [ordered]@{
                leg = $legIndex; group = $gid; profile = $lane
                bank = ""; ok = $false; minutes = 0
                note = "skipped: server did not come up for boot group $gid"
            }
        }
        Write-Receipt
        continue
    }

    foreach ($lane in $g.lanes) {
        $legIndex++
        $slot = $BANKS[($legIndex - 1) % $BANKS.Count]
        $label = "VIDEOSOAK{0:d2} group=$gid profile=$lane bank=$($slot.bank)" -f $legIndex
        Write-Host "[videosoak] leg $legIndex START $label"
        Wait-QueueIdle | Out-Null
        $started = Get-Date
        $legLog = Join-Path $REPO ("tmp\otr_videosoak_leg{0:d2}.log" -f $legIndex)
        # `*> $legLog` writes UTF-16 on PowerShell 5.1, which makes every leg
        # log un-greppable by ordinary tooling (a plain `grep RESULT` finds
        # nothing on a leg that plainly succeeded) and breaks the project's
        # UTF-8-no-BOM rule. Capture the streams and write them ourselves.
        $legOut = & $VENV $RUNNER --act-count 1 --source-bank $slot.bank `
            --visual-style $slot.style --profile $lane `
            --creative-model $E2B --technical-model $E2B `
            --timeout $TimeoutSeconds 2>&1 | Out-String
        $rc = $LASTEXITCODE
        [System.IO.File]::WriteAllText($legLog, $legOut, (New-Object System.Text.UTF8Encoding($false)))
        $elapsed = ((Get-Date) - $started).TotalMinutes
        # Judge the captured text, not a re-read of the file: one less place
        # for an encoding round-trip to change the answer.
        $ok = $legOut -match "RESULT SUCCESS"
        Write-Host ("[videosoak] leg {0} {1} {2:N1} min rc={3}" -f $legIndex, $(if ($ok) { "PASS" } else { "FAIL" }), $elapsed, $rc)
        $script:Results += [ordered]@{
            leg = $legIndex; group = $gid; profile = $lane
            bank = $slot.bank; visual_style = $slot.style
            creative_model = $E2B; technical_model = $E2B
            ok = $ok; rc = $rc; minutes = [math]::Round($elapsed, 1)
            log = $legLog
        }
        Write-Receipt
    }
}

$passed = @($script:Results | Where-Object { $_.ok }).Count
Write-Host "[videosoak] DONE $passed/$($script:Results.Count) passed. Receipt: $RECEIPT"
