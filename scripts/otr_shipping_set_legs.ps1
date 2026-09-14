# Run the shipped graphs through a live ComfyUI, one act each, cheapest first.
#
# WHY THIS EXISTS. The files in workflows/variants/ are the product now --
# otr_<arch>_<tier>.json, one per machine class and episode kind. A --check that
# proves they re-derive from the canonical is not the same as an episode
# landing in otr/obs/, and the operator's bar is the second thing. This walks
# a named list of them through scripts/otr_canonical_api_run.py against a
# server that is already up, and writes one log per leg so the receipt facts
# can be read back from the leg log rather than from memory.
#
# ONE ACT, ON PURPOSE (operator, 2026-09-13): the shipped graphs carry three
# acts and three characters; a smoke of every graph at one act finishes in a
# night, and --act-count overrides only that widget.
#
# NEVER --title: the harness label becomes the on-screen title card.
#
# usage:  powershell -File scripts\otr_shipping_set_legs.ps1 -Url http://127.0.0.1:8000
#         -Graphs otr_16gb_low,otr_16gb_still      (default: the 8 GB + 16 GB NVIDIA set)
param(
    [string]$Url = "http://127.0.0.1:8000",
    [string[]]$Graphs = @(
        "otr_8gb_low", "otr_16gb_low",
        "otr_8gb_still", "otr_16gb_still",
        "otr_8gb_animatediff", "otr_16gb_animatediff",
        "otr_8gb_video", "otr_16gb_video",
        "otr_16gb_mime",
        "otr_16gb_foley"
    ),
    [string]$ActCount = "1",
    # Pin every leg to ONE source bank instead of letting each roll. Empty (the
    # default) leaves the graph's own saved value alone, which for all 17
    # shipped graphs is 'roll (any eligible bank)'.
    #
    # WHY IT IS HERE. On 2026-09-14 the operator asked to see a My Story
    # episode and the only way to guarantee one was to bypass the roll -- six
    # legs rolling uniformly over six banks is not a test of any one of them.
    # The runner has always had --source-bank; this harness simply never
    # passed it, so proving a single lane meant abandoning the harness.
    [string]$SourceBank = "",
    # The runner's default observation window is 5400s and a one-act LTX 2.5
    # leg can brush it; a leg that times out here is still rendering, and the
    # next leg would queue behind it and time out too. Watch longer, and
    # clear the server before moving on if a leg did not reach SUCCESS.
    [int]$TimeoutSec = 9000,
    # Where the server publishes finished episodes on this box.
    [string]$ObsDir = "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs",
    # The ComfyUI venv python that runs the API runner. The defaults above
    # and here are the 5080's; the 4060 passes its own.
    [string]$Python = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = $Python
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# A parent that launches this file as `powershell -File ... -Graphs a,b,c` hands
# the whole comma-joined list over as ONE string (only an in-process call
# binds it as an array), so split whatever arrived. The first night run
# skipped all six graphs as "no such variant" for exactly this reason.
$Graphs = @($Graphs | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logdir = Join-Path $root "otr\legs\shipping_set_$stamp"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$summary = Join-Path $logdir "SUMMARY.txt"
# One writer for the summary: Tee-Object appends UTF-16 under PowerShell 5.1,
# which turned the first summary into spaced-out characters after a UTF-8 header.
function Note($m) { $m | Out-File -FilePath $summary -Append -Encoding utf8; Write-Output $m }
$bankNote = if ($SourceBank) { "source_bank=$SourceBank (PINNED)" } else { "source_bank=per-graph (roll)" }
"shipping-set legs  $stamp  url=$Url  act_count=$ActCount  $bankNote" | Out-File -FilePath $summary -Encoding utf8
Note ("graphs: " + ($Graphs -join ", "))

# Counted by POSITION, not matched by NAME (see the LAST LEG note below): a
# caller may legitimately list the same graph twice, and `$g -eq $Graphs[-1]`
# would call the FIRST occurrence the last leg and skip the queue cleanup the
# second occurrence still needs.
#
# And counted over the legs that will ACTUALLY RUN, not the names requested: a
# graph whose variant file is missing is skipped below, so if the LAST name in
# the list does not exist, no leg would ever match "last" and the real final
# render would be /interrupt'd -- the exact 2.5-hour loss the LAST LEG rule
# exists to stop.
$totalLegs = @($Graphs | Where-Object { Test-Path (Join-Path $root "workflows\variants\$_.json") }).Count
$legIndex = 0
foreach ($g in $Graphs) {
    $wf = Join-Path $root "workflows\variants\$g.json"
    if (-not (Test-Path $wf)) {
        Note "$g  SKIP  no such variant"
        continue
    }
    # AFTER the skip, so the index counts legs that run, not names requested.
    $legIndex++
    $log = Join-Path $logdir "$g.log"
    $t0 = Get-Date
    Note "$g  START  $($t0.ToString('HH:mm:ss'))"
    # Built as an ARRAY, not an inline argument list, so an unset -SourceBank
    # contributes nothing at all. Passing an empty string through as
    # `--source-bank ""` would pin the bank to the empty string, which is a
    # different and much worse thing than not pinning it.
    $runArgs = @(
        "scripts\otr_canonical_api_run.py",
        "--workflow", $wf,
        "--act-count", $ActCount,
        "--comfyui-url", $Url,
        "--timeout", $TimeoutSec
    )
    if ($SourceBank) { $runArgs += @("--source-bank", $SourceBank) }
    & $py @runArgs 2>&1 | Out-File -FilePath $log -Encoding utf8
    $rc = $LASTEXITCODE
    $mins = [Math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $result = (Select-String -Path $log -Pattern "RESULT (SUCCESS|FAIL\w*|TIMEOUT|ERROR)" | Select-Object -Last 1).Matches.Value
    # otr/obs is the success signal: the server publishes there, so count the
    # episodes that landed since this leg started (the runner's own log never
    # carries the server's "obs_publish OK" line).
    if (Test-Path $ObsDir) {
        $landed = @(Get-ChildItem $ObsDir -Filter *.mp4 | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime)
        $obsNote = "obs=$($landed.Count)  " + (($landed | ForEach-Object { $_.Name }) -join ", ")
    } else {
        # A remote server publishes on its own disk; pull that folder to read the signal.
        $obsNote = "obs=n/a (remote; pull the server's otr/obs)"
    }
    if (-not $result) { $result = "NO-RESULT-LINE" }
    Note "$g  $result  rc=$rc  ${mins}min  $obsNote"
    if ($result -ne "RESULT SUCCESS") {
        # Leave the server empty for the NEXT leg: a render that outlived the
        # observation window, or one wedged mid-graph, must not become the next
        # leg's queue-mate.
        #
        # NEVER ON THE LAST LEG, and this cost a real render. On 2026-09-14
        # otr_16gb_video (LTX 2.5, Q3_K_M) hit the 150-minute window as the
        # FINAL graph. The runner said so in as many words -- "BUT THE RENDER IS
        # STILL ALIVE: the server reports 1 running / 0 pending ... the episode
        # should still publish to otr/obs on its own" -- and this block then
        # POSTed /interrupt and killed it. Two and a half hours of render, no
        # episode, because the cleanup was protecting a next leg that did not
        # exist. A timeout means "I stopped watching", not "it is wedged", and
        # with nothing queued behind it there is nothing to protect.
        $isLastLeg = ($legIndex -eq $totalLegs)
        if ($isLastLeg) {
            Note "$g  LAST LEG -- leaving the render alone. It may still be running and may still publish to $ObsDir on its own; check there before calling this a failure. Re-run this graph with a larger -TimeoutSec (or the runner's --timeout 0) to watch it to a terminal result."
        } else {
            try {
                $q = Invoke-RestMethod -Uri "$Url/queue" -TimeoutSec 10
                $ids = @($q.queue_pending | ForEach-Object { $_[1] })
                if ($ids.Count) { Invoke-RestMethod -Uri "$Url/queue" -Method Post -ContentType "application/json" -Body (@{ delete = $ids } | ConvertTo-Json -Compress) | Out-Null }
                if ($q.queue_running.Count) { Invoke-RestMethod -Uri "$Url/interrupt" -Method Post | Out-Null; Start-Sleep -Seconds 20 }
                Note "$g  cleared the server (was running=$($q.queue_running.Count) pending=$($ids.Count))"
            } catch { Note "$g  could not clear the server: $($_.Exception.Message)" }
        }
    }
}
Note "DONE  $(Get-Date -Format 'HH:mm:ss')"
