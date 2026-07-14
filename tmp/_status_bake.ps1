# Bakeoff status across every sweep: bake420 (10 banks) + bake420b (codex re-leg)
# + bake720. One line per finished leg, plus the live leg's elapsed time.
param([int]$Words = 420)

$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

# GROUND TRUTH: a leg is green only if it PUBLISHED an episode. RESULT SUCCESS is
# not proof -- OTR_MasterAudioMux used to swallow its own fail-closed errors and
# return an empty path, so a 25-minute render that published nothing was recorded
# green (live 2026-07-14, scifi_codex 5ab3884b). `obs_publish OK` in that leg's own
# server log is the only thing that cannot lie.
function Test-Published([string]$bank, [int]$words) {
    $legLog = Join-Path $repo ("tmp\leg_{0}_{1}w.log" -f $bank, $words)
    if (-not (Test-Path -LiteralPath $legLog)) { return $false }
    $portHit = Select-String -Path $legLog -Pattern "port=(\d+)" | Select-Object -Last 1
    if (-not $portHit) { return $false }
    $srv = Join-Path $repo ("tmp\otr_headless_{0}.log" -f $portHit.Matches[0].Groups[1].Value)
    if (-not (Test-Path -LiteralPath $srv)) { return $false }
    return [bool](Select-String -Path $srv -Pattern "obs_publish OK" -Quiet)
}

function Show-Sweep([string]$name, [int]$w) {
    $sum = Join-Path $repo ("tmp\{0}_{1}w_summary.txt" -f $name, $w)
    if (-not (Test-Path -LiteralPath $sum)) { return }
    Write-Host ""
    Write-Host ("--- {0} ({1}w) ---" -f $name, $w)
    $lines = Get-Content -LiteralPath $sum

    foreach ($l in ($lines | Where-Object { $_ -match "END" })) {
        if ($l -match "END\s+(\S+)\s+\d+w\s+::\s+EXIT=(\d+)\s+::.*RESULT (\w+).*::\s+(\d+)s") {
            $bank = $Matches[1]
            $ok = ($Matches[3] -eq "SUCCESS")

            # RESULT SUCCESS IS NOT PROOF. OTR_MasterAudioMux used to swallow its own
            # fail-closed errors and return an empty path, so the graph "completed",
            # ComfyUI logged "Prompt executed", the harness recorded SUCCESS -- and no
            # episode existed (live 2026-07-14, scifi_codex re-leg 5ab3884b). The only
            # ground truth is `obs_publish OK` in the leg's own server log. A SUCCESS
            # with no publish is a PHANTOM and must never be counted green.
            $secs = $Matches[4]
            if ($ok -and -not (Test-Published $bank $w)) {
                "{0} {1,-22} {2,6}s  {3}" -f "PHNT", $bank, $secs, `
                    "<- RESULT SUCCESS but NO obs_publish -- NO EPISODE. Re-leg required."
                continue
            }

            $mark = if ($ok) { "OK  " } else { "FAIL" }
            # A red leg that a later re-leg turned green is HISTORY, not a problem.
            # Say so, or a stale red trains the eye to ignore reds that are real.
            $note = ""
            if (-not $ok) {
                # A re-leg supersedes the original FAIL only if it actually PUBLISHED.
                # A phantom re-leg (RESULT SUCCESS, no obs_publish) supersedes nothing --
                # counting it would launder a dead render into a green.
                $superseded = $false
                foreach ($suffix in @("b", "c", "d")) {
                    $releg = Join-Path $repo ("tmp\{0}{1}_{2}w_summary.txt" -f $name, $suffix, $w)
                    if (-not (Test-Path -LiteralPath $releg)) { continue }
                    $hit = Get-Content -LiteralPath $releg |
                           Where-Object { $_ -match ("END\s+" + [regex]::Escape($bank) + "\s.*RESULT SUCCESS") }
                    if ($hit -and (Test-Published $bank $w)) { $superseded = $true }
                }
                if ($superseded) { $mark = "OK  "; $note = "  (re-leg green + asset verified)" }
                else { $note = "  <- re-leg pending" }
            }
            # $secs, NOT $Matches[4] -- the -match inside the supersede loop above
            # clobbers $Matches, and the leg time silently prints blank.
            "{0} {1,-22} {2,6}s{3}" -f $mark, $bank, $secs, $note
        } else { $l }
    }

    $started = $lines | Where-Object { $_ -match "START" } | Select-Object -Last 1
    $ended   = $lines | Where-Object { $_ -match "END"   } | Select-Object -Last 1
    if ($started -match "START\s+(\S+)") {
        $live = $Matches[1]
        $isLive = $true
        if ($ended -match "END\s+(\S+)") { if ($Matches[1] -eq $live) { $isLive = $false } }
        if ($isLive) {
            $lg = Join-Path $repo ("tmp\leg_{0}_{1}w.log" -f $live, $w)
            $t = "?"
            if (Test-Path -LiteralPath $lg) {
                $hit = Select-String -Path $lg -Pattern "t=(\d+)s" | Select-Object -Last 1
                if ($hit) { $t = $hit.Matches[0].Groups[1].Value }
            }
            "..   {0,-22} LIVE t={1}s" -f $live, $t
        }
    }
    if (Test-Path -LiteralPath (Join-Path $repo ("tmp\{0}_{1}w_ALLDONE.txt" -f $name, $w))) {
        Write-Host "    [sweep complete]"
    }
}

Show-Sweep "bake420"  420
Show-Sweep "bake420b" 420
Show-Sweep "bake420c" 420
Show-Sweep "bake420d" 420
Show-Sweep "bake720"  720

Write-Host ""
Write-Host ("gpu: " + (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader))
Write-Host ("now: " + (Get-Date -Format 'HH:mm:ss'))
