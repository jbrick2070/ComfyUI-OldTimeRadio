# Bakeoff status across every sweep: bake420 (10 banks) + bake420b (codex re-leg)
# + bake720. One line per finished leg, plus the live leg's elapsed time.
param([int]$Words = 420)

$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

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
            if ($ok) {
                $legLog = Join-Path $repo ("tmp\leg_{0}_{1}w.log" -f $bank, $w)
                $published = $false
                if (Test-Path -LiteralPath $legLog) {
                    $portHit = Select-String -Path $legLog -Pattern "port=(\d+)" |
                               Select-Object -Last 1
                    if ($portHit) {
                        $srv = Join-Path $repo ("tmp\otr_headless_{0}.log" -f $portHit.Matches[0].Groups[1].Value)
                        if (Test-Path -LiteralPath $srv) {
                            if (Select-String -Path $srv -Pattern "obs_publish OK" -Quiet) {
                                $published = $true
                            }
                        }
                    }
                }
                if (-not $published) {
                    "{0} {1,-22} {2,6}s  {3}" -f "PHNT", $bank, $Matches[4], `
                        "<- RESULT SUCCESS but NO obs_publish -- NO EPISODE. Re-leg required."
                    continue
                }
            }

            $mark = if ($ok) { "OK  " } else { "FAIL" }
            # A red leg that a later re-leg turned green is HISTORY, not a problem.
            # Say so, or a stale red trains the eye to ignore reds that are real.
            $note = ""
            if (-not $ok) {
                $releg = Join-Path $repo ("tmp\{0}b_{1}w_summary.txt" -f $name, $w)
                if (Test-Path -LiteralPath $releg) {
                    $fixed = Get-Content -LiteralPath $releg |
                             Where-Object { $_ -match ("END\s+" + [regex]::Escape($bank) + "\s.*RESULT SUCCESS") }
                    if ($fixed) { $mark = "OK  "; $note = "  (re-leg green; original FAIL superseded)" }
                    else { $note = "  <- re-leg pending" }
                } else { $note = "  <- re-leg pending" }
            }
            "{0} {1,-22} {2,6}s{3}" -f $mark, $bank, $Matches[4], $note
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
Show-Sweep "bake720"  720

Write-Host ""
Write-Host ("gpu: " + (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader))
Write-Host ("now: " + (Get-Date -Format 'HH:mm:ss'))
