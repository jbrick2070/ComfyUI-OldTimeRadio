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
            $mark = if ($Matches[3] -eq "SUCCESS") { "OK  " } else { "FAIL" }
            "{0} {1,-22} {2,6}s" -f $mark, $Matches[1], $Matches[4]
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
