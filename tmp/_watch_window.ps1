# Live bakeoff monitor -- runs in its OWN visible window. Read-only: it never
# touches the render, it just re-prints the status every 20 seconds.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo
$Host.UI.RawUI.WindowTitle = "OTR BAKEOFF -- live monitor"

while ($true) {
    Clear-Host
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host "  OTR BAKEOFF -- 420 then 720, all 10 banks" -ForegroundColor Cyan
    Write-Host "  refreshes every 20s   |   Ctrl+C to close" -ForegroundColor DarkGray
    Write-Host "=========================================================" -ForegroundColor Cyan

    & (Join-Path $repo "tmp\_status_bake.ps1")

    # The live leg's most recent server activity -- proves it is MOVING, not hung.
    $live = Get-ChildItem (Join-Path $repo "tmp\leg_*_420w.log"),
                          (Join-Path $repo "tmp\leg_*_720w.log") -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime | Select-Object -Last 1
    if ($live) {
        $age = [int]((Get-Date) - $live.LastWriteTime).TotalSeconds
        Write-Host ""
        Write-Host ("live leg log : " + $live.Name)
        $color = if ($age -gt 300) { "Red" } else { "Green" }
        Write-Host ("last write   : ${age}s ago") -ForegroundColor $color
        if ($age -gt 300) {
            Write-Host "  ^ NO HEARTBEAT FOR 5+ MIN -- possible stall" -ForegroundColor Red
        }
        # "status=pending" only means the API is still waiting. The REAL progress
        # is in the server log: which pass the writer is on, and whether TTS /
        # render / obs_publish have started.
        $port = $null
        $hit = Select-String -Path $live.FullName -Pattern "server pid=\d+ port=(\d+)" |
               Select-Object -Last 1
        if ($hit) { $port = $hit.Matches[0].Groups[1].Value }

        if ($port) {
            $srv = Join-Path $repo ("tmp\otr_headless_{0}.log" -f $port)
            if (Test-Path -LiteralPath $srv) {
                Write-Host ""
                Write-Host "--- where it actually is (server log) ---" -ForegroundColor Yellow
                $marks = Select-String -Path $srv -Pattern `
                    "attempt \d+/\d+|coerced \d+ over-long|WORD_BUDGET|Prompt executed|obs_publish|TTS|render|RESULT|Exception|ERROR" |
                    Select-Object -Last 6
                foreach ($m in $marks) {
                    $t = $m.Line -replace "\x1b\[[0-9;]*m", ""   # strip ANSI colour
                    if ($t.Length -gt 150) { $t = $t.Substring(0, 150) }
                    $c = if ($t -match "ERROR|Exception") { "Red" }
                         elseif ($t -match "obs_publish|Prompt executed") { "Green" }
                         else { "Gray" }
                    Write-Host ("  " + $t.Trim()) -ForegroundColor $c
                }
            }
        }

        Write-Host ""
        Get-Content -LiteralPath $live.FullName -Tail 1 | ForEach-Object {
            Write-Host ("  " + $_) -ForegroundColor DarkGray
        }
    }

    # Failed legs, split into the two kinds that MATTER DIFFERENTLY:
    #   TRIAGED  -- root-caused, fix shipped, re-leg queued. Nothing for you to do.
    #   NEEDS TRIAGE -- new, unexplained. THIS is the one that wants a human.
    # A bare red FAIL cannot tell those apart, and a stale red that never clears
    # is how a real failure gets ignored.
    $triaged = @{
        "scifi_codex|420" = "P4 audit died on its own 240-char rationale -> FIXED 80e30c2e, re-leg queued"
    }

    $fails = Get-ChildItem (Join-Path $repo "tmp\bake*_summary.txt") -ErrorAction SilentlyContinue |
             ForEach-Object { Get-Content $_ } |
             Where-Object { $_ -match "RESULT FAIL" }

    # A later SUCCESS for the same bank+length supersedes an earlier failure.
    $passed = @{}
    Get-ChildItem (Join-Path $repo "tmp\bake*_summary.txt") -ErrorAction SilentlyContinue |
        ForEach-Object { Get-Content $_ } |
        Where-Object { $_ -match "END\s+(\S+)\s+(\d+)w.*RESULT SUCCESS" } |
        ForEach-Object {
            if ($_ -match "END\s+(\S+)\s+(\d+)w") { $passed[($Matches[1] + "|" + $Matches[2])] = $true }
        }

    if ($fails) {
        $open = @()
        $known = @()
        foreach ($f in $fails) {
            if ($f -notmatch "END\s+(\S+)\s+(\d+)w") { continue }
            $key = $Matches[1] + "|" + $Matches[2]
            if ($passed[$key]) { continue }                 # re-legged green: gone
            if ($triaged.ContainsKey($key)) { $known += ,@($key, $triaged[$key]) }
            else { $open += $key }
        }

        if ($known.Count -gt 0) {
            Write-Host ""
            Write-Host "--- TRIAGED (fix shipped, re-leg queued -- no action) ---" -ForegroundColor Yellow
            foreach ($k in $known) {
                Write-Host ("  " + $k[0].Replace("|", "  ") + "w") -ForegroundColor Yellow
                Write-Host ("      " + $k[1]) -ForegroundColor DarkYellow
            }
        }
        if ($open.Count -gt 0) {
            Write-Host ""
            Write-Host "*** NEEDS TRIAGE -- NEW FAILURE ***" -ForegroundColor Red
            foreach ($o in $open) {
                Write-Host ("  " + $o.Replace("|", "  ") + "w") -ForegroundColor Red
            }
        }
    }

    if (Test-Path (Join-Path $repo "tmp\bake720_GATE_BLOCKED.txt")) {
        Write-Host ""
        Write-Host "*** 720 GATE BLOCKED ***" -ForegroundColor Red
        Get-Content (Join-Path $repo "tmp\bake720_GATE_BLOCKED.txt") |
            ForEach-Object { Write-Host ("  " + $_) -ForegroundColor Red }
    }

    Start-Sleep -Seconds 20
}
