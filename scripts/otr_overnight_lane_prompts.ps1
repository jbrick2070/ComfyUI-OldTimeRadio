# OVERNIGHT -- one act per lane, to put the new per-lane prompts on screen.
#
# WHY THIS EXISTS. Every video lane now composes its own prompt through the
# compose_prompt seam, and only ONE of them (ltx25_foley_plus) has actually been
# watched. The rest shipped proven-reachable by log line, which is not the same
# as proven-good by eye. This queue renders one act on each so there is
# something in otr/obs/ to judge in the morning.
#
# A WATCHER TIMEOUT IS NOT A RENDER DEATH, AND THIS SCRIPT NOW KNOWS IT
# (2026-08-28, learned live at 04:14:40). The canonical runner's own
# classifier (Bible 12.140, cebe7c75) printed "...BUT THE RENDER IS STILL
# ALIVE: the server reports 1 running / 0 pending ... the episode should
# still publish to otr/obs on its own" -- and the first version of this queue
# read ONLY the exit code, declared the leg failed, and the NEXT leg's boot
# reset killed a healthy foley render one second later. A correct diagnostic
# printed into a log changes nothing when the caller acts on the exit code
# alone. So now: a non-zero runner exit with nothing in obs enters a WAIT
# loop -- publish, or a genuinely idle GPU, is what ends a leg. Never an
# impatient watcher.
#
# ORDER IS DELIBERATE -- most-unseen first. If the night dies at leg 3, the legs
# that ran are the ones worth having.
#
# THE 5-MINUTE RULE APPLIES PER LEG (operator): a leg with nothing in otr/obs/
# after it truly finishes is a FAIL to be read in the log, not explained away.
#
# Every leg loads workflows/otr_canonical.json -- there is no second path.
# The harness resets the box itself between legs; do not add external kills.
param(
    [string[]]$Profiles = @(
        "otr_ltx25_high_mime",
        "otr_ltx25_high_foley_plus",
        "otr_ltx25_high_video",
        "otr_w45_minimax_h3_video",
        "otr_w45_minimax_h3_audio_in",
        "otr_16gb_ltx_audio_in"
    ),
    # 3h. Mime took 1h39m and foley exceeded 2h -- 7200 was sized to the
    # fastest lane, which is exactly how a watcher under-times a slow one.
    [int]$LegTimeout = 10800,
    # How long the STILL-ALIVE wait loop will trail a timed-out watcher
    # before giving up on the render publishing by itself.
    [int]$TrailMinutes = 150
)
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

# THE REAL OBS IS UNDER COMFYUI'S OUTPUT BASE, NOT UNDER THE REPO. The repo has
# an `otr\` directory too, which is exactly what makes the mistake convincing --
# it holds 0 mp4 files while the real one holds twenty. CLAUDE.md section 6.
$OBS = "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"

function Get-ObsCount {
    @(Get-ChildItem $OBS -Filter *.mp4 -ErrorAction SilentlyContinue).Count
}

function Get-GpuUtil {
    try {
        $raw = (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) | Select-Object -First 1
        return [int]($raw.Trim())
    } catch { return -1 }   # unreadable -> treated as "do not declare dead"
}

$before = Get-ObsCount
Write-Host "[OVERNIGHT] obs holds $before mp4 files at start"
Write-Host "[OVERNIGHT] legs: $($Profiles -join ', ')  timeout=${LegTimeout}s trail=${TrailMinutes}m"

$n = 0
foreach ($profile in $Profiles) {
    $n++
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "LEG $n/$($Profiles.Count)  $profile"
    Write-Host "  started $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "================================================================"

    & powershell -ExecutionPolicy Bypass -File "scripts\otr_headless_canonical.ps1" `
        -Profile $profile -Acts 1 -Timeout $LegTimeout
    $rc = $LASTEXITCODE

    $now = Get-ObsCount
    if (($now - $before) -eq 0 -and $rc -ne 0) {
        # The watcher gave up, or the runner really died -- the exit code
        # cannot tell those apart (Bible 12.140). Ask the machine: a render
        # in flight keeps the GPU busy; a finished-but-resident server idles
        # at ~1% (CLAUDE.md section 5). Wait for a publish or for real
        # quiet; only then call it.
        Write-Host "[LEG $n] runner exit=$rc with nothing in obs -- trailing the render instead of trusting the watcher"
        $idleStreak = 0
        $deadline = (Get-Date).AddMinutes($TrailMinutes)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 60
            $now = Get-ObsCount
            if (($now - $before) -gt 0) {
                Write-Host "[LEG $n] the render published on its own at $(Get-Date -Format 'HH:mm:ss') -- the watcher was the only thing that quit"
                break
            }
            $util = Get-GpuUtil
            if ($util -ge 0 -and $util -lt 5) { $idleStreak++ } else { $idleStreak = 0 }
            if ($idleStreak -ge 5) {
                Write-Host "[LEG $n] GPU idle 5 consecutive minutes with no publish -- the render is genuinely dead"
                break
            }
        }
        $now = Get-ObsCount
    }

    $landed = $now - $before
    $before = $now
    if ($landed -gt 0) {
        $newest = Get-ChildItem $OBS -Filter *.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $mb = [math]::Round($newest.Length / 1MB, 1)
        Write-Host "LEG $n RESULT: exit=$rc  PUBLISHED -> $($newest.Name)  ($mb MB)"
    } else {
        Write-Host "LEG $n RESULT: exit=$rc  *** NOTHING REACHED OBS -- THIS LEG FAILED ***"
    }
    Write-Host "  finished $(Get-Date -Format 'HH:mm:ss')"
}

Write-Host ""
Write-Host "OVERNIGHT QUEUE COMPLETE -- obs now holds $before mp4 files"
