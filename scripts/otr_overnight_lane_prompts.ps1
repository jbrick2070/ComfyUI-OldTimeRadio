# OVERNIGHT -- one act per lane, to put the new per-lane prompts on screen.
#
# WHY THIS EXISTS. Every video lane now composes its own prompt through the
# compose_prompt seam, and only ONE of them (ltx25_foley_plus) has actually been
# watched. The rest shipped proven-reachable by log line, which is not the same
# as proven-good by eye. This queue renders one act on each so there is
# something in otr/obs/ to judge in the morning.
#
# ORDER IS DELIBERATE -- most-unseen first. If the night dies at leg 3, the legs
# that ran are the ones worth having: mime has NEVER been seen, foley has been
# seen once, and the audio-in lanes are the ones a decision is pending on.
#
# THE 5-MINUTE RULE APPLIES PER LEG (operator): a leg with nothing in otr/obs/
# after it finishes is a FAIL to be read in the log, not a result to explain
# away. Each leg's obs state is checked and recorded below as it lands.
#
# Every leg loads workflows/otr_canonical.json -- there is no second path.
# The harness resets the box itself between legs; do not add external kills.
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

# THE REAL OBS IS UNDER COMFYUI'S OUTPUT BASE, NOT UNDER THE REPO. The repo has
# an `otr\` directory too, which is exactly what makes the mistake convincing --
# it holds 0 mp4 files while the real one holds twenty. CLAUDE.md section 6 says
# so outright: "If otr\ actually resolves to ComfyUI's real output\ base on
# disk, use that base." Pointing at the repo copy made every leg report
# "NOTHING REACHED OBS" while it was in fact publishing perfectly.
$OBS = "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
$before = @(Get-ChildItem $OBS -Filter *.mp4 -ErrorAction SilentlyContinue).Count
Write-Host "[OVERNIGHT] obs holds $before mp4 files at start"

$legs = @(
    @{ p = "otr_ltx25_high_mime";        why = "NEVER SEEN -- new mime prompt, scored ambience, no speech" },
    @{ p = "otr_ltx25_high_foley_plus";  why = "seen once at 2 acts -- confirm at 1 act" },
    @{ p = "otr_ltx25_high_video";       why = "plain ltx25 video lane, new motion prompt" },
    @{ p = "otr_w45_minimax_h3_video";   why = "H3 silent -- this is the lane that logged prompt source=m4" },
    @{ p = "otr_w45_minimax_h3_audio_in";why = "H3 audio-in -- separate prompt from silent, per operator" },
    @{ p = "otr_16gb_ltx_audio_in";      why = "the one lane still on the shared path -- baseline before it moves" }
)

$n = 0
foreach ($leg in $legs) {
    $n++
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "LEG $n/$($legs.Count)  $($leg.p)"
    Write-Host "  $($leg.why)"
    Write-Host "  started $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "================================================================"

    & powershell -ExecutionPolicy Bypass -File "scripts\otr_headless_canonical.ps1" `
        -Profile $leg.p -Acts 1 -Timeout 7200
    $rc = $LASTEXITCODE

    $now = @(Get-ChildItem $OBS -Filter *.mp4 -ErrorAction SilentlyContinue).Count
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
