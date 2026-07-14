# Wait for bake420c (scifi_codex vs the mux fix) to drain, then re-leg scifi_fable2
# against the conversational-wrapper strip.
#
# fable2's critic fix WORKED: the critic now passes and returns notes, which advanced
# the lane to a pass that had never run before -- the REVISION. Aion opened it with
# "I will now produce the complete revised radio play episode, applying all critic
# notes..." -> SKELETON_BREAK (TITLE is not the first line). The markup ladder is a
# raw-text pass with no schema to constrain shape, so it re-asked five times and the
# model announced itself every single time. Python now strips the wrapper: the
# format's own skeleton is TITLE: first, END. last, so anything outside that is not
# script.
#
# ONE GPU -- must not overlap bake420c.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

$cDone = Join-Path $repo "tmp\bake420c_420w_ALLDONE.txt"
$deadline = (Get-Date).AddHours(4)
while (-not (Test-Path -LiteralPath $cDone) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
}

"[chain-d] bake420c drained at $(Get-Date -Format 'HH:mm:ss'); re-legging scifi_fable2" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg_d.log") -Encoding utf8

# IN-PROCESS call with a REAL array -- `powershell -File` cannot carry an array.
& (Join-Path $repo "tmp\_sweep.ps1") -Banks @("scifi_fable2") -Words 420 -SweepName "bake420d"

"[chain-d] done at $(Get-Date -Format 'HH:mm:ss')" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg_d.log") -Encoding utf8 -Append
